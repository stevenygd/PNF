import torch
import trimesh
import numpy as np
from torch.utils import data
from pykdtree.kdtree import KDTree
from torch.utils.data import Dataset
from datasets.single_image_datasets import init_np_seed


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        # we lose texture information here
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                  for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh

    mesh.fix_normals()
    return mesh


class SingleShape(Dataset):

    def __init__(self, cfg, cfgdata):
        self.cfg = cfg
        self.cfgdata = cfgdata
        self.path = cfg.path
        self.num_coarse_pnts = cfgdata.num_coarse_pnts
        self.num_fine_pnts = cfgdata.num_fine_pnts
        self.coarse_scale = cfgdata.coarse_scale
        self.fine_scale = cfgdata.fine_scale
        self.noise_type = getattr(self.cfgdata, "noise_type", "lap")
        self.length = int(cfgdata.length)
        self.dim = 3
        self.mult_scalar = float(getattr(cfg, "mult_scalar", 1))
        self.sample_method = getattr(cfg, "sample_method", "from_vertex")
        self.tree_type = getattr(cfg, "tree_type", "from_vertex")

        # Load and normalize meshes
        mesh = as_mesh(trimesh.load_mesh(self.path))
        verts = self.normalize(mesh.vertices.reshape(-1, 3))
        print(verts.max(), verts.min(), verts.mean())
        self.mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces)
        self.mesh.fix_normals()
        self.return_mesh = getattr(cfgdata, "return_mesh", False)
        self.return_pcl = getattr(cfgdata, "return_pcl", False)

        # Load KD Tree
        self.make_tree()
        # self.v = np.array(self.mesh.vertices)
        # self.n = np.array(self.mesh.vertex_normals)
        # n_norm = (np.linalg.norm(self.n, axis=-1)[:, None])
        # n_norm[n_norm == 0] = 1.
        # self.n = self.n / n_norm
        # self.tree = KDTree(self.v)

        if self.return_pcl:
            npoints = int(cfgdata.num_gtr_pcl)
            self.gtr_pcl, fidx = trimesh.sample.sample_surface(
                self.mesh, npoints)
            self.gtr_sfn = self.mesh.face_normals[fidx]

    def make_tree(self):
        if self.tree_type == "from_vertex":
            self.v = np.array(self.mesh.vertices)
            self.n = np.array(self.mesh.vertex_normals)
        elif self.tree_type == "from_sample":
            print("Tree_type= from_sample")
            nv = self.mesh.vertices.shape[0]
            v, fid = trimesh.sample.sample_surface(self.mesh, nv)
            self.v = np.array(v)
            self.n = self.mesh.face_normals[fid]
        else:
            raise NotImplementedError

        n_norm = (np.linalg.norm(self.n, axis=-1)[:, None])
        n_norm[n_norm == 0] = 1.
        self.n = self.n / n_norm
        self.tree = KDTree(self.v)

    def normalize(self, coords):
        coords -= np.mean(coords, axis=0, keepdims=True)
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
        # [0, 0.9]
        coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
        # [-0.45, 0.45]
        coords -= 0.45
        # NOTE: we use -1, 1 so we scale it back to [-0.9, 0.9]
        coords *= self.mult_scalar
        return coords

    def sample_surface(self, npoints, scale=2e-6):
        if self.sample_method in ['from_face', 'from_mesh']:
            points = self.mesh.sample(npoints)
        elif self.sample_method == 'from_vertex':
            idx = np.random.randint(0, self.v.shape[0], npoints)
            points = self.v[idx]
        else:
            raise ValueError
        if self.noise_type == 'lap':
            points += np.random.laplace(scale=scale, size=points.shape)
        elif self.noise_type == 'gau':
            points += np.random.normal(scale=scale, size=points.shape)
        else:
            raise NotImplemented

        points[points > 0.5] -= 1
        points[points < -0.5] += 1

        # use KDTree to get distance to surface and estimate the normal
        sdf, idx = self.tree.query(points, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((points - self.v[idx][:, 0]) * avg_normal, axis=-1)

        points = torch.from_numpy(points).float().reshape(-1, 3)
        sdf =  torch.from_numpy(sdf).float().reshape(-1, 1)
        return points, sdf

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # Sample points, Coarse
        c_xyz, c_sdf = self.sample_surface(
            self.num_coarse_pnts, scale=self.coarse_scale)

        # Sample points, Fine
        f_xyz, f_sdf = self.sample_surface(
            self.num_fine_pnts, scale=self.fine_scale)

        # [xyz] range from [-1, 1]
        res = {
            'idx': torch.tensor(idx).long(),
            # Coarse
            'c_xyz': c_xyz,
            'c_sdf': c_sdf,
            # Fine
            'f_xyz': f_xyz,
            'f_sdf': f_sdf,
        }
        if self.return_mesh:
            res['mesh'] = self.mesh
        if self.return_pcl:
            res['pcl'] = self.gtr_pcl
            res['sfn'] = self.gtr_sfn
        return res


def mesh_collate_fn(batch):
    out_batch = {}
    for d in batch:
        for k, v in d.items():
            if k not in out_batch:
                out_batch[k] = []
            out_batch[k].append(v)

    for k in out_batch.keys():
        if k != 'mesh':
            out_batch[k] = torch.stack(out_batch[k])

    return out_batch


def get_data_loaders(cfg, args):
    tr_dataset = SingleShape(cfg, cfg.train)
    te_dataset = SingleShape(cfg, cfg.val)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=getattr(cfg.train, "num_workers", cfg.num_workers),
        drop_last=True, worker_init_fn=init_np_seed,
        collate_fn=mesh_collate_fn)
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=cfg.val.batch_size, shuffle=False,
        num_workers=getattr(cfg.val, "num_workers", cfg.num_workers),
        drop_last=False, worker_init_fn=init_np_seed,
        collate_fn=mesh_collate_fn)

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders
