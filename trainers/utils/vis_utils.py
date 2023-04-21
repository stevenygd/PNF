from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imageio
import PIL

import cv2
import math
import tqdm
import cmapy
import torch
import trimesh
import skimage
import matplotlib
import numpy as np
import skimage.measure
import mcubes

matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D  # nofa: #401


# Visualization
def visualize_point_clouds_3d(pcl_lst, title_lst=None):
    # pts, gtr, inp):
    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax1 = fig.add_subplot(1, len(pcl_lst), 1 + idx, projection='3d')
        ax1.set_title(title)
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


# Visualization
def visualize_point_clouds_3d_scan(pcl_lst, title_lst=None):
    # pts, gtr, inp):
    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax1 = fig.add_subplot(1, len(pcl_lst), 1 + idx, projection='3d')
        ax1.set_title(title)
        ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 1], s=0.5)
        # print(min(pts[:, 0]), max(pts[:, 0]), min(pts[:, 1]), max(pts[:, 1]), min(pts[:, 2]), max(pts[:, 2]))
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(-1, 1)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


# Visualization moving field and likelihood
def get_grid(x, k=10):
    # TODO: set the range of field
    # x = self.get_prior(
    #        1, num_points, self.cfg.models.scorenet.dim).cuda()
    # ind_x = np.arange(x[:,:,0].min(),x[:,:,0].max(),3/k)
    # ind_y = np.arange(x[:,:,1].min(),x[:,:,0].max(),3/k)
    ind_x = np.arange(-1.5, 1.5, 3 / k)
    ind_y = np.arange(-1.5, 1.5, 3 / k)
    X, Y = np.meshgrid(ind_x, ind_y)
    X = torch.tensor(X).view(k * k).to(x)
    Y = torch.tensor(Y).view(k * k).to(x)

    point_grid = torch.ones((1, k * k, 2), dtype=torch.double).to(x)
    point_grid[0, :, 1] = point_grid[0, :, 1] * X
    point_grid[0, :, 0] = point_grid[0, :, 0] * Y
    point_grid = point_grid.float()
    point_grid = point_grid.expand(x.size(0), -1, -1)
    return point_grid


def visualize_point_clouds_2d_overlay(pcl_lst, title_lst=None, path=None):
    # pts, gtr, inp):
    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    fig = plt.figure(figsize=(3, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title(title_lst[0])
    for idx, pts in enumerate(pcl_lst):
        ax1.scatter(pts[:, 0], pts[:, 1], s=5)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))
    if path:
        plt.savefig(path)
    plt.close()
    return res


def visualize_field(gtr, grid, field, k, label='field'):
    grid_ = np.reshape(grid.cpu().detach().numpy(), (1, k * k, 2))
    if field.size(-1) == 2:
        field = np.reshape(field.cpu().detach().numpy(), (1, k * k, 2))
        fig = plt.figure(figsize=(int(k / 100) * 2, int(k / 100)))
        plt.title(label)
        cs = fig.add_subplot(1, 2, 1)
        field_val = np.sqrt(np.reshape((field ** 2).sum(axis=-1), (1, k * k, 1)))
    else:
        fig = plt.figure(figsize=(int(k / 100), int(k / 100)))
        plt.title(label)
        cs = fig.add_subplot(1, 1, 1)
        field_val = np.reshape(field.cpu().detach().numpy(), (1, k * k, 1))

    gt = gtr.cpu().detach().numpy()

    for i in range(np.shape(field_val)[0]):
        # cs = fig.add_subplot(1, 2, 1)
        X = np.reshape(grid_[i, :, 0], (k, k))
        Y = np.reshape(grid_[i, :, 1], (k, k))
        cs.contourf(X, Y, np.reshape(field_val[i, :], (k, k)), 20,
                    vmin=min(field_val[i, :]), vmax=max(field_val[i, :]),
                    cmap=cm.coolwarm)
        print(min(field_val[i, :]), max(field_val[i, :]))
        m = plt.cm.ScalarMappable(cmap=cm.coolwarm)
        m.set_array(np.reshape(field_val[i, :], (k, k)))
        m.set_clim(min(field_val[i, :]), max(field_val[i, :]))

    if np.shape(field)[-1] == 2:
        for i in range(np.shape(field_val)[0]):
            ax = fig.add_subplot(1, 2, 2)
            scale = 20
            indx = np.array([np.arange(0, k, scale) + t * k for t in range(0, k, scale)])
            X = np.reshape(grid_[i, indx, 0], int(k * k / scale / scale))
            Y = np.reshape(grid_[i, indx, 1], int(k * k / scale / scale))
            u = np.reshape(field[i, indx, 0], int(k * k / scale / scale))
            v = np.reshape(field[i, indx, 1], int(k * k / scale / scale))

            color = np.sqrt(v ** 2 + u ** 2)
            field_norm = field / field_val
            u = np.reshape(field_norm[i, indx, 0], int(k * k / scale / scale))
            v = np.reshape(field_norm[i, indx, 1], int(k * k / scale / scale))
            ax.quiver(X, Y, u, v, color, alpha=0.8, cmap=cm.coolwarm)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_aspect('equal')
            ax.scatter(gt[:, 0], gt[:, 1], s=1, color='r')

    return matplotlib_fig2img(fig)


def matplotlib_fig2img(fig, transpose=True):
    fig.canvas.draw()
    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)  # (h, w, c)
    if transpose:
        res = np.transpose(res, (2, 0, 1))  # (c, h, w)
    plt.close()
    return res


def visualize_procedure(sigmas, fig_list, gtr, num_vis, cfg, name="Rec_gt"):
    all_imgs = []
    sigmas = np.append([0], sigmas)
    for idx in range(num_vis):
        img = visualize_point_clouds_3d(
            [fig_list[i][idx] for i in
             range(0, len(fig_list), 1)] + [gtr[idx]],
            [(name + " step" +
              str(i * int(getattr(cfg.inference, "num_steps", 5))) +
              " sigma%.3f" % sigmas[i])
             for i in range(0, len(fig_list), 1)] + ["gt shape"])
        all_imgs.append(img)
    img = np.concatenate(all_imgs, axis=1)
    return img


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)


def vis_img_lap(lapl):
    vis = torch.from_numpy(cv2.cvtColor(cv2.applyColorMap(
        to_uint8(rescale_img(lin2img(lapl), perc=2).permute(
            0, 2, 3, 1).squeeze(0).detach().cpu().numpy()),
        cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB))
    return vis


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def imf2mesh(imf, res=256, threshold=0.0, batch_size = 10000, verbose=True,
             use_double=False, normalize=False, norm_type='res',
             return_stats=False, bound=1.):
    xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
    grid = np.concatenate([
        ys[..., np.newaxis],
        xs[..., np.newaxis],
        zs[..., np.newaxis]
    ], axis=-1).astype(np.float)
    grid = (grid / float(res) - 0.5) * 2 * bound
    grid = grid.reshape(-1, 3)

    dists_lst = []
    pbar = range(0, grid.shape[0], batch_size)
    if verbose:
        pbar = tqdm.tqdm(pbar)
    for i in pbar:
        sidx, eidx = i, i + batch_size
        eidx = min(grid.shape[0], eidx)
        with torch.no_grad():
            xyz = torch.from_numpy(
                grid[sidx:eidx, :]).float().cuda().view(1, -1, 3)
            if use_double:
                xyz = xyz.double()
            distances = imf(xyz)
            distances = distances.cpu().numpy()
        dists_lst.append(distances.reshape(-1))

    dists = np.concatenate(
        [x.reshape(-1, 1) for x in dists_lst], axis=0).reshape(-1)
    field = dists.reshape(res, res, res)
    try:
        vert, face, _, _ = skimage.measure.marching_cubes(
            field, level=threshold)

        if normalize:
            if norm_type == 'norm':
                center = vert.mean(axis=0).view(1, -1)
                vert_c = vert - center
                length = np.linalg.norm(vert_c, axis=-1).max()
                vert = vert_c / length
            elif norm_type == 'res':
                vert = (vert * 2 - res) / float(res) * bound
            else:
                raise ValueError
        new_mesh = trimesh.Trimesh(vertices=vert, faces=face)
    except ValueError as e:
        print(field.max(), field.min())
        print(e)
        new_mesh = None
    except RuntimeError as e:
        print(field.max(), field.min())
        print(e)
        new_mesh = None

    if return_stats:
        if new_mesh is not None:
            area = new_mesh.area
            vol = (field < threshold).astype(np.float).mean() * (2 * bound) ** 3
        else:
            area = 0
            vol = 0
        return new_mesh, {
            'vol': vol,
            'area': area
        }

    return new_mesh


def make_2d_grid(r, add_noise=False):
    xs, ys = torch.meshgrid(torch.arange(r), torch.arange(r), indexing='ij')
    xy = torch.cat([ys.reshape(-1, 1), xs.reshape(-1, 1)], dim=-1).float()
    if add_noise:
        xy += torch.rand_like(xy)
    else:
        xy += 0.5
    xy = (xy / float(r) - 0.5) * 2
    return xy


def make_3d_grid(r, add_noise=False):
    xs, ys, zs = torch.meshgrid(
        torch.arange(r), torch.arange(r), torch.arange(r))
    xyz = torch.cat([
        ys.reshape(-1, 1), xs.reshape(-1, 1), zs.reshape(-1, 1)
    ], dim=-1).float()
    if add_noise:
        xyz += torch.rand_like(xyz)
    else:
        xyz += 0.5
    xy = (xyz / float(r) - 0.5) * 2
    return xy



def imf2img(imf, res=256, add_noise=False, batch_size=10000, threshold=0.,
            verbose=False, grid=None, return_stats=False, bound=1):
    if grid is None:
        grid = make_2d_grid(res, add_noise=add_noise).view(-1, 2)
    dists_lst = []
    pbar = range(0, grid.shape[0], batch_size)
    if verbose:
        pbar = tqdm.tqdm(pbar)
    for i in pbar:
        sidx, eidx = i, i + batch_size
        eidx = min(grid.shape[0], eidx)
        with torch.no_grad():
            xyz = grid[sidx:eidx, :].cuda().view(1, -1, 2)
            n = xyz.size(1)
            distances = imf(xyz)
            distances = distances.cpu().numpy()
        dists_lst.append(distances.reshape(n, -1))
    dists = np.concatenate(
        [x for x in dists_lst], axis=0)
    img = dists.reshape(res, res, -1)
    if return_stats:
        area = (img < threshold).astype(np.float).mean() * 2 ** 2
        contours = skimage.measure.find_contours(
            img.reshape(res, res), level=threshold)
        total_length = 0
        for vert in contours:
            n_v_c = vert.shape[0]
            n_v_c_idx = np.array(
                (np.arange(n_v_c).astype(np.int) + 1) % n_v_c).astype(np.int)
            v_next = vert[n_v_c_idx, :]
            v_next = v_next.reshape(n_v_c, 2)
            diff = (vert - v_next) / float(res)
            dist = np.linalg.norm(diff, axis=-1).sum()
            total_length += dist
        return img, {
            'area' : area,
            'len': total_length,
            'contours': contours
        }
    return img



def imf2mesh_multires(
        imf, levels=None, res=256, threshold=0.0, batch_size = 10000, verbose=True,
        use_double=False, normalize=False, norm_type='res',
        return_stats=False, bound=1.):

    xs, ys, zs = np.meshgrid(np.arange(res), np.arange(res), np.arange(res))
    grid = np.concatenate([
        ys[..., np.newaxis],
        xs[..., np.newaxis],
        zs[..., np.newaxis]
    ], axis=-1).astype(np.float)
    grid = (grid / float(res) - 0.5) * 2 * bound
    grid = grid.reshape(-1, 3)

    print("Gather all evaluations")
    dists_dict = {}
    pbar = range(0, grid.shape[0], batch_size)
    if verbose:
        pbar = tqdm.tqdm(pbar)
    for i in pbar:
        sidx, eidx = i, i + batch_size
        eidx = min(grid.shape[0], eidx)
        with torch.no_grad():
            xyz = torch.from_numpy(
                grid[sidx:eidx, :]).float().cuda().view(1, -1, 3)
            if use_double:
                xyz = xyz.double()
            distances = imf(xyz)
        if levels is None:
            levels = list(range(len(distances)))
        for level in levels:
            if level not in dists_dict:
                dists_dict[level] = []
            dist_level = distances[level].cpu().numpy()
            dists_dict[level].append(dist_level.reshape(-1))

    print("Meshing all levels")
    mesh_dict, mesh_stat_dict = {}, {}
    for level in levels:
        print("\tLevel %d" % level)
        dists_level = np.concatenate(
            [x.reshape(-1, 1) for x in dists_dict[level]], axis=0).reshape(-1)
        field = dists_level.reshape(res, res, res)
        try:
            # TODO: there is a mask option which we can used to mask out regions
            #       and save compute time
            # vert, face, _, _ = skimage.measure.marching_cubes(
            #     field, level=threshold)
            # Newer version
            vert, face, _, _ = skimage.measure.marching_cubes_lewiner(
                field, level=threshold)
            # vert, face = mcubes.marching_cubes(-field, threshold)

            if normalize:
                if norm_type == 'norm':
                    center = vert.mean(axis=0).view(1, -1)
                    vert_c = vert - center
                    length = np.linalg.norm(vert_c, axis=-1).max()
                    vert = vert_c / length
                elif norm_type == 'res':
                    vert = (vert * 2 - res) / float(res) * bound
                else:
                    raise ValueError
            new_mesh = trimesh.Trimesh(vertices=vert, faces=face)
        except ValueError as e:
            print(field.max(), field.min())
            print(e)
            new_mesh = None
        except RuntimeError as e:
            print(field.max(), field.min())
            print(e)
            new_mesh = None
        mesh_dict[level] = new_mesh

        if new_mesh is not None:
            area = new_mesh.area
            vol = (field < threshold).astype(np.float).mean() * (2 * bound) ** 3
        else:
            area = 0
            vol = 0
        mesh_stat_dict[level] = {
            'vol': vol,
            'area': area
        }

    if return_stats:
        return mesh_dict, mesh_stat_dict
    else:
        return mesh_dict


def make_2d_grid(r, add_noise=False):
    """
    Return grid with resolution [r] and bounded within [-1, 1]
    :param r:
    :param add_noise:
    :return:
    """
    xs, ys = torch.meshgrid(torch.arange(r), torch.arange(r))
    xy = torch.cat([xs.reshape(-1, 1), ys.reshape(-1, 1)], dim=-1).float()
    if add_noise:
        xy += torch.rand_like(xy)
    else:
        xy += 0.5
    xy = (xy / float(r) - 0.5) * 2
    return xy


def imf2img(imf, res=256, add_noise=False, batch_size=10000, threshold=0.,
            verbose=False, grid=None, return_stats=False, bound=1):
    if grid is None:
        grid = make_2d_grid(res, add_noise=add_noise).view(-1, 2)
    dists_lst = []
    pbar = range(0, grid.shape[0], batch_size)
    if verbose:
        pbar = tqdm.tqdm(pbar)
    for i in pbar:
        sidx, eidx = i, i + batch_size
        eidx = min(grid.shape[0], eidx)
        with torch.no_grad():
            xyz = grid[sidx:eidx, :].cuda().view(1, -1, 2)
            n = xyz.size(1)
            distances = imf(xyz)
            distances = distances.cpu().numpy()
        dists_lst.append(distances.reshape(n, -1))
    dists = np.concatenate(
        [x for x in dists_lst], axis=0)
    img = dists.reshape(res, res, -1)
    if return_stats:
        area = (img < threshold).astype(np.float).mean() * 2 ** 2
        contours = skimage.measure.find_contours(
            img.reshape(res, res), level=threshold)
        total_length = 0
        for vert in contours:
            n_v_c = vert.shape[0]
            n_v_c_idx = np.array(
                (np.arange(n_v_c).astype(np.int) + 1) % n_v_c).astype(np.int)
            v_next = vert[n_v_c_idx, :]
            v_next = v_next.reshape(n_v_c, 2)
            diff = (vert - v_next) / float(res)
            dist = np.linalg.norm(diff, axis=-1).sum()
            total_length += dist
        return img, {
            'area' : area,
            'len': total_length,
            'contours': contours
        }
    return img


def make_gif(imgs_path, ext='.png'):
    fname_lst = []
    for fname in os.listdir(imgs_path):
        if not fname.endswith(ext):
            continue
        fname_lst.append(os.path.join(imgs_path, fname))
    fname_lst = sorted(fname_lst)

    out_video_name = os.path.join(imgs_path, "video.gif")

    video_out = imageio.get_writer(
        out_video_name, 
        fps=60, 
    )
    for fname in fname_lst:
        img = np.array(PIL.Image.open(fname))
        video_out.append_data(img)
    video_out.close()

    return out_video_name


def visualize_2d_sdf_samples(pts, sdf, xlim=(-1, 1), ylim=(-1, 1), img_transpose=True):
    """
    :param pts: (bs, npts, 2)
    :param sdf:
    :param nrows:
    :return:
    """
    bs, npts = pts.shape[0], pts.shape[1]
    pts = pts.reshape(bs, npts, 2)
    sdf = sdf.reshape(bs, npts)
    nrows = np.ceil(bs / float(4.))
    fig = plt.figure(figsize=(12, 3 * nrows))
    for idx in range(bs):
        plt.subplot(nrows, 4, 1 + idx)
        plt.scatter(pts[idx, :, 0], pts[idx, :, 1], c=sdf[idx], s=1)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
    plt.tight_layout()
    return matplotlib_fig2img(fig, transpose=img_transpose)


def visualize_2d_occ_contour(occ, img_transpose=True, return_fig=False):
    bs, res = occ.shape[0], occ.shape[-1]
    occ = occ.reshape(bs, res, res)
    nrows = np.ceil(bs / float(4.))
    if int(bs) <= 4:
        ncols = int(bs)
    else:
        ncols = 4
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    for idx in range(bs):
        plt.subplot(nrows, ncols, 1 + idx)
        plt.contourf(occ[idx], levels=[0, 0.5], cmap='gray')
        plt.contour(occ[idx], levels=[0, 0.5], colors='r', linewidths=2)
    plt.tight_layout()
    if return_fig:
        return fig
    else:
        return matplotlib_fig2img(fig, transpose=img_transpose)



def visualize_2d_sdf_contour(sdfs, img_transpose=True, return_fig=False):
    bs, res = sdfs.shape[0], sdfs.shape[-1]
    sdfs = sdfs.reshape(bs, res, res)

    nrows = np.ceil(bs / float(4.))
    if int(bs) <= 4:
        ncols = int(bs)
    else:
        ncols = 4
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    for idx in range(bs):
        plt.subplot(nrows, ncols, 1 + idx)
        plt.contour(sdfs[idx])
        plt.contour(sdfs[idx], levels=[-np.inf, 0.], colors='r', linewidths=1)
    plt.tight_layout()
    if return_fig:
        return fig
    else:
        return matplotlib_fig2img(fig, transpose=img_transpose)



def visualize_2d_occ_partbb(
        partc, parts, occs, img_transpose=True, return_fig=False):
    bs, res = occs.shape[0], occs.shape[-1]
    occs = occs.reshape(bs, res, res)
    nparts = partc.shape[1]

    nrows = int(np.ceil(bs / float(4.)))
    if int(bs) <= 4:
        ncols = int(bs)
    else:
        ncols = 4
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    for idx in range(bs):
        ax = plt.subplot(nrows, ncols, 1 + idx)
        ax.contourf(occs[idx], levels=[0.5], cmap='gray')
        ax.contour(occs[idx], levels=[-np.inf, 0.5], colors='r', linewidths=2)
        for parti in range(nparts):
            ci = (partc[idx, parti].reshape(2) + 1) * res * 0.5
            si = parts[idx, parti] * res * 0.5
            xy = ci - si * 0.5
            rect = patches.Rectangle(
                (xy[0], xy[1]), si[0], si[1], color='red', fill=False)
            ax.add_patch(rect)
            ax.scatter([ci[0]], [ci[1]], color='red')

            # The coordinate frame
            rect = patches.Rectangle(
                (0, 0), res, res,
                color='blue', linestyle='--', fill=False)
            ax.add_patch(rect)

    plt.tight_layout()
    if return_fig:
        return fig
    else:
        return matplotlib_fig2img(fig, transpose=img_transpose)


def visualize_2d_sdf_partbb(
        partc, parts, sdfs, img_transpose=True, return_fig=False):
    bs, res = sdfs.shape[0], sdfs.shape[-1]
    sdfs = sdfs.reshape(bs, res, res)
    nparts = partc.shape[1]

    nrows = int(np.ceil(bs / float(4.)))
    if int(bs) <= 4:
        ncols = int(bs)
    else:
        ncols = 4
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    for idx in range(bs):
        ax = plt.subplot(nrows, ncols, 1 + idx)
        ax.contour(sdfs[idx])
        ax.contour(sdfs[idx], levels=[-np.inf, 0.], colors='r', linewidths=5)
        for parti in range(nparts):
            ci = (partc[idx, parti].reshape(2) + 1) * res * 0.5
            si = parts[idx, parti] * res * 0.5
            xy = ci - si * 0.5
            rect = patches.Rectangle(
                (xy[0], xy[1]), si[0], si[1], color='red', fill=False)
            ax.add_patch(rect)
            ax.scatter([ci[0]], [ci[1]], color='red')

            # The coordinate frame
            rect = patches.Rectangle(
                (0, 0), res, res,
                color='blue', linestyle='--', fill=False)
            ax.add_patch(rect)

    plt.tight_layout()
    if return_fig:
        return fig
    else:
        return matplotlib_fig2img(fig, transpose=img_transpose)


def visualize_2d_sdf_part(
        partc, parts, part_sdf, final_sdfs,
        img_transpose=True, return_fig=False):
    bs, res = final_sdfs.shape[0], final_sdfs.shape[-1]
    if bs > 4:
        bs = 4
    partc = partc[:bs]
    parts = parts[:bs]
    part_sdf = part_sdf[:bs]
    final_sdfs = final_sdfs[:bs]

    sdfs = final_sdfs.reshape(bs, res, res)
    nparts = part_sdf.shape[1]

    nrows = int(bs)
    ncols = int(nparts)
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    for idx in range(bs):
        for parti in range(nparts):
            ax = plt.subplot(nrows, ncols, int(1 + idx * nparts + parti))
            ax.title.set_text("Batch:%d Part:%d" % (idx, parti))
            ax.contour(part_sdf[idx, parti])
            ax.contour(
                part_sdf[idx, parti], levels=[-np.inf, 0.], colors='b', linewidths=2)

    plt.tight_layout()
    if return_fig:
        return fig
    else:
        return matplotlib_fig2img(fig, transpose=img_transpose)


def visualize_2d_occ_part(
        partc, parts, part_occ, final_occ,
        img_transpose=True, return_fig=False):
    bs, res = final_occ.shape[0], final_occ.shape[-1]
    if bs > 4:
        bs = 4
    part_occ = part_occ[:bs]
    nparts = part_occ.shape[1]

    nrows = int(bs)
    ncols = int(nparts)
    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
    for idx in range(bs):
        for parti in range(nparts):
            ax = plt.subplot(nrows, ncols, int(1 + idx * nparts + parti))
            ax.title.set_text("Batch:%d Part:%d" % (idx, parti))
            ax.contourf(part_occ[idx, parti])
            ax.contour(
                part_occ[idx, parti], levels=[-np.inf, 0.5], colors='r', linewidths=2)

    plt.tight_layout()
    if return_fig:
        return fig
    else:
        return matplotlib_fig2img(fig, transpose=img_transpose)


def compute_psnr(x, y, eps=1e-8):
    # NOTE: bacon used code use something like this
    from skimage.metrics import peak_signal_noise_ratio as psnr

    x = x.clamp(0, 1).detach().cpu().numpy()
    y = y.clamp(0, 1).detach().cpu().numpy()
    return torch.tensor(psnr(x, y, data_range=1))

    # return 10 * torch.log10(1 / (torch.mean((x - y) ** 2) + eps))

def compute_ssim(x, y):
    # x, y : (res, res, C)
    from skimage.metrics import structural_similarity as ssim

    x = x.clamp(0, 1).detach().cpu().numpy()
    y = y.clamp(0, 1).detach().cpu().numpy()
    return ssim(x, y, data_range=1, multichannel=True, channel_axis=2)


def compute_fft(x, eps=1e-8):
    H, W, C = x.shape
    # dim = x.size(-1)
    # x_flat = x.reshape(-1, dim)
    # res = int(np.sqrt(float(x_flat.shape[0])))
    out_lst = []
    # for i in range(dim):
    for i in range(C):
        # x_i = x_flat[:, i].reshape(res, res)
        x_i = x[..., i]
        spec_x_i = torch.fft.fftshift(torch.fft.fft2(x_i))
        spec_x_i = torch.log(1 + torch.abs(spec_x_i))

        spec_x_i_min, spec_x_i_max = spec_x_i.min(), spec_x_i.max()
        spec_x_i = (spec_x_i - spec_x_i_min) / (spec_x_i_max - spec_x_i_min)

        # out_lst.append(spec_x_i.reshape(res, res, 1))
        out_lst.append(spec_x_i.reshape(H, W, 1))
    return out_lst

