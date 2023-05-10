import os
import torch
import importlib
import numpy as np
import os.path as osp
from pprint import pprint
import torch.nn.functional as F
from trainers.utils.diff_ops import gradient
from trainers.utils.vis_utils import imf2mesh_multires
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
# from evaluation.mesh_metrics import compute_all_mesh_metrics
from evaluation.mesh_metrics import compute_all_mesh_metrics_with_opcl


class Trainer(BaseTrainer):

    def __init__(self, cfg, args, *nkwargs, **kwargs):
        super().__init__(cfg, args, *nkwargs)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        sn_lib = importlib.import_module(cfg.models.net.type)
        self.net = sn_lib.Net(cfg, cfg.models.net)
        self.net.cuda()

        self.multi_gpu = getattr(cfg.trainer, "multi_gpu", False)
        if self.multi_gpu:
            self.multi_gpu_wrapper(torch.nn.DataParallel)

        print("Net:")
        print(self.net)
        print("%.5fM Parameters" %
              (sum([p.numel() for p in self.net.parameters()]) * 1e-6))

        # The optimizer
        self.cfg.trainer.opt = self.cfg.trainer.opt
        self.opt, self.scheduler_dec = get_opt(
            self.net.parameters(), self.cfg.trainer.opt)

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "meshes"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "val"), exist_ok=True)

        # Prepare variable for summy
        self.oracle_res = None
        self.mc_levels = getattr(cfg.viz, "mc_levels", None)
        self.loss_levels = getattr(cfg.trainer, "loss_levels", None)

        # Loss weights
        self.coarse_loss_weight = float(getattr(
            self.cfg.trainer, "coarse_loss_weight", 1.))
        self.fine_loss_weight = float(getattr(
            self.cfg.trainer, "fine_loss_weight", 1.))
        self.mse_reduction = getattr(
            self.cfg.trainer, "mse_reduction", 'mean')

        # Clip gradient
        self.clip_grad = getattr(
            self.cfg.trainer, "clip_grad", True)
        self.clip_grad_thr = getattr(
            self.cfg.trainer, "clip_grad_thr", None)

        self.out_key = getattr(cfg.trainer, "out_key", "out_lst")

    def _compute_loss_(self, xyz, out, sdf):
        bs = sdf.size(0)
        loss_y_sdf_i = F.mse_loss(
            out.view(*sdf.shape), sdf,
            reduction=self.mse_reduction)
        # loss_y_sdf_i = ((out.view(*sdf.shape) - sdf) ** 2).sum()

        # Eikonal loss
        grad_norm_weight = float(getattr(
            self.cfg.trainer, "grad_norm_weight", 0.))
        grad_norm_num_points = int(getattr(
            self.cfg.trainer, "grad_norm_num_points", 0))
        if grad_norm_weight > 0. and grad_norm_num_points > 0:
            grad_norm = gradient(out, xyz).view(
                bs, -1, xyz.size(-1)).norm(dim=-1)
            loss_unit_grad_norm_i = F.mse_loss(
                grad_norm, torch.ones_like(grad_norm)) * grad_norm_weight
        else:
            loss_unit_grad_norm_i = torch.zeros(1).cuda().float()

        return loss_y_sdf_i, loss_unit_grad_norm_i

    def update(self, data, *args, **kwargs):
        self.net.train()
        self.opt.zero_grad(set_to_none=True)

        c_xyz, c_sdf = data['c_xyz'].cuda(), data['c_sdf'].cuda()
        f_xyz, f_sdf = data['f_xyz'].cuda(), data['f_sdf'].cuda()

        loss = 0
        results = {}
        for weight, prefix, xyz, sdf in [
            (self.coarse_loss_weight, 'coarse', c_xyz, c_sdf),
            (self.fine_loss_weight, 'fine', f_xyz, f_sdf)
        ]:
            bs = xyz.size(0)
            xyz = xyz.view(bs, -1, xyz.size(-1))
            xyz.requires_grad = True
            xyz.retain_grad()
            res = self.net(xyz, None)

            acc_lst, loss_lst, loss_y_sdf_lst, loss_ekn_lst = [], [], [], []

            # SDF Loss
            out_lst = res[self.out_key]
            for i, out in enumerate(out_lst):
                loss_y_sdf_i, loss_unit_grad_norm_i = self._compute_loss_(
                    xyz=xyz, out=out, sdf=sdf)

                loss_y_sdf_lst.append(loss_y_sdf_i)
                loss_ekn_lst.append(loss_unit_grad_norm_i)
                loss_lst.append(loss_y_sdf_i + loss_unit_grad_norm_i)

                # Compute accuracy
                with torch.no_grad():
                    acc_i = ((out > 0) == (sdf > 0)).float().mean()
                    acc_lst.append(acc_i)

            loss_y_sdf = sum(loss_y_sdf_lst) / float(len(loss_y_sdf_lst))
            loss_ekn = sum(loss_ekn_lst) / float(len(loss_ekn_lst))
            curr_ttl_loss = loss_y_sdf + loss_ekn
            loss += curr_ttl_loss * weight

            results.update({
                ('scalar/ttl/%s_loss' % prefix): curr_ttl_loss.detach().cpu().item(),
                ('scalar/ttl/%s_weight' % prefix): weight,
                ('scalar/%s/loss_sdf' % prefix): loss_y_sdf.detach().cpu().item(),
                ('scalar/%s/loss_ekn' % prefix): loss_ekn.detach().cpu().item(),
                # Weighted
                ('scalar/ttl/%s_loss_weighted' % prefix): \
                    (curr_ttl_loss * weight).detach().cpu().item(),
                ('scalar/%s/loss_sdf_weighted' % prefix): loss_y_sdf.detach().cpu().item() * weight,
                ('scalar/%s/loss_ekn_weighted' % prefix): loss_ekn.detach().cpu().item() * weight,
                ('scalar/%s/acc' % prefix): torch.stack(acc_lst).mean().detach().cpu().item(),
            })
            for n, lst in [('ekn_loss', loss_ekn_lst),
                           ('sdf_loss', loss_y_sdf_lst),
                           ('ttl_loss', loss_lst),
                           ('acc', acc_lst)]:
                for i, loss_curr in enumerate(lst):
                    results["scalar/%s_level/%s/%d" % (prefix, n, i)] = \
                        loss_curr.detach().cpu().item()

        loss.backward()
        if self.clip_grad:
            if self.clip_grad_thr is None:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), max_norm=1.)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), max_norm=self.clip_grad_thr)
        self.opt.step()

        # Total loss
        results.update({
            'loss': loss.detach().cpu().item(),
            'scalar/ttl/loss': loss.detach().cpu().item(),
        })

        return results

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        super().log_train(
            train_info, train_data, writer=writer,
            step=step, epoch=epoch, visualize=visualize, **kwargs)

        writer_step = step if step is not None else epoch
        if visualize:
            with torch.no_grad():
                print("MC at %s" % step)
                res = int(getattr(self.cfg.viz, "mc_res", 128))
                thr = float(getattr(self.cfg.viz, "mc_thr", 0.))
                bound = float(getattr(self.cfg.viz, "mc_bound", 1.))
                mc_bs = int(getattr(self.cfg.viz, "mc_bs", 100000))
                print("   config:res=%d thr=%s bound=%s" % (res, thr, bound))

                mesh_dict, mesh_stat_dict = imf2mesh_multires(
                    lambda x:  self.net(x, None)[self.out_key],
                    levels=self.mc_levels, return_stats=True,
                    res=res, threshold=thr, bound=bound, batch_size=mc_bs,
                    normalize=True, norm_type='res'
                )
                for level in mesh_dict.keys():
                    level_out_dir = osp.join(
                        self.cfg.save_dir, "meshes", "level_%d" % level)
                    os.makedirs(level_out_dir, exist_ok=True)
                    if mesh_dict[level] is not None:
                        save_name = "mesh_level%d_%diters.obj" % (level, writer_step)
                        mesh_dict[level].export(osp.join(level_out_dir, save_name))
                        for k, v in mesh_stat_dict[level].items():
                            writer.add_scalar('vis/mesh/%s' % k, v, writer_step)

    def validate(self, test_loader, epoch, *args, **kwargs):
        writer_step = epoch
        print("Validation at ecpoh: %d" % epoch)

        val_results = {}
        with torch.no_grad():
            for data in test_loader:
                break
            metric_npnts = int(getattr(self.cfg.val, "metric_npnts", 300000))
            if 'mesh' in data:
                gtr_mesh = data['mesh'][0]
                print("Gtr mesh:",
                      gtr_mesh.vertices.max(),
                      gtr_mesh.vertices.mean(),
                      gtr_mesh.vertices.min(),
                      )
                import trimesh
                pcl, fidx = trimesh.sample.sample_surface(gtr_mesh, metric_npnts)
                sfn = gtr_mesh.face_normals[fidx]
            else:
                assert 'sfn' in data and 'pcl' in data
                sfn = data['sfn']
                pcl = data['pcl']

            res = int(getattr(self.cfg.val, "mc_res", 512))
            thr = float(getattr(self.cfg.val, "mc_thr", 0.))
            bound = float(getattr(self.cfg.val, "mc_bound", 0.5))
            # BACON: 10k takes 4G GPU memory
            batch_size = int(getattr(self.cfg.val, "mc_bs", 100000))
            print("   config:res=%d thr=%s bound=%s bs=%d"
                  % (res, thr, bound, batch_size))

            mesh_dict, mesh_stat_dict = imf2mesh_multires(
                lambda x: self.net(x, None)[self.out_key],
                levels=self.mc_levels, return_stats=True,
                res=res, threshold=thr, bound=bound, batch_size=batch_size,
                normalize=True, norm_type='res'
            )
            for level in mesh_dict.keys():
                level_out_dir = osp.join(
                    self.cfg.save_dir, "val", "level_%d" % level)
                os.makedirs(level_out_dir, exist_ok=True)
                if mesh_dict[level] is not None:
                    mesh_at_level = mesh_dict[level]
                    print("[level %d] Export mesh" % level)
                    print("           ",
                          mesh_at_level.vertices.max(),
                          mesh_at_level.vertices.mean(),
                          mesh_at_level.vertices.min(),
                          )
                    save_name = "mesh_level%d_%diters.obj" % (level, writer_step)
                    mesh_at_level.export(osp.join(level_out_dir, save_name))
                    for k, v in mesh_stat_dict[level].items():
                        val_results[
                            'scalar/val/mesh_stats_%s/%d' % (k, level)] = v
                    # TODO: I don't know the order of gtr/output, it will mess
                    #       up with precision/recall, but not fscore.
                    print("Compute validation metrics at level %d" % level)
                    points1, fidx = trimesh.sample.sample_surface(mesh_at_level, metric_npnts)
                    normals1 = mesh_at_level.face_normals[fidx]
                    level_val_res = compute_all_mesh_metrics_with_opcl(
                        points1, pcl, normals1, sfn)
                    # level_val_res = compute_all_mesh_metrics(
                    #     gtr_mesh, mesh_dict[level], sample_count=metric_npnts)
                    pprint(level_val_res)
                    for k, v in level_val_res.items():
                        val_results[
                            'scalar/val/mesh_metrics_%s/%d' % (k, level)] = v
        pprint(val_results)
        return val_results

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt': self.opt.state_dict(),
            'dec': self.net.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if self.scheduler_dec is not None:
            d['sch'] = self.scheduler_dec.state_dict()
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)

        save_name = "latest.pt"
        path = os.path.join(self.cfg.save_dir, save_name)
        torch.save(d, path)

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt['dec'], strict=strict)
        self.opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch']
        if self.scheduler_dec is not None:
            self.scheduler_dec.load_state_dict(ckpt['sch'])
        return start_epoch

    def multi_gpu_wrapper(self, wrapper):
        self.net = wrapper(self.net)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.scheduler_dec is not None:
            self.scheduler_dec.step(epoch)
            if writer is not None:
                writer.add_scalar(
                    'lr', self.scheduler_dec.get_last_lr()[0], epoch)
