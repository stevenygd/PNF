import os
import torch
import pprint
import importlib
import os.path as osp
from torchvision.utils import save_image, make_grid


class BaseTrainer():

    def __init__(self, cfg, args, loaders=None):
        self.cfg = cfg
        self.args = args
        self.loaders = loaders
        pass

    def get_data_loaders(self):
        data_lib = importlib.import_module(self.cfg.data.type)
        loaders = data_lib.get_data_loaders(self.cfg.data, self.args)
        return loaders

    def print_params(self):

        params_by_layer = getattr(self.cfg.trainer, "params_by_layer", None)
        fix_only_output = getattr(self.cfg.trainer, "fix_only_output", None)

        if params_by_layer is None:
            params = list(self.net.parameters())
        else:
            params = self.net.get_parameters_by_layer(
                params_by_layer, fix_only_output)

        return sum([p.numel() for p in params]) * 1e-6

    def update(self, data, *args, **kwargs):
        raise NotImplementedError("Trainer [update] not implemented.")

    def epoch_end(self, epoch, writer=None, **kwargs):
        # Signal now that the epoch ends....
        pass

    def multi_gpu_wrapper(self, wrapper):
        raise NotImplementedError(
            "Trainer doesn't support multiGPU yet.")

    def _write_info_(self, info, writer, writer_step, visualize):
        # Log training information to tensorboard
        for k, v in info.items():
            h, kn = k.split("/")[0], "/".join(k.split("/")[1:])
            if h == 'scalar':
                writer.add_scalar(kn, v, writer_step)
            elif h == 'images':
                v = v.clamp(0, 1)
                if visualize:
                    writer.add_images(kn, v, writer_step, dataformats='NHWC')

                if getattr(self.cfg.viz, "save_imgs", False):
                    img_dir = osp.join(self.cfg.save_dir, "tb_images", kn)
                    os.makedirs(img_dir, exist_ok=True)
                    bs, H, W, C = v.size(0), v.size(1), v.size(2), v.size(3)
                    v = v.view(bs, H * W, C).transpose(1, 2).view(
                        bs, C, H, W).contiguous()
                    vgrid = make_grid(v)
                    save_image(vgrid,
                               osp.join(img_dir, "%d.png" % writer_step))
            elif h == 'image':
                v = v.clamp(0, 1)
                if visualize:
                    writer.add_image(kn, v, writer_step, dataformats='HWC')

                if getattr(self.cfg.viz, "save_imgs", False):
                    img_dir = osp.join(self.cfg.save_dir, "tb_images", kn)
                    os.makedirs(img_dir, exist_ok=True)
                    H, W, C = v.size(0), v.size(1), v.size(-1)
                    v = v.view(H * W, C).transpose(0, 1).view(
                        1, C, H, W).contiguous()
                    vgrid = make_grid(v)
                    save_image(vgrid,
                               osp.join(img_dir, "%d.png" % writer_step))
            elif h == 'hist':
                if self.ttl_iter % getattr(
                    self.cfg.viz, "log_hist_freq", 1) == 0:
                    writer.add_histogram(kn, v, writer_step)
            elif h == 'loss':
                writer.add_scalar('loss', v, writer_step)
            else:
                print("Skip", h, kn)

    def log_train(self, train_info, train_data,
                  writer=None, step=None, epoch=None, visualize=False,
                  **kwargs):
        if writer is None:
            return
        writer_step = step if step is not None else epoch
        self._write_info_(train_info, writer, writer_step, visualize)

        if getattr(self.cfg.viz, "log_param", False):
            for n, p in self.net.named_parameters():
                writer.add_histogram("param_hist/%s" % n, p, writer_step)

        if getattr(self.cfg.viz, "log_grad", False):
            for n, p in self.net.named_parameters():
                if p.grad is not None:
                    writer.add_histogram(
                        "grad_hist/%s" % n, p.grad, writer_step)
                else:
                    writer.add_histogram(
                        "grad_hist/%s" % n, torch.zeros_like(p), writer_step)

    def validate(self, test_loader, epoch, *args, **kwargs):
        print("Trainer [validate] not implemented.")
        return {}

    def log_val(self, val_info, writer=None, step=None, epoch=None, **kwargs):
        if writer is None:
            return
        writer_step = step if step is not None else epoch
        self._write_info_(val_info, writer, writer_step, True)

        val_info_scalar = {
            k: v for k, v in val_info.items() if k.startswith('scalar')}
        for k, v in val_info_scalar.items():
            if k.startswith("scalar"):
                print(k, v)

        if hasattr(self.cfg, "log_dir"):
            os.makedirs(osp.join(self.cfg.log_dir, "val"), exist_ok=True)
            save_fname = osp.join(
                self.cfg.log_dir, "val", "results_%d.txt" % writer_step)
            with open(save_fname, "w") as f:
                f.write(pprint.pformat(val_info_scalar))

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        raise NotImplementedError("Trainer [save] not implemented.")

    def resume(self, path, strict=True, **kwargs):
        raise NotImplementedError("Trainer [resume] not implemented.")