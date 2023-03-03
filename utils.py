import os
import time
import yaml
import torch
import argparse
import importlib
import numpy as np
import os.path as osp
from torch.utils.tensorboard import SummaryWriter as TBSummaryWriter
from torchvision.utils import save_image, make_grid


class SummaryWriter(TBSummaryWriter):

    def __init__(self, small_log_dir=None, **kwargs):
        assert 'log_dir' in kwargs
        log_dir = kwargs['log_dir']
        if small_log_dir is None:
            small_log_dir = kwargs['log_dir']
        self.small_writer = TBSummaryWriter(log_dir=small_log_dir)
        super().__init__(**kwargs)

    def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
        double_precision=False,
    ):
        self.small_writer.add_scalar(
            tag, scalar_value, global_step, walltime,
            new_style, double_precision)
        return super().add_scalar(
            tag, scalar_value, global_step, walltime,
            new_style, double_precision)

    def add_scalars(self, main_tag, tag_scalar_dict,
                    global_step=None, walltime=None):
        self.small_writer.add_scalars(
            main_tag, tag_scalar_dict, global_step, walltime)
        return super().add_scalars(
            main_tag, tag_scalar_dict, global_step, walltime)


class SummaryWriterToFile(TBSummaryWriter):

    def __init__(self, small_log_dir=None, **kwargs):
        assert 'log_dir' in kwargs
        run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
        self.file_log_dir = osp.join(kwargs['log_dir'], "tb", run_time)
        os.makedirs(self.file_log_dir, exist_ok=True)
        super().__init__(**kwargs)

    def add_scalar(
            self,
            tag,
            scalar_value,
            global_step=None,
            walltime=None,
            new_style=False,
            double_precision=False,
    ):
        fname = osp.join(self.file_log_dir, "scalar", tag + ".csv")
        os.makedirs(osp.dirname(fname), exist_ok=True)
        if not osp.isfile(fname):  # add header
            with open(fname, "w") as f:
                f.write("step,walltime,value\n")

        with open(fname, "a") as f:
            f.write("%d,%s,%s\n" % (global_step, walltime, scalar_value))

        return super().add_scalar(
            tag, scalar_value, global_step, walltime,
            new_style, double_precision)

    def add_scalars(self, main_tag, tag_scalar_dict,
                    global_step=None, walltime=None):
        raise NotImplementedError

    def add_image(self, tag, img_tensor, global_step=None,
                  walltime=None, dataformats='CHW'):
        img_dir = osp.join(self.file_log_dir, "image", tag)
        os.makedirs(img_dir, exist_ok=True)

        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor).float()
        if dataformats == 'CHW':
            img_tensor_out = img_tensor.unsqueeze(0)  # (B, C, H, W)
        elif dataformats == 'HWC':
            H, W, C = img_tensor.size()
            img_tensor_out = img_tensor.view(
                    H * W, C).transpose(0, 1).reshape(1, C, H, W)
        else:
            raise ValueError
        vgrid = make_grid(img_tensor_out)
        run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
        save_dir = osp.join(img_dir, "%d_%s.png" % (global_step, run_time))
        save_image(vgrid, save_dir)
        print(save_dir)

        return super().add_image(
            tag, img_tensor, global_step, walltime, dataformats)

    def add_images(self, tag, img_tensor, global_step=None,
                   walltime=None, dataformats='NCHW'):
        img_dir = osp.join(self.file_log_dir, "images", tag)
        os.makedirs(img_dir, exist_ok=True)

        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor).float()
        if dataformats == 'NCHW':
            img_tensor_out = img_tensor
        elif dataformats == 'NHWC':
            N, H, W, C = img_tensor.size()
            img_tensor_out = img_tensor.view(
                    N, H * W, C).transpose(1, 2).reshape(N, C, H, W)
        else:
            raise ValueError
        vgrid = make_grid(img_tensor_out)
        run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
        save_dir = osp.join(img_dir, "%d_%s.png" % (global_step, run_time))
        save_image(vgrid, save_dir)
        print(save_dir)
        return super().add_image(
            tag, img_tensor, global_step, walltime, dataformats)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def dict2namespace(config):
    if isinstance(config, argparse.Namespace):
        return config
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_imf(log_path=None, config_fpath=None, epoch=None,
             verbose=False, load_dataloaders=False,
             hparam_lst=None, strict=True, resume=False,
             return_loaders=False, return_cfg=True, return_trainer=False, ):
    # Load configuration
    if config_fpath is None:
        config_fpath = os.path.join(log_path, "config", "config.yaml")
    with open(config_fpath) as f:
        cfg = dict2namespace(yaml.load(f, Loader=yaml.Loader))
    if hparam_lst is not None:
        cfg, _ = update_cfg_hparam_lst(cfg, hparam_lst, strict=strict)
    print(cfg)

    cfg.save_dir = "logs"
    if load_dataloaders:
        data_lib = importlib.import_module(cfg.data.type)
        loaders = data_lib.get_data_loaders(cfg.data, None)
    else:
        loaders = None
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, None, loaders)

    # Load pretrained checkpoints
    if resume:
        if log_path is not None:
            ep2file = {}
            last_file, last_ep = None, -1
            checkpoint_path = os.path.join(log_path, "checkpoints")
            if os.path.isdir(checkpoint_path):
                for f in os.listdir(checkpoint_path):
                    ep = int(f.split("_")[1])
                    if verbose:
                        print(ep, f)
                    ep2file[ep] = os.path.join(log_path, "checkpoints", f)
                    if ep > last_ep:
                        last_ep = ep
                        last_file = os.path.join(log_path, "checkpoints", f)
                if epoch is not None:
                    last_file = ep2file[epoch]
            print(last_file)
            if last_file is None:
                last_file = osp.join(log_path, "latest.pt")
            print(last_file)
            trainer.resume(last_file)

    ret_tpl = []
    if return_cfg:
        ret_tpl.append(cfg)
    if return_trainer:
        ret_tpl.append(trainer)

    if hasattr(trainer, "net"):
        imf = trainer.net
    elif hasattr(trainer, "decoder"):
        imf = trainer.decoder
    else:
        raise ValueError
    ret_tpl.append(imf)

    if return_loaders:
        ret_tpl.append(loaders)
    return ret_tpl

def parse_hparams(hparam_lst):
    print("=" * 80)
    print("Parsing:", hparam_lst)
    out_str = ""
    out = {}
    for i, hparam in enumerate(hparam_lst):
        hparam = hparam.strip()
        k, v = hparam.split("=")[:2]
        k = k.strip()
        v = v.strip()
        print(k, v)
        out[k] = v
        out_str += "%s=%s_" % (k, v.replace("/", "-"))
    print(out)
    print(out_str)
    print("=" * 80)
    return out, out_str


def update_cfg_with_hparam(cfg, k, v, strict=True):
    k_path = k.split(".")
    cfg_curr = cfg
    v = eval(str(v))
    for k_curr in k_path[:-1]:
        assert hasattr(cfg_curr, k_curr), "%s not in %s" % (k_curr, cfg_curr)
        cfg_curr = getattr(cfg_curr, k_curr)
    k_final = k_path[-1]
    if strict:
        assert hasattr(cfg_curr, k_final), \
            "Final: %s not in %s" % (k_final, cfg_curr)
        v_type = type(getattr(cfg_curr, k_final))
        v_to_be_set = v_type(v)
    else:
        v_to_be_set = v
    setattr(cfg_curr, k_final, v_to_be_set)


def update_cfg_hparam_lst(cfg, hparam_lst, strict=True):
    hparam_dict, hparam_str = parse_hparams(hparam_lst)
    for k, v in hparam_dict.items():
        update_cfg_with_hparam(cfg, k, v, strict=strict)
    return cfg, hparam_str
