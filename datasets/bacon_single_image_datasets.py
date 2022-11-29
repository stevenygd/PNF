import os
import torch
import errno
import urllib
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from datasets.single_img_datasets import init_np_seed


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


class SingleImage(Dataset):
    def __init__(self, cfg, cfgdata):
        # def __init__(self, filename, grayscale=False, resolution=None,
        #              root_path=None, crop_square=True, url=None):
        super().__init__()
        self.cfg = cfg
        self.cfgdata = cfgdata
        filename = cfg.path
        grayscale = getattr(cfg, "grayscale", False)
        resolution = getattr(cfgdata, "res", None)
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        crop_square = getattr(cfgdata, "crop_square", True)
        url = getattr(cfg, "url", None)

        if not os.path.exists(filename):
            if url is None:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), filename)
            else:
                print('Downloading image file...')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                urllib.request.urlretrieve(url, filename)

        self.img = Image.open(filename)
        if grayscale:
            self.img = self.img.convert('L')
        else:
            self.img = self.img.convert('RGB')

        self.img_channels = len(self.img.mode)
        self.resolution = self.img.size

        if crop_square:  # preserve aspect ratio
            self.img = crop_max_square(self.img)

        if resolution is not None:
            self.resolution = resolution
            self.img = self.img.resize(resolution, Image.ANTIALIAS)

        self.img = np.array(self.img)
        self.img = self.img.astype(np.float32)/255.

        if len(self.img.shape) == 2:
            H, W = self.img.shape
            C = 1
        else:
            H, W, C = self.img.shape

        print(self.img.shape)
        self.value = torch.from_numpy(self.img.reshape(H * W, C)).float()

        coor = torch.cat(
            [x.reshape(-1, 1) for x in torch.meshgrid(torch.arange(H), torch.arange(W))],
            dim=-1
        ).float()
        xyz = ((coor + 0.5) / float(H)) * 2 - 1.  # ranger [-1, 1]
        self.xyz = xyz.view(-1, 2)

    def __len__(self):
        return getattr(self.cfgdata, "length", 1)

    def __getitem__(self, idx):
        return {
            'idx': idx,
            'xyz': self.xyz,  # (H * W, 2)
            'value': self.value,  # (H * W)
        }




def get_data_loaders(cfg, args):
    tr_dataset = SingleImage(cfg, cfg.train)
    te_dataset = SingleImage(cfg, cfg.val)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=cfg.val.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders
