import os
import torch
import importlib
import numpy as np
import torch.nn.functional as F
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import get_opt, set_random_seed
from trainers.utils.vis_utils import make_2d_grid, compute_psnr, compute_ssim, compute_fft
from nerf_utils import clip_styler_utils
from torchvision import transforms, models

class Trainer(BaseTrainer):

    net: torch.nn.Module

    def __init__(self, cfg, args, *nkwargs, **kwargs):
        super().__init__(cfg, args, *nkwargs)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        sn_lib = importlib.import_module(cfg.models.net.type)
        self.net = sn_lib.Net(cfg, cfg.models.net)
        self.net.cuda()
        print("Net:")
        print(self.net)

        params_by_layer = getattr(cfg.trainer, "params_by_layer", None)
        fix_only_output = getattr(cfg.trainer, "fix_only_output", None)
        if params_by_layer is None:
            params = list(self.net.parameters())
        else:
            params = self.net.get_parameters_by_layer(params_by_layer, fix_only_output)

        print("%.5fM Parameters" %
              (sum([p.numel() for p in params]) * 1e-6))

        self.opt, self.scheduler = get_opt(
            params, self.cfg.trainer.opt)

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.save_dir, "val"), exist_ok=True)

        self.ttl_iter = 0

        # Loss type
        self.loss_type = getattr(cfg.trainer, "loss_type", 'mse')
        self.output_key = getattr(cfg.trainer, "loss_key", 'out_lst')
        self.loss_batch = getattr(cfg.trainer, "loss_batch", None)
        if isinstance(self.loss_batch, str):
            self.loss_batch = int(eval(self.loss_batch))

        if self.loss_type == "clip_styler":
            from CLIP import clip

            device = "cuda"

            self.clip_model, self.preprocess = clip.load('ViT-B/32', device, jit=False)

            self.VGG = models.vgg19(pretrained=True).features
            self.VGG.to(device)

            with torch.no_grad():
                content_image = clip_styler_utils.load_image2(self.cfg.data.content_img,
                                                              img_height=cfg.data.img_height,
                                                              img_width=cfg.data.img_width).to(device)
                self.content_features = clip_styler_utils.get_features(clip_styler_utils.img_normalize(content_image),
                                                                       self.VGG)

                template_text = clip_styler_utils.compose_text_with_templates(self.cfg.data.text_prompt)
                tokens = clip.tokenize(template_text).to(device)
                self.text_features = self.clip_model.encode_text(tokens).detach()
                self.text_features = self.text_features.mean(axis=0, keepdim=True)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

                template_source = clip_styler_utils.compose_text_with_templates("a Photo")
                tokens_source = clip.tokenize(template_source).to(device)
                self.text_source = self.clip_model.encode_text(tokens_source).detach()
                self.text_source = self.text_source.mean(axis=0, keepdim=True)
                self.text_source /= self.text_source.norm(dim=-1, keepdim=True)
                self.source_features = self.clip_model.encode_image(clip_styler_utils.clip_normalize(content_image, device)).detach()
                self.source_features /= (self.source_features.clone().norm(dim=-1, keepdim=True))

    def print_params(self):

        params_by_layer = getattr(self.cfg.trainer, "params_by_layer", None)
        fix_only_output = getattr(self.cfg.trainer, "fix_only_output", None)

        if params_by_layer is None:
            params = list(self.net.parameters())
        else:
            params = self.net.get_parameters_by_layer(params_by_layer, fix_only_output)

        return sum([p.numel() for p in params]) * 1e-6

    def compute_loss(self, out_b, val_b):
        if self.loss_type == 'mse':
            loss_ndf = F.mse_loss(out_b.view(*val_b.shape), val_b)
        elif self.loss_type == 'l1':
            loss_ndf = F.l1_loss(out_b.view(*val_b.shape), val_b)
        elif self.loss_type == 'bce':
            loss_ndf = F.binary_cross_entropy(
                F.sigmoid(out_b.view(*val_b.shape)),
                (val_b > 0.5).float()
            )
        elif self.loss_type == "clip_similarity":
            cdim = out_b.size(-1)
            res = int(np.sqrt(out_b.size(-2)))
            out_b = out_b.view(res, res, cdim)
            from nerf_utils.loss_functions import CLIPLoss
            loss_ndf = CLIPLoss(res)(out_b, self.cfg.data.text_prompt)
        elif self.loss_type == "clip_styler":
            device = "cuda"

            cdim = out_b.size(-1)
            res = int(np.sqrt(out_b.size(-2)))
            target = out_b.view(res, res, cdim)
            target = target.permute(2, 0, 1).unsqueeze(dim=0)

            target_features = clip_styler_utils.get_features(clip_styler_utils.img_normalize(target), self.VGG)

            content_loss = 0

            content_loss += torch.mean((target_features['conv4_2'] - self.content_features['conv4_2']) ** 2)
            content_loss += torch.mean((target_features['conv5_2'] - self.content_features['conv5_2']) ** 2)

            loss_patch = 0
            img_proc = []
            for n in range(self.cfg.data.num_crops):
                target_crop = clip_styler_utils.get_cropper(self.cfg.data.crop_size)(target)
                target_crop = clip_styler_utils.get_augmenter()(target_crop)
                img_proc.append(target_crop)

            img_proc = torch.cat(img_proc, dim=0)
            img_aug = img_proc

            image_features = self.clip_model.encode_image(clip_styler_utils.clip_normalize(img_aug, device))
            image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

            img_direction = (image_features - self.source_features)
            img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

            text_direction = (self.text_features - self.text_source).repeat(image_features.size(0), 1)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)
            loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
            loss_temp[loss_temp < self.cfg.data.thresh] = 0
            loss_patch += loss_temp.mean()

            glob_features = self.clip_model.encode_image(clip_styler_utils.clip_normalize(target, device))
            glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

            glob_direction = (glob_features - self.source_features)
            glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

            loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

            reg_tv = self.cfg.data.lambda_tv * clip_styler_utils.get_image_prior_losses(target)

            loss_ndf = self.cfg.data.lambda_patch * loss_patch + \
                       self.cfg.data.lambda_c * content_loss + \
                       reg_tv + \
                       self.cfg.data.lambda_dir * loss_glob

        elif self.loss_type == "style_transfer":
            import torchvision.models as models
            from nerf_utils.loss_functions import get_style_model_and_losses, image_loader
            device = "cuda"

            cdim = out_b.size(-1)
            res = int(np.sqrt(out_b.size(-2)))
            input_img = out_b.view(res, res, cdim)
            self.patch_size = getattr(self.cfg.trainer, "patch_size", None)

            if self.patch_size is None:
                imsize = res
            else:
                imsize = self.patch_size
            style_img = image_loader(self.cfg.data.style_img, imsize=imsize)
            content_img = image_loader(self.cfg.data.content_img, imsize=imsize)

            if self.patch_size is not None:
                low = int(res/2) - int(self.patch_size/2)
                high = int(res/2) + int(self.patch_size/2)
                input_img = input_img[low:high, low:high, :]

            cnn = models.vgg19(pretrained=True).features.to(device).eval()
            cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

            model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                             cnn_normalization_mean,
                                                                             cnn_normalization_std, style_img,
                                                                             content_img, device)
            model.requires_grad_(False)

            input_img = input_img.permute(2, 0, 1).unsqueeze(dim=0)
            model(input_img)

            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            content_weight = getattr(self.cfg.trainer, "content_weight", 1)
            style_weight = getattr(self.cfg.trainer, "style_weight", 1000000)
            style_score *= style_weight
            content_score *= content_weight

            loss_ndf = style_score + content_score
        else:
            raise NotImplementedError
        return {'': loss_ndf}  # to match the default

    def update(self, data, *args, **kwargs):
        self.ttl_iter += 1
        self.net.train()
        self.opt.zero_grad()

        xyz, value = data['xyz'].float().cuda(), data['value'].float().cuda()
        xyz.requires_grad = True
        bs, npts = xyz.size(0), xyz.size(1)
        assert int(bs) == 1
        xyz = xyz.view(-1, xyz.size(-1))
        value = value.view(npts, -1)

        if self.loss_batch is None:
            loss_batch = npts
        else:
            loss_batch = self.loss_batch

        loss = 0.
        loss_lst, all_out_dict = {}, {}
        for i in range(0, npts, loss_batch):
            j = min(npts, i + loss_batch)
            xyz_b = xyz[i:j, :]
            val_b = value[i:j, :]

            out_dict = self.net(xyz_b, None)
            out_lst = out_dict[self.output_key]
            curr_loss = 0.
            for level, out_b in enumerate(out_lst):
                all_losses = self.compute_loss(out_b, val_b)
                for loss_name, loss_curr in all_losses.items():
                    curr_loss += loss_curr  # for back propogation, sum across levels

                    # Record the loss
                    loss_level_name = "%s%d" % (loss_name, level)
                    if loss_level_name not in loss_lst:
                        loss_lst[loss_level_name] = 0
                    loss_lst[loss_level_name] += loss_curr.item() * (j - i)
                    loss += loss_curr.item() * (j - i)

                # Record the output
                if level not in all_out_dict:
                    all_out_dict[level] = []
                all_out_dict[level].append(out_b.view(j - i, -1).detach().cpu())

            # Backward the current gradients
            curr_loss.backward()

        # All gradients received, backprop
        self.opt.step()
        loss /= float(npts)
        loss_lst = {l:(v/float(npts)) for l, v in loss_lst.items()}
        out_lst = {l:(torch.cat(v, dim=0)) for l, v in all_out_dict.items()}

        # Record the PSNR
        psnr_lst = {}
        for level, out in out_lst.items():
            with torch.no_grad():
                psnr = compute_psnr(value.to(out), out.view(*value.shape))
                psnr_lst[level] = psnr.cpu().item()

        results = {'loss': loss }
        for i, loss_ndf in loss_lst.items():
            results['scalar/loss_%s' % i] = loss_ndf

        for i, psnr in psnr_lst.items():
            results['scalar/psnr_%s' % i] = psnr

        for i, out_i in out_lst.items():
            cdim = out_i.size(-1)
            res = int(np.sqrt(out_i.size(-2)))
            out_i = out_i.view(res, res, cdim)
            results['image/out/out_%s' % i] = out_i
            if getattr(self.cfg.viz, "log_fft", False) and \
                    self.ttl_iter % int(getattr(self.cfg.viz, "log_fft_freq", 1)) == 0:
                with torch.no_grad():
                    spec_out_i = compute_fft(out_i)
                    for j, spec_out_i_img in enumerate(spec_out_i):
                        results['image/fft/img_%s_%d' % (i, j)] = spec_out_i_img
                        results['hist/fft/hist_%s_%d' % (i, j)] = \
                            spec_out_i_img.reshape(-1)

        if getattr(self.cfg.viz, "log_fft", False) and \
            self.ttl_iter % int(getattr(self.cfg.viz, "log_fft_freq", 1)) == 0:
            with torch.no_grad():
                spec_gtr = compute_fft(value.reshape(res, res, -1))
                for j, spec_gtr_j_img in enumerate(spec_gtr):
                    results['hist/gtr_fft/hist_%d' % j] = \
                        spec_gtr_j_img.reshape(-1)
                    results['image/gtr_fft/img_%d' % j] = spec_gtr_j_img

        cdim = value.size(-1)
        value = value.view(1, -1, cdim)
        res = int(np.sqrt(value.size(-2)))
        value_img = value.view(res, res, cdim)
        results['image/gtr'] = value_img.detach().cpu()

        k_lst = []
        if getattr(self.cfg.viz, "log_z", False):
            k_lst.append('z_lst')
        if getattr(self.cfg.viz, "log_zm", False):
            k_lst.append('zm_lst')
        if getattr(self.cfg.viz, "log_g", False):
            k_lst.append('g_lst')
        for k in k_lst:
            kh = k.split("_")[0]
            for i, lst_i in enumerate(out_dict[k]):
                results['hist/%s/%s_%d' % (kh, k, i)] = lst_i.detach().cpu()

        return results

    def _net_batch_forward_(self, xyz, batch_size, out_key):
        bs, npts = xyz.size(0), xyz.size(1)
        if batch_size is None:
            batch_size = npts
        all_out_lst = {}
        for i in range(0, npts, batch_size):
            j = min(npts, i + batch_size)
            curr_out_lst = self.net(xyz[:, i:j, :], None)[out_key]
            for k, out in enumerate(curr_out_lst):
                if k not in all_out_lst:
                    all_out_lst[k] = []
                all_out_lst[k].append(out.view(bs, j-i, -1))
        out_lst = []
        for i in range(len(all_out_lst)):
            out_lst.append(torch.cat(all_out_lst[i], dim=1))
        return out_lst

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        writer_step = step if step is not None else epoch
        if visualize:
            if getattr(self.cfg.viz, "log_all_levels", True):
                print("Log all levels at step : %d" % writer_step)
                alllvl_res = getattr(self.cfg.viz, "alllvl_res", 512)
                alllvl_keys = getattr(self.cfg.viz, "alllvl_keys", ['all_out_lst'])
                for alllvl_key in alllvl_keys:
                    with torch.no_grad():
                        grid = make_2d_grid(alllvl_res)
                        grid = grid.view(1, alllvl_res ** 2, 2).cuda().float()
                        out_lst = self._net_batch_forward_(
                            grid, self.loss_batch, alllvl_key)
                        img_lst, fft_lst = [], []
                        for out in out_lst:
                            cdim = out.size(-1)
                            img_lst.append(out.view(1, alllvl_res, alllvl_res, cdim))
                            if getattr(self.cfg.viz, "log_fft_all_levels", True):
                                spec_out_i = compute_fft(
                                    out.reshape(alllvl_res, alllvl_res, cdim))
                                for j, spec_out_i_img in enumerate(spec_out_i):
                                    fft_lst.append(spec_out_i_img.reshape(
                                        1, alllvl_res, alllvl_res, 1))
                        img_lst = torch.cat(img_lst, dim=0)
                        train_info["images/all_levels/%s/out" % alllvl_key] = img_lst
                        if getattr(self.cfg.viz, "log_fft_all_levels", True):
                            fft_lst = torch.cat(fft_lst, dim=0)
                            train_info['images/all_levels/%s/fft' %alllvl_key] = fft_lst

            if getattr(self.cfg.viz, "log_extrap", True):
                print("Log extrapolation at step :%d" % writer_step)
                extrap_mult = getattr(self.cfg.viz, "extrap_mult", 4)
                extrap_res = getattr(self.cfg.viz, "extrap_res", 512)
                with torch.no_grad():
                    grid = make_2d_grid(extrap_res) * extrap_mult
                    grid = grid.view(1, extrap_res ** 2, 2).cuda().float()
                    out_lst = self._net_batch_forward_(
                        grid, self.loss_batch, self.output_key)
                    img_lst = []
                    for i, out in enumerate(out_lst):
                        cdim = out.size(-1)
                        img_lst.append(out.view(1, extrap_res, extrap_res, cdim))
                    img_lst = torch.cat(img_lst, dim=0)
                    train_info['images/extrapolation/out'] = img_lst
                    # img_grid = make_grid(img_lst)
                    # writer.add_image("extrapolation/out", img_grid, writer_step)

        super().log_train(
            train_info, train_data, writer=writer,
            step=step, epoch=epoch, visualize=visualize, **kwargs)

    def validate(self, test_loader, epoch, *args, **kwargs):
        print("Validation at epoch : %d" % epoch)
        results = {}
        with torch.no_grad():
            for data in test_loader:
                break
            xyz = data['xyz'].cuda().float()
            bs = xyz.size(0)
            assert bs == 1
            gtr = data['value'].cuda().float()
            cdim = gtr.size(-1)
            gtr = gtr.reshape(-1, cdim)
            res = int(np.sqrt(gtr.shape[0]))
            assert res ** 2 == gtr.shape[0]
            out_lst = self._net_batch_forward_(
                xyz, self.loss_batch, self.output_key)

            img_lst, psnr_lst = [], []
            img_lst.append(gtr.reshape(1, res, res, cdim))
            for i, out in enumerate(out_lst):
                img_lst.append(out.view(1, res, res, cdim))
                psnr = compute_psnr(
                    out.reshape(res**2, cdim), gtr.reshape(res ** 2, cdim)
                ).cpu().item()
                results['scalar/val/psnr_%d' % i] = psnr

                ssim = compute_ssim(
                    out.reshape(res, res, cdim), gtr.reshape(res, res, cdim))
                results['scalar/val/ssim_%d' % i] = ssim
            img_lst = torch.cat(img_lst, dim=0)
            results["images/val/img"] = img_lst
        return results

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'opt': self.opt.state_dict(),
            'net': self.net.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)
        torch.save(d, os.path.join(self.cfg.save_dir, "latest.pt"))

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt['net'], strict=strict)
        load_opt = getattr(self.cfg.trainer, "load_opt", True)
        if load_opt:
            self.opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def multi_gpu_wrapper(self, wrapper):
        self.net = wrapper(self.net)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.scheduler is not None:
            self.scheduler.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/lr',
                    self.scheduler.get_last_lr()[0],
                    epoch)
        else:
            lr = self.opt.param_groups[0]['lr']
            if writer is not None:
                writer.add_scalar('train/lr', lr, epoch)
