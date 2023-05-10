import torch
import numpy as np
import torch.nn as nn
from models.sbn_2dfan import subband_combine
import itertools


class SubbSineNDQuad(nn.Module):

    def __init__(self, inp_dim, hid_dim, lr, ur,
                 fan_uniform=False, quantize=False,
                 ring_type=['circular']):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.fan_uniform = fan_uniform
        self.quantize = quantize
        self.ring_type = ring_type

        # Step 1: initialize weights for each subband
        self.nsubband = 0
        freq, phase = self._compute_params_(
            lr, ur, self.ring_type)

        # Step 2: stack weights together, put into the Conv1D layer
        self.layer = nn.Conv1d(2, hid_dim * self.nsubband, 1)
        self.layer.weight.data = freq
        self.layer.bias.data = phase

    def _compute_params_(self, lr, ur, ring_type_lst):
        freq_lst, phase_lst = [], []
        for ring_type in ring_type_lst:
            if ring_type in ['circular', 'rect', 'rectangle']:
                self.nsubband += (2 ** self.inp_dim) // 2
                exists = set()
                for signs in itertools.product(*([[1, -1]] * self.inp_dim)):
                    if tuple(signs) in exists:
                        continue # TODO: should rotate 45 degree!
                    exists.add(tuple(signs))
                    exists.add(tuple([(-x) for x in signs]))
                    freq, phase = self._compute_quad_params_signs_(
                        lr, ur, signs, self.hid_dim, ring_type)
                    freq_lst.append(freq)
                    phase_lst.append(phase)
            elif ring_type in ['triangle', 'trig']:
                self.nsubband += self.inp_dim
                for i in range(self.inp_dim):
                    freq, phase = self._compute_trig_params_dims_(
                        lr, ur, i, self.hid_dim)
                    freq_lst.append(freq)
                    phase_lst.append(phase)
            else:
                raise ValueError("ring_type: %s" % ring_type)

        assert len(freq_lst) == self.nsubband and len(phase_lst) == self.nsubband
        ttl_out = self.hid_dim * self.nsubband
        freq = torch.cat(freq_lst, dim=0).reshape(
            ttl_out, self.inp_dim, 1)  # (#subband x #hid, 2)
        phase = torch.cat(
            phase_lst, dim=0).reshape(ttl_out)  # (#subband x #hid, 1)
        if self.quantize:
            freq = torch.round(freq / (2 * np.pi)) * 2 * np.pi

        return freq, phase

    def _compute_trig_params_dims_(self, lr, ur, dim, n):
        ur, lr = np.pi * ur, np.pi * lr
        f_i = torch.rand(n, 1) * (ur - lr) + lr
        f = []
        if dim > 0:
            f.append((torch.rand(n, dim) * 2 - 1) * f_i)
        f.append(f_i)
        if dim < self.inp_dim - 1:
            f.append((torch.rand(n, self.inp_dim - 1 - dim) * 2 - 1) * f_i)
        freq = torch.cat(f, dim=-1)
        assert freq.shape[1] == self.inp_dim

        phase = (torch.rand(n) - 0.5) * 2 * np.pi  # (self.n)
        return freq, phase

    def _compute_quad_params_signs_(self, lr, ur, signs, n, ring_type):
        if ring_type == 'circular':
            signs = torch.from_numpy(
                np.array(signs)).float().view(1, self.inp_dim, 1)
            vect = torch.abs(torch.randn(n, self.inp_dim + 1, 1))
            vect = (vect / vect.norm(dim=1, keepdim=True))[:, :self.inp_dim, :] * signs
            radi = vect.norm(dim=1, keepdim=True)
            dirc = vect / radi
            assert ur >= lr
            radi = np.pi * (radi * (ur - lr) + lr)
            freq = dirc * radi  # (self.inp_dim, self.n, 2)

        elif ring_type in ['rect', 'rectangle']:
            signs = torch.from_numpy(
                np.array(signs)).float().view(1, self.inp_dim, 1)
            lr = lr * np.pi
            ur = ur * np.pi
            all_region_p, all_region_w = [], []
            for region_d in itertools.product(*([[1, -1]] * self.inp_dim)):
                region_d = np.array(region_d)
                if np.array(region_d).max() < 0:  # rule out all -1 case
                    continue
                region_lb, region_ub = [], []
                for vd in region_d:
                    if vd > 0:  # choose [lr, ur]
                        region_lb.append(lr)
                        region_ub.append(ur)
                    else:       # choose [0, lr]
                        region_lb.append(0)
                        region_ub.append(lr)
                region_lb = np.array(region_lb).reshape(1, self.inp_dim)
                region_ub = np.array(region_ub).reshape(1, self.inp_dim)
                points = np.random.rand(n, self.inp_dim)
                points = points * (region_ub - region_lb) + region_lb
                all_region_p.append(points)  # (n, self.inp_dim)

                # compute weight
                region_w = np.prod(region_ub - region_lb) / float(n)
                if region_w == 0:
                    region_w = 1 / float(n)
                region_w = np.array([region_w] * n).reshape(n)
                all_region_w.append(region_w)  # (n, )

            # For each region, sample points according to the probability
            points = np.concatenate(all_region_p, axis=0)  # (#region * n, self.inp_dim)
            probability = np.concatenate(all_region_w, axis=0)  # (#region * n)
            probability /= probability.sum()
            points_idx = np.random.choice(
                points.shape[0], n, p=probability, replace=False)
            freq = torch.from_numpy(points[points_idx]).reshape(n, self.inp_dim)
            freq *= signs.reshape(1, self.inp_dim)
            freq = freq.float()
        else:
            raise ValueError("ring type" % ring_type)

        # Phase
        phase = (torch.rand(n) - 0.5) * 2 * np.pi  # (self.n)
        return freq, phase

    def forward(self, x):
        """
        :param x: (bs, N, C), [C] is the spatial dimension,
                  [N] is the number of points, [bs] is batch size.
                  Assume x range from [-0.5, 0.5]
        Return
            (bs x 2, #subband x #hiddim, N), stacking of two tensors, each of size
            (bs, #subband x #hiddim, N), with sin and cos respectively.
        """
        x = x.transpose(1, 2).contiguous()
        y = self.layer(x)
        out = torch.cat([torch.sin(y), torch.cos(y)], dim=0)
        return out


class MixLayer(nn.Module):

    def __init__(self, inp_dim, out_dim, nsubband, init_type='none',
                 shared=False):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.nsubband = nsubband
        self.shared = shared
        if self.shared:
            print("Mix layer with shared weights")
            self.layer = nn.Conv1d(
                inp_dim, out_dim, 1, groups=1, bias=None)
        else:
            self.layer = nn.Conv1d(
                inp_dim * nsubband, out_dim * nsubband, 1,
                groups=nsubband, bias=None)

        if init_type == 'bacon':
            c = np.sqrt(6. / float(inp_dim))
            self.layer.weight.data.uniform_(-c, c)
        elif init_type == 'sbn_gaussian':
            c = np.sqrt(3. / float(inp_dim))
            self.layer.weight.data.uniform_(-c, c)
        else:
            assert init_type == 'none'

    def forward(self, x):
        """
        :param x: (bs x 2, #sub x #inp, N)
        Return:
            (bs x 2, #sub x #out, N)
        """
        if self.shared:
            bs2, _, npts = x.size(0), x.size(1), x.size(2)
            # TODO: not sure which is more correct :)
            # x = x.reshape(bs2, self.inp_dim, self.nsubband * npts)
            x = x.reshape(bs2 * self.nsubband, self.inp_dim, npts)
            y = self.layer(x)
            y = y.reshape(bs2, self.nsubband * self.out_dim, npts)
            return y
        else:
            return self.layer(x)


class MultLayer(nn.Module):

    def forward(self, x, y):
        """
        :param x: (bs x 2, #sub x #inp, N)
        :param y: (bs x 2, #sub x #inp, N)
        Return: (bs x 2, #sub x #inp, N)
        """
        bs2 = x.size(0)
        assert bs2 % 2 == 0
        bs = bs2 // 2
        sin_x, cos_x = torch.split(x, bs, dim=0)
        sin_y, cos_y = torch.split(y, bs, dim=0)
        xpy_sin, xpy_cos = subband_combine(sin_x, cos_x, sin_y, cos_y)
        xpy = torch.cat([xpy_sin, xpy_cos], dim=0)
        return xpy


class OutputLayer(nn.Module):

    def __init__(self, inp_dim, out_dim, nsubband, init_type='none'):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.nsubband = nsubband
        self.layer = nn.Conv1d(inp_dim * nsubband, out_dim * nsubband, 1,
                               groups=nsubband)
        if init_type == 'bacon':
            c = np.sqrt(6. / float(inp_dim))
            self.layer.weight.data.uniform_(-c, c)
        elif init_type == 'sbn_gaussian':
            c = np.sqrt(3. / float(inp_dim) / float(self.nsubband))
            self.layer.weight.data.uniform_(-c, c)
        else:
            assert init_type == 'none'

    def forward(self, x):
        """
        :param x: (bs x 2, #sub x #inp, N)
        Return: (bs, N, out_dim)
        """
        bs2, C, npts = x.size(0), x.size(1), x.size(2)
        assert bs2 % 2 == 0
        bs = bs2 // 2
        y_all = self.layer(x)  # (bs * 2, out * nsuband, npts)
        sin_y, cos_y = torch.split(y_all, bs, dim=0)
        y = sin_y + cos_y
        y = y.transpose(1, 2).view(
            bs, npts, self.nsubband, self.out_dim).contiguous()
        return y, y.sum(dim=2, keepdims=False)


class OutputLayerLegacy(nn.Module):

    def __init__(self, inp_dim, out_dim, nsubband, init_type='none'):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.nsubband = nsubband
        self.layer = nn.Conv1d(inp_dim * nsubband * 2, out_dim * nsubband, 1,
                               groups=nsubband)
        if init_type == 'bacon':
            c = np.sqrt(6. / float(inp_dim))
            self.layer.weight.data.uniform_(-c, c)
        elif init_type == 'sbn_gaussian':
            c = np.sqrt(3. / float(inp_dim) / float(self.nsubband))
            self.layer.weight.data.uniform_(-c, c)
        else:
            assert init_type == 'none'

    def forward(self, x):
        """
        :param x: (bs x 2, #sub x #inp, N)
        Return: (bs, N, out_dim)
        """
        bs2, C, npts = x.size(0), x.size(1), x.size(2)
        assert bs2 % 2 == 0
        bs = bs2 // 2
        sin_x, cos_x = torch.split(x, bs, dim=0)
        # (bs, 2C=2*#inp*#sub, npts)
        x = torch.cat([sin_x.view(bs, C, npts), cos_x.view(bs, C, npts)], dim=1)
        x = x.view(bs, 2, C, npts).reshape(bs, 2 * C, npts)
        y = self.layer(x)  # (bs * 2, out * nsuband, npts)
        y = y.transpose(1, 2).view(
            bs, npts, self.nsubband, self.out_dim).contiguous()
        return y, y.sum(dim=2, keepdims=False)



def accumulate(acc_method, i, curr_acc, curr_out, curr_out_lst, prev_acc_lst):
    """
    :param i:
    :param curr_acc:
    :param curr_out:
    :param curr_out_lst: All output, including the current one
    :param prev_acc_lst: All previous accumulator
    :return:
    """
    if acc_method == 'avg':
        return (curr_acc * i + curr_out) / float(i + 1)
    elif acc_method == 'avg_sqrt':
        # acc output var == 1, but each layer contributes the same
        return (curr_acc * np.sqrt(i) + curr_out) / np.sqrt(float(i + 1))
    elif acc_method == 'exp_decay_half':
        n = len(curr_out_lst)
        acc = 0
        for j in range(n):
            w_j = np.sqrt(2 ** (-j) / (2 * (1 - 0.5 ** n)))
            acc += curr_out_lst[j] * w_j
        return acc
    elif acc_method == 'sum':
        # acc output var == sum(all distr std)
        return curr_acc + curr_out
    else:
        raise ValueError("Invalid acc_method: %s" % acc_method)



class SubbandNet(nn.Module):
    """ Subband Net: generalized MFN that allow precise subband tiling
    # TODO: parameters groups for progressive training

    Example configuration:
        dim: 2              # input dim
        out_dim: 1          # output dim
        hid_dim: 128        # hidden dim (BACON / (2^2) for 4 subbands)

        # Subband tiling (orientation)
        n_subbands: 4       # number of subband (or orientations)
        sb_agl_range: 0.25     # '0.25 * np.pi', angle within each subband
        sb_agl_delta: 0.25     # '0.25 * np.pi', the angle between subbands

        # Subband tiling (magnitude)
        max_bw: 256         # maximum bandwidth in unit of cycles
        bws:
            - lb: 0         # '0 * max_bw'
              ub: 1/8       # 1/8*max_bw
            - lb: 0         # '0 * max_bw'
              ub: 1/8       # 1/8*max_bw

        # Other subband information
        quantize: True
    """
    def __init__(self, _, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.out_dim = cfg.out_dim

        # Band width per output layer
        self.max_bw = float(eval(str(cfg.max_bw)))
        self.bws = [
            (float(eval(str(l))) * self.max_bw,
             float(eval(str(u))) * self.max_bw)
            for l, u in cfg.bws
        ]
        self.hid_dims = getattr(cfg, "hid_dims", None)
        if self.hid_dims is None:
            self.hid_dims = [int(cfg.hid_dim)] * len(self.bws)
        self.mix_shared_idxs = getattr(cfg, "mix_shared_idxs", [])

        self.acc_method = getattr(cfg, "acc_method", "sum")
        self.fan_uniform = getattr(cfg, "fan_uniform", False)
        self.quantize = getattr(cfg, "quantize", False)
        self.quantize_train = getattr(cfg, "quantize_train", False)
        self.n_subbands = int(getattr(cfg, "n_subbands", 4))
        self.sb_agl_range = float(eval(
            str(getattr(cfg, "sb_agl_range", 0.25)))) * np.pi
        self.sb_agl_delta = float(eval(
            str(getattr(cfg, "sb_agl_delta", 0.25)))) * np.pi
        self.fan_types = getattr(cfg, "fan_types", [('ring', None)])
        self.fan_opt = getattr(cfg, "fan_opt", True)

        self.fans = nn.ModuleList()  # FAN-shape filter layers
        self.mixs = nn.ModuleList()  # Mix layers
        self.mult = nn.ModuleList()  # "Multi" layers
        self.outs = nn.ModuleList()  # Output layers
        self.fans.append(SubbandSine2DFAN(
            self.hid_dims[0], self.bws[0][0], self.bws[0][1],
            self.n_subbands, self.sb_agl_range, self.sb_agl_delta,
            fan_uniform=self.fan_uniform,
            quantize=self.quantize,
            quantize_train=self.quantize_train,
            fan_types=self.fan_types,
            fan_opt=self.fan_opt
        ))
        assert self.n_subbands == self.fans[0].nsubband
        for i, (lr, ur) in enumerate(self.bws[1:]):
            self.fans.append(
                SubbandSine2DFAN(
                    # lr, ur, la, ua, hiddim, quantize=quantize
                    self.hid_dims[i + 1], lr, ur,
                    self.n_subbands, self.sb_agl_range, self.sb_agl_delta,
                    fan_uniform=self.fan_uniform,
                    quantize=self.quantize,
                    quantize_train=self.quantize_train,
                    fan_types=self.fan_types,
                    fan_opt=self.fan_opt
                ))
            self.mixs.append(MixLayer(
                self.hid_dims[i], self.hid_dims[i + 1], self.n_subbands,
                shared=(i in self.mix_shared_idxs)
            ))
            self.mult.append(MultLayer())
            if getattr(cfg, "legacy_outlayer", True):
                self.outs.append(OutputLayerLegacy(
                    self.hid_dims[i + 1], self.out_dim, self.n_subbands))
            else:
                self.outs.append(OutputLayer(
                    self.hid_dims[i + 1], self.out_dim, self.n_subbands))

        # Output level
        self.out_levels = getattr(cfg, "out_levels", None)

        # Input and output scale
        self.inp_mult_const = float(getattr(cfg, "inp_mult_const", 0.5))

        # default scale from [-1, 1]

    def get_parameters_by_layer(self, layers, fix_only_output):

        params = []

        for layer in range(len(self.bws)):
            if layer in layers:
                params += list(self.mixs[layer].parameters())
                params += list(self.outs[layer].parameters())
            elif fix_only_output:
                params += list(self.mixs[layer].parameters())

        return params

    def get_modules(self):

        return list(self.mixs)

    def prune_model(self, proportion, n):

        from torch.nn.utils import prune

        num_params_to_prune = 0
        modules = self.get_modules()

        for module in modules:
            num_params_to_prune += sum([p.numel() for p in module.parameters()]) * 1e-6
            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')

        return proportion * num_params_to_prune

    # def prune_model(self, proportion, n):
    #
    #     from torch.nn.utils import prune
    #
    #     num_params_to_prune = 0
    #     modules = self.get_modules()
    #
    #     for module in modules:
    #         num_params_to_prune += sum([p.numel() for p in module.parameters()]) * 1e-6
    #         prune.ln_structured(module, 'weight', proportion, n=n, dim=0)
    #         prune.remove(module, 'weight')
    #
    #         prune.ln_structured(module, 'weight', proportion, n=n, dim=1)
    #         prune.remove(module, 'weight')
    #
    #     return num_params_to_prune - (num_params_to_prune * ((1-proportion)**2))


    def forward(self, x, z, fst_n=None, out_levels=None,
                retain_inter_grad=False, compute_output=True):
        """
        # TODO: progressive training
        :param x: (bs, npoints, self.dim) Input coordinate (xyz), range [-1, 1]
        :param z: (bs, self.zdim) Shape latent code + sigma
        TODO: will ignore [z] for now
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        out_levels = self.out_levels if out_levels is None else out_levels
        xdim = len(x.shape)
        if xdim == 2:
            x = x.unsqueeze(0)
        x = x * self.inp_mult_const

        # Just like MFN, we go forward with groupped convolution
        # We keep track of several things:
        # 1. xf_lst: x output after filtered at each layer
        # 2. z_lst:  intermediate activation as mixture of basis
        # 3. out_lst: z -> out_layer -> out (this accumulated across different FANs)
        # 4. acc_lst: accumulated all the outs from different levels
        xf_lst, z_lst, all_out_lst, out_lst, acc_lst, acc = [], [], [], [], [], 0

        # The banks before the output layer (bsx2, hdim * nsubbands, npts)
        out_act_lst = []

        z = xf = self.fans[0](x)
        xf_lst.append(xf)
        for i, (fan, mix, mult, out) in enumerate(
                zip(self.fans[1:], self.mixs, self.mult, self.outs)):
            # Create new FAN samples
            yf = fan(x)
            xf_lst.append(yf)

            # Mix
            z = mix(z)
            z_lst.append(z)

            # Multiplication
            z = mult(z, yf)
            z_lst.append(z)

            # Create output
            if compute_output:
                curr_out_all, curr_out = out(z)
                all_out_lst.append(curr_out_all)
                out_lst.append(curr_out)
                acc = accumulate(
                    self.acc_method, i, curr_acc=acc, curr_out=curr_out,
                    curr_out_lst=out_lst, prev_acc_lst=acc_lst)
                acc_lst.append(acc)
                if fst_n is not None and i >= fst_n:
                    break
            else:
                out_act_lst.append(z)

        if compute_output:
            out_acc_lst = acc_lst
            if out_levels is not None:
                out_acc_lst = [acc_lst[i] for i in out_levels]

            return {
                'out_lst': out_acc_lst,  # the output for supervision
                'all_out_lst': out_lst,  # the outpu in all levels
                'acc_lst': acc_lst,      # accumulation in all levels
                'all_acc_lst': acc_lst,  # accumulation in all levels
                'xf_lst': xf_lst,
                'z_lst': z_lst,
                'all_out_fan_lst': all_out_lst,
                # Lower bounded regions
            }
        else:
            return out_act_lst


Decoder = SubbandNet
Net = Decoder
