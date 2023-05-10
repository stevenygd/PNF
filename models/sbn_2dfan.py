import torch
import numpy as np
import torch.nn as nn


class SubbandSine2DFAN(nn.Module):

    def __init__(self, lr, ur, la, ua, nsmp, quantize=False, fan_uniform=False):
        super().__init__()
        self.lr = lr  # lower bound in radius
        self.ur = ur  # upper bound in radius
        self.la = la  # lower bound in angle
        self.ua = ua  # upper bound in angel
        self.n = nsmp  # number of neurons/samples
        self.quantize = quantize

        R1 = np.pi * self.lr
        R2 = np.pi * self.ur
        if fan_uniform:
            self.radius = nn.Parameter(
                # NOTE: sqrt to be uniform to the area,
                #       derivation in the notes already
                (np.sqrt(torch.rand(self.n)) * (R2 - R1) + R1),
                requires_grad=True
            )
        else:
            self.radius = nn.Parameter(
                (torch.rand(self.n) * (R2 - R1) + R1),
                requires_grad=True
            )
        self.angles = nn.Parameter(
            torch.rand(self.n) * (self.ua - self.la) + self.la,
            requires_grad=True
        )

        # Phase
        self.phase = nn.Parameter(
            (torch.rand(self.n) - 0.5) * 2 * np.pi,
            requires_grad=True
        )

    def _get_freq_(self):
        direction = torch.cat([
            torch.cos(self.angles).view(1, -1, 1),
            torch.sin(self.angles).view(1, -1, 1)
        ], dim=-1)  # (1, self.n, 2)
        out = self.radius.view(1, -1, 1) * direction  # (1, self.n, 2)
        if self.quantize:
            out = torch.round(out / np.pi / 2.) * np.pi * 2
        return out

    def __repr__(self):
        return "SubbandSine2DFAN(r=(%d, %d), a=(%.2f, %.2f)pi, hid=%d, quant=%s)" % \
               (self.lr, self.ur, self.la / np.pi, self.ua / np.pi, self.n, self.quantize)

    def forward(self, x):
        """
        :param x: (bs, npts, 2), assume range [-1, 1]
        :return:
            sin/cosx : (bs, npts, hidim)
        """
        bs, npts = x.size(0), x.size(1)
        x = x.view(bs, npts, 1, 2)                   # (bs, npts, 1,    2)
        w = self._get_freq_().view(1, 1, -1, 2)      # (1,  1,    nsmp, 2)
        b = self.phase.view(1, 1, -1)                # (1,  1, nsmp)
        o = (w * x).sum(dim=-1, keepdims=False) + b  # (bs, npts, nsmp)
        sinx, cosx = torch.sin(o), torch.cos(o)
        return sinx, cosx


def sbn_init_mix(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # c = np.sqrt(6 / num_input) # Too big :(
            # c = np.sqrt(4 / num_input) # Too big :(
            c = np.sqrt(3 / num_input) # roughly constant, but not std Gaussian, VAR=2
            # c = np.sqrt(1.5 / num_input) # First layer STD Gaussian, later layers std too small
                                           # This means the combine layer changed the distribution :(
            m.weight.uniform_(-c, c)
        # if hasattr(m, "bias") and m.bias is not None:
        #     torch.nn.init.constant_(m.bias, 0)


def sbn_init_out(nfans):
    def init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                c = np.sqrt(6 / num_input / nfans)
                m.weight.uniform_(-c, c)
    return init


def subband_combine(sin_x, cos_x, sin_y, cos_y):
    """
    Args:
        sin/cos_x:  (bs, npts, hdim)
        sin/cos_lb: (bs, npts, hdim)
    Return:
        sin/cos_xpy: (bs, npts, hdim)
    """
    sin_xpy = sin_x * cos_y + cos_x * sin_y
    cos_xpy = cos_x * cos_y - sin_x * sin_y
    return sin_xpy, cos_xpy


class SubbandSeries(nn.Module):

    def __init__(self, bws, hiddim, outdim, la, ua, nfans, quantize=False,
                 mix_init_type='none', out_init_type='none',
                 acc_method='sum', fan_uniform=False):
        """ Subband Series (fourier basis within a subband, and output layers)
        :param bws: list of bandwidth configurations
        :param hiddim: hidden dimension of the network
                       TODO: we assume the hidden size of the layers remain the same
        :param outdim: output dimension of the network
        :param la:  lower bound in angle
        :param ua:  upper bound in angle
        :param nfans: int, total number of fans
        :param quantize:  boolean, whether we will quantize the frequencyes
        """

        super().__init__()
        self.bws = bws
        self.fans = nn.ModuleList()
        self.mixs = nn.ModuleList()  # Mix layers
        self.outs = nn.ModuleList()  # Output layers
        self.fans.append(SubbandSine2DFAN(
            bws[0][0], bws[0][1], la, ua, hiddim, quantize=quantize,
            fan_uniform=fan_uniform
        ))
        for lr, ur in bws[1:]:
            self.fans.append(
                SubbandSine2DFAN(lr, ur, la, ua, hiddim, quantize=quantize))
            self.mixs.append(
                nn.Linear(hiddim, hiddim, bias=False))  # TODO: my calculation is bias is bad here
            self.outs.append(
                nn.Linear(hiddim * 2, outdim))
        if mix_init_type == 'forward':
            self.mixs.apply(sbn_init_mix)
        if out_init_type == 'forward':
            self.outs.apply(sbn_init_out(nfans))

        self.acc_method = acc_method

    def accumulate(self, i, curr_acc, curr_out, curr_out_lst, prev_acc_lst):
        """

        :param i:
        :param curr_acc:
        :param curr_out:
        :param curr_out_lst: All output, including the current one
        :param prev_acc_lst: All previous accumulator
        :return:
        """
        if self.acc_method == 'avg':
            return (curr_acc * i + curr_out) / float(i + 1)
        elif self.acc_method == 'avg_sqrt':
            # acc output var == 1, but each layer contributes the same
            return (curr_acc * np.sqrt(i) + curr_out) / np.sqrt(float(i + 1))
        elif self.acc_method == 'exp_decay_half':
            n = len(curr_out_lst)
            acc = 0
            for j in range(n):
                w_j = np.sqrt(2 ** (-j) / (2 * (1 - 0.5 ** n)))
                acc += curr_out_lst[j] * w_j
            return acc
        elif self.acc_method == 'sum':
            # acc output var == sum(all distr std)
            return curr_acc + curr_out
        else:
            raise ValueError("Invalid acc_method: %s" % self.acc_method)

    def forward(self, x, fst_n=None, retain_inter_grad=False):
        """
        :param x: (bs, npts, dim)
        :return:
            sin/cosx : (bs, npts, hdim)
        """
        # basis output (after mix & comb.)
        sin_x_lst, cos_x_lst = [], []

        # Final output (before and after aggregation)
        out_lst, acc_lst, acc = [], [], 0

        sin_x, cos_x = self.fans[0](x)
        sin_fx, cos_fx = [sin_x], [cos_x]   # output from the filters
        for i, (fan, mix, out) in enumerate(
                zip(self.fans[1:], self.mixs, self.outs)):
            # Create new FAN samples
            sin_y, cos_y = fan(x)
            sin_fx.append(sin_y)
            cos_fx.append(cos_y)

            # Mix and combine
            sin_x, cos_x = mix(sin_x), mix(cos_x)
            sin_x, cos_x = subband_combine(sin_x, cos_x, sin_y, cos_y)
            sin_x_lst.append(sin_x)
            cos_x_lst.append(cos_x)

            curr_out = out(torch.cat([sin_x, cos_x], dim=-1))
            if retain_inter_grad:
                curr_out.retain_grad()
            out_lst.append(curr_out)
            # acc = acc + curr_out
            acc = self.accumulate(
                i, curr_acc=acc, curr_out=curr_out,
                curr_out_lst=out_lst, prev_acc_lst=acc_lst)
            if retain_inter_grad:
                acc.retain_grad()
            acc_lst.append(acc)
            if fst_n is not None and i >= fst_n:
                break

        return {
            'sin_fx': sin_fx,
            'cos_fx': cos_fx,
            'sin_x': sin_x_lst,
            'cos_x': cos_x_lst,
            'sb_out': out_lst,
            'sb_acc': acc_lst
        }


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
        self.hid_dim = cfg.hid_dim

        # Band width per output layer
        self.bw_span_diag = getattr(cfg, "bw_span_diag", False)
        self.max_bw = float(eval(str(cfg.max_bw)))
        if self.bw_span_diag:
            self.max_bw *= np.sqrt(2.)
        self.bws = cfg.bws
        self.bws = [
            (float(eval(str(l))) * self.max_bw,
             float(eval(str(u))) * self.max_bw)
            for l, u in self.bws
        ]

        self.quantize = getattr(cfg, "quantize_freq", False)
        self.n_subbands = int(getattr(cfg, "n_subbands", 4))
        self.sb_agl_range = float(eval(
            str(getattr(cfg, "sb_agl_range", 0.25)))) * np.pi
        self.sb_agl_delta = float(eval(
            str(getattr(cfg, "sb_agl_delta", 0.25)))) * np.pi

        self.sbs = nn.ModuleList()
        for i in range(self.n_subbands):
            la = i * self.sb_agl_delta - 0.5 * self.sb_agl_range
            ua = la + self.sb_agl_range
            self.sbs.append(SubbandSeries(
                self.bws, self.hid_dim, self.out_dim,
                la, ua, self.n_subbands, quantize=self.quantize,
                mix_init_type=getattr(cfg, "mix_init_type", "none"),
                out_init_type=getattr(cfg, "out_init_type", "none"),
                acc_method=getattr(cfg, "acc_method", "sum"),
                fan_uniform=getattr(cfg, "fan_uniform", False)
            ))

        # OUtput level
        self.out_levels = getattr(cfg, "out_levels", None)

        # Input and output scale
        self.inp_mult_const = float(getattr(cfg, "inp_mult_const", 0.5))  # default scale from [-1, 1]
    def get_modules(self):

        return self.sbs

    def prune_model(self, proportion, n):

        from torch.nn.utils import prune

        num_params_to_prune = 0
        modules = self.get_modules()

        for module in modules:
            num_params_to_prune += sum([p.numel() for p in module.parameters()]) * 1e-6
            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')

        return proportion * num_params_to_prune

    def forward(self, x, fst_n=None, out_levels=None,
                retain_inter_grad=False):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz), range [-1, 1]
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        out_levels = self.out_levels if out_levels is None else out_levels
        xdim = len(x.shape)
        if xdim == 2:
            x = x.unsqueeze(0)
        x = x * self.inp_mult_const

        # Gather all subband results
        # key -> list of list, all_res[sb_i][res_i] -> tensor
        all_sb_res = {}
        for sbs_i in self.sbs:
            sbs_out = sbs_i(x, fst_n=fst_n,
                            retain_inter_grad=retain_inter_grad)
            for k, lst in sbs_out.items():
                if k not in all_sb_res:
                    all_sb_res[k] = []
                all_sb_res[k].append(lst)

        # Reorg s.t. all_res index with all_res[res_i][sb_i] -> tensor
        all_res = {}
        for key, sb_lst in all_sb_res.items():  # sb_lst[sb_i][res_i] -> tensor
            assert len(sb_lst) == self.n_subbands
            all_res[key] = []
            for res_i in range(len(sb_lst[0])):
                lst = []
                for sb_i in range(self.n_subbands):
                    lst.append(sb_lst[sb_i][res_i])
                all_res[key].append(lst)

        # Compute the final output and the final accumulator by summing together
        # outputs from all subbands
        for key in ['out', 'acc']:
            key_all = "all_%s" % key

            all_res[key_all] = []
            for lst in all_res['sb_%s' % key]:
                slst = sum(lst)
                if retain_inter_grad:
                    slst.retain_grad()
                all_res[key_all].append(slst)

            if out_levels is not None:
                all_res[key] = [all_res[key_all][i] for i in out_levels]
            else:
                all_res[key] = all_res[key_all]
        # Match the key for trainer
        all_res['out_lst'] = all_res['acc']
        all_res['all_out_lst'] = all_res['all_acc']
        return all_res

Decoder = SubbandNet
Net = Decoder
