import numpy as np
import torch.nn as nn
from models.utils import MultLayer, MixLayer, OutputLayer, OutputLayerLegacy, SubbSineNDQuad


class SubbandNet(nn.Module):
    """ Subband Net: generalized MFN that allow precise subband tiling
    # TODO: parameters groups for progressive training

    Example configuration:
        dim: 2              # input dim
        out_dim: 1          # output dim
        hid_dim: 128        # hidden dim (BACON / (2^2) for 4 subbands)

        # Subband tiling (magnitude)
        max_bw: 256         # maximum bandwidth in unit of cycles per interval (i.e. 1)
        # list of (basis upper bound, output lower bound);
        # if 'none', then ignore output in this layer
        # Ex: we want [0, 1/8], [1/8, 1/4], [1/4, 1/2], [1/2, 1]
        #     the widths are 1/8, 1/8, 1/4, 1/2
        #     the basis will be [0, 1/8], [0, 1/8], [0, 1/4], [0, 1/2]
        #     the freq will be 1/16 (skip), 1/16, 1/16 (skip),
        bws:
            # With overlap!
            - [1/16, 'none'] # B=1/16
            - [1/16, 0]      # B=1/8,  L=0,  [0, 1/8]
            - [1/16, 1/8]    # B=3/16, L=1/16 [1/16, 1/4]
            - [3/16, 1/8]    # B=6/16, L=1/8 [2/16, 8/16]
            - [6/16,  1/2]    # B=3/4,  L=1/4, [1/2, 1]
    """
    def __init__(self, _, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.inp_dim = cfg.dim
        self.out_dim = cfg.out_dim
        self.hid_dim = cfg.hid_dim

        # Band width per output layer
        # *0.5 to match the bandwidth of the definition in BACON
        self.max_bw = float(eval(str(cfg.max_bw)))
        self.bw0 = float(eval(str(cfg.bw0))) * self.max_bw
        self.bws = [
            (float(eval(str(l))) * self.max_bw,
             float(eval(str(u))) * self.max_bw)
            for l, u in cfg.bws
        ]

        self.acc_method = getattr(cfg, "acc_method", "sum")
        self.fan_uniform = getattr(cfg, "fan_uniform", False)
        self.quantize = getattr(cfg, "quantize_freq", False)
        self.ring_type = ring_type = getattr(cfg, "ring_type", ["rect"])

        self.width_fans = nn.ModuleList()  # FAN-shape filter layers (width)
        self.lower_fans = nn.ModuleList()  # FAN-shape filter layers (lower bound)
        self.mixs_width = nn.ModuleList()  # Mix layers
        self.mixs_lower = nn.ModuleList()  # Mix layers
        self.mult_width = nn.ModuleList()  # "Multi" layers (for growing the width)
        self.mult_lower = nn.ModuleList()  # "Multi" layers (for setting the lower bound)
        self.outs = nn.ModuleList()  # Output layers
        filter_width = SubbSineNDQuad(
            self.inp_dim, self.hid_dim, 0, self.bw0,
            fan_uniform=self.fan_uniform, quantize=self.quantize,
            ring_type=ring_type
        )
        self.n_subbands = filter_width.nsubband
        assert filter_width.nsubband == self.n_subbands
        self.width_fans.append(filter_width)
        for lim_w, lim_l in self.bws:
            filter_width = SubbSineNDQuad(
                self.inp_dim, self.hid_dim, 0, lim_w,
                fan_uniform=self.fan_uniform, quantize=self.quantize,
                ring_type=ring_type
            )
            assert filter_width.nsubband == self.n_subbands
            self.width_fans.append(filter_width)

            self.mixs_width.append(MixLayer(
                self.hid_dim, self.hid_dim, self.n_subbands,
                init_type=getattr(cfg, "mix_width_init_type", "sbn_gaussian")
            ))
            self.mult_width.append(MultLayer())

            self.mixs_lower.append(MixLayer(
                self.hid_dim, self.hid_dim, self.n_subbands,
                init_type=getattr(cfg, "mix_lower_init_type", "sbn_gaussian")
            ))
            filter_lower = SubbSineNDQuad(
                self.inp_dim, self.hid_dim, lim_l, lim_l,
                fan_uniform=self.fan_uniform, quantize=self.quantize,
                ring_type=ring_type
            )
            self.lower_fans.append(filter_lower)
            self.mult_lower.append(MultLayer())

            if getattr(cfg, "legacy_outlayer", True):
                self.outs.append(OutputLayerLegacy(
                    self.hid_dim, self.out_dim, self.n_subbands,
                    init_type=getattr(cfg, "output_init_type", "sbn_gaussian")
                ))
            else:
                self.outs.append(OutputLayer(
                    self.hid_dim, self.out_dim, self.n_subbands,
                    init_type=getattr(cfg, "output_init_type", "sbn_gaussian")
                ))

        # Output level
        self.out_levels = getattr(cfg, "out_levels", None)

        # Input and output scale
        # default scale from [-1, 1]
        # This architecture assume range [-0.5, 0.5]
        self.inp_mult_const = float(getattr(cfg, "inp_mult_const", 0.5))

    def get_parameters_by_layer(self, layers, fix_only_output):

        params = []

        for layer in layers:
            # params += list(self.width_fans[1 + layer].parameters())
            params += list(self.mixs_width[layer].parameters())
            params += list(self.mixs_lower[layer].parameters())
            # params += list(self.lower_fans[layer].parameters())
            params += list(self.outs[layer].parameters())

        if fix_only_output:
            for layer in range(len(self.bws)):
                if layer not in layers:
                    # params += list(self.width_fans[1 + layer].parameters())
                    params += list(self.mixs_width[layer].parameters())
                    params += list(self.mixs_lower[layer].parameters())
                    # params += list(self.lower_fans[layer].parameters())

        return params

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

    def get_modules(self):

        modules = []
        for module in list(self.mixs_width) + list(self.mixs_lower):
            modules.append(module.layer)
        return modules

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
                retain_inter_grad=False):
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
        # NOTE: x is assumed to be ranged from [-0.5, 0.5] before going into
        #       the filter layers to get the right distribution.
        x = x * self.inp_mult_const

        # Just like MFN, we go forward with groupped convolution
        # We keep track of several things:
        # 1. xf_lst: x output after filtered at each layer
        # 2. z_lst:  intermediate activation as mixture of basis
        # 3. out_lst: z -> out_layer -> out (this accumulated across different FANs)
        # 4. acc_lst: accumulated all the outs from different levels
        xf_lst, z_lst, all_out_lst, out_lst, acc_lst, acc = [], [], [], [], [], 0

        # another pass that stored the lifting intermediate outputs
        xfl_lst, zl_lst, zlm_lst = [], [], []

        z = xf = self.width_fans[0](x)
        xf_lst.append(xf)
        for i in range(len(self.width_fans) - 1):
            # Create new FAN samples
            yf_w = self.width_fans[1 + i](x)
            xf_lst.append(yf_w)

            # Mix
            z = self.mixs_width[i](z)
            z_lst.append(z)

            # Multiplication
            z = self.mult_width[i](z, yf_w)
            z_lst.append(z)

            # Create lower bound
            zl = self.mixs_lower[i](z)
            zl_lst.append(zl)
            yf_l = self.lower_fans[i](x)
            xfl_lst.append(yf_l)
            zl = self.mult_lower[i](zl, yf_l)
            zlm_lst.append(zl)

            # Create output
            curr_out_all, curr_out = self.outs[i](zl)
            all_out_lst.append(curr_out_all)
            out_lst.append(curr_out)
            acc = self.accumulate(
                i, curr_acc=acc, curr_out=curr_out,
                curr_out_lst=out_lst, prev_acc_lst=acc_lst)
            acc_lst.append(acc)
            if fst_n is not None and i >= fst_n:
                break

        out_acc_lst = acc_lst
        if self.out_levels is not None:
            out_acc_lst = [acc_lst[i] for i in self.out_levels]

        return {
            'out_lst': out_acc_lst,  # the output for supervision
            'all_out_lst': out_lst,  # the outpu in all levels
            'acc_lst': acc_lst,      # accumulation in all levels
            'all_acc_lst': acc_lst,  # accumulation in all levels
            'xf_lst': xf_lst,
            'z_lst': z_lst,
            'all_out_fan_lst': all_out_lst,
            # Lower bounded regions
            "zl_lst": zl_lst,
            "zlm_lst": zlm_lst,
            "xfl_lst": xfl_lst
        }

Decoder = SubbandNet
Net = Decoder
