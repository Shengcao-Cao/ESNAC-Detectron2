import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import Conv2d, BatchNorm2d, FrozenBatchNorm2d
from detectron2.modeling.backbone import ESNACLayer, ESNACIdentity, ESNACFlatten
from ESNAC.graph import get_links, get_groups
import ESNAC.options as opt

def init_layer_rep(layer):
    b = layer.base
    lt = opt.ar_supported_layer.index(layer.base_type)
    lr = [0] * opt.ar_n_supported_layer
    lr[lt] = 1
    k = getattr(b, 'kernel_size', 0)
    k = k[0] if type(k) is tuple else k
    s = getattr(b, 'stride', 0)
    s = s[0] if type(s) is tuple else s
    p = getattr(b, 'padding', 0)
    p = p[0] if type(p) is tuple else p
    g = getattr(b, 'groups', 0)
    i = layer.in_shape[1]
    o = layer.out_shape[1]
    layer.rep = lr + [k, s, p, g, i, o]
    return layer.rep

def init_arch_rep(arch):
    n = arch.n
    V = arch.V
    base_mat = [init_layer_rep(V[i]) for i in range(n)]
    in_mat = [([0] * opt.ar_max_layers) for i in range(n)]
    out_mat = [([0] * opt.ar_max_layers) for i in range(n)]
    for i in range(n):
        for j in arch.in_links[i]:
            in_mat[i][i - j] = 1
        for j in arch.out_links[i]:
            out_mat[i][j - i] = 1
    arch.rep = [(base_mat[i] + in_mat[i] + out_mat[i]) for i in range(n)]
    arch.rep = torch.tensor(arch.rep, dtype=torch.float, device=opt.device)
    return arch.rep

def init_arch_groups(arch):
    groups = get_groups(arch.V)
    arch.groups = groups
    return groups

def replace_layer(layer, base):
    base_type = base.__class__.__name__
    if base_type not in opt.ar_supported_layer:
        raise NotImplementedError('Unknown base layer!')
    layer.base = base
    layer.base_type = base_type
    init_layer_rep(layer)

def shrink_layer(layer, Fi, Fo):
    in_shape = list(layer.in_shape)
    in_shape[1] = Fi
    layer.in_shape = torch.Size(in_shape)
    out_shape = list(layer.out_shape)
    out_shape[1] = Fo
    layer.out_shape = torch.Size(out_shape)

    b = layer.base
    if layer.base_type == 'Conv2d':
        groups = b.groups
        if (groups == b.in_channels and b.in_channels == b.out_channels and
            Fi == Fo):
            groups = Fi
        norm = None
        if b.norm is not None:
            norm = BatchNorm2d(Fo)
            norm.weight = nn.Parameter(b.norm.weight[:Fo].clone().detach())
            norm.bias = nn.Parameter(b.norm.bias[:Fo].clone().detach())
            norm.running_mean = b.norm.running_mean[:Fo].clone().detach()
            norm.running_var = b.norm.running_var[:Fo].clone().detach()
        conv = Conv2d(Fi, Fo, b.kernel_size, stride=b.stride, padding=b.padding,
            dilation=b.dilation, groups=groups, bias=(b.bias is not None),
            norm=norm, activation=b.activation)
        conv.weight = nn.Parameter(b.weight[:Fo, :(Fi // groups)].clone().detach())
        if b.bias is not None:
            conv.bias = nn.Parameter(b.bias[:Fo].clone().detach())
        replace_layer(layer, conv)

    elif layer.base_type == 'Linear':
        ln = nn.Linear(Fi, Fo, bias=(b.bias is not None))
        ln.weight = nn.Parameter(b.weight[:Fo, :Fi].clone().detach())
        if b.bias is not None:
            ln.bias = nn.Parameter(b.bias[:Fo].clone().detach())
        replace_layer(layer, ln)

    else:
        init_layer_rep(layer)

def comp_action_rand(arch):
    n = arch.n
    V = arch.V
    action = []
    p1 = random.choice(opt.ar_p1)
    for i in range(n):
        if random.random() < p1 and V[i].in_shape == V[i].out_shape:
            action.append(1.0)
        else:
            action.append(0.0)

    for i in range(len(arch.groups)):
        action.append(random.uniform(*opt.ar_p2))

    p3 = random.choice(opt.ar_p3)
    for i in range(n):
        for j in range(i + 1, n):
            if V[i].out_shape == V[j].in_shape:
                if random.random() < p3 and not action[j]:
                    action.append(1.0)
                else:
                    action.append(0.0)

    return np.array(action)

def comp_rep(arch, action):
    n = arch.n
    V = arch.V
    p = 0
    base_mat = [(V[i].rep.copy()) for i in range(n)]
    in_mat = [([0] * opt.ar_max_layers) for i in range(n)]
    out_mat = [([0] * opt.ar_max_layers) for i in range(n)]
    for i in range(n):
        if action[p]:
            for j in range(opt.ar_n_supported_layer + 4):
                base_mat[i][j] = 0
            base_mat[i][0] = 1  # For identity layer
        p += 1

    for g in arch.groups:
        F = max(1, int((1.0 - action[p]) * g.F))
        for j in g.in_layers:
            base_mat[j][opt.ar_n_supported_layer + 4] = F
        for j in g.out_layers:
            base_mat[j][opt.ar_n_supported_layer + 5] = F
        p += 1

    for i in range(n):
        for j in range(i + 1, n):
            if V[i].out_shape == V[j].in_shape:
                if arch.E[i][j] or action[p]:
                    in_mat[j][j - i] = 1
                    out_mat[i][j - i] = 1
                p += 1

    rep = [(base_mat[i] + in_mat[i] + out_mat[i]) for i in range(n)]
    rep = torch.tensor(rep, dtype=torch.float, device=opt.device)
    return rep

def comp_arch(arch_, action):
    arch = copy.deepcopy(arch_)
    n = arch.n
    V = arch.V
    p = 0
    for i in range(n):
        if action[p]:
            replace_layer(V[i], ESNACIdentity())
        p += 1

    in_shapes = [V[i].in_shape for i in range(n)]
    out_shapes = [V[i].out_shape for i in range(n)]
    for g in arch.groups:
        F = max(1, int((1.0 - action[p]) * g.F))
        for j in g.inter:
            shrink_layer(V[j], F, F)
        for j in g.in_only:
            Fo = V[j].out_shape[1]
            shrink_layer(V[j], F, Fo)
        for j in g.out_only:
            Fi = V[j].in_shape[1]
            shrink_layer(V[j], Fi, F)
        p += 1

    for i in range(n):
        for j in range(i + 1, n):
            if out_shapes[i] == in_shapes[j]:
                if action[p]:
                    arch.E[i][j] = True
                p += 1

    arch.in_links, arch.out_links = get_links(arch.E)
    init_arch_rep(arch)
    return arch

def replace_resnet(model, arch):
    model = copy.deepcopy(model)
    model.backbone.bottom_up = arch
    in_channels = [arch.V[i].out_shape[1] for i in arch.out_layers]
    in_channels = in_channels[::-1]
    for i in range(len(in_channels)):
        b = model.backbone.lateral_convs[i]
        Fi = in_channels[i]
        Fo = b.out_channels
        # Copied from shrink_layer
        groups = b.groups
        if (groups == b.in_channels and b.in_channels == b.out_channels and
            Fi == Fo):
            groups = Fi
        norm = None
        if b.norm is not None:
            norm = BatchNorm2d(Fo)
            norm.weight = nn.Parameter(b.norm.weight[:Fo].clone().detach())
            norm.bias = nn.Parameter(b.norm.bias[:Fo].clone().detach())
            norm.running_mean = b.norm.running_mean[:Fo].clone().detach()
            norm.running_var = b.norm.running_var[:Fo].clone().detach()
        conv = Conv2d(Fi, Fo, b.kernel_size, stride=b.stride, padding=b.padding,
            dilation=b.dilation, groups=groups, bias=(b.bias is not None),
            norm=norm, activation=b.activation)
        conv.weight = nn.Parameter(b.weight[:Fo, :(Fi // groups)].clone().detach())
        if b.bias is not None:
            conv.bias = nn.Parameter(b.bias[:Fo].clone().detach())
        model.backbone.lateral_convs[i] = conv
    return model
