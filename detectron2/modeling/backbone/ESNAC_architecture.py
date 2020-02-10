import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from .ESNAC_layer import ESNACIdentity, ESNACFlatten, ESNACLayer

__all__ = [
    "ESNACArchitecture",
]

def _get_links(n, E):
    in_links = [[] for i in range(n)]
    out_links = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if E[i][j]:
                in_links[j].append(i)
                out_links[i].append(j)
    return in_links, out_links

class ESNACArchitecture(Backbone):
    def __init__(self, n=0, V=[], E=[], out_layers={}):
        super(ESNACArchitecture, self).__init__()
        self.n = n
        self.V = V
        for i in range(n):
            self.add_module('layer_%d' % (i), V[i])
        self.E = E
        self.in_links, self.out_links = _get_links(n, E)
        self.out_layers = out_layers

    def _get_layer(self, i):
        return getattr(self, 'layer_%d' % (i))

    def forward(self, x):
        y = [None] * self.n
        ret = {}
        y[0] = self._get_layer(0)(x)
        for j in range(1, self.n):
            x = []
            for i in self.in_links[j]:
                x.append(y[i])
                if j == self.out_links[i][-1]:
                    y[i] = None
            y[j] = self._get_layer(j)(sum(x))
            if j in self.out_layers:
                ret[self.out_layers[j]] = y[j]
        return ret

    '''
    TODO: fix this up
    def _V_to_desc(self):
        ret = []
        for i in range(self.n):
            layer = self.V[i]
            desc = {
                'base_type': layer.base_type,
                'in_shape': layer.in_shape,
                'out_shape': layer.out_shape,
            }
            if layer.base_type in ['ESNACIdentity', 'ESNACFlatten']:
                desc['base'] = layer.base_type
            elif layer.base_type == 'Conv2d':
                b = layer.base
                desc['base'] = {
                    'in_channels': b.in_channels,
                    'out_channels': b.out_channels,
                    'kernel_size': b.kernel_size,
                    'stride': b.stride,
                    'padding': b.padding,
                    'dilation': b.dilation,
                    'groups': b.groups,
                    'bias': b.bias is not None,
                    'padding_mode': b.padding_mode,
                }
    '''

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + 'n'] = self.n
        destination[prefix + 'V'] = self.V
        destination[prefix + 'E'] = self.E
        destination[prefix + 'in_links'] = self.in_links
        destination[prefix + 'out_links'] = self.out_links
        destination[prefix + 'out_layers'] = self.out_layers
        super(ESNACArchitecture, self)._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.n = state_dict.pop(prefix + 'n')
        self.V = state_dict.pop(prefix + 'V')
        self.E = state_dict.pop(prefix + 'E')
        self.in_links = state_dict.pop(prefix + 'in_links')
        self.out_links = state_dict.pop(prefix + 'out_links')
        self.out_layers = state_dict.pop(prefix + 'out_layers')
        for i in range(self.n):
            self.add_module('layer_%d' % (i), self.V[i])
        super(ESNACArchitecture, self)._load_from_state_dict(state_dict, prefix,
            local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
