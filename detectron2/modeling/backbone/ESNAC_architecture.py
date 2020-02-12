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
        if 0 in self.out_layers:
            ret[self.out_layers[0]] = y[0]
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
