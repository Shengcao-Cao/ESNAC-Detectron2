import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ESNACIdentity",
    "ESNACFlatten",
    "ESNACLayer",
]

class ESNACIdentity(nn.Module):
    def __init__(self):
        super(ESNACIdentity, self).__init__()

    def forward(self, x):
        return x

class ESNACFlatten(nn.Module):
    def __init__(self):
        super(ESNACFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ESNACLayer(nn.Module):
    def __init__(self, base, in_shape=None, out_shape=None):
        super(ESNACLayer, self).__init__()
        self.base = base
        self.base_type = base.__class__.__name__
        self.in_shape = torch.Size([-1]) + in_shape[1:]
        self.out_shape = torch.Size([-1]) + out_shape[1:]

    def forward(self, x):
        return self.base(x)
