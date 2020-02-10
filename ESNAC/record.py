import torch

import ESNAC.options as opt

class Record(object):
    def __init__(self):
        super(Record, self).__init__()
        self.n = 0
        self.x = []
        self.y = torch.tensor([], dtype=torch.float, device=opt.device,
                              requires_grad=False)
        self.reward_best = 0.0

    def add_sample(self, xn, yn):
        self.x.append(xn)
        self.y = torch.cat((self.y, torch.tensor([yn], dtype=torch.float,
                                                 device=opt.device,
                                                 requires_grad=False)))
        if yn > self.reward_best:
            self.reward_best = yn
        self.n += 1
