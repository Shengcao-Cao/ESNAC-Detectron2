import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.modeling.backbone import ESNACArchitecture
from ESNAC.utils import seed_everything

def test_DDP_worker(rank):
    seed_everything(0)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='NCCL', init_method='tcp://127.0.0.1:49062', world_size=2, rank=rank)
    conv = nn.Conv2d(1, 2, 3)
    arch = ESNACArchitecture(1, [conv], [[True]], {0:'0'}).to('cuda')
    arch = DDP(arch, device_ids=[rank])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(arch.parameters(), lr=0.1, weight_decay=5e-4)
    for i in range(100):
        input = torch.rand(1, 1, 4, 4).to('cuda')
        output = arch(input)
        target = torch.rand(1, 2, 2, 2).to('cuda')
        optimizer.zero_grad()
        loss = criterion(target, output['0'])
        loss.backward()
        optimizer.step()

def test_DDP():
    mp.spawn(test_DDP_worker, nprocs=2, args=())

if __name__ == '__main__':
    conv = nn.Conv2d(1, 2, 3)
    arch = ESNACArchitecture(1, [conv], [[True]], {0:'0'}).to('cuda')
    print(arch.state_dict())
    test_DDP()