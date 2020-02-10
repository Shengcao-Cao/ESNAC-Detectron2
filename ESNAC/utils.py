import os
import random
import numpy as np
import torch

def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def save_model(model, save_path):
    path = os.path.dirname(save_path)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, save_path)

def param_n(model):
    return sum(x.numel() for x in model.parameters())
