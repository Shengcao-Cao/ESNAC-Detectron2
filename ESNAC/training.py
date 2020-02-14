import copy
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.evaluation import inference_on_dataset
from detectron2.modeling.backbone import ESNACArchitecture

from ESNAC.architecture import replace_resnet
from ESNAC.utils import seed_everything, load_model, load_train, load_val, to_cuda
import ESNAC.options as opt

def train_worker(rank, world_size, dist_url, seed, student_path, config_file, model_weights, ims_per_batch, iterations, lr=1e-4, weight_decay=5e-4):
    torch.cuda.set_device(rank)
    seed_everything(seed)
    dist.init_process_group(backend='NCCL', init_method=dist_url, world_size=world_size, rank=rank)

    teacher = load_model(config_file, model_weights, 'cuda')
    student_arch = torch.load(student_path)
    student = replace_resnet(teacher, student_arch).to('cuda')
    teacher_backbone = DDP(teacher.backbone, device_ids=[rank], find_unused_parameters=True)
    student_backbone = DDP(student.backbone, device_ids=[rank], find_unused_parameters=True)
    teacher_backbone.train()
    student_backbone.train()
    preprocess = teacher.preprocess_image
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student_backbone.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = load_train(config_file, ims_per_batch)

    for data, iteration in zip(train_loader, range(iterations)):
        iteration += 1
        images = preprocess(data).tensor

        with torch.no_grad():
            teacher_features = teacher_backbone(images)

        student_features = student_backbone(images)
        optimizer.zero_grad()
        loss = sum([criterion(teacher_features[key], student_features[key]) for key in teacher_features])
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', flush=True)
            return
        loss.backward()
        optimizer.step()

    dist.barrier()
    if rank == 0:
        torch.save(student.state_dict(), student_path)
        dist.destroy_process_group()

def evaluate(student):
    start_time = time.time()
    student_path = os.path.join(opt.savedir, 'temp')
    if not os.path.exists(student_path):
        os.makedirs(student_path)
    student_path = os.path.join(student_path, 'student.pth')
    torch.save(student.backbone.bottom_up, student_path)

    mp.spawn(train_worker, nprocs=opt.n_gpus,
        args=(
            opt.n_gpus,
            opt.tr_dist_url,
            opt.seed,
            student_path,
            opt.config_file,
            opt.model_weights,
            opt.tr_ims_per_batch,
            opt.tr_iterations,
        ))

    end_time = time.time()
    print('Training time', end_time - start_time)
    start_time = end_time

    student.load_state_dict(torch.load(student_path))
    student = to_cuda(student)
    val_loader, val_evaluator = load_val(opt.config_file, os.path.join(opt.savedir, 'temp', 'inference'))
    results = inference_on_dataset(student, val_loader, val_evaluator)
    student.acc = results['bbox']['AP']

    end_time = time.time()
    print('Test time', end_time - start_time)
    start_time = end_time

    return student
