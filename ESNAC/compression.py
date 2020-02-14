import argparse
import os
from operator import attrgetter
import random
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from detectron2.modeling.backbone import ESNACArchitecture

from ESNAC.acquisition import random_search
from ESNAC.architecture import init_arch_rep, init_arch_groups, comp_arch, replace_resnet
from ESNAC.graph import get_graph_resnet
from ESNAC.kernel import Kernel
from ESNAC.record import Record
from ESNAC.training import evaluate
from ESNAC.utils import seed_everything, save_model, param_n, load_model
import ESNAC.options as opt

def new_kernels(arch, record, kernel_n, alpha=opt.co_alpha,
                beta=opt.co_beta, gamma=opt.co_gamma):
    start_time = time.time()
    kernels = []
    for i in range(kernel_n):
        kernel = Kernel(arch.rep, 0.0)
        indices = []
        for j in range(record.n):
            if random.random() < gamma:
                indices.append(j)
        if len(indices) > 0:
            x = [record.x[i] for i in indices]
            indices = torch.tensor(indices, dtype=torch.long, device=opt.device)
            y = torch.index_select(record.y, 0, indices)
            kernel.add_batch(x, y)
        ma = 0.0
        for j in range(100):
            ll = kernel.opt_step()
            opt.writer.add_scalar('step_%d/kernel_%d_loglikelihood' % (opt.i, i),
                                  ll, j)
            ma = (alpha * ll + (1 - alpha) * ma) if j > 0 else ll
            if j > 5 and abs(ma - ll) < beta:
                break
        kernels.append(kernel)
    opt.writer.add_scalar('compression/kernel_time',
                          time.time() - start_time, opt.i)
    return kernels

def next_samples(teacher, backbone, kernels, kernel_n):
    start_time = time.time()
    n = kernel_n
    reps_best, acqs_best, students_best = [], [], []
    for i in range(n):
        action, rep, acq = random_search(backbone, kernels[i])
        reps_best.append(rep)
        acqs_best.append(acq)
        arch = comp_arch(backbone, action)
        student = replace_resnet(teacher, arch)
        students_best.append(student)

        opt.writer.add_scalar('compression/acq', acq, opt.i * n + i - n + 1)
    opt.writer.add_scalar('compression/sampling_time',
                        time.time() - start_time, opt.i)
    return students_best, reps_best

def reward(teacher, students, target_acc=opt.tr_target_acc):
    start_time = time.time()
    n = len(students)
    students = [evaluate(student) for student in students]
    rs = []
    for j in range(n):
        acc = students[j].acc
        c = 1.0 - 1.0 * param_n(students[j]) / param_n(teacher)
        a = 1.0 * acc / target_acc
        r = c * (2 - c) * a
        opt.writer.add_scalar('compression/compression_score', c,
                              opt.i * n - n + 1 + j)
        opt.writer.add_scalar('compression/accuracy_score', a,
                              opt.i * n - n + 1 + j)
        opt.writer.add_scalar('compression/reward', r,
                              opt.i * n - n + 1 + j)
        rs.append(r)
        students[j].comp = c
        students[j].acc = acc
        students[j].reward = r
    opt.writer.add_scalar('compression/evaluating_time',
                          time.time() - start_time, opt.i)
    return students, rs

def compression(teacher, backbone,
                record, archs_best=[], step_start=1, step_end=opt.co_step_n,
                kernel_n=opt.co_kernel_n, best_n=opt.co_best_n):
    for i in range(step_start, step_end + 1):
        print('Search step %d/%d' % (i, step_end))
        start_time = time.time()
        opt.i = i
        kernels = new_kernels(backbone, record, kernel_n)
        students, xi = next_samples(teacher, backbone, kernels, kernel_n)
        students, yi = reward(teacher, students)
        for j in range(kernel_n):
            record.add_sample(xi[j], yi[j])
            if yi[j] == record.reward_best:
                opt.writer.add_scalar('compression/reward_best', yi[j], i)
            print('Arch %d: comp %.2f\tacc %.2f\treward %.2f' %
                (j, students[j].comp, students[j].acc, students[j].reward))
        students = [student.to('cpu') for student in students]
        archs_best.extend(students)
        archs_best.sort(key=attrgetter('reward'), reverse=True)
        archs_best = archs_best[:best_n]
        ckpt = {
            'step': i,
            'archs_best': [(arch.backbone.bottom_up, arch.state_dict()) for arch in archs_best],
            'record': record
        }
        save_model(ckpt, os.path.join(opt.savedir, 'ckpt.pth'))
        print('Step time', time.time() - start_time)
        opt.writer.add_scalar('compression/step_time',
                              time.time() - start_time, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learnable Embedding Space for Efficient Neural Architecture Compression')

    parser.add_argument('--config-file', type=str, default='configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml', help='path to model config file')
    parser.add_argument('--model-weights', type=str, default='models/model_final_280758.pkl', help='pretrained model weights')
    parser.add_argument('--n-gpus', type=int, default=2, help='number of GPUs used for training')
    parser.add_argument('--ims-per-batch', type=str, default='4', help='training batch size')
    parser.add_argument('--name', type=str, help='name of experiment')

    args = parser.parse_args()

    seed_everything()
    mp.set_start_method('spawn')

    opt.config_file = args.config_file
    opt.model_weights = args.model_weights
    opt.n_gpus = args.n_gpus
    opt.tr_ims_per_batch = args.ims_per_batch
    opt.savedir = 'save/%s' % (args.name)
    opt.writer = SummaryWriter('runs/%s' % (args.name))

    if os.path.exists(os.path.join(opt.savedir, 'ckpt.pth')):
        ckpt = torch.load(os.path.join(opt.savedir, 'ckpt.pth'))
        step_start = ckpt['step'] + 1
        archs_best = ckpt['archs_best']
        record = ckpt['record']
    else:
        step_start = 1
        archs_best = []
        record = Record()

    fasterrcnn = load_model(opt.config_file, opt.model_weights, 'cpu')
    resnet = fasterrcnn.backbone.bottom_up
    resnet = ESNACArchitecture(*(get_graph_resnet(resnet)))
    init_arch_rep(resnet)
    init_arch_groups(resnet)

    print ('Compression %s starts. Saved checkpoint at save/%s. Log file at runs/%s.' %
        (args.name, args.name, args.name))

    compression(fasterrcnn, resnet, record, archs_best, step_start)
