import os
import random
import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model

import ESNAC.options as opt

def seed_everything(seed=opt.seed):
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

def load_model(config_file, model_weights, model_device):
    print('Loading cfg')
    args = [
        '--config-file', config_file,
        'MODEL.WEIGHTS', model_weights,
        'MODEL.DEVICE', model_device,
    ]
    args = default_argument_parser().parse_args(args)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    print('Loading model')
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
    return model

def load_train(config_file, ims_per_batch):
    print('Loading cfg')
    args = [
        '--config-file', config_file,
        'SOLVER.IMS_PER_BATCH', str(ims_per_batch),
    ]
    args = default_argument_parser().parse_args(args)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    print('Loading train')
    train_loader = build_detection_train_loader(cfg)
    return train_loader

def load_val(config_file, evaluator_path):
    print('Loading cfg')
    args = [
        '--config-file', config_file,
    ]
    args = default_argument_parser().parse_args(args)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    print('Loading data')
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    if not os.path.exists(evaluator_path):
        os.makedirs(evaluator_path)
    val_evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, evaluator_path)
    return val_loader, val_evaluator

def to_cuda(model):
    device = torch.device('cuda')
    model.to(device)
    model.device = device

    pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor([1.0, 1.0, 1.0]).to(device).view(3, 1, 1)
    model.normalizer = lambda x: (x - pixel_mean) / pixel_std

    for conv in model.backbone.lateral_convs:
        conv.to(device)
    for conv in model.backbone.output_convs:
        conv.to(device)

    return model