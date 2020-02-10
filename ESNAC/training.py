import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.modeling.backbone import ESNACArchitecture

from ESNAC.architecture import replace_resnet
import ESNAC.options as opt

def preparation():
    print('Loading cfg')
    args = [
        '--config-file', opt.config_file,
        'MODEL.WEIGHTS', opt.model_weights,
        'SOLVER.IMS_PER_BATCH', str(opt.tr_batch_size),
    ]
    args = default_argument_parser().parse_args(args)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    print('Loading data')
    train_loader = build_detection_train_loader(cfg)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    val_evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, os.path.join(opt.savedir, 'inference_temp'))

    print('Loading model')
    fasterrcnn = build_model(cfg)
    checkpointer = DetectionCheckpointer(fasterrcnn, cfg.OUTPUT_DIR)
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

    return train_loader, val_loader, val_evaluator, fasterrcnn

def evaluate_sub(save_path, idx,
                 iterations=opt.tr_iterations, lr=1e-4, weight_decay=5e-4):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)
    train_loader, val_loader, val_evaluator, teacher = preparation()
    arch = ESNACArchitecture()
    arch.load_state_dict(torch.load(os.path.join(save_path, 'student_%d_state.pth' % (idx))))
    student = replace_resnet(teacher, arch).to(opt.device)
    teacher.train()
    student.train()
    preprocess = teacher.preprocess_image
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student.backbone.parameters(), lr=lr, weight_decay=weight_decay)

    for data, iteration in zip(train_loader, range(iterations)):
        images = preprocess(data).tensor

        with torch.no_grad():
            teacher_features = teacher.backbone(images)

        student_features = student.backbone(images)
        loss = sum([criterion(teacher_features[key], student_features[key]) for key in teacher_features])
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training')
            return
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if opt.writer:
            opt.writer.add_scalar('step_%d/sample_%d_loss' % (opt.i, idx), loss.item(), iteration)

    torch.cuda.empty_cache()
    results = inference_on_dataset(student, val_loader, val_evaluator)
    state_dict = student.state_dict()
    state_dict['acc'] = results['bbox']['AP']
    torch.save(state_dict, os.path.join(save_path, 'student_%d_state.pth' % (idx)))

def evaluate(students):
    n = len(students)
    processes = []
    save_path = os.path.join(opt.savedir, 'temp')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, student in enumerate(students):
        torch.save(student.backbone.bottom_up.state_dict(), os.path.join(save_path, 'student_%d_state.pth' % (idx)))
        p = mp.Process(target=evaluate_sub, args=(save_path, idx))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    for idx in range(n):
        state_dict = torch.load(os.path.join(save_path, 'student_%d_state.pth' % (idx)))
        acc = state_dict.pop('acc')
        students[idx].load_state_dict(state_dict, strict=False)
        students[idx].acc = acc
    return students
