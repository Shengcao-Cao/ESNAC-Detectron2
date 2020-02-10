from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.modeling import build_model

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

args = [
    '--config-file', 'configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
    'MODEL.WEIGHTS', 'models/model_final_280758.pkl',
]
args = default_argument_parser().parse_args(args)
cfg = setup(args)
model = build_model(cfg)

from detectron2.checkpoint import DetectionCheckpointer
checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True).get("iteration", -1) + 1

import torch
from ESNAC.graph import get_graph_resnet, get_plot
from detectron2.modeling.backbone import ESNACArchitecture

resnet = model.backbone.bottom_up
arch = ESNACArchitecture(*get_graph_resnet(resnet))

'''
input = torch.rand(1, 3, 256, 256)
o1 = resnet(input)
o2 = arch(input)
print(torch.sum(torch.abs(o1['res2'] - o2['res2'])))
print(torch.sum(torch.abs(o1['res3'] - o2['res3'])))
print(torch.sum(torch.abs(o1['res4'] - o2['res4'])))
print(torch.sum(torch.abs(o1['res5'] - o2['res5'])))
'''

from ESNAC.architecture import init_arch_rep, init_arch_groups, comp_action_rand, comp_rep, comp_arch, replace_resnet

init_arch_rep(arch)
init_arch_groups(arch)
input = torch.rand(1, 3, 256, 256)
for i in range(1):
    action = comp_action_rand(arch)
    rep_ = comp_rep(arch, action)
    arch_ = comp_arch(arch, action)
    print(torch.sum(torch.abs(rep_ - arch_.rep)))
    output = arch_(input.to('cuda'))
    for k, v in output.items():
        print(k, v.shape)

model_ = replace_resnet(model, arch)
output = model_.backbone(input.to('cuda'))
for k, v in output.items():
    print(k, v.shape)

model_ = replace_resnet(model, arch_)
output = model_.backbone(input.to('cuda'))
for k, v in output.items():
    print(k, v.shape)

backbone = model_.backbone
torch.save(backbone.state_dict(), 'test.pth')
backbone.bottom_up = ESNACArchitecture()
backbone.load_state_dict(torch.load('test.pth'))
print(backbone.bottom_up.n)
output_ = model_.backbone(input.to('cuda'))
print(torch.sum(torch.abs(output['p2'] - output_['p2'])))
print(torch.sum(torch.abs(output['p3'] - output_['p3'])))
print(torch.sum(torch.abs(output['p4'] - output_['p4'])))
print(torch.sum(torch.abs(output['p5'] - output_['p5'])))
print(torch.sum(torch.abs(output['p6'] - output_['p6'])))

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader

# train_loader = build_detection_train_loader(cfg)
val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
val_evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, 'inference_temp')

results = inference_on_dataset(model, val_loader, val_evaluator)
