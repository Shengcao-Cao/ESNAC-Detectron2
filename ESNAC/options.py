import torch

# global
config_file = 'configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
model_weights = 'models/model_final_280758.pkl'
device = torch.device('cuda')                               # used device, cuda only for now
n_gpus = 2
seed = 0
savedir = None                                              # save directory
writer = None                                               # record writer for tensorboardX
i = 0                                                       # sample index in search

# acquisition.py
ac_search_n = 1000                                          # number of randomly sampled architectures when optimizing acquisition function (see 3.2)

# architecture.py
ar_supported_layer = [
    'ESNACIdentity',
    'ESNACFlatten',
    'Conv2d',
    'MaxPool2d',
    'AvgPool2d',
    'ReLU',
    'Linear'
]
ar_n_supported_layer = len(ar_supported_layer)
ar_max_layers = 256                                         # maximum number of layers of the original architecture

# hyper-params for random sampling in search space (see 3.2 & 6.5)
ar_p1 = [0.3, 0.4, 0.5, 0.6, 0.7]                           # for layer removal
ar_p2 = [0.0, 1.0]                                          # for layer shrinkage
ar_p3 = [0.003, 0.005, 0.01, 0.03, 0.05]                    # for adding skip connections

# compression.py
# hyper-params for multiple kernel strategy (see 3.3 & 6.3)
co_step_n = 50                                              # number of search steps
co_kernel_n = 4                                             # number of kernels, as well as evaluated architectures in each search step
co_best_n = 4                                               # number of saved best architectures during search, all of which will be fully trained
# hyper-params for stopping criterion of kernel optimization
co_alpha = 0.5
co_beta = 0.001
co_gamma = 0.5

# kernel.py
# hyper-params for kernels (see 3.1)
ke_alpha = 0.01
ke_beta = 0.05
ke_gamma = 1
ke_input_size = ar_n_supported_layer + 6 + ar_max_layers * 2
ke_hidden_size = 64
ke_num_layers = 4
ke_bidirectional = True
ke_lr = 0.001
ke_weight_decay = 5e-4

# training.py
tr_target_acc = 40.22
tr_iterations = 4000
tr_ims_per_batch = 4
tr_dist_url = 'tcp://127.0.0.1:49062'
