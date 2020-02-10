import torch
import torch.nn as nn

from detectron2.modeling.backbone import ESNACLayer
import ESNAC.options as opt

def get_graph_resnet(resnet):
    n = 0
    V = []
    x = []
    prev = []
    out_layers = {}

    def add_node(node, prev_node=None):
        nonlocal n
        if prev_node is None:
            prev_node = n - 1
        if prev_node == -1:
            input = torch.rand(1, 3, 256, 256)
        elif isinstance(prev_node, list):
            input = sum([x[j] for j in prev_node])
        else:
            input = x[prev_node]
        output = node(input)
        x.append(output)
        in_shape = input.shape
        out_shape = output.shape

        V.append(ESNACLayer(node, in_shape, out_shape))
        prev.append(prev_node)
        n += 1

    def parse_bottleneck(btn):
        last_node = n - 1
        btn.conv1.activation = nn.ReLU()
        add_node(btn.conv1)
        btn.conv2.activation = nn.ReLU()
        add_node(btn.conv2)
        add_node(btn.conv3)
        bn3_node = n - 1
        if btn.shortcut is not None:
            add_node(btn.shortcut, last_node)
            add_node(nn.ReLU(), [bn3_node, n - 1])
        else:
            add_node(nn.ReLU(), [last_node, bn3_node])

    def parse_stage(stg, name):
        for i in range(len(stg)):
            parse_bottleneck(stg[i])
        out_layers[n - 1] = name

    resnet.stem.conv1.activation = nn.ReLU()
    add_node(resnet.stem.conv1)
    add_node(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    parse_stage(resnet.res2, 'res2')
    parse_stage(resnet.res3, 'res3')
    parse_stage(resnet.res4, 'res4')
    parse_stage(resnet.res5, 'res5')

    E = [([False] * n) for i in range(n)]
    for i in range(1, n):
        if isinstance(prev[i], list):
            for j in prev[i]:
                E[j][i] = True
        else:
            E[prev[i]][i] = True

    return n, V, E, out_layers

class LayerGroup(object):
    def __init__(self, F, in_layers, out_layers):
        self.F = F
        self.in_layers = set(in_layers)
        self.out_layers = set(out_layers)
        self.union = self.in_layers | self.out_layers
        self.inter = self.in_layers & self.out_layers
        self.in_only = self.union - self.out_layers
        self.out_only = self.union - self.in_layers

def get_groups(V):
    n = len(V)
    vis = [([False] * 2) for i in range(n)]
    vis[0][0] = True
    groups = []
    for i in range(n):
        for j in range(2):
            if not vis[i][j]:
                F = V[i].out_shape[1] if j else V[i].in_shape[1]
                in_layers = []
                out_layers = []
                for k in range(n):
                    if not vis[k][0] and V[k].in_shape[1] == F:
                        in_layers.append(k)
                        vis[k][0] = True
                    if not vis[k][1] and V[k].out_shape[1] == F:
                        out_layers.append(k)
                        vis[k][1] = True
                groups.append(LayerGroup(F, in_layers, out_layers))
    return groups

def get_links(E):
    n = len(E)
    in_links = [[] for i in range(n)]
    out_links = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if E[i][j]:
                in_links[j].append(i)
                out_links[i].append(j)
    return in_links, out_links

def get_plot(name, n, V, E, reduced=False):
    from graphviz import Digraph
    dot = Digraph(name=name)
    colors = {
        'ESNACIdentity': 'gray',
        'ESNACFlatten': 'gray',
        'Conv2d': 'red',
        'MaxPool2d': 'yellow',
        'AvgPool2d': 'yellow',
        'ReLU': 'green',
        'Linear': 'blue',
    }
    for i, v in enumerate(V):
        node_name = '%d %s %s->%s' % (i, v.base_type, 
            str(v.in_shape[1:]), str(v.out_shape[1:]))
        if v.base_type != 'Identity' or not reduced:
            color = colors[v.base_type]
            dot.node(str(i), node_name, shape='box', color=color)
    if reduced:
        for i in range(n):
            if V[i].base_type == 'Identity':
                in_links = []
                out_links = []
                for j in range(n):
                    if E[j][i]:
                        in_links.append(j)
                        E[j][i] = False
                    if E[i][j]:
                        out_links.append(j)
                        E[i][j] = False
                for u in in_links:
                    for v in out_links:
                        E[u][v] = True
    for i in range(n):
        for j in range(n):
            if E[i][j]:
                dot.edge(str(i), str(j))
    dot.view()
