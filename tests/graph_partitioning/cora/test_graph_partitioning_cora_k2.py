import sys
import os
import torch
import torch.nn as nn
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  

from src import (
    Layer, LayerType, DualHeadNet, Datasets, 
    init, get_device, from_file_to_hypergraph, run_pubo,loss_partitioning_onehot_pubo
)
init(cuda_index=0, reproducibility=True)
device = get_device()
print(f"Using device: {device}")
data_path = Datasets.Hypergraph_Cora.path
hg = from_file_to_hypergraph(data_path, True).to(device)

num_nodes = hg.num_v
input_dim = 1024
k=2
X = torch.randn(num_nodes, input_dim).to(device)

layers = [
    [
        Layer(LayerType.HGNNPCONV, input_dim, 512, hidden_channels=512, num_layers=2, jk="last"),
    ],
    [],
    [
        Layer(LayerType.LINEAR, 512, k, use_bn=True)
    ],
    [
        Layer(LayerType.LINEAR,512, k, use_bn=True)
    ],
]

net = DualHeadNet(layers[0], layers[1], layers[2], layers[3]).to(device)
print(net)
gini_cof_lambda = lambda e, n: (-800 + 0.25 * e) / 1000
obj_cof_lambda = lambda e, n: e / 900
cons_cof_lambda = lambda e, n: e / 1000 +e*0.01


loss, outs,res = run_pubo(
    type="partitioning", 
    net=net, 
    X=X, 
    hypergraph=hg, 
    num_epochs=5000,
    loss_func=loss_partitioning_onehot_pubo, # 直接传入
    lr=3e-4, 
    opt='AdamW', 
    gini_cof_lambda=gini_cof_lambda,
    obj_cof_lambda=obj_cof_lambda,
    cons_cof_lambda=cons_cof_lambda,
    clip_grad=False,
)
# print(f"\nResult: Cuts = {res['cuts']}, Balance = {res['blce']}")