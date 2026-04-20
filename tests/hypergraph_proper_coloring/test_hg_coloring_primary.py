import sys
import os
import time
import torch
import dhg
from src import DualHeadNet, Datasets, Layer, LayerType
from src import init, get_device, from_file_to_hypergraph, run_pubo

if __name__ == "__main__":
    # Best sol primary proper coloring: 31 in 19.5s
    init(cuda_index=1 , reproducibility=False)
    data_path = Datasets.Hypergraph_primary.path
    hg = from_file_to_hypergraph(data_path, True).to(get_device())
    init_dim = 1024
    layers = [
        [Layer(LayerType.GRAPHSAGE, init_dim, 512, hidden_channels=1024, num_layers=3, jk="last", drop_rate=0)],
        [],
        [Layer(LayerType.LINEAR, 512, 35, use_bn=True, dropout=0)],
        [Layer(LayerType.LINEAR, 512, 35, use_bn=True, dropout=0)],
    ]
    x = torch.rand(hg.num_v, init_dim)
    net = DualHeadNet(layers[0], layers[1], layers[2], layers[3])
    
    gini_cons_lambda = lambda e, n: (-200 + 0.35 * e) / 1000
    obj_cof_lambda = lambda e, n: e / 80
    obj_cons_cof_lambda = lambda e, n: e / 100
    cons_cof_lambda = lambda e, n: e / 10

    loss, outs,rs = run_pubo(
        "coloring",
        net,
        x,
        hg,
        500,
        4e-4,
        "AdamW",
        clip_grad=True,
        simple=True,
        gini_cons_cof_lambda=gini_cons_lambda,
        obj_cof_lambda=obj_cof_lambda,
        cons_cof_lambda=cons_cof_lambda,
        obj_cons_cof_lambda=obj_cons_cof_lambda,
    )
