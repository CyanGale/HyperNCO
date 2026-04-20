import sys
import os
from src import from_file_to_graph, init, get_device, run_qubo
from src.core import Layer, LayerType
from src.maxcut import loss_maxcut_onehot_qubo  

import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import time
import pandas as pd
from src import generate_data
from src.maxcut import loss_maxcut_onehot_qubo, Net
from src import from_file_to_graph, init, get_device, run_qubo, Layer, LayerType, Datasets

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DirectProbModel(nn.Module):
    def __init__(self, num_nodes, num_classes):
        super().__init__()
        self.h = nn.Parameter(torch.randn(num_nodes, num_classes)) 

    def forward(self, X, *args, **kwargs):
        p = torch.softmax(self.h, dim=-1)
        return (p,)

if __name__ == "__main__":

    init(cuda_index=1, reproducibility=False)
    device = get_device()
    data_path = Datasets.Graph_eat.path
    data = from_file_to_graph(data_path, True, True).to(get_device())
    num_group = 6
    num_nodes = data.num_v
    total_runs = 10  
    all_cuts = []    

    for run_idx in range(1, total_runs + 1):
        set_seed(run_idx)
        net = DirectProbModel(num_nodes, num_group).to(device)
        x = torch.randn(1, 1).to(device)
        loss, outs, eval_result = run_qubo(
            "maxcut",
            net,
            x,
            data,
            15000,
            1.0,
            "rmsprop",
            True,
            loss_maxcut_onehot_qubo,
        )
        cut_value = eval_result["cut_edges"]
        all_cuts.append({
            "round": run_idx,
            "seed": run_idx,
            "cut": cut_value
        })
    for item in all_cuts:
        print(f" {item['round']} |  {item['seed']} | cut = {item['cut']}")