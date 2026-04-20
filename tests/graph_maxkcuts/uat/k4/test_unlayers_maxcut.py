import sys
import os

# 强行把项目根目录加入路径（绝对不会错）
ROOT = "/home/guohao/k-grouping"
sys.path.insert(0, ROOT)

sys.path.append('.')
from src import from_file_to_graph, init, get_device, run_qubo
from src.core import Layer, LayerType
from src.maxcut import loss_maxcut_onehot_qubo  # 补上你漏的

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
        self.h = nn.Parameter(torch.randn(num_nodes, num_classes))  # 这里受种子控制！

    def forward(self, X, *args, **kwargs):
        p = torch.softmax(self.h, dim=-1)
        return (p,)

if __name__ == "__main__":

    init(cuda_index=1, reproducibility=False)
    device = get_device()

   
    data_path = Datasets.Graph_uat.path
    data = from_file_to_graph(data_path, True, True).to(get_device())

    print("\n" + "="*50)
    print("📊 图信息：")
    print(f"节点数: {data.num_v}")
    print(f"总边数: {data.num_e}")
    print("="*50 + "\n")

    num_group = 4
    num_nodes = data.num_v
    total_runs = 10  
    all_cuts = []    

    for run_idx in range(1, total_runs + 1):
        print(f"\n\n{'='*60}")
        print(f"🚀 第 {run_idx} 轮 | 随机种子 = {run_idx}")
        print('='*60)


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

    print("\n\n" + "="*70)
    print("📊 10 轮实验 cut 值汇总")
    print("="*70)
    for item in all_cuts:
        print(f"轮次 {item['round']} | 种子 {item['seed']} | cut = {item['cut']}")