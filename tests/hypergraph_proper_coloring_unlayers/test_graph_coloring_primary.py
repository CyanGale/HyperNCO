import sys
import os
import time

BASE_DIR = "/home/guohao/k-grouping"
sys.path.append(BASE_DIR)
import time
import random
import numpy as np
import torch
import torch.nn as nn

from src import loss_coloring_onehot_qubo
from src import from_file_to_graph, init, get_device, run_qubo,run_pubo, from_file_to_hypergraph
from src import Layer, LayerType, Datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DirectProbModel(nn.Module):
    def __init__(self, num_nodes, num_classes):
        super().__init__()
        self.node_color = nn.Parameter(torch.randn(num_nodes, num_classes))

    def forward(self, X, *args, **kwargs):
        out_cons = torch.softmax(self.node_color, dim=-1)  # [V, K]

        out_obj = torch.sigmoid(out_cons.max(dim=0, keepdim=True)[0])  # [1, K]

        return out_cons, out_obj

if __name__ == "__main__":
    init(cuda_index=1, reproducibility=False)
    results = []  


    for run_idx in range(1, 11):
        set_seed(run_idx * 10)
        hg = from_file_to_hypergraph(Datasets.Hypergraph_primary.path, True).to(get_device())
        net = DirectProbModel(hg.num_v, 100).to(get_device())
        
        gini_cons_lambda = lambda e, n: (-200 + 0.35 * e) / 1000
        obj_cof_lambda = lambda e, n: e / 80
        obj_cons_cof_lambda = lambda e, n: e /10000
        cons_cof_lambda = lambda e, n: e / 10

        start_time = time.time()

        loss, outs, res = run_pubo(
            type="coloring",
            net=net,
            X=torch.randn(1, 1).to(get_device()),
            hypergraph=hg,
            num_epochs=500,
            lr=0.3,
            opt='AdamW',
            clip_grad=True,
            gini_cons_cof_lambda=gini_cons_lambda,
            obj_cof_lambda=obj_cof_lambda,
            cons_cof_lambda=cons_cof_lambda,
            obj_cons_cof_lambda=obj_cons_cof_lambda,
            evaluate=True
        )

        elapsed_time = round(time.time() - start_time, 2)

        results.append([
            run_idx,
            res["num_color"],        
            res["accuracy"],         
            res["correct_edges"],    
            res["total_edges"],      
            elapsed_time             
        ])


    for idx, color, acc, correct, total, t in results:
         print(f"{idx:2d} | colors={color:2d} | accuracy={acc:.1%} | correct_edges={correct}/{total} | times={t:4.2f}s")
