import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
import random
import numpy as np
import torch
import torch.nn as nn

from src import loss_partitioning_onehot_qubo
from src import from_file_to_graph, init, get_device, run_qubo
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
        self.h = nn.Parameter(torch.randn(num_nodes, num_classes))

    def forward(self, X, *args, **kwargs):
        p = torch.softmax(self.h, dim=-1)
        return (p,)

if __name__ == "__main__":
    init(cuda_index=1, reproducibility=False)
    results = []
    for run_idx in range(1, 11):
        set_seed(run_idx * 10)
        graph = from_file_to_graph(Datasets.Graph_uat.path, True, True).to(get_device())
        k=3
        net = DirectProbModel(graph.num_v, k).to(get_device())

        gini_cof_lambda = lambda e, n: (-800 + 0.25 * e) / 1000
        obj_cof_lambda = lambda e, n: e / 400
        cons_cof_lambda = lambda e, n: e / 950

        loss, outs, res = run_qubo(
            type="partitioning", net=net, X=torch.randn(1,1).to(get_device()),
            graph=graph, num_epochs=10000, loss_func=loss_partitioning_onehot_qubo,
            lr=0.1, 
            opt='AdamW', 
            gini_cof_lambda=gini_cof_lambda,
            obj_cof_lambda=obj_cof_lambda,
            cons_cof_lambda=cons_cof_lambda,
            clip_grad=False, 
            evaluate=True
        )

        results.append([run_idx, res["cuts"], res["blce"]])


    for idx, c, b in results:
        print(f"{idx:2d}| cuts={c:4d} | blce={b}")
    