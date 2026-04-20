import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
import random
import numpy as np
import torch
import torch.nn as nn

from src import loss_partitioning_onehot_qubo
from src import from_file_to_graph, init, get_device, run_qubo, from_file_to_hypergraph, run_pubo
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
    all_cuts = [] 
    for run_idx in range(1, 11):
        set_seed(run_idx * 10)
        data_path = Datasets.Hypergraph_Cora.path
        hg = from_file_to_hypergraph(data_path, True).to(get_device())

        net = DirectProbModel(hg.num_v, 2).to(get_device())
        gini_cof_lambda = lambda e, n: (-200 + 0.25 * e) / 10

        loss, outs, eval_result = run_pubo(
            type="maxcut", net=net, X=torch.randn(1,1).to(get_device()),
            hypergraph=hg, num_epochs=5000,
            lr=0.9, 
            opt='AdamW', 
            gini_cof_lambda=gini_cof_lambda,
            clip_grad=False, 
            evaluate=True
        )
        cut_value = eval_result["cut_edges"]
        all_cuts.append({
            "round": run_idx,
            "seed": run_idx,
            "cut": cut_value
        })


    for item in all_cuts:
        print(f" {item['round']} |  {item['seed']} | cut = {item['cut']}")