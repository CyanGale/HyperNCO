import torch

from torch.nn.modules import GELU, ReLU, LeakyReLU
from src import Net, loss_partitioning_onehot_qubo
from src import from_file_to_graph, init, get_device, run_qubo, Layer, LayerType, Datasets


if __name__ == "__main__":
    init(cuda_index=1, reproducibility=False)

    data_path = Datasets.Graph_Citeseer.path
    graph = from_file_to_graph(data_path, True, True).to(get_device())

    init_feature_dim = 1024
    k = 6
    x = torch.rand((graph.num_v, init_feature_dim))
    
    layers = [
        Layer(LayerType.GAT, init_feature_dim, 256, hidden_channels=1024, num_layers=3, jk="cat", drop_rate=0, v2=True, act=GELU()),
        Layer(LayerType.LINEAR, 256, k, use_bn=True, drop_rate=0.2),
    ]

    gini_cof_lambda = lambda e, n: (-750 + 0.22 * e) / 1000 if e < 4000 else (-800 + 0.25 * e) / 1000 +(e-4000)*0.01
    obj_cof_lambda = lambda e, n: e / 900
    cons_cof_lambda = lambda e, n: e / 1150
    cons_cof_lambda = lambda e, n: e / 1150 if e < 4000 else e / 1150+(e-4000)*0.02

    net = Net(layers)
    loss, outs = run_qubo(
        "partitioning",
        net,
        x,
        graph,
        5000,
        loss_partitioning_onehot_qubo,
        1e-4,
        opt="AdamW",
        gini_cof_lambda=gini_cof_lambda,
        obj_cof_lambda=obj_cof_lambda,
        cons_cof_lambda=cons_cof_lambda,
        clip_grad=False,
    )
