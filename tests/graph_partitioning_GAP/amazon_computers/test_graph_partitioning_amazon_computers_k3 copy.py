import torch

from torch.nn.modules import GELU, ReLU, LeakyReLU
from src import Net, loss_partitioning_onehot_qubo
from src import from_file_to_graph, init, get_device, run_qubo, Layer, LayerType, Datasets


if __name__ == "__main__":
    init(cuda_index=1, reproducibility=False)

    data_path = Datasets.Graph_Amazon_PC.path
    graph = from_file_to_graph(data_path, True, True).to(get_device())

    init_feature_dim = 4096
    k = 3
    x = torch.rand((graph.num_v, init_feature_dim))
    
    layers = [
        Layer(LayerType.GAT, init_feature_dim, 1024, hidden_channels=2048, num_layers=2, jk="cat", drop_rate=0.5, v2=True, act=GELU()),
        #Layer(LayerType.LINEAR, 1024, 512, use_bn=False, drop_rate=0.2),
        Layer(LayerType.LINEAR, 1024, k, use_bn=True, drop_rate=0.2),
    ]

    gini_cof_lambda = lambda e, n: (-850 + 0.22 * e) / 1000
    obj_cof_lambda = lambda e, n: e / 900
    cons_cof_lambda = lambda e, n: e / 1150

    net = Net(layers)
    loss, outs = run_qubo(
        "partitioning",
        net,
        x,
        graph,
        8000,
        loss_partitioning_onehot_qubo,
        1e-4,
        opt="AdamW",
        gini_cof_lambda=gini_cof_lambda,
        obj_cof_lambda=obj_cof_lambda,
        cons_cof_lambda=cons_cof_lambda,
        clip_grad=False,
    )
