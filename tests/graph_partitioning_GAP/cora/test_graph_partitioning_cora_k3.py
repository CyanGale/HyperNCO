import torch
from torch.nn.modules import GELU, ReLU, LeakyReLU
from src import Net, loss_partitioning_onehot_qubo
from src import from_file_to_graph, init, get_device, run_qubo, Layer, LayerType, Datasets
from src.utils import edge_weight

if __name__ == "__main__":
    # Find Best sol Cora k3: CUTS: 260 BLCE 0.33
    init(cuda_index=0, reproducibility=False)

    data_path = Datasets.Graph_Cora.path
    graph = from_file_to_graph(data_path, True, False).to(get_device())

    init_feature_dim = 2048
    k = 3
    x = torch.rand((graph.num_v, init_feature_dim))

    layers = [
        Layer(LayerType.GAT, init_feature_dim, 512, hidden_channels=2048, num_layers=3, jk="last", drop_rate=0.1, v2=True, act=GELU()),
        Layer(LayerType.LINEAR, 512, 256, use_bn=False, drop_rate=0),
        Layer(LayerType.LINEAR, 256, 64, use_bn=False, drop_rate=0),
        Layer(LayerType.LINEAR, 64, k, use_bn=True, drop_rate=0),
    ]

    # gini_cof_lambda = lambda e, n: (-490 + 0.25 * e) / 1000
    # obj_cof_lambda = lambda e, n: e / 2000
    # cons_cof_lambda = lambda e, n: obj_cof_lambda(e, n) / 100

    gini_cof_lambda = lambda e, n: (-490 + 0.25 * e) * 1e-3
    obj_cof_lambda = lambda e, n: e / 2000
    cons_cof_lambda = lambda e, n: e / 2_000_00

    net = Net(layers)
    loss, outs = run_qubo(
        "partitioning",
        net,
        x,
        graph,
        3500,
        loss_partitioning_onehot_qubo,
        1e-4,
        opt="Adam",
        gini_cof_lambda=gini_cof_lambda,
        obj_cof_lambda=obj_cof_lambda,
        cons_cof_lambda=cons_cof_lambda,
        clip_grad=True,
        edge_weight=edge_weight(graph.e[0], 40),
    )
