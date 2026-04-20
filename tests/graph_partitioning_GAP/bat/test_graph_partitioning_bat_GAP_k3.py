import torch

from src import Net, loss_partitioning_onehot_qubo
from src import from_file_to_graph, init, get_device, run_qubo, Layer, LayerType, Datasets


if __name__ == "__main__":
    init(cuda_index=1, reproducibility=False)

    data_path = Datasets.Graph_bat.path
    graph = from_file_to_graph(data_path, True, True).to(get_device())

    init_feature_dim = 2048
    k = 3
    x = torch.rand((graph.num_v, init_feature_dim))
    
    layers = [
        Layer(LayerType.GAT, init_feature_dim, 256, hidden_channels=512, num_layers=4, jk="cat", drop_rate=0.5, v2=True, act="leaky_relu"),
        Layer(LayerType.LINEAR, 256, k, use_bn=True, drop_rate=0.2),
    ]

    gini_cof_lambda = lambda e, n: 0
    obj_cof_lambda = lambda e, n: e / 900
    cons_cof_lambda = lambda e, n: e / 1000 +e*0.004

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
