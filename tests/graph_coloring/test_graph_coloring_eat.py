import torch

from src import DualHeadNet, loss_coloring_onehot_qubo
from src import from_file_to_graph, init, get_device, run_qubo, Layer, LayerType, Datasets


if __name__ == "__main__": 
    # Find Best sol bat: 29 1.5s
    init(cuda_index=0, reproducibility=False)
    init_feature_dim = 512
    data_path = Datasets.Graph_eat.path
    graph = from_file_to_graph(data_path, True, True).to(get_device())
    x = torch.rand((graph.num_v, init_feature_dim))

    gnn_layers = [Layer(LayerType.GRAPHSAGE, init_feature_dim, 512, hidden_channels=512, num_layers=2, jk="last", drop_rate=0)]
    shared_layers = []
    cons_layers = [Layer(LayerType.LINEAR, 512, 100, use_bn=True, drop_rate=0)]
    obj_layers = [Layer(LayerType.LINEAR, 512, 100, use_bn=True, drop_rate=0.2)]

    net = DualHeadNet(gnn_layers, shared_layers, cons_layers, obj_layers)

    gini_cons_lambda = lambda e, n: (-200 + 1 * e) / 50
    obj_cof_lambda = lambda e, n: e / 110
    obj_cons_cof_lambda = lambda e, n: e / 100
    cons_cof_lambda = lambda e, n: e / 10
    
    loss, outs = run_qubo(
        "coloring",
        net,
        x,
        graph,
        600,
        loss_coloring_onehot_qubo,
        4e-4,
        clip_grad=True,
        gini_cons_cof_lambda=gini_cons_lambda,
        obj_cof_lambda=obj_cof_lambda,
        cons_cof_lambda=cons_cof_lambda,
        obj_cons_cof_lambda=obj_cons_cof_lambda,
    )