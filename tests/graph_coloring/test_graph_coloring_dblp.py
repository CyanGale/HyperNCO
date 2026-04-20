import torch

from src import loss_coloring_onehot_qubo, from_file_to_graph, init, get_device, run_qubo
from src import Layer, LayerType, Datasets, DualHeadNet

if __name__ == "__main__":
    # Find Best sol dblp: 7 3.9s
    # if reproducibility as `False` can find better sol 7 sometime but can get 8 as `True` 
    init(cuda_index=0, reproducibility=True)
    init_feature_dim = 1024
    data_path = Datasets.Graph_dblp.path
    graph = from_file_to_graph(data_path, True, True).to(get_device())
    x = torch.rand((graph.num_v, init_feature_dim))

    gnn_layers = [Layer(LayerType.GRAPHSAGE, init_feature_dim, 512, hidden_channels=512, num_layers=2, jk="last", drop_rate=0)]
    shared_layers = []
    cons_layers = [Layer(LayerType.LINEAR, 512, 100, drop_rate=0.05)]
    obj_layers = [Layer(LayerType.LINEAR, 512, 100, drop_rate=0.2)]

    net = DualHeadNet(gnn_layers, shared_layers, cons_layers, obj_layers)

    gini_cons_lambda = lambda e, n: (-150 + 0.25 * e) / 1000
    obj_cof_lambda = lambda e, n: e / 450
    cons_cof_lambda = lambda e, n: e / 448
    obj_cons_cof_lambda = lambda e, n: e / 1000

    loss, outs = run_qubo(
        "coloring",
        net,
        x,
        graph,
        1000,
        loss_coloring_onehot_qubo,
        3e-4,
        clip_grad=True,
        gini_cons_cof_lambda=gini_cons_lambda,
        obj_cof_lambda=obj_cof_lambda,
        cons_cof_lambda=cons_cof_lambda,
        obj_cons_cof_lambda=obj_cons_cof_lambda,
    )