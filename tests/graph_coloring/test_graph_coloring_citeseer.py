import torch
from src import DualHeadNet, loss_coloring_onehot_qubo
from src import from_file_to_graph, init, get_device, run_qubo, Layer, LayerType, Datasets


if __name__ == "__main__":
    # Find Best sol Citeseer: 7 20s
    init(cuda_index=0, reproducibility=True)
    init_feature_dim = 1024
    data_path = Datasets.Graph_Citeseer.path
    graph = from_file_to_graph(data_path, True, True).to(get_device())
    x = torch.rand((graph.num_v, init_feature_dim))

    gnn_layers = [Layer(LayerType.GRAPHSAGE, init_feature_dim, 512, hidden_channels=512, num_layers=2, jk="last", drop_rate=0)]
    shared_layers = [Layer(LayerType.LINEAR, 512, 256, use_bn=False, drop_rate=0.4)]
    cons_layers = [Layer(LayerType.LINEAR, 256, 100, use_bn=True, drop_rate=0)]
    obj_layers = [Layer(LayerType.LINEAR, 256, 100, use_bn=True, drop_rate=0)]

    net = DualHeadNet(gnn_layers, shared_layers, cons_layers, obj_layers)

    gini_cons_lambda = lambda e, n: (-200 + 0.25 * e) / 1000
    obj_cof_lambda = lambda e, n: e / 110
    obj_cons_cof_lambda = lambda e, n: e / 140
    cons_cof_lambda = lambda e, n: e / 160
    
    loss, outs = run_qubo(
        type="coloring",
        net=net,
        X=x,
        graph=graph,
        num_epochs=4000,
        loss_func=loss_coloring_onehot_qubo,
        lr=1e-4,
        opt='AdamW',
        clip_grad=True,
        gini_cons_cof_lambda=gini_cons_lambda,
        obj_cof_lambda=obj_cof_lambda,
        cons_cof_lambda=cons_cof_lambda,
        obj_cons_cof_lambda=obj_cons_cof_lambda,
        evaluate=False
    )
