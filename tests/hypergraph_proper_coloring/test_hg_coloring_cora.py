import torch
import dhg
from src import DualHeadNet, Datasets, Layer, LayerType
from src import init, get_device, from_file_to_hypergraph, run_pubo

if __name__ == "__main__":
    # Best sol Cora proper coloring: 4 in 8s
    # scip: 3 in 29.27s
    init(cuda_index=0, reproducibility=True)
    data_path = Datasets.Hypergraph_Cora.path
    hg = from_file_to_hypergraph(data_path, True).to(get_device())
    g = dhg.Graph.from_hypergraph_clique(hg)
    init_dim = 1024
    layers = [
        [
        Layer(LayerType.GRAPHSAGE, init_dim, 256, hidden_channels=512, num_layers=3, jk="last", drop_rate=0)
        ],
        [],
        [Layer(LayerType.LINEAR, 256, 100, use_bn=True, dropout=0)],
        [Layer(LayerType.LINEAR, 256, 100, use_bn=True, dropout=0)],
    ]
    x = torch.rand(hg.num_v, init_dim)
    net = DualHeadNet(layers[0], layers[1], layers[2], layers[3])

    gini_cons_lambda = lambda e, n: (-200 + 0.25 * e) / 1000
    obj_cof_lambda = lambda e, n: 200 - e / 100  
    obj_cons_cof_lambda = lambda e, n:  100 - e / 100 
    cons_cof_lambda = lambda e, n: 1 + e / 100

    loss, outs = run_pubo(
        "coloring",
        net,
        x,
        hg,
        1200,
        3e-4,
        "AdamW",
        clip_grad=True,
        gini_cons_cof_lambda=gini_cons_lambda,
        obj_cof_lambda=obj_cof_lambda,
        cons_cof_lambda=cons_cof_lambda,
        obj_cons_cof_lambda=obj_cons_cof_lambda,
    )
