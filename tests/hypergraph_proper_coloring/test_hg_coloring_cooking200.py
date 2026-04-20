import torch
import dhg
from src import DualHeadNet, Datasets, Layer, LayerType
from src import init, get_device, from_file_to_hypergraph, run_pubo

if __name__ == "__main__":
    # Best sol cooking200 proper coloring:  3 in 6 s
    # scip 3 in 313.82s 2 in 851.19s
    init(cuda_index=0, reproducibility=True)
    data_path = Datasets.Hypergraph_cooking200.path
    hg = from_file_to_hypergraph(data_path, True).to(get_device())
    init_dim = 1024
    layers = [
        [Layer(LayerType.GRAPHSAGE, init_dim, 512, hidden_channels=512, num_layers=2, jk="last", drop_rate=0)],
        [],
        [Layer(LayerType.LINEAR, 512, 5, use_bn=True, dropout=0.2)],
        [Layer(LayerType.LINEAR, 512, 5, use_bn=True, dropout=0)],
    ]
    x = torch.rand(hg.num_v, init_dim)
    net = DualHeadNet(layers[0], layers[1], layers[2], layers[3])

    gini_cons_lambda = lambda e, n: (-200 + 1 * e) / 1000
    obj_cof_lambda = lambda e, n: 250 - e / 100
    obj_cons_cof_lambda = lambda e, n: 100 - e / 100
    cons_cof_lambda = lambda e, n:  5 + e / 1000

    loss, outs = run_pubo(
        "coloring",
        net,
        x,
        hg,
        500,
        4e-4,
        "AdamW",
        clip_grad=True,
        simple=True,
        gini_cons_cof_lambda=gini_cons_lambda,
        obj_cof_lambda=obj_cof_lambda,
        cons_cof_lambda=cons_cof_lambda,
        obj_cons_cof_lambda=obj_cons_cof_lambda,
    )
