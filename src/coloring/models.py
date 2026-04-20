from typing import List, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool


class DualHeadNet(nn.Module):
    def __init__(
        self,
        gnn_layers,
        shared_layers,
        cons_layers,
        obj_layers,
    ):
        super().__init__()
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.shared_layers = nn.ModuleList(shared_layers)
        self.cons_layers = nn.ModuleList(cons_layers)
        self.obj_layers = nn.ModuleList(obj_layers)
        self.act = nn.ModuleDict({"softmax": nn.Softmax(dim=1), "sigmoid": nn.Sigmoid()})

    def forward(self, x: torch.Tensor, graph: object, edge_index, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.gnn_layers:
            x = layer(x, graph, edge_index)
        for layer in self.shared_layers:
            x = layer(x)
        cons = x
        if len(self.cons_layers) != 0:
            for layer in self.cons_layers:
                cons = layer(cons)

        obj = x
        if len(self.obj_layers) != 0:
            for layer in self.obj_layers:
                obj = layer(obj)

        cons = self.act["softmax"](cons)
        # obj = global_max_pool(self.softmax(obj), batch=torch.zeros(obj.shape[0], dtype=torch.long))
        obj = self.act["sigmoid"](global_max_pool(obj, batch=torch.zeros(obj.shape[0], dtype=torch.long)))

        return (cons, obj)


class StreamNet(nn.Module):
    def __init__(
        self,
        layers,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.act = nn.ModuleDict({"softmax": nn.Softmax(dim=1), "sigmoid": nn.Sigmoid()})

    def forward(self, x: torch.Tensor, graph: object, edge_index, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x, graph, edge_index)

        cons = self.act["softmax"](x)

        obj = global_max_pool(cons, batch=torch.zeros(cons.shape[0], dtype=torch.long))

        return (cons, obj)


class DualHeadAttentionNet(nn.Module):
    def __init__(
        self,
        shared_layers,
        cons_layers,
        obj_layers,
    ):
        super().__init__()
        self.shared_layers = nn.ModuleList(shared_layers)
        self.cons_layers = nn.ModuleList(cons_layers)
        self.obj_layers = nn.ModuleList(obj_layers)
        self.act = nn.ModuleDict({"softmax": nn.Softmax(dim=1), "sigmoid": nn.Sigmoid()})

    def forward(self, x: torch.Tensor, graph: object, edge_index, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.shared_layers:
            x = layer(x, graph, edge_index)

        cons = x
        if len(self.cons_layers) != 0:
            for layer in self.cons_layers:
                cons = layer(cons)

        obj = x.t()
        if len(self.obj_layers) != 0:
            for layer in self.obj_layers:
                obj = layer(obj)

        cons = self.act["softmax"](cons)
        obj = self.act["sigmoid"](obj.squeeze(1))
        return (cons, obj)
    
class DirectProbModel(nn.Module):
    def __init__(self, num_nodes, num_classes):
        super().__init__()
        self.node_color = nn.Parameter(torch.randn(num_nodes, num_classes))

    def forward(self, X, *args, **kwargs):
        out_cons = torch.softmax(self.node_color, dim=-1)  # [V, K]

        out_obj = torch.sigmoid(out_cons.max(dim=0, keepdim=True)[0])  # [1, K]

        return out_cons, out_obj
