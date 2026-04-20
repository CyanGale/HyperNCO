from .. import Layer
from typing import List

import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, layers: List[Layer]):
        super(Net, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, graph=None, edge_index=None, edge_weight=None, **kwargs):
        for layer in self.layers:
            x = layer(x, graph, edge_index, edge_weight, **kwargs)
        x = self.softmax(x)
        return (x,)
class DirectProbModel(nn.Module):
    def __init__(self, num_nodes, num_classes):
        super().__init__()
        self.h = nn.Parameter(torch.randn(num_nodes, num_classes))  

    def forward(self, X, *args, **kwargs):
        p = torch.softmax(self.h, dim=-1)
        return (p,)
