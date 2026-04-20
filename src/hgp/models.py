import torch
import torch.nn as nn
import dhg
from dhg.nn import HGNNPConv
from dhg.nn import UniSAGEConv, UniGATConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ParameterDict(dict):
    # fmt: off
    r"""
    A dictionary that only accepts string keys and dictionary values with required keys.  
    
    **We should use this class to define the parameters of the HGNNP model, activally HGNNP's each layers**
    
    ## Example:
    
    .. code-block:: python
        from models import HGNNP, ParameterDict
        
        parameters = ParameterDict()
        parameters["convlayer1"] = {"in_channels": 512, "out_channels": 256, "use_bn": True, "drop_rate": 0.5}
        parameters["sagelayer2"] = {"in_channels": 256, "out_channels": 3, "use_bn": True, "drop_rate": 0.5}
        net = CHGNN(parameters)


    ## HGNNPConv Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels{x}`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """
    # fmt: on

    def __init__(self) -> None:
        super().__init__()
        self.REQUIRED_KEYS = {"in_channels", "out_channels", "use_bn", "drop_rate"}

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError(f"Key must be a string. Got {type(key).__name__}.")

        if not isinstance(value, dict):
            raise ValueError(f"Value must be a dictionary. Got {type(value).__name__}.")

        super().__setitem__(key, value)


class BDLiner(nn.Module):
    r"""
    a simple linear model with batch normalization, dropout and relu activation function.
    """

    def __init__(self, parameters: ParameterDict) -> None:
        
        super().__init__()
        self.layers = nn.ModuleList()
        self.parameterDict = parameters
        for idx, (k, v) in enumerate(parameters.items()):
            if str.startswith(k, "liner"):
                self.layers.append(nn.BatchNorm1d(num_features=v["in_channels"])) if v["use_bn"] else None
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(v["drop_rate"]))
                self.layers.append(
                    nn.Linear(
                        in_features=v["in_channels"], out_features=v["out_channels"]
                    )
                )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X)

        return X


class CHGNN(nn.Module):
    r"""
    A complex model which can contains multiple HGNNPConv layers , TransformerEncoder layers and UniSAGEConv layers.
    Args:
        ``parameters`` (:class: `ParameterDict` ): A dictionary that contains the parameters of the HGNNP model.
    """

    def __init__(
        self,
        parameters: ParameterDict,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.parameterDict = parameters

        for idx, (k, v) in enumerate(parameters.items()):
            is_last = idx == len(parameters) - 1
            if str.startswith(k, "conv"):
                self.layers.append(
                    HGNNPConv(
                        v["in_channels"],
                        v["out_channels"],
                        use_bn=v["use_bn"],
                        drop_rate=v["drop_rate"],
                        is_last=is_last,
                    )
                )
            elif str.startswith(k, "tf"):
                en_layer = TransformerEncoderLayer(
                    v["channels"],
                    nhead=v["nhead"],
                    dim_feedforward=v["dim_feedforward"],
                    dropout=v["drop_rate"],
                )
                self.layers.append(
                    TransformerEncoder(
                        encoder_layer=en_layer, num_layers=v["num_layers"]
                    )
                )
            elif str.startswith(k, "sage"):
                self.layers.append(
                    UniSAGEConv(
                        v["in_channels"],
                        v["out_channels"],
                        drop_rate=v["drop_rate"],
                        is_last=is_last,
                    )
                )
            else:
                assert "NOT TOUCHABLE"

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for (k, _), layer in zip(self.parameterDict.items(), self.layers):
            if str.startswith(k, "conv"):
                X = layer(X, hg)
            elif str.startswith(k, "tf"):
                X = layer(X)
            elif str.startswith(k, "sage"):
                X = layer(X, hg)
        return X


class HGNNP(nn.Module):
    def __init__(
        self,
        parameters: ParameterDict,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.parameterDict = parameters
        for idx, (k, v) in enumerate(parameters.items()):
            is_last = idx == len(parameters) - 1
            self.layers.append(
                HGNNPConv(
                    v["in_channels"],
                    v["out_channels"],
                    use_bn=v["use_bn"],
                    drop_rate=v["drop_rate"],
                    is_last=is_last,
                )
            )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


from dhg.nn import GCNConv
class GCN(nn.Module):
    r"""
    The GCN model proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).
    """
    def __init__(self, parameterDict) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.parameterDict = parameterDict
        for idx, (k, v) in enumerate(parameterDict.items()):
            is_last = idx == len(parameterDict) - 1
            self.layers.append(
                GCNConv(
                    v["in_channels"],
                    v["out_channels"],
                    use_bn=v["use_bn"],
                    drop_rate=v["drop_rate"],
                    is_last=is_last,
                )
            )
    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """        
        for layer in self.layers:
            X = layer(X, g)
        return X

from dhg.nn import GraphSAGEConv
class GraphSAGE(nn.Module):
    r"""
    The GraphSAGE model proposed in `Inductive Representation Learning on Large Graphs <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>`_ paper (NIPS 2017).
    """

    def __init__(
        self,
        parameterDict,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.parameterDict = parameterDict
        for idx, (k, v) in enumerate(parameterDict.items()):
            is_last = idx == len(parameterDict) - 1
            self.layers.append(
                GraphSAGEConv(
                    v["in_channels"],
                    v["out_channels"],
                    aggr=aggr,
                    use_bn=v["use_bn"],
                    drop_rate=v["drop_rate"],
                    is_last=is_last,
                )
            )

    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, g)
        return X