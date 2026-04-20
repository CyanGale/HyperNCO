from typing import Tuple
import torch
from tqdm import tqdm


def loss_partitioning_onehot_qubo(outs: torch.Tensor, Q: Tuple, **kwargs):
    """Loss function for graph partitioning problem formulated as One-Hot QUBO (OH-QUBO).

    The optimization goal is to minimize the number of cut, see [wiki](https://en.wikipedia.org/wiki/Graph_partition#Problem).

    Args:
        outs (torch.Tensor): Assignment tensor of shape ``[V, K]`` representing vertex-part assignments. Each row
            is a one-hot vector. V indicates the number of vertices. K indicates K parts for partitioning.
        Q (Tuple): see `src/partitioning/utils/partitioning_construct_Q`.
        kwargs: Additional parameters including
            - epoch (int): Current training epoch.
            - num_epochs (int): Total number of training epochs.
            - gini_cof_lambda (Callable): Function to compute Gini coefficient weight,
                accepts (current_epoch, total_epochs).
            - cons_cof_lambda (Callable): Function to compute constraint weight,
                accepts (current_epoch, total_epochs).
            - obj_cof_lambda (Callable): Function to compute objective weight,
                accepts (current_epoch, total_epochs).
    """
    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)
    obj_cof_lambda = kwargs.get("obj_cof_lambda", lambda e, n: 1.0)
    cons_cof_lambda = kwargs.get("cons_cof_lambda", lambda e, n: 1.0)
    gini_cof_lambda = kwargs.get("gini_cof_lambda", lambda e, n: 0)

    obj_cof = obj_cof_lambda(epoch, num_epochs)
    cons_cof = cons_cof_lambda(epoch, num_epochs)
    gini_cof = gini_cof_lambda(epoch, num_epochs)

    q_diag = Q[0]
    _q = Q[1]
    loss_obj = (outs.t().mm(_q) * outs.t()).sum() + (outs * (q_diag.unsqueeze(1).repeat(1, outs.shape[1]))).sum()
    loss_cons = loss_partitioning_constraints(outs)
    loss_gini = _gini_annealed_loss(outs)

    if epoch % 1000 == 0:
        tqdm.write(
            f"Epoch: {epoch} | "
            f"partitoning Loss: {loss_obj.item():.2f} | "
            f"balance Loss: {loss_cons.item():.2f} | "
            f"gini Loss: {loss_gini.item():.2f}"
        )

    loss_obj = obj_cof * loss_obj
    loss_cons = cons_cof * loss_cons
    loss_gini = gini_cof * loss_gini

    return loss_obj + loss_cons + loss_gini


def loss_partitioning_onehot_pubo(outs: torch.Tensor, H: torch.Tensor, **kwargs):
    """Loss function for hypergraph partitioning problem formulated as One-Hot PUBO (OH-PUBO).

    The optimization goal is to minimize the number of cut, see [wiki](https://en.wikipedia.org/wiki/Graph_partition#Problem).

    Args:
        outs (torch.Tensor): Assignment tensor of shape ``[V, K]`` representing vertex-part assignments. Each row
            is a one-hot vector. V indicates the number of vertices. K indicates K parts for partitioning.
        H (torch.Tensor): The hypergraph [incidence matrix](https://en.wikipedia.org/wiki/Incidence_matrix).
        kwargs: Additional parameters including
            - epoch (int): Current training epoch.
            - num_epochs (int): Total number of training epochs.
            - gini_cof_lambda (Callable): Function to compute Gini coefficient weight,
                accepts (current_epoch, total_epochs).
            - cons_cof_lambda (Callable): Function to compute constraint weight,
                accepts (current_epoch, total_epochs).
            - obj_cof_lambda (Callable): Function to compute objective weight,
                accepts (current_epoch, total_epochs).
    """
    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)
    obj_cof_lambda = kwargs.get("obj_cof_lambda", lambda e, n: 1.0)
    cons_cof_lambda = kwargs.get("cons_cof_lambda", lambda e, n: 1.0)
    gini_cof_lambda = kwargs.get("gini_cof_lambda", lambda e, n: 0.0)

    obj_cof = obj_cof_lambda(epoch, num_epochs)
    cons_cof = cons_cof_lambda(epoch, num_epochs)
    gini_cof = gini_cof_lambda(epoch, num_epochs)

    H = H.to_dense()
    edges_weight = H.sum(dim=0)
    # Expand dimensions to match broadcasting
    X_ = outs.t().unsqueeze(-1)
    H_ = H.unsqueeze(0)

    masked_X = X_.mul(H_)
    first_order_terms_sum = (masked_X * (1 / edges_weight)).sum()
    # higher_order_terms_sum = (masked_X + (torch.ones_like(H) - H)).prod(dim=1).sum()
    higher_order_terms_sum = (masked_X + (1 - H)).prod(dim=1).sum()
    
    loss_obj = first_order_terms_sum - higher_order_terms_sum
    loss_cons = loss_partitioning_constraints(outs)
    loss_gini = _gini_annealed_loss(outs)

    if epoch % 2000 == 0: 
        tqdm.write(
            f"Epoch: {epoch} | "
            f"partitioning Loss: {loss_obj.item():.2f} | "
            f"balance Loss: {loss_cons.item():.2f} | "
            f"gini Loss: {loss_gini.item():.2f}"
        )

    loss_obj = obj_cof * loss_obj
    loss_cons = cons_cof * loss_cons
    loss_gini = gini_cof * loss_gini

    return loss_obj + loss_cons + loss_gini


def loss_partitioning_constraints(outs: torch.Tensor):
    """
    Constraints for graph/hypergraph partitioning, the number of vertices in each part approaches similarity
    """ 
    constraints = torch.var(torch.sum(outs, dim=0))
    return constraints


def _gini_annealed_loss(outs: torch.Tensor):
    """Gini Coefficient-Based Annealing Algorithm"""
    p = outs.pow(2).sum(dim=1)
    gini = (torch.ones_like(p) - p).sum()
    return gini
