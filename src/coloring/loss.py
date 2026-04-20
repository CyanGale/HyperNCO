import torch
from tqdm import tqdm


def loss_coloring_onehot_qubo(outs_cons: torch.Tensor, outs_obj: torch.Tensor, Q: torch.Tensor, **kwargs):
    """Loss function for graph coloring problem formulated as One-Hot QUBO (OH-QUBO).

    The optimization goal is to minimize the number of colors used while ensuring valid graph coloring.

    Args:
        outs_cons (torch.Tensor): Assignment tensor of shape ``[V, K]`` representing vertex-color assignments. Each row
            is a one-hot vector. V indicates the number of vertices.
        outs_obj (torch.Tensor): Binary tensor of shape ``[1, K]`` indicating color usage where elements ∈ {0, 1}.
            K represents the maximum allowed colors (e.g., total number of vertices).
        Q (torch.Tensor): see `src/coloring/utils/coloring_construct_Q`.
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

    gini_cons_cof_lambda = kwargs.get("gini_cons_cof_lambda", lambda e, n: 0)
    cons_cof_lambda = kwargs.get("cons_cof_lambda", lambda e, n: 1.0)
    obj_cof_lambda = kwargs.get("obj_cof_lambda", lambda e, n: 1.0)
    obj_cons_cof_lambda = kwargs.get("obj_cons_cof_lambda", obj_cof_lambda)

    gini_cons_cof = gini_cons_cof_lambda(epoch, num_epochs)
    cons_cof = cons_cof_lambda(epoch, num_epochs)
    obj_cof = obj_cof_lambda(epoch, num_epochs)
    obj_cons_cof = obj_cons_cof_lambda(epoch, num_epochs)

    loss_cons_coloring = (outs_cons.t().mm(Q) * outs_cons.t()).sum()
    loss_obj = outs_obj.sum()
    loss_cons_obj = ((torch.ones_like(loss_obj) - outs_obj) * outs_cons).sum()
    loss_gini_cons = _gini_annealed_loss_cons(outs_cons)

    if epoch % 1000 == 0:
        tqdm.write(
            f"Epoch: {epoch} | "
            f"coloring Loss: {loss_cons_coloring.item():.2f} | "
            f"K Loss: {loss_obj.item():.2f} | "
            f"obj cons: {loss_cons_obj.item():.2f} | "
            f"gini cons Loss: {loss_gini_cons.item():.2f}"
        )

    loss_cons_coloring = cons_cof * loss_cons_coloring
    loss_obj = obj_cof * loss_obj
    loss_cons_obj = obj_cons_cof * loss_cons_obj
    loss_gini_cons = gini_cons_cof * loss_gini_cons
    
    return loss_cons_coloring + loss_obj + loss_cons_obj + loss_gini_cons


def loss_coloring_onehot_pubo(outs_cons: torch.Tensor, outs_obj: torch.Tensor, H: torch.Tensor, **kwargs):
    """Loss function for hypergraph proper coloring problem formulated as One-Hot PUBO (OH-PUBO).

    The optimization goal is to minimize the number of colors used while ensuring valid [hypergraph proper coloring](https://www.youtube.com/watch?v=plqtIpRKBRs).

    Args:
        outs_cons (torch.Tensor): Assignment tensor of shape ``[V, K]`` representing vertex-color assignments. Each row
            is a one-hot vector. V indicates the number of vertices.
        outs_obj (torch.Tensor): Binary tensor of shape ``[1, K]`` indicating color usage where elements ∈ {0, 1}.
            K represents the maximum allowed colors (e.g., total number of vertices).
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

    gini_cons_cof_lambda = kwargs.get("gini_cons_cof_lambda", lambda e, n: 0)
    cons_cof_lambda = kwargs.get("cons_cof_lambda", lambda e, n: 1.0)
    obj_cof_lambda = kwargs.get("obj_cof_lambda", lambda e, n: 1.0)
    obj_cons_cof_lambda = kwargs.get("obj_cons_cof_lambda", obj_cof_lambda)

    gini_cons_cof = gini_cons_cof_lambda(epoch, num_epochs)
    cons_cof = cons_cof_lambda(epoch, num_epochs)
    obj_cof = obj_cof_lambda(epoch, num_epochs)
    obj_cons_cof = obj_cons_cof_lambda(epoch, num_epochs)

    mask = H.transpose(0, 1).unsqueeze(-1)
    # If the point is not within this edge, set it to 1 to correctly participate in the multiplication operation
    masked_tensor = torch.where(mask == 1, outs_cons, torch.ones_like(outs_cons)) 

    loss_cons_coloring = masked_tensor.prod(dim=1).sum()
    
    loss_obj = outs_obj.sum()
    loss_cons_obj = ((1 - outs_obj) * outs_cons).sum()
    loss_gini_cons = _gini_annealed_loss_cons(outs_cons)

    if epoch % 200 == 0:
        tqdm.write(
            f"Epoch: {epoch} | "
            f"coloring Loss: {loss_cons_coloring.item():.2f} | "
            f"K Loss: {loss_obj.item():.2f} | "
            f"obj cons: {loss_cons_obj.item():.2f} | "
            f"gini cons Loss: {loss_gini_cons.item():.2f}"
        )

    loss_cons_coloring = cons_cof * loss_cons_coloring
    loss_obj = obj_cof * loss_obj
    loss_cons_obj = obj_cons_cof * loss_cons_obj
    loss_gini_cons = gini_cons_cof * loss_gini_cons

    return loss_cons_coloring + loss_obj + loss_cons_obj + loss_gini_cons


def _gini_annealed_loss_cons(outs_cons: torch.Tensor):
    """Gini Coefficient-Based Annealing Algorithm"""
    p = outs_cons.pow(2).sum(dim=1)
    gini = (torch.ones_like(p) - p).sum()
    return gini
