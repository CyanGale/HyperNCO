from tqdm import tqdm
import torch


def loss_maxcut_onehot_qubo(outs: torch.Tensor, Q, **kwargs):
    r"""Loss function for graph maxcut problem formulated as One-Hot QUBO (OH-QUBO)."""
    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)
    gini_cof_lambda = kwargs.get("gini_cof_lambda", lambda e, n: 0)

    gini_cof = gini_cof_lambda(epoch, num_epochs)

    loss_gini = _gini_annealed_loss(outs)

    # loss_obj = (outs.t().mm(Q) * outs.t()).sum()  !!!!!!!!!!!!!!!Warning!!!!!!!!!!!!! x^2 != x
    Q_diag = Q.diag()
    Q_nodiag = Q.clone()
    Q_nodiag.fill_diagonal_(0)
    loss_obj = ((outs.T.mm(Q_nodiag) * outs.T) + (outs.T * Q_diag)).sum()

    if epoch % 2000 == 0:
        tqdm.write(f"Epoch: {epoch:.2f} | " f"obj Loss: {loss_obj:.2f} | " f"gini Loss: {loss_gini:.2f} | ")

    loss_gini = gini_cof * loss_gini

    return loss_gini + loss_obj


def loss_maxcut_onehot_pubo(outs: torch.Tensor, H, **kwargs):
    r"""Loss function for hypergraph proper coloring problem formulated as One-Hot PUBO (OH-PUBO)"""

    epoch = kwargs.get("epoch", 1)
    num_epochs = kwargs.get("num_epochs", None)
    gini_cof_lambda = kwargs.get("gini_cof_lambda", lambda e, n: 0)

    gini_cof = gini_cof_lambda(epoch, num_epochs)

    X_ = outs.t().unsqueeze(-1)
    H_ = H.unsqueeze(0)
    weight = H.sum(dim=0)
    mid = X_ * H_
    sum = (mid * (1 / weight)).sum()
    sub = (mid + (1 - H)).prod(dim=1).sum()  # Set the irrelevant position to 1 so that it cannot participate in multiplication

    loss_obj = (sum - sub) * -1
    loss_gini = _gini_annealed_loss(outs)

    if epoch % 2000 == 0:
        tqdm.write(f"Epoch: {epoch:.2f} | " f"obj Loss: {loss_obj:.2f} | " f"gini Loss: {loss_gini:.2f} | ")

    loss_gini = gini_cof * loss_gini

    return loss_obj + loss_gini


def _gini_annealed_loss(outs: torch.Tensor):
    """Gini Coefficient-Based Annealing Algorithm"""
    p = outs.pow(2).sum(dim=1)
    gini = (torch.ones_like(p) - p).sum()
    return gini
