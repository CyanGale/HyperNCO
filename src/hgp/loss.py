import torch
from torch.types import _device


def loss_bs(outs, hg, device: _device | str):
    loss_1 = torch.zeros(outs.shape[1]).to(device) 
    edges, _ = hg.e
    for vertices in edges:
        vertices = list(vertices)
        loss_1 = loss_1 + (
            torch.sum(outs[vertices], dim=0) / len(vertices)
            - torch.prod(outs[vertices], dim=0)
        )

    loss_1 = 1e-4 * loss_1.sum()
    loss_2 = torch.var(torch.sum(outs, dim=0)).to(device)

    total_loss = loss_1 + loss_2

    return total_loss, loss_1, loss_2


def loss_bs_matrix(outs, hg, device: _device | str):
    H = hg.H.to_dense().to(device)
    outs = outs.to(device)
    nn = torch.matmul(outs, (1 - torch.transpose(outs, 0, 1)))
    ne_k = torch.matmul(nn, H)
    ne_k = ne_k.mul(H)

    H_degree = torch.sum(H, dim=0)
    H_degree = H_degree - 0.4

    H_1 = ne_k / H_degree
    a2 = 1 - H_1
    a3 = torch.prod(a2, dim=0)
    a3 = a3.sum()
    loss_1 = -1 * a3

    # pun = torch.mul(ne_k, H)

    # loss_1 = pun.sum()
    loss_2 = torch.var(torch.sum(outs, dim=0)).to(device)

    loss = 50 * loss_1 + loss_2
    return loss, loss_1, loss_2

def loss_bs_matrix_x(outs, adj, device: _device | str):
    
    outs = outs.to(device)
    nn = torch.matmul(outs, (1 - torch.transpose(outs, 0, 1)))
    ne_k = torch.matmul(nn, adj)

    pun = torch.mul(ne_k, adj)

    loss_1 = pun.sum()
    loss_2 = torch.var(torch.sum(outs, dim=0)).to(device)

    loss = loss_1 + loss_2
    return loss, loss_1, loss_2

def loss_bs_matrix_mega(outs, hg, de: torch.Tensor, device: _device | str):

    H = hg.H.to_dense().to(device)
    outs = outs.to(device)
    nn = torch.matmul(outs, (1 - torch.transpose(outs, 0, 1)))
    ne_k = torch.matmul(nn, H)

    pun = torch.mul(ne_k, H)
    pun = pun.sum(dim=0)
    pun = pun.mul(de.sqrt()) 
    loss_1 = pun.sum()
    loss_2 = torch.var(torch.sum(outs, dim=0)).to(device)

    loss = loss_1 + loss_2
    return loss, loss_1, loss_2
