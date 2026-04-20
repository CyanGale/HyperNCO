from typing import Any
import dhg
import numpy as np
import torch
import os
import pickle as pkl
from dhg import Hypergraph


def from_file_to_hypergraph(file_path: str, reset_vertex_index=False) -> Hypergraph:
    # fmt: off
    
    r"""
    
    Read a hypergraph from a file.
    
    ---
    Args:
        ``file_path``(`str`):  The path to the file containing the hypergraph.  
        ``reset_vertex_index``(`bool`):  Whether to reset the vertex index to start from 0.

    Returns:
        HG(`Hypergraph`):  The hypergraph read from the file.

    ## Note:
        The file should be in the following format:   
        ``{nums_edges} {nums_vertices}`` 
        ``7008 8428 3566 38 1606 4146 5855 1014 7722 1739 7716 5817``    
        ``5056 4114 12483 10073 8546 10045``   
        ``11070 2289 4114 10073 1747 1628``  
        The numbers represent vertices, and each line of numbers represents a hyperedge.
    """
    # fmt: on
    with open(file_path, "r") as file:
        lines = file.readlines()
        lines = [
            line for line in lines if line.strip() and not line.strip().startswith("#")
        ]
        lines = lines[1:]
        edges = [list(map(int, line.split())) for line in lines]
        all_vertices = sorted(set(vertex for edge in edges for vertex in edge))

        vertex_mapping = {
            old_vertex: new_vertex
            for new_vertex, old_vertex in enumerate(all_vertices, start=1)
        }
        new_edges = [[vertex_mapping[vertex] for vertex in edge] for edge in edges]

        return Hypergraph(
            num_v=len(all_vertices), e_list=new_edges if reset_vertex_index else edges
        )


    
def from_pickle_to_hypergraph(dataset: str) -> Any:
    # fmt: off
    
    r"""
    
    Read a hypergraph from a pickle file.
    
    ---
    Args:
        ``file_path``(`str`):  The path to the pickle file containing the hypergraph.

    Returns:
        HG(`Hypergraph`):  The hypergraph read from the pickle file.  
    
    Example:
    
    .. code-block:: python
        hg = from_pickle_to_hypergraph("data/test_hypergraph")
    """
    # fmt: on
    from dhg import Hypergraph

    data_path = os.path.join(dataset)

    with open(os.path.join(data_path, f"H.pickle"), "rb") as f:
        H = pkl.load(f)
    l: dict[int, list] = {}
    for i, j in zip(H[0], H[1]):
        i, j = i.item(), j.item()
        if l.get(j):
            l[j].append(i)
        else:
            l[j] = [i]
    sorted_l = {k: v for k, v in sorted(l.items(), key=lambda item: item[0])}
    num_v = H[0].max().item() + 1
    e_list = list(sorted_l.values())
    return Hypergraph(num_v, e_list, merge_op="mean")


def from_pickle_to_adj(dataset: str, unique:bool) -> Any:
    r"""
    
    Read a hypergraph from a pickle file and convert it to an adjacency matrix.
    
    ---
    Args:
        ``dataset``(`str`):  The path to the pickle file containing the hypergraph.
        ``unique``(`bool`):  Whether to remove duplicate columns in the adjacency matrix, same meaning remove duplicate hyperedges.
    
    Returns:
        ``A``(`torch.Tensor`):  The adjacency matrix of the hypergraph Size ``(V, E)``
    """
    data_path = os.path.join(dataset)

    with open(os.path.join(data_path, f"H.pickle"), "rb") as f:
        a = pkl.load(f)
    vertex_indices, edge_indices  = a
    
    num_hyperedges = edge_indices.max().item() + 1  
    num_vertices = vertex_indices.max().item() + 1 
    
    H = torch.zeros((num_vertices, num_hyperedges), dtype=torch.int32)
    H[vertex_indices, edge_indices] = 1
    if unique:
        H, indices = torch.unique(H, dim=1, return_inverse=True)
    return H

def from_hypergraph_to_clique_adj(H:torch.Tensor):
    H = H.to(torch.float32).to_sparse()
    H_T = H.t()
    num_v = H.shape[0]
    miu = 1.0
    adj = miu * H.mm(H_T).coalesce().cpu().clone()
    adj = adj.to_dense()
    mask = torch.eye(num_v, dtype=torch.bool)
    adj[mask] = 0
    return adj

def from_hypergraph_to_clique(H:torch.Tensor):
    H = H.to(torch.float32).to_sparse()
    H_T = H.t()
    num_v = H.shape[0]
    miu = 1.0
    adj = miu * H.mm(H_T).coalesce().cpu().clone()
    src_idx, dst_idx = adj._indices()
    edge_mask = src_idx != dst_idx
    edge_list = torch.stack([src_idx[edge_mask], dst_idx[edge_mask]]).t().cpu().numpy()
    edge_tuple_list = [(src, dst) for src, dst in edge_list]
    # if weighted:
    #     e_weight = adj._values()[edge_mask].numpy().tolist()
    #     _g = Graph(num_v, edge_list, e_weight, merge_op="sum", device=device)
    # else:
    #     _g = Graph(num_v, edge_list, merge_op="mean", device=device)
    return num_v, edge_tuple_list

def from_hypergraph_to_kahypar(data:dhg.data.BaseData):
    r"""
    change the pickle file to the format that kahypar can read
    
    Returns:
        Tuple: (num_nodes(`int`),  
        num_nets(`int`),  
        hyperedge_indices(`list`), 
        hyperedges(`list`),  
        edge_weights(`list`), 
        node_weights(`list`)) 
        
    """
    el = data["edge_list"]
    e_index = []
    e_list = []

    e_index.append(0)
    for x in el:
        e_list.extend(list(x))
        e_index.append(e_index[-1] + len(x))
    num_nodes = max(e_list) + 1
    num_nets = data["num_edges"]
    edge_weights = [1] * num_nets
    node_weights = [1] * num_nodes
    
    return num_nodes, num_nets, e_index, e_list, edge_weights, node_weights

def from_pickle_to_kahypar(Hpickle_filepath, weights:bool):
    with open(Hpickle_filepath, "rb") as f:
        H = pkl.load(f)

    l: dict[int, list] = {}
    for i, j in zip(H[0], H[1]):
        i, j = i.item(), j.item()
        if l.get(j):
            l[j].append(i)
        else:
            l[j] = [i]
    sorted_l = {k: v for k, v in sorted(l.items(), key=lambda item: item[0])}
    el = list(sorted_l.values())
    e_index = []
    e_list = []

    e_index.append(0)
    for x in el:
        e_list.extend(list(x))
        e_index.append(e_index[-1] + len(x))
    num_nodes = max(e_list) + 1
    num_nets = len(el)
    edge_weights = [1] * num_nets
    node_weights = [1] * num_nodes
    if weights:
        return num_nodes, num_nets, e_index, e_list, edge_weights, node_weights
    else:
        return num_nodes, num_nets, e_index, e_list