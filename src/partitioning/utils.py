import logging
import torch
import time
from pyscipopt import Model, quicksum, Eventhdlr, SCIP_EVENTTYPE
from ..core import BaseSCIPSolver

logger = logging.getLogger(__name__)


def partitioning_construct_Q(graph):
    """Constructs graph partitioning as an `OH-QUBO` formulation requiring matrix Q.

    The partitioning problem is modeled as `Reduce_sum(X^T·Q⊙X)`, where:
    - `Reduce_sum` performs element-wise matrix summation
    - X represents the solution vector/matrix
    - Q is the problem-specific design matrix
        + In praph partitioning problem, diagonal elements of the Q matrix are the degrees of each vertex, with other elements being negative

    Nont:
        The diagonal of Q must handled separately in the loss function.
    """
    A = graph.A.to_dense().fill_diagonal_(0)
    diag = A.sum(dim=0)  # diagonal matrix constructed from the degree of each vertex
    q = diag - A  # Output, but please read `Note` in `partitioning_construct_Q`
    return (diag, -A)


class PartitioningSCIPSolver(BaseSCIPSolver):
    def __init__(self, edge_list, max_k, pre_solve=True, balance_tolerance=1):
        super().__init__(edge_list, pre_solve)
        self.max_k = max_k
        self.balance_tolerance = balance_tolerance
        self.x = None  # x[v, k]
        self.y = None  # y[e]
        self.z = None  # z[e, k]
        self.max_size = None
        self.min_size = None

    def _add_variables(self):
        self.x = {(v, k): self.model.addVar(vtype="B", name=f"x_{v}_{k}") for v in self.V for k in range(1, self.max_k + 1)}
        self.y = {tuple(e): self.model.addVar(vtype="B", name=f"y_{e}") for e in self.edge_list}
        self.z = {(tuple(e), k): self.model.addVar(vtype="B", name=f"z_{e}_{k}") for e in self.edge_list for k in range(1, self.max_k + 1)}
        self.max_size = self.model.addVar(vtype="I", name="max_size")
        self.min_size = self.model.addVar(vtype="I", name="min_size")

    def _add_constraints(self):
        # onthot constraint
        for v in self.V:
            self.model.addCons(quicksum(self.x[v, k] for k in range(1, self.max_k + 1)) == 1, name=f"vertex_{v}_assignment")

        for e in self.edge_list:
            e_tuple = tuple(e)
            for k in range(1, self.max_k + 1):
                for v in e:
                    self.model.addCons(self.z[(e_tuple, k)] <= self.x[v, k], name=f"z_{e_tuple}_{k}_vertex_{v}_constraint")
                sum_x = quicksum(self.x[v, k] for v in e)
                self.model.addCons(sum_x >= len(e) * self.z[(e_tuple, k)] - (len(e) - 1), name=f"z_{e_tuple}_{k}_sum_constraint")

        # balance constraint
        for e in self.edge_list:
            e_tuple = tuple(e)
            sum_z = quicksum(self.z[(e_tuple, k)] for k in range(1, self.max_k + 1))
            self.model.addCons(sum_z + self.y[e_tuple] >= 1, name=f"y_{e_tuple}_lower")
            self.model.addCons(sum_z <= 1 - self.y[e_tuple], name=f"y_{e_tuple}_upper")

        for k in range(1, self.max_k + 1):
            sum_vars = quicksum(self.x[v, k] for v in self.V)
            self.model.addCons(sum_vars <= self.max_size, name=f"balance_max_{k}")
            self.model.addCons(sum_vars >= self.min_size, name=f"balance_min_{k}")

        self.model.addCons(self.max_size - self.min_size <= self.balance_tolerance, name="balance_tolerance_constraint")

    def _set_objective(self):
        total_cut = quicksum(self.y[e_tuple] for e_tuple in self.y)
        self.model.setObjective(total_cut, "minimize")

    def _get_solution_metrics(self, sol):
        # cut_count = sum(round(self.model.getSolVal(sol, var)) for var in self.y.values())
        # balance = [sum(round(self.model.getSolVal(sol, self.x[v, k])) for v in self.V)
        #            for k in range(1, self.max_k + 1)]
        # return {'cut': cut_count, 'balance': balance}
        assignment = {}
        for v in self.V:
            for k in range(1, self.max_k + 1):
                if round(self.model.getSolVal(sol, self.x[v, k])) > 0.5:
                    assignment[v] = k
                    break

        cut_count = 0
        for e in self.edge_list:
            groups = [assignment[v] for v in e]
            if len(set(groups)) > 1:
                cut_count += 1

        balance = [sum(round(self.model.getSolVal(sol, self.x[v, k])) for v in self.V) for k in range(1, self.max_k + 1)]

        return {
            "cut": cut_count,
            "balance": balance,
            "consistency_check": [cut_count, sum(round(self.model.getSolVal(sol, var)) for var in self.y.values())],
        }

    def _extract_solution(self, sol):
        assignment = {}
        for v in self.V:
            for k in range(1, self.max_k + 1):
                if round(self.model.getSolVal(sol, self.x[v, k])) > 0.5:
                    assignment[v] = k
                    break
        return assignment


def partitioning_evaluate(outs: torch.Tensor, graph, threshold=0.6):
    cuts = 0
    not_converged = 0

    max_values, max_indices = torch.max(outs, dim=1)
    outs_max = torch.zeros_like(outs, dtype=torch.int)
    outs_max.scatter_(1, max_indices.unsqueeze(1), 1)

    not_converged = (max_values <= threshold).sum().item()

    for edge in graph.e[0]:
        ids = list(edge)
        outs_ids = outs_max[ids]
        if outs_ids.prod(dim=0).sum() < 1e-4:
            cuts += 1
    blce = outs_max.sum(dim=0).cpu().numpy()
    print(f"+------------[Evaluation Result]------------+")
    print(f"Cuts: {cuts}\n" f"Blce: {blce}\n" f"Not converged nodes: {not_converged}")
    print(f"+-------------------------------------------+")
    return {"cuts": cuts, "blce": blce, "not_converged": not_converged}
