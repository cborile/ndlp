import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from typing import Iterable, Final

################################################################################
# UNDIRECTED model layers: BASIC version
################################################################################
class GCNConv(MessagePassing):
    def  __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype) # in-degree
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    
class GCNMeanConv(MessagePassing):
    def  __init__(self, in_channels, out_channels):
        super(GCNMeanConv, self).__init__(aggr='sum')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row]
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j):
        return x_j
    
################################################################################
# DIRECTED model layers: BASIC version
################################################################################
class DirectedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha=0.5, beta=0.5, self_loops=True, adaptive=False):
        super(DirectedGCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

        # if adaptive is True:
        #     self.alpha = torch.nn.Parameter(torch.Tensor([alpha]))
        #     self.beta  = torch.nn.Parameter(torch.Tensor([beta]))
        # else:
        #     self.alpha      = alpha
        #     self.beta       = beta

        self.alpha      = alpha
        self.beta       = beta

        self.self_loops = self_loops
        self.adaptive   = adaptive

    
    def forward(self, x, edge_index):
        if self.self_loops is True:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col  = edge_index

        in_degree  = degree(col)
        out_degree = degree(row)

        alpha = self.alpha
        beta  = self.beta 

        in_norm_inv  = pow(in_degree,  -alpha)
        out_norm_inv = pow(out_degree, -beta)

        in_norm  = in_norm_inv[col]
        out_norm = out_norm_inv[row]
        norm     = in_norm * out_norm

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# Addpted from NCNC
class PureConv(torch.nn.Module):
    aggr: Final[str]
    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim == outdim:
            self.lin = torch.nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        elif self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        elif self.aggr == "sum":
            return spmm_add(adj_t, x)
        elif self.aggr == "gcn":
            norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
            x = norm * x
            x = spmm_add(adj_t, x) + x
            x = norm * x
            return x