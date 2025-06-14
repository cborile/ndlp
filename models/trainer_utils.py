import torch
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected, from_scipy_sparse_matrix


def compute_weights_binary_classification(edge_label):
    tot_train_edges = edge_label.size(0)
    tot_pos_train_edges = int(edge_label.sum())
    if tot_train_edges==tot_pos_train_edges:  # negative sampling
        tot_train_edges = 2*tot_train_edges
        tot_neg_edges_train = tot_pos_train_edges
    else:
        tot_neg_edges_train = tot_train_edges - tot_pos_train_edges
    pos_weight = torch.tensor(tot_neg_edges_train / tot_pos_train_edges)
    norm = tot_train_edges / (tot_neg_edges_train + pos_weight * tot_pos_train_edges)
    return norm, pos_weight

def remove_reciprocal_edges(edge_index, return_removed_reciprocal_edges = False):
    edge_index_symm = to_undirected(edge_index)
    adj_sparse_symm        = to_scipy_sparse_matrix(edge_index_symm) 
    adj_sparse = to_scipy_sparse_matrix(edge_index)
    adj_tilde       =  (adj_sparse_symm - adj_sparse).T
    adj_tilde.eliminate_zeros()
    
    edge_index_no_reciprocal, _ = from_scipy_sparse_matrix(adj_tilde) 
    edge_index = edge_index_no_reciprocal

    if not return_removed_reciprocal_edges:
        return edge_index
    else:
        removed_reciprocals_sp = (adj_sparse - adj_tilde).T
        removed_reciprocals_sp.eliminate_zeros()
        removed_reciprocals, _ = from_scipy_sparse_matrix(removed_reciprocals_sp)
        return edge_index, removed_reciprocals
    

# Assumes self-loops have already been added
def get_multicass_lp_edge_label(edge_label_index, edge_label, device):
    # 0. = negative bidirectional
    # 1. = positives unidirectional
    # 2. = positives bidirectional
    # 3. = negatives unidirectional

    edge_label_index_wout_reciprocals, removed_reciprocals = remove_reciprocal_edges(edge_label_index, return_removed_reciprocal_edges = True)
    
    edge_label_class_index = torch.cat([
        edge_label_index[:, edge_label==0].to(device),
        edge_label_index_wout_reciprocals.to(device),
        removed_reciprocals.to(device),
        edge_label_index_wout_reciprocals.flip(dims=(0,)).to(device)
    ], dim=1
    )

    edge_label_class = torch.cat(
        [
            torch.zeros(edge_label_index[:, edge_label==0].size(1)).to(device),
            torch.ones(edge_label_index_wout_reciprocals.size(1)).to(device),
            torch.ones(removed_reciprocals.size(1)).to(device)*2, 
            torch.ones(edge_label_index_wout_reciprocals.size(1)).to(device)*3,
        ]
    ).type(torch.int64)
    return edge_label_class, edge_label_class_index

def compute_weights_multiclass_classification(edge_label):
    _, classes_sizes_train = edge_label.unique(return_counts = True)
    classes_sizes_train_max = torch.max(classes_sizes_train)
    loss_weights_train = classes_sizes_train_max / classes_sizes_train
    norm = classes_sizes_train.sum() / (loss_weights_train * classes_sizes_train).sum()
    return norm, loss_weights_train