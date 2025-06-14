import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric import seed_everything
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import networkx as nx

def torch_identity(n):
    return torch.sparse.spdiags(torch.ones(n), torch.tensor([0]), (n, n))

# find all unidrectional edges in networkx graph
def find_unidirectional_edges(graph):
    unidirectional_edges = []
    for edge in tqdm(graph.edges()):
        if (edge[1], edge[0]) not in graph.edges():
            unidirectional_edges.append(edge)
    return np.array(unidirectional_edges)

class Dataset():

    def __init__(self, dataset_name, random_seed=42):
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        self.adj = None
        self.features = None
        seed_everything(random_seed)

    def load_data(self):
        pass

    def to_pyg(self):
        self.num_nodes = self.adj.shape[0]
        edge_index = torch.Tensor(np.array(self.adj.nonzero()).T)
        # OHE features
        x = torch_identity(self.num_nodes).type(torch.float32)
        return Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes)
    
    def to(self, device):
        for value in self.__dict__.values():
            if isinstance(value, torch.Tensor) or isinstance(value, Data):
                value.to(device)
    
    def split_3_tasks(self, test_frac=0.1, val_frac=0.05, test_bd_frac=0.3, val_bd_frac=0.15):
        
        # DIRECTIONAL POSITIVES
        if self.dataset_name == 'citation':
            G = nx.from_scipy_sparse_matrix(self.adj, create_using=nx.DiGraph)
            positive_directionals_edge_index_t = find_unidirectional_edges(G)    
        else:
            positive_directionals_edge_index_t = np.array(self.adj.multiply(self.adj.T == 0).nonzero()).T
        num_positives_directionals = positive_directionals_edge_index_t.shape[0]
        num_positives_directional_test =  int(num_positives_directionals * test_frac)
        num_positives_directional_val =  int(num_positives_directionals * val_frac)

        train_val_directional_positives, test_directional_positives = train_test_split(positive_directionals_edge_index_t, test_size = num_positives_directional_test, shuffle = True)
        train_directional_positives, val_directional_positives = train_test_split(train_val_directional_positives, test_size = num_positives_directional_val, shuffle = True)
        print(f"positive unidirectional edges for train: {train_directional_positives.shape}")
        print(f"positive unidirectional edges for validation: {val_directional_positives.shape}")
        print(f"positive unidirectional edges for test: {test_directional_positives.shape}")

        # BIDIRECTIONAL POSITIVES
        positive_bidirectionals_edge_index = np.array(self.adj.multiply(self.adj.T).nonzero())
        positive_bidirectionals_one_way_edge_index_t = positive_bidirectionals_edge_index[:, positive_bidirectionals_edge_index[0,:] > positive_bidirectionals_edge_index[1,:] ].T
        num_positives_bidirectionals = positive_bidirectionals_one_way_edge_index_t.shape[0]
        num_positives_bidirectional_test =  int(num_positives_bidirectionals * test_bd_frac)
        num_positives_bidirectional_val = int(num_positives_bidirectionals * val_bd_frac)
        train_val_bidirectional_positives, test_bidirectional_positives = train_test_split(positive_bidirectionals_one_way_edge_index_t, test_size = num_positives_bidirectional_test, shuffle = True)
        train_bidirectional_positives, val_bidirectional_positives = train_test_split(train_val_bidirectional_positives, test_size = num_positives_bidirectional_val, shuffle = True)
        print(f"positive bidirectional edges for train: {train_bidirectional_positives.shape}")
        print(f"positive bidirectional edges for validation: {val_bidirectional_positives.shape}")
        print(f"positive bidirectional edges for test: {test_bidirectional_positives.shape}")

        # PT DIRECTIONAL
        # edge index
        train_directional_edge_label_index = torch.cat((
            torch.tensor(train_directional_positives.T),
            torch.tensor(train_directional_positives.T).flip(dims = (0,))
        ), dim = 1)
        val_directional_edge_label_index = torch.cat((
            torch.tensor(val_directional_positives.T),
            torch.tensor(val_directional_positives.T).flip(dims = (0,))
        ), dim = 1)
        test_directional_edge_label_index = torch.cat((
            torch.tensor(test_directional_positives.T),
            torch.tensor(test_directional_positives.T).flip(dims = (0,))
        ), dim = 1)
        # edge_label
        num_positives_directional_train = train_directional_positives.shape[0]
        train_directional_edge_label = torch.cat((
            torch.ones(num_positives_directional_train),
            torch.zeros(num_positives_directional_train)
        ))
        val_directional_edge_label = torch.cat((
            torch.ones(num_positives_directional_val),
            torch.zeros(num_positives_directional_val)
        ))
        test_directional_edge_label = torch.cat((
            torch.ones(num_positives_directional_test),
            torch.zeros(num_positives_directional_test)
        ))

        # PT BIDIRECTIONAL
        num_positives_bidirectional_train = train_bidirectional_positives.shape[0]
        train_positive_unidirectionals_edge_index_t_idx_sample = torch.randperm(num_positives_directional_train)[:num_positives_bidirectional_train]
        train_bidirectional_negatives =  train_directional_positives[train_positive_unidirectionals_edge_index_t_idx_sample,:]

        sample_positive_unidirectionals_edge_index_t_idx = torch.randperm(num_positives_directionals)[:num_positives_bidirectional_val + num_positives_bidirectional_test]
        sample_positive_unidirectionals_edge_index_t =  positive_directionals_edge_index_t[sample_positive_unidirectionals_edge_index_t_idx,:]
        val_bidirectional_negatives = sample_positive_unidirectionals_edge_index_t[:num_positives_bidirectional_val, :]
        test_bidirectional_negatives = sample_positive_unidirectionals_edge_index_t[num_positives_bidirectional_val:, :]

        # edge index
        train_bidirectional_edge_label_index = torch.cat((
            torch.tensor(train_bidirectional_positives.T),
            torch.tensor(train_bidirectional_negatives.T).flip(dims = (0,))
        ), dim = 1)
        val_bidirectional_edge_label_index = torch.cat((
            torch.tensor(val_bidirectional_positives.T),
            torch.tensor(val_bidirectional_negatives.T).flip(dims = (0,))
        ), dim = 1)
        test_bidirectional_edge_label_index = torch.cat((
            torch.tensor(test_bidirectional_positives.T),
            torch.tensor(test_bidirectional_negatives.T).flip(dims = (0,))
        ), dim = 1)

        # edge_label 
        train_bidirectional_edge_label = torch.cat((
            torch.ones(num_positives_bidirectional_train),
            torch.zeros(num_positives_bidirectional_train)
        ))
        val_bidirectional_edge_label = torch.cat((
            torch.ones(num_positives_bidirectional_val),
            torch.zeros(num_positives_bidirectional_val)
        ))
        test_bidirectional_edge_label = torch.cat((
            torch.ones(num_positives_bidirectional_test),
            torch.zeros(num_positives_bidirectional_test)
        ))

        # PT GENERAL
        # TRAIN POSITIVES
        positive_bidirectionals_other_way_edge_index = positive_bidirectionals_edge_index[:, positive_bidirectionals_edge_index[0,:] <= positive_bidirectionals_edge_index[1,:] ]
        train_general_positives_edge_index = torch.cat((
            torch.tensor(train_directional_positives.T),
            torch.tensor(train_bidirectional_positives.T),
            torch.tensor(positive_bidirectionals_other_way_edge_index)
        ), dim = 1)   

        val_test_general_negatives = negative_sampling(torch.tensor(np.array(self.adj.nonzero())), num_nodes = self.num_nodes, num_neg_samples = num_positives_directional_val + num_positives_directional_test + num_positives_bidirectional_val + num_positives_bidirectional_test)

        val_general_negatives  = val_test_general_negatives[:, :(num_positives_directional_val + num_positives_bidirectional_val)]
        test_general_negatives = val_test_general_negatives[:, (num_positives_directional_val + num_positives_bidirectional_val):]
        
        val_general_edge_label_index = torch.cat((torch.tensor(val_directional_positives.T),  torch.tensor(val_bidirectional_positives.T), val_general_negatives ), dim = 1)
        val_general_edge_label = torch.cat((torch.ones(num_positives_directional_val + num_positives_bidirectional_val), torch.zeros(num_positives_directional_val + num_positives_bidirectional_val)), dim = 0)

        test_general_edge_label_index = torch.cat((torch.tensor(test_directional_positives.T),  torch.tensor(test_bidirectional_positives.T), test_general_negatives), dim = 1)
        test_general_edge_label = torch.cat((torch.ones(num_positives_directional_test + num_positives_bidirectional_test), torch.zeros(num_positives_directional_test + num_positives_bidirectional_test)), dim = 0)

        print(f"positive general edges for train: {train_general_positives_edge_index.shape}")
        print(f"general edges for validation: {val_general_edge_label_index.shape}")
        print(f"general edges for test: {test_general_edge_label_index.shape}")

        # OHE features
        x = torch_identity(self.num_nodes).type(torch.float32)

        train_general_positives_edge_label = torch.ones(train_general_positives_edge_index.size(1))
        self.train_general_pos = Data(x = x, edge_label = train_general_positives_edge_label.reshape(-1,1),  edge_label_index=train_general_positives_edge_index, edge_index = train_general_positives_edge_index)
        self.val_general   = Data(x = x, edge_label = val_general_edge_label.reshape(-1,1),  edge_label_index=val_general_edge_label_index, edge_index = train_general_positives_edge_index)
        self.test_general  = Data(x = x, edge_label = test_general_edge_label.reshape(-1,1),  edge_label_index=test_general_edge_label_index, edge_index = train_general_positives_edge_index)

        self.train_directional = Data(x = x, edge_label = train_directional_edge_label.reshape(-1,1),  edge_label_index=train_directional_edge_label_index, edge_index = train_general_positives_edge_index)
        self.val_directional   = Data(x = x, edge_label = val_directional_edge_label.reshape(-1,1),  edge_label_index=val_directional_edge_label_index, edge_index = train_general_positives_edge_index)
        self.test_directional  = Data(x = x, edge_label = test_directional_edge_label.reshape(-1,1),  edge_label_index=test_directional_edge_label_index, edge_index = train_general_positives_edge_index)


        self.train_bidirectional = Data(x = x, edge_label = train_bidirectional_edge_label.reshape(-1,1),  edge_label_index=train_bidirectional_edge_label_index, edge_index = train_general_positives_edge_index)
        self.val_bidirectional   = Data(x = x, edge_label = val_bidirectional_edge_label.reshape(-1,1),  edge_label_index=val_bidirectional_edge_label_index, edge_index = train_general_positives_edge_index)
        self.test_bidirectional  = Data(x = x, edge_label = test_bidirectional_edge_label.reshape(-1,1),  edge_label_index=test_bidirectional_edge_label_index, edge_index = train_general_positives_edge_index)
        
        
        return None



