from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
import scipy.sparse as sp
import networkx as nx

from datasets.dataset_utils import Dataset


class CitationDataset(Dataset):

    def load_data(self, path=None):
        print(f"Loading citation2 from ogbl")
        dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                     transform=T.ToSparseTensor())
        data = dataset[0]
        
        # adj = nx.adjacency_matrix(nx.read_edgelist(path,
        #                                            delimiter='\t',
        #                                            create_using=nx.DiGraph()))
        
        adj = data.adj_t.t().to_scipy(layout='coo')
        features = sp.identity(adj.shape[0])
        self.adj = adj
        self.features = features
        self.num_nodes = adj.shape[0]

        # self.data = self.to_pyg()
        self.split_3_tasks()