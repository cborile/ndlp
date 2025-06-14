import scipy.sparse as sp
import networkx as nx

from datasets.dataset_utils import Dataset


class GoogleDataset(Dataset):

    def load_data(self):
        path = "data/"+self.dataset_name+".cites"
        print(f"Loading {path}")
        adj = nx.adjacency_matrix(nx.read_edgelist(path,
                                                   delimiter='\t',
                                                   create_using=nx.DiGraph()))
        features = sp.identity(adj.shape[0])
        self.adj = adj
        self.features = features
        self.num_nodes = adj.shape[0]

        # self.data = self.to_pyg()
        self.split_3_tasks()