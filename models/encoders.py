import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GCNConv, DirectedGCNConv, PureConv
from functools import partial
from torch_geometric.nn import GCNConv as GCNConvPyG


################################################################################
# ENCODERS for UNDIRECTED models
################################################################################
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    

# Encoder for MPLP taken from official repo
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0., xdropout=0., use_feature=True, jk=False, gcn_name='pure', embedding=None):
        super(GCN, self).__init__()

        self.use_feature = use_feature
        self.embedding = embedding
        self.dropout = dropout
        self.xdropout = xdropout
        self.input_size = 0
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))
        if self.use_feature:
            self.input_size += in_channels
        if self.embedding is not None:
            self.input_size += embedding.embedding_dim
        self.convs = torch.nn.ModuleList()
        
        if self.input_size > 0:
            if gcn_name == 'gcn':
                conv_func = partial(GCNConvPyG, cached=False)
                # conv_func = GCNConv
            elif 'pure' in gcn_name:
                conv_func = partial(PureConv, aggr='gcn')
            self.xemb = nn.Sequential(nn.Identity()) # nn.Sequential(nn.Dropout(xdropout)) #
            if ("pure" in gcn_name or num_layers==0):
                self.xemb.append(nn.Linear(self.input_size, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
                self.input_size = hidden_channels
            self.convs.append(conv_func(self.input_size, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(
                    conv_func(hidden_channels, hidden_channels))
            self.convs.append(conv_func(hidden_channels, out_channels))


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, adj_t):
        if self.input_size > 0:
            xs = []
            if self.use_feature:
                xs.append(x)
            if self.embedding is not None:
                xs.append(self.embedding.weight)
            x = torch.cat(xs, dim=1)
            x = self.xemb(x)
            jkx = []
            for conv in self.convs:
                x = conv(x, adj_t)
                # x = F.relu(x) # FIXME: not using nonlinearity in Sketching
                if self.jk:
                    jkx.append(x)
            if self.jk: # JumpingKnowledge Connection
                jkx = torch.stack(jkx, dim=0)
                sftmax = self.jkparams.reshape(-1, 1, 1)
                x = torch.sum(jkx*sftmax, dim=0)
        return x

################################################################################
# ENCODERS for DIRECTED models
################################################################################
class SourceGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(SourceGCNConvEncoder, self).__init__()
        self.conv1 = DirectedGCNConv(in_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        self.conv2 = DirectedGCNConv(hidden_channels, out_channels, alpha, beta, self_loops, adaptive)
        

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, edge_index))
        # x = self.conv1(x, edge_index)
        
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, torch.flip(edge_index, [0]))

        return x

class TargetGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(TargetGCNConvEncoder, self).__init__()
        self.conv1 = DirectedGCNConv(in_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        self.conv2 = DirectedGCNConv(hidden_channels, out_channels, alpha, beta, self_loops, adaptive)
        

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, torch.flip(edge_index, [0])))
        # x = self.conv1(x, torch.flip(edge_index, [0]))

        # x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv2(x, edge_index)

        return x

class DirectedGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=0.5, beta=0.5, self_loops=True, adaptive=False):
        super(DirectedGCNConvEncoder, self).__init__()
        self.source_conv = SourceGCNConvEncoder(in_channels, hidden_channels, out_channels, alpha, beta, self_loops, adaptive)
        self.target_conv = TargetGCNConvEncoder(in_channels, hidden_channels, out_channels, alpha, beta, self_loops, adaptive)

    def forward(self, s, t, edge_index):
        s = self.source_conv(s, edge_index)
        t = self.target_conv(t, edge_index)
        return s, t