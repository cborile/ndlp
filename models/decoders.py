import torch
from .mplp_utils import MLP, NodeLabel


################################################################################
# DECODER for UNDDIRECTED models
################################################################################
class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

################################################################################
# DECODER for DIRECTED models
################################################################################
class DirectedInnerProductDecoder(torch.nn.Module):
    def forward(self, s, t, edge_index, sigmoid=True):
        value = (s[edge_index[0]] * t[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, s, t, sigmoid=True):
        adj = torch.matmul(s, t.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GravityDecoder(torch.nn.Module):
    def __init__(self, l, EPS = 1e-2, CLAMP = None, train_l = True): 
        super().__init__()
        self.l_initialization = l
        self.l = torch.nn.Parameter(torch.tensor([l], dtype=torch.float64), requires_grad=train_l)
        self.EPS = EPS
        self.CLAMP = CLAMP

    def decode_all(self, z, sigmoid=True):
        m_i = z[:,-1].reshape(-1,1).expand((-1,z.size(0))).t()
        r = z[:,:-1]
        norm = (r * r).sum(dim = 1, keepdim = True)
        r1r2 = torch.matmul(r, r.t())
        r2 = norm - 2*r1r2 + norm.t() 
        logr2 = torch.log(r2 + self.EPS)

        if self.CLAMP is not None:
            logr2 = logr2.clamp(min = -self.CLAMP, max = self.CLAMP)
        adj = (m_i -  self.l * logr2)
        return torch.sigmoid(adj) if sigmoid else adj

    def forward(self, z, edge_label_index, sigmoid=True):
        adj = self.decode_all(z)
        value = adj[edge_label_index[0,:], edge_label_index[1,:]].reshape(-1,1)
        return torch.sigmoid(value) if sigmoid else value
    

################################################################################
# DECODER for MULTICLASS DIRECTED models
################################################################################
def get_multiclass_from_logits(logits_uv, logits_vu):
    sigma_uv = torch.sigmoid(logits_uv)
    sigma_vu = torch.sigmoid(logits_vu)
    p_nu = ((1. - sigma_uv)*sigma_vu).reshape(-1,1)
    p_pu = (sigma_uv*(1.-sigma_vu)).reshape(-1,1)
    p_pb = (sigma_uv*sigma_vu).reshape(-1,1)
    p_nb = ((1.-sigma_uv)*(1.-sigma_vu)).reshape(-1,1)
    probs = torch.cat((p_nb, p_pu, p_pb, p_nu), dim = 1)
    log_probs = torch.log(probs.clamp(min = 1e-10, max = 1.)) 

    return probs, log_probs

class DirectedMCInnerProductDecoder(torch.nn.Module):
    def forward(self, s, t, edge_index, sigmoid=True, binary=False):
        if binary:
           value = (s[edge_index[0]] * t[edge_index[1]]).sum(dim=1)
           return torch.sigmoid(value) if sigmoid else value 
        logit_uv = (s[edge_index[0]] * t[edge_index[1]]).sum(dim=1)
        logit_vu = (s[edge_index[1]] * t[edge_index[0]]).sum(dim=1)
        probs, log_probs = get_multiclass_from_logits(logit_uv, logit_vu) 

        return probs if sigmoid else log_probs

    def forward_all(self, s, t, sigmoid=True, binary=False):
        print("=====> HERE!")
        adj = torch.matmul(s, t.t())
        if binary:
            return torch.sigmoid(adj) if sigmoid else adj
        probs, log_probs = get_multiclass_from_logits(adj) 
        return probs if sigmoid else log_probs


class GravityMCDecoder(torch.nn.Module):
    def __init__(self, l, EPS = 1e-2, CLAMP = None, train_l = True): 
        super().__init__()
        self.l_initialization = l
        self.l = torch.nn.Parameter(torch.tensor([l], dtype=torch.float64), requires_grad=train_l)
        self.EPS = EPS
        self.CLAMP = CLAMP

    def decode_all(self, z, sigmoid=True, binary=False):
        m_i = z[:,-1].reshape(-1,1).expand((-1,z.size(0))).t()
        r = z[:,:-1]
        norm = (r * r).sum(dim = 1, keepdim = True)
        r1r2 = torch.matmul(r, r.t())
        r2 = norm - 2*r1r2 + norm.t() 
        logr2 = torch.log(r2 + self.EPS)

        if self.CLAMP is not None:
            logr2 = logr2.clamp(min = -self.CLAMP, max = self.CLAMP)
        adj = (m_i -  self.l * logr2)
        
        if binary:
            return torch.sigmoid(adj) if sigmoid else adj
        
        probs, log_probs = get_multiclass_from_logits(adj) 
        return probs if sigmoid else log_probs
    

    def forward(self, z, edge_label_index, sigmoid=True, binary=False):
        adj = self.decode_all(z, sigmoid=False, binary=True)
        if binary:
            value = adj[edge_label_index[0,:], edge_label_index[1,:]].reshape(-1,1)
            return torch.sigmoid(value) if sigmoid else value 
        logit_uv = adj[edge_label_index[0,:], edge_label_index[1,:]].reshape(-1,1)
        logit_vu = adj[edge_label_index[1,:], edge_label_index[0,:]].reshape(-1,1)  
        probs, log_probs = get_multiclass_from_logits(logit_uv, logit_vu) 
        return probs if sigmoid else log_probs
    

################################################################################
# DECODER for MPLP UNDIRECTED model
################################################################################
class MPLPDecoder(torch.nn.Module):
    def __init__(self, in_channels,
                 feat_dropout=0., label_dropout=0., num_hops=2, prop_type='combine', signature_sampling='torchhd', use_degree='RA',
                 signature_dim=1024, minimum_degree_onehot=50, batchnorm_affine=True,
                 feature_combine="hadamard",adj2=False):
        super(MPLPDecoder, self).__init__()

        self.in_channels = in_channels
        self.feat_dropout = feat_dropout
        self.label_dropout = label_dropout
        self.num_hops = num_hops
        self.prop_type = prop_type # "MPLP+exactly","MPLP+prop_only","MPLP+combine"
        self.signature_sampling=signature_sampling
        self.use_degree = use_degree
        self.feature_combine = feature_combine
        self.adj2 = adj2

        print("=====================================")
        print(self.in_channels,
        self.feat_dropout,
        self.label_dropout,
        self.num_hops,
        self.prop_type,
        self.signature_sampling,
        self.use_degree,
        self.feature_combine,
        self.adj2)
        print("=====================================")

        if self.use_degree == 'mlp':
            self.node_weight_encode = MLP(2, in_channels + 1, 32, 1, feat_dropout, norm_type="batch", affine=batchnorm_affine)
        if self.prop_type in ['prop_only', 'precompute']:
            struct_dim = 8
        elif self.prop_type == 'exact':
            struct_dim = 5
        elif self.prop_type == 'combine':
            struct_dim = 15
            
        self.nodelabel = NodeLabel(signature_dim, signature_sampling=self.signature_sampling, prop_type=self.prop_type,
                               minimum_degree_onehot= minimum_degree_onehot)
        self.struct_encode = MLP(1, struct_dim, struct_dim, struct_dim, self.label_dropout, "batch", tailnormactdrop=True, affine=batchnorm_affine)

        dense_dim = struct_dim + in_channels
        if in_channels > 0:
            if feature_combine == "hadamard":
                feat_encode_input_dim = in_channels
            elif feature_combine == "plus_minus":
                feat_encode_input_dim = in_channels * 2
            elif feature_combine == "cat":
                feat_encode_input_dim = in_channels * 2
            self.feat_encode = MLP(2, feat_encode_input_dim, in_channels, in_channels, self.feat_dropout, "batch", tailnormactdrop=True, affine=batchnorm_affine)
        self.classifier = torch.nn.Linear(dense_dim, 1)


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
    
    def forward(self, x, adj, edges, cache_mode=None, adj2=None, sigmoid=True):
        """
        Args:
            x: [N, in_channels] node embedding after GNN
            adj: [N, N] adjacency matrix
            edges: [2, E] target edges
            fast_inference: bool. If True, only caching the message-passing without calculating the structural features
        """

        # if cache_mode is None and self.prop_type == "precompute":
        #     # when using precompute, forward always use cache_mode == 'use'
        #     cache_mode = 'use'
        # if cache_mode in ["use","delete"]:
        #     # no need to compute node_weight
        #     node_weight = None
        # elif self.use_degree == 'none':
        #     node_weight = None
        # elif self.use_degree == 'mlp': # 'mlp' for now
        if self.use_degree == 'mlp': # 'mlp' for now
            xs = []
            if self.in_channels > 0:
                xs.append(x)
            degree = adj.sum(dim=1).view(-1,1).to(adj.device())
            xs.append(degree)
            node_weight_feat = torch.cat(xs, dim=1)
            node_weight = self.node_weight_encode(node_weight_feat).squeeze(-1) + 1 # like residual, can be learned as 0 if needed
        else:
            # AA or RA
            degree = adj.sum(dim=1).view(-1,1).to(adj.device()).squeeze(-1) + 1 # degree at least 1. then log(degree) > 0.
            if self.use_degree == 'AA':
                node_weight = torch.sqrt(torch.reciprocal(torch.log(degree)))
            elif self.use_degree == 'RA':
                node_weight = torch.sqrt(torch.reciprocal(degree))
            node_weight = torch.nan_to_num(node_weight, nan=0.0, posinf=0.0, neginf=0.0)

        # if cache_mode in ["build","delete"]:
        #     propped = self.nodelabel(edges, adj, node_weight=node_weight, cache_mode=cache_mode)
        #     return
        # else:
        propped = self.nodelabel(edges, adj, node_weight=node_weight, cache_mode=cache_mode, adj2=adj2)
        propped_stack = torch.stack([*propped], dim=1)
        out = self.struct_encode(propped_stack)

        if self.in_channels > 0:
            x_i = x[edges[0]]
            x_j = x[edges[1]]
            if self.feature_combine == "hadamard":
                x_ij = x_i * x_j
            elif self.feature_combine == "plus_minus":
                x_ij = torch.cat([x_i+x_j, torch.abs(x_i-x_j)], dim=1)
            elif self.feature_combine == "cat":
                x_ij = torch.cat([x_i, x_j], dim=1)

            x_ij = self.feat_encode(x_ij)
            x = torch.cat([x_ij, out], dim=1)
        else:
            x = out
        logit = self.classifier(x)
        return torch.sigmoid(logit) if sigmoid else logit

    def precompute(self, adj):
        self(None, adj, None, cache_mode="build")
        return self
    
################################################################################
# DECODER for MULTICLASS MPLP UNDIRECTED model
################################################################################
class MPLPMCDecoder(torch.nn.Module):
    def __init__(self, in_channels,
                 feat_dropout=0., label_dropout=0., num_hops=2, prop_type='combine', signature_sampling='torchhd', use_degree='mlp',
                 signature_dim=1024, minimum_degree_onehot=50, batchnorm_affine=True,
                 feature_combine="hadamard",adj2=False):
        super(MPLPMCDecoder, self).__init__()

        self.in_channels = in_channels
        self.feat_dropout = feat_dropout
        self.label_dropout = label_dropout
        self.num_hops = num_hops
        self.prop_type = prop_type # "MPLP+exactly","MPLP+prop_only","MPLP+combine"
        self.signature_sampling=signature_sampling
        self.use_degree = use_degree
        self.feature_combine = feature_combine
        self.adj2 = adj2

        print("=====================================")
        print(self.in_channels,
        self.feat_dropout,
        self.label_dropout,
        self.num_hops,
        self.prop_type,
        self.signature_sampling,
        self.use_degree,
        self.feature_combine,
        self.adj2)
        print("=====================================")

        if self.use_degree == 'mlp':
            self.node_weight_encode = MLP(2, in_channels + 1, 32, 1, feat_dropout, norm_type="batch", affine=batchnorm_affine)
        if self.prop_type in ['prop_only', 'precompute']:
            struct_dim = 8
        elif self.prop_type == 'exact':
            struct_dim = 5
        elif self.prop_type == 'combine':
            struct_dim = 15
            
        self.nodelabel = NodeLabel(signature_dim, signature_sampling=self.signature_sampling, prop_type=self.prop_type,
                               minimum_degree_onehot= minimum_degree_onehot)
        self.struct_encode = MLP(1, struct_dim, struct_dim, struct_dim, self.label_dropout, "batch", tailnormactdrop=True, affine=batchnorm_affine)

        dense_dim = struct_dim + in_channels
        if in_channels > 0:
            if feature_combine == "hadamard":
                feat_encode_input_dim = in_channels
            elif feature_combine == "plus_minus":
                feat_encode_input_dim = in_channels * 2
            elif feature_combine == "cat":
                feat_encode_input_dim = in_channels * 2
            self.feat_encode = MLP(2, feat_encode_input_dim, in_channels, in_channels, self.feat_dropout, "batch", tailnormactdrop=True, affine=batchnorm_affine)
        self.classifier = torch.nn.Linear(dense_dim, 1)


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
    
    def _decode(self, x, adj, edges, cache_mode=None, adj2=None):
        """
        Args:
            x: [N, in_channels] node embedding after GNN
            adj: [N, N] adjacency matrix
            edges: [2, E] target edges
            fast_inference: bool. If True, only caching the message-passing without calculating the structural features
        """
        if cache_mode is None and self.prop_type == "precompute":
            # when using precompute, forward always use cache_mode == 'use'
            cache_mode = 'use'
        if cache_mode in ["use","delete"]:
            # no need to compute node_weight
            node_weight = None
        elif self.use_degree == 'none':
            node_weight = None
        elif self.use_degree == 'mlp': # 'mlp' for now
            xs = []
            if self.in_channels > 0:
                xs.append(x)
            degree = adj.sum(dim=1).view(-1,1).to(adj.device())
            xs.append(degree)
            node_weight_feat = torch.cat(xs, dim=1)
            node_weight = self.node_weight_encode(node_weight_feat).squeeze(-1) + 1 # like residual, can be learned as 0 if needed
        else:
            # AA or RA
            degree = adj.sum(dim=1).view(-1,1).to(adj.device()).squeeze(-1) + 1 # degree at least 1. then log(degree) > 0.
            if self.use_degree == 'AA':
                node_weight = torch.sqrt(torch.reciprocal(torch.log(degree)))
            elif self.use_degree == 'RA':
                node_weight = torch.sqrt(torch.reciprocal(degree))
            node_weight = torch.nan_to_num(node_weight, nan=0.0, posinf=0.0, neginf=0.0)

        if cache_mode in ["build","delete"]:
            propped = self.nodelabel(edges, adj, node_weight=node_weight, cache_mode=cache_mode)
            return
        else:
            propped = self.nodelabel(edges, adj, node_weight=node_weight, cache_mode=cache_mode, adj2=adj2)
        propped_stack = torch.stack([*propped], dim=1)
        out = self.struct_encode(propped_stack)

        if self.in_channels > 0:
            x_i = x[edges[0]]
            x_j = x[edges[1]]
            if self.feature_combine == "hadamard":
                x_ij = x_i * x_j
            elif self.feature_combine == "plus_minus":
                x_ij = torch.cat([x_i+x_j, torch.abs(x_i-x_j)], dim=1)
            elif self.feature_combine == "cat":
                x_ij = torch.cat([x_i, x_j], dim=1)
            x_ij = self.feat_encode(x_ij)
            x = torch.cat([x_ij, out], dim=1)
        else:
            x = out
        logit = self.classifier(x)
        return logit
        
        
    
    def forward(self, x, adj, edges, cache_mode=None, adj2=None, sigmoid=True, binary=False):
        x_ = torch.clone(x)
        adj_ = adj.clone()
        edges_ = torch.clone(edges)
        
        logit_uv = self._decode(x, adj, edges)
        if binary:
            return torch.sigmoid(logit_uv) if sigmoid else logit_uv
        
        logit_vu = self._decode(x_, adj_, edges_.flip(dims = (0,)))
        probs, log_probs = get_multiclass_from_logits(logit_uv, logit_vu)  
        return probs if sigmoid else log_probs

    def precompute(self, adj):
        self(None, adj, None, cache_mode="build")
        return self