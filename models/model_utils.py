import functools
from torch.nn import Module
from torch_geometric import seed_everything

class Model(Module):
    def __init__(self,
                 model_name,
                 optimizer,
                 random_seed = 42):
        super().__init__()
        self.model_name = model_name
        self.random_seed = random_seed
        self._optimizer = optimizer
        seed_everything(random_seed)

    @property
    def optimizer(self):
        if isinstance(self._optimizer, functools.partial):
            # Partial function needs to be called to instantiate the optimizer
            self._optimizer = self._optimizer(self.parameters())
        return self._optimizer

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


############################################
# OLD CM
############################################
# class GNN_FB(Module):
#     def __init__(self, gnn_layers,  preprocessing_layers = [], postprocessing_layers = []):
#         super().__init__()

#         self.gnn_layers = ModuleList(gnn_layers)
#         self.preprocessing_layers = ModuleList(preprocessing_layers)
#         self.postprocessing_layers = ModuleList(postprocessing_layers)
#         # self.net = torch.nn.Sequential(*preprocessing_layers, *gnn_layers, *postprocessing_layers )
    
#     def forward(self, x, edge_index):
#         # x, edge_index = batch.x, batch.edge_index
#         for layer in self.preprocessing_layers:
#             x = layer(x, edge_index)
#         for layer in self.gnn_layers:
#             x = layer(x, edge_index)
#         for layer in self.postprocessing_layers:
#             x = layer(x, edge_index)
#         return x

# class EncoderDecoderGAE(Model):

#     def __init__(self, model_name, encoder, decoder):
#         super().__init__(model_name)
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, x, edge_index, edge_label_index):
#         x = self.encoder(x, edge_index)
#         return self.decoder(x, edge_label_index)
    
# class DecoderDotProduct(Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, z, edge_label_index):

#         if edge_label_index in ["full_graph", "salha_biased"]:
            
#             z = torch.matmul(z, z.t()).reshape(-1,1) 

#         else:
#             z = ( z[edge_label_index[0,:],:] * z[edge_label_index[1,:],:]).sum(dim = 1).reshape(-1,1)
        
#         return z

# # This conv expects self loops to have already been added, and that the rows of the adjm are NOT normalized by the inverse out-degree +1. Such normalization will be done using the "mean" aggregation inherited from MessagePassing The adjm in question should be the transpose of the usual adjm.
# class Conv(MessagePassing):

#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr = "mean")
#         self.W = LazyLinear(out_channels, bias = False)

#     def message_and_aggregate(self, adj_t, x):
#         return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

#     def message(self, x_j):
#         return x_j

#     def forward(self, x, edge_index):
#         transformed = self.W(x)

#         transformed_aggregated_normalized = self.propagate(edge_index, x = transformed)  

#         return transformed_aggregated_normalized

#     def reset_parameters(self):
#         super().reset_parameters()
#         self.W.reset_parameters()

# class LayerWrapper(Module):
#     def __init__(self, layer, normalization_before_activation = None, activation = None, normalization_after_activation =None, dropout_p = None, _add_remaining_self_loops = False, uses_sparse_representation = False  ):
#         super().__init__()
#         self.activation = activation
#         self.normalization_before_activation = normalization_before_activation
#         self.layer = layer
#         self.normalization_after_activation = normalization_after_activation
#         self.dropout_p = dropout_p
#         self._add_remaining_self_loops = _add_remaining_self_loops
#         self.uses_sparse_representation = uses_sparse_representation

#     def forward(self, x, edge_index):

#         edge_index = copy.copy(edge_index).to(x.device)
#         if self._add_remaining_self_loops and not self.uses_sparse_representation:
#             edge_index, _  = add_remaining_self_loops(edge_index)
#         elif self._add_remaining_self_loops and self.uses_sparse_representation:
#             edge_index = torch_sparse.fill_diag(edge_index, 2)

#         if not self.uses_sparse_representation:
#             x = self.layer(x = x, edge_index = edge_index)
#         else:
#             x = self.layer(x = x, edge_index = edge_index)

#         if self.normalization_before_activation is not None:
#             x = self.normalization_before_activation(x)
#         if self.activation is not None:
#             x = self.activation(x)
#         if self.normalization_after_activation is not None:
#             x =  self.normalization_after_activation(x)
#         if self.dropout_p is not None:
#             x =  dropout(x, p=self.dropout_p, training=self.training)

#         return x