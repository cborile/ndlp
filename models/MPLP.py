import torch
from .model_utils import Model, reset
from .decoders import MPLPDecoder, MPLPMCDecoder

from torch_sparse import SparseTensor

EPS        = 1e-15
MAX_LOGSTD = 10


class MPLP(Model):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.
    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self,
                 encoder, decoder=None,
                 in_channels=None, use_degree='mlp',
                 feature_combine="hadamard",
                 minimum_degree_onehot=50,
                 **kwargs):
        super(MPLP, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = MPLPDecoder(in_channels=in_channels,
                                   use_degree=use_degree,
                                   feature_combine=feature_combine,
                                   minimum_degree_onehot=minimum_degree_onehot
                                   ) if decoder is None else decoder
        MPLP.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, x, edge_index):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(x, edge_index)

    def decode(self, z, adj_t, edge_label_index, sigmoid):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(z, adj_t, edge_label_index, sigmoid=sigmoid)
    
    def forward(self, x, edge_index, edge_label_index, sigmoid):
        r"""Runs the forward pass."""
        adj_t = SparseTensor(row=edge_index[0].type(torch.long), col=edge_index[1].type(torch.long), sparse_sizes=(x.size(1), x.size(1))).t()
        z = self.encode(x, adj_t)
        return self.decode(z, adj_t, edge_label_index, sigmoid=sigmoid).view(-1)
    

class MCMPLP(Model):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.
    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None, in_channels=None, use_degree='mlp', feature_combine="hadamard", **kwargs):
        super(MCMPLP, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = MPLPMCDecoder(in_channels=in_channels, use_degree=use_degree, feature_combine=feature_combine) if decoder is None else decoder
        MPLP.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, x, edge_index):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(x, edge_index)

    def decode(self, z, adj_t, edge_label_index, sigmoid, binary=False):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(z, adj_t, edge_label_index, sigmoid=sigmoid, binary=binary)
    
    def forward(self, x, edge_index, edge_label_index, sigmoid, binary=False):
        r"""Runs the forward pass."""
        adj_t = SparseTensor(row=edge_index[0].type(torch.long), col=edge_index[1].type(torch.long), sparse_sizes=(x.size(1), x.size(1))).t()
        z = self.encode(x, adj_t)
        return self.decode(z, adj_t, edge_label_index, sigmoid=sigmoid, binary=binary) #.view(-1)