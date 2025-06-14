import torch
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

from .decoders import GravityDecoder, GravityMCDecoder
from .model_utils import Model, reset


EPS        = 1e-15
MAX_LOGSTD = 10


class GravityGAE(Model):
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
    def __init__(self, encoder, decoder=None, **kwargs):
        super(GravityGAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = GravityDecoder(l=2., CLAMP=10, train_l=False) if decoder is None else decoder
        GravityGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)
    
    def forward(self, x, edge_index, edge_label_index, sigmoid):
        r"""Runs the forward pass."""
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index, sigmoid=sigmoid).view(-1)
    


class GravityMCGAE(Model):
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
    def __init__(self, encoder, decoder=None, **kwargs):
        super(GravityMCGAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = GravityMCDecoder(l=2., CLAMP=10, train_l=False) if decoder is None else decoder
        GravityMCGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)
    
    def forward(self, x, edge_index, edge_label_index, sigmoid, binary=False):
        r"""Runs the forward pass."""
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index, sigmoid=sigmoid, binary=binary) #.view(-1)