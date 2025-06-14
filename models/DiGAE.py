import torch
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

from .decoders import DirectedInnerProductDecoder, DirectedMCInnerProductDecoder
from .model_utils import Model, reset

EPS        = 1e-15
MAX_LOGSTD = 10


class DirectedGAE(Model):
    def __init__(self, encoder, decoder=None, **kwargs):
        super(DirectedGAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = DirectedInnerProductDecoder() if decoder is None else decoder
        DirectedGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    
    def forward_all(self, x, edge_index):
        s, t = self.encoder(x, x, edge_index)
        adj_pred = self.decoder.forward_all(s, t)
        return adj_pred

    
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    
    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
    
    def forward(self, x, edge_index, edge_label_index, sigmoid):
        s, t = self.encode(x, x, edge_index)
        return self.decode(s, t, edge_label_index, sigmoid=sigmoid).view(-1)

    
    def recon_loss(self, s, t, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decoder(s, t, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, s.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(s, t, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss
    

class DirectedMCGAE(Model):
    def __init__(self, encoder, decoder=None, **kwargs):
        super(DirectedMCGAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = DirectedMCInnerProductDecoder() if decoder is None else decoder
        DirectedMCGAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    
    def forward_all(self, x, edge_index):
        s, t = self.encoder(x, x, edge_index)
        adj_pred = self.decoder.forward_all(s, t)
        return adj_pred

    
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    
    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
    
    def forward(self, x, edge_index, edge_label_index, sigmoid, binary=False):
        s, t = self.encode(x, x, edge_index)
        return self.decode(s, t, edge_label_index, sigmoid=sigmoid, binary=binary) #.view(-1)

    
    def recon_loss(self, s, t, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decoder(s, t, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, s.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(s, t, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss