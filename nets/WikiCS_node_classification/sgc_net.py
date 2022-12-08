import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import dgl
import dgl.function as fn
import numpy as np

from layers.mlp_readout_layer import MLPReadout


class SGCNet(nn.Module):

    def __init__(self, net_params) -> None:
        super().__init__()


        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.conv_layers = dgl.nn.pytorch.conv.SGConv(hidden_dim, n_classes,n_layers)

    def forward(self, g, h, e):
        h = self.embedding_h(h)

        h = self.in_feat_dropout(h)

        h_out = self.conv_layers(g, h)

        return h_out
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss