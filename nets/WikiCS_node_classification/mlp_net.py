import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.mlp_readout_layer import MLPReadout
from loss.triplet_loss import TripletLoss


class MLPNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.gated = net_params['gated']
        self.n_classes = n_classes
        self.device = net_params['device']
        if 'loss_type' in net_params.keys():
            self.loss_type = net_params['loss_type']
        else:
            self.loss_type = 'cross_entropy'
        
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        feat_mlp_modules = [
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers-1):
            feat_mlp_modules.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            feat_mlp_modules.append(nn.ReLU())
            feat_mlp_modules.append(nn.Dropout(dropout))
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)
        
        if self.gated:
            self.gates = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.readout_mlp = MLPReadout(hidden_dim, n_classes)


    def forward(self, g, h, e):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # MLP
        h = self.feat_mlp(h)
        if self.gated:
            h = torch.sigmoid(self.gates(h)) * h
        
        # output
        h_out = self.readout_mlp(h)

        return h_out

        
    def loss(self, pred, label):

        if self.loss_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif self.loss_type == 'triplet_loss':
            criterion = TripletLoss()
        loss = criterion(pred, label)

        return loss



        
