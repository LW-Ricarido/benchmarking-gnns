import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import dgl.function as fn
import numpy as np

from layers.mlp_readout_layer import MLPReadout


class DAGNNConv(nn.Module):
    def __init__(self, in_dim, k):
        super(DAGNNConv, self).__init__()

        self.s = Parameter(torch.FloatTensor(in_dim, 1))
        self.k = k

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):

        with graph.local_scope():
            results = [feats]

            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata["h"]
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = F.sigmoid(torch.matmul(H, self.s))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()

            return H

class DAGNNNet(nn.Module):

    def __init__(self, net_params):
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
        learning_layers = net_params['learning_layers']

        self.embedding_h = nn.Linear(in_dim_node, n_classes) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList()
        # for i in range(learning_layers - 2):
        #     self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        #     self.layers.append(nn.ReLU())
        #     self.layers.append(nn.Dropout(dropout))
        # self.layers.append(nn.Linear(hidden_dim, out_dim))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.Dropout(dropout))
        self.graphConv = DAGNNConv(n_classes, n_layers)
        # self.MLP_layer = MLPReadout(50, n_classes)
    
    def forward(self, g, h, e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for layer in self.layers:
            h = layer(h)
        
        h = self.graphConv(g, h)

        h_out = h #self.MLP_layer(h)
        return h_out
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss