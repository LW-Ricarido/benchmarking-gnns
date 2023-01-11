import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from utils.tools import sparse_mx_to_torch_sparse_tensor, torch_sparse_tensor_to_sparse_mx
import dgl
import numpy as np
import pdb

class NDLS(nn.Module):

    def __init__(self, net_params) -> None:
        super().__init__()
        self.preprocessed = False
        self.k1 = net_params['k1']
        self.k2 = net_params['k2']
        self.epsilon1 = net_params['epsilon1']
        self.epsilon2 = net_params['epsilon2']
        self.alpha = net_params['alpha']
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
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.extend([nn.Dropout(dropout), nn.Linear(in_dim_node,hidden_dim),nn.ReLU()])
            for i in range(n_layers - 1):
                self.layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            self.layers.append(nn.Linear(hidden_dim, n_classes))
        else:
            self.layers.append(nn.Linear(in_dim_node, n_classes))
    
    def ndls_f(self,g, feature, adjecant_matrix):
        node_sum = g.num_nodes()
        edge_sum = g.num_edges()
        adj_sp_np = torch_sparse_tensor_to_sparse_mx(adjecant_matrix)
        self_loop_degree = (adj_sp_np.sum(1) + 1)
        norm_a_inf = self_loop_degree / (2 * edge_sum + node_sum)
        
        d_inv = np.array(np.power(self_loop_degree, -1.0)).flatten()
        d_mat = sp.diags(d_inv)
        adj_norm = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(d_mat.dot(adj_sp_np + sp.eye(adj_sp_np.shape[0])))).to(self.device)

        feature = F.normalize(feature, p=1)
        feature_list = []
        feature_list.append(feature)
        for i in range(1, self.k1):
            feature_list.append(torch.mm(adj_norm, feature_list[-1]))
        norm_a_inf = torch.Tensor(norm_a_inf).view(-1, node_sum).to(self.device)
        norm_fea_inf = torch.mm(norm_a_inf, feature)

        hops = torch.Tensor([0] * adjecant_matrix.shape[0]).to(self.device)
        mask_before = torch.Tensor([False] * adjecant_matrix.shape[0]).bool().to(self.device)

        for i in range(self.k1):
            dist = (feature_list[i] - norm_fea_inf).norm(2, 1)
            mask = (dist < self.epsilon1).masked_fill_(mask_before, False)
            mask_before.masked_fill_(mask, True)
            hops.masked_fill_(mask, i)
        mask_final = torch.Tensor([True] * adjecant_matrix.shape[0]).bool().to(self.device)
        mask_final.masked_fill_(mask_before, False)
        hops.masked_fill_(mask_final, self.k1 - 1)


        input_feature = []
        for i in range(adjecant_matrix.shape[0]):
            hop = hops[i].int().item()
            if hop == 0:
                fea = feature_list[0][i].unsqueeze(0)
            else:
                fea = 0
                for j in range(hop):
                    fea += (1 - self.alpha) * feature_list[j][i].unsqueeze(0) + self.alpha * feature_list[0][i].unsqueeze(0)
                fea = fea / hop
            input_feature.append(fea)
        input_feature = torch.cat(input_feature, dim=0)
        # pdb.set_trace()
        self.preprocessed = True
        self.cached_feature = input_feature
        return input_feature

    def ndls_l(self):
        pass

    def forward(self, g, h, e):
        with torch.no_grad():
            if self.preprocessed is False:
                adjecant_matrix = g.adj().to(self.device)
                h = self.ndls_f(g, h, adjecant_matrix)
                print('===============passs ndls_f')
            else:
                h = self.cached_feature
        for layer in self.layers:
            h = layer(h)
        h = F.log_softmax(h, dim=1)
        return h
    
    def loss(self, pred, label):

        criterion = nn.NLLLoss()
        loss = criterion(pred, label)

        return loss