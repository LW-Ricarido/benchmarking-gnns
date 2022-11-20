import imp
import numpy as np

import dgl
from scipy import sparse as sp
import torch
from ogb.nodeproppred import DglNodePropPredDataset

class OgbProductsDataset(torch.utils.data.Dataset):

    def __init__(self, DATASET_NAME='ObgProducts', path='data/Ogb') -> None:
        self.name = DATASET_NAME
        self.data = DglNodePropPredDataset(name='ogbn-products')
        
        self.g, self.labels = None, None
        self.train_masks, self.val_masks, self.test_mask = None, None, None
        self.num_classes, self.n_feats = None, None
    
    def _load(self):
        self.g, self.labels = self.data[0]
        split_idx = self.data.get_idx_split()
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        self.train_masks = torch.zeros(self.g.number_of_nodes(), dtype=torch.bool) 
        for i in train_idx:
            self.train_masks[i] = True
        self.train_masks = [self.train_masks]
        self.val_masks = torch.zeros(self.g.number_of_nodes(), dtype=torch.bool)
        for i in valid_idx:
            self.val_masks[i] = True
        self.val_masks = [self.val_masks]
        self.test_mask = torch.zeros(self.g.number_of_nodes(), dtype=torch.bool)
        for i in test_idx:
            self.test_mask[i] = True
        self.n_feats = self.g.ndata['feat'].shape[1]
        self.num_classes = len(set(self.g.ndata['label'].tolist()))
        self.g.ndata['feat'] = self.g.ndata['feat']
        self.g.edata['feat'] = torch.zeros(self.g.number_of_edges(), 1)

    def _add_positional_encodings(self, pos_enc_dim):
        g = self.g

        # Laplacian
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
        
        self.g = g