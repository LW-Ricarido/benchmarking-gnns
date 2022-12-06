import torch
import torch.utils.data
import numpy as np

import dgl
from scipy import sparse as sp

class CoraDataset(torch.utils.data.Dataset):
    
    def __init__(self, DATASET_NAME='Cora', path="data/Cora/"):
        self.name = DATASET_NAME
        self.data = dgl.data.CoraGraphDataset()
        
        self.g, self.labels = None, None
        self.train_masks, self.val_masks, self.test_mask = None, None, None
        self.num_classes, self.n_feats = None, None
        
        self._load()
    
    def _load(self):
        self.g = self.data[0]
        self.labels = self.g.ndata['label']
        self.train_masks = [self.g.ndata['train_mask']]
        self.val_masks = [self.g.ndata['val_mask']]

        self.test_mask = self.g.ndata['test_mask']
        
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
