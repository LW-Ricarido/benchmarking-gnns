import imp
import numpy as np

import dgl
from scipy import sparse as sp
import torch
from ogb.nodeproppred import DglNodePropPredDataset

class OgbArxivDataset(torch.utils.data.Dataset):

    def __init__(self, DATASET_NAME='ObgArxiv', path='/home/wei/dataset') -> None:
        self.name = DATASET_NAME
        self.data = DglNodePropPredDataset(name='ogbn-arxiv',root=path)
        
        self.g, self.labels = None, None
        self.train_masks, self.val_masks, self.test_mask = None, None, None
        self.num_classes, self.n_feats = None, None
        
        self._load()
    
    def _load(self):
        self.g, self.labels = self.data[0]
        self.g = dgl.add_self_loop(self.g)
        self.labels = self.labels.reshape(self.labels.shape[0])
        self.g.ndata['label'] = self.labels
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
        # self.g.ndata['feat'] = self.g.ndata['feat']
        self.g.edata['feat'] = torch.zeros(self.g.number_of_edges(), 1)

        print("NumNodes:{}\n NumEdges:{}\n NumFeats:{}\n NumClasses:{}\n NumTraining:{}\n NumValid:{}\n NumTest:{}\n".format(
            self.g.number_of_nodes(), self.g.number_of_edges(), self.n_feats, self.num_classes,
            len(train_idx), len(valid_idx),len(test_idx)
        ))


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

    def split_dataset(self):
        split_idx = self.data.get_idx_split()
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        trainDataSet = OgbArxivNodeDataset(self.g.ndata['feat'].index_select(0,train_idx), self.labels.index_select(0,train_idx))
        valDataSet = OgbArxivNodeDataset(self.g.ndata['feat'].index_select(0,valid_idx), self.labels.index_select(0,valid_idx))
        testDataSet = OgbArxivNodeDataset(self.g.ndata['feat'].index_select(0,test_idx), self.labels.index_select(0,test_idx))
        return trainDataSet, valDataSet, testDataSet

class OgbArxivNodeDataset(torch.utils.data.Dataset):

    def __init__(self, node_features, labels) -> None:
        self.node_features = node_features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.node_features[idx], self.labels[idx]