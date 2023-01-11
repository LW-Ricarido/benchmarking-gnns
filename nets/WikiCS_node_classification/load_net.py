"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.WikiCS_node_classification.gated_gcn_net import GatedGCNNet
from nets.WikiCS_node_classification.gcn_net import GCNNet
from nets.WikiCS_node_classification.gat_net import GATNet
from nets.WikiCS_node_classification.graphsage_net import GraphSageNet
from nets.WikiCS_node_classification.mlp_net import MLPNet
from nets.WikiCS_node_classification.gin_net import GINNet
from nets.WikiCS_node_classification.mo_net import MoNet as MoNet_
from nets.WikiCS_node_classification.DAGNN_net import DAGNNNet
from nets.WikiCS_node_classification.sgc_net import SGCNet
from nets.WikiCS_node_classification.NDLS import NDLS as NDLS_Net

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def DAGNN(net_params):
    return DAGNNNet(net_params)

def SGC(net_params):
    return SGCNet(net_params)

def NDLS(net_params):
    return NDLS_Net(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'MLP': MLP,
        'GIN': GIN,
        'MoNet': MoNet,
        'DAGNN': DAGNN,
        'SGC': SGC,
        'NDLS': NDLS
    }
        
    return models[MODEL_NAME](net_params)