"""
    File to load dataset based on user control from main file
"""
from data.superpixels import SuperPixDataset
from data.molecules import MoleculeDataset
from data.TUs import TUsDataset
from data.SBMs import SBMsDataset
from data.TSP import TSPDataset
from data.COLLAB import COLLABDataset
from data.CSL import CSLDataset
from data.cycles import CyclesDataset
from data.graphtheoryprop import GraphTheoryPropDataset
from data.WikiCS import WikiCSDataset
from data.Cora import CoraDataset
from data.CiteSeer import CiteSeerDateset
from data.Pubmed import PubmedDataset
from data.OgbProducts import OgbProductsDataset
from data.OgbArxiv import OgbArxivDataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for MNIST or CIFAR Superpixels
    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)
    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME in ['ZINC', 'ZINC-full', 'AQSOL']:
        return MoleculeDataset(DATASET_NAME)

    # handling for the TU Datasets
    TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS_full']
    if DATASET_NAME in TU_DATASETS: 
        return TUsDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS: 
        return SBMsDataset(DATASET_NAME)
    
    # handling for TSP dataset
    if DATASET_NAME == 'TSP':
        return TSPDataset(DATASET_NAME)

    # handling for COLLAB dataset
    if DATASET_NAME == 'OGBL-COLLAB':
        return COLLABDataset(DATASET_NAME)

    # handling for the CSL (Circular Skip Links) Dataset
    if DATASET_NAME == 'CSL': 
        return CSLDataset(DATASET_NAME)
    
    # handling for the CYCLES Dataset from https://github.com/cvignac/SMP
    if DATASET_NAME == 'CYCLES': 
        return CyclesDataset(DATASET_NAME)
    
    # handling for the GraphTheoryProp Dataset, which is the multitask dataset from https://github.com/lukecavabarrett/pna
    if DATASET_NAME == 'GraphTheoryProp' or DATASET_NAME == 'GRAPHTHEORYPROP':
        return GraphTheoryPropDataset('GraphTheoryProp')
    
    if DATASET_NAME == 'WikiCS':
        return WikiCSDataset(DATASET_NAME)
    
    if DATASET_NAME == 'Cora':
        return CoraDataset(DATASET_NAME)
    
    if DATASET_NAME == 'CiteSeer':
        return CiteSeerDateset(DATASET_NAME)

    if DATASET_NAME == "Pubmed":
        return PubmedDataset(DATASET_NAME)
    
    if DATASET_NAME == 'OgbProducts':
        return OgbProductsDataset(DATASET_NAME)

    if DATASET_NAME == 'OgbArxiv':
        return OgbArxivDataset(DATASET_NAME)