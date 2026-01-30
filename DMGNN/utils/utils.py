import torch
import numpy as np
import os
import random
import sys
src_dir = os.path.dirname(os.path.dirname('__file__'))
sys.path.append(src_dir)
from seed import seed

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return 
seed_everything(seed)

def create_otf_edges(node_features,feature_mask):
    #print('feature_mask ', feature_mask.sum())
    assert node_features.shape[1] == feature_mask.shape[1]
    assert node_features.shape[0] == feature_mask.shape[0]
    otf_edge_index = feature_mask.nonzero(as_tuple=True)
    otf_edge_index = (otf_edge_index[0].to(node_features.device), otf_edge_index[1].to(node_features.device))
    
    otf_edge_attr = node_features[otf_edge_index[0],otf_edge_index[1]].reshape(otf_edge_index[0].shape[0], -1)
    otf_edge_index = torch.cat((otf_edge_index[0].unsqueeze(0),otf_edge_index[1].unsqueeze(0)), dim=0)
    # print('otf_edge_index otf_edge_attr', otf_edge_index.shape,otf_edge_attr.shape)
    return otf_edge_index,otf_edge_attr

def get_feature_mask(rate, n_nodes, n_features, type="uniform"):
    """ Return mask of shape [n_nodes, n_features] indicating whether each feature is present or missing"""
    if type == "structural":  # either remove all of a nodes features or none
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes)).bool().unsqueeze(1).repeat(1, n_features)
    else:
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()




