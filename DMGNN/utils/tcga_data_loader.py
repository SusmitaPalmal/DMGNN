from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
import sys
import os
import pathlib

def load_tcga_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/TCGA-BRCA"
    print(data_dir)
    print("loading TCGA-BRCA data")

    cnv = pd.read_csv(data_dir + "/Copy_Number_Variation_Data.csv")
    gene = pd.read_csv(data_dir + "/Gene_Expression_Data.csv")
    clinical = pd.read_csv(data_dir + "/Clinical_Data.csv")
    target = pd.read_csv(data_dir + "/Class_label.csv")

    clinical.drop(columns=['submitter_id.samples'], inplace=True)
    cnv.drop(columns=['submitter_id.samples'], inplace=True)
    gene.drop(columns=['submitter_id.samples'], inplace=True)
    target.drop(columns=['submitter_id.samples'], inplace=True)

    target.replace({'Short_term': 1, 'Long_term': 0}, inplace=True)
    merged_data = pd.concat([clinical, gene, cnv], axis=1)

    edge_list  = [(i, i) for i in range(merged_data.shape[0])]
    edge_array = np.array(edge_list).T
    edge_index = torch.tensor(edge_array, dtype=torch.long)
    print(edge_index.shape)

    Y = np.array(target)
    node_feats = np.array(merged_data)
    print(Y.shape,node_feats.shape)
    print(sum(node_feats.sum(axis=1)==0))

    node_feats = torch.tensor(node_feats,dtype=torch.float) ## boolean features present or absent
    print(node_feats.shape)

    data = Data(
            x=node_feats,
            edge_index=edge_index,
            y=torch.tensor(Y,dtype=torch.long),
            train_mask = torch.ones(len(Y),dtype=torch.bool), ### placeholder for masks
            val_mask = torch.ones(len(Y),dtype=torch.bool), ### placeholder for masks
            test_mask = torch.ones(len(Y),dtype=torch.bool), ### placeholder for masks
            num_nodes = node_feats.shape[0],
            num_features = node_feats.shape[1]

    )
    return data
