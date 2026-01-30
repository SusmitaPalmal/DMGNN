from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
import sys
import os
from sklearn.random_projection import GaussianRandomProjection 
import pathlib

def load_metabrics_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer data")
    cnv = pd.read_csv(data_dir + "/METABRIC_cnv_1980.txt", delimiter=' ', header=None)
    gene_expression = pd.read_csv(data_dir + "/METABRIC_gene_exp_1980.txt", delimiter=' ', header=None)
    clinical = pd.read_csv(data_dir + "/METABRIC_clinical_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)
    merged_data = pd.concat([clinical, gene_expression, cnv], axis=1)

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
def load_metabrics_clinical_data():
    print("============running Clinical Only============")
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer Clinical data")
    clinical_data = pd.read_csv(data_dir + "/METABRIC_clinical_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)

    edge_list  = [(i, i) for i in range(clinical_data.shape[0])]
    edge_array = np.array(edge_list).T
    edge_index = torch.tensor(edge_array, dtype=torch.long)
    print(edge_index.shape)

    Y = np.array(target)
    node_feats = np.array(clinical_data)
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

def load_metabrics_cnv_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer CNV data")
    cnv = pd.read_csv(data_dir + "/METABRIC_cnv_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)

    edge_list  = [(i, i) for i in range(cnv.shape[0])]
    edge_array = np.array(edge_list).T
    edge_index = torch.tensor(edge_array, dtype=torch.long)
    print(edge_index.shape)

    Y = np.array(target)
    node_feats = np.array(cnv)
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

def load_metabrics_gene_exp_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer Gene_Expression data")
    gene_expression = pd.read_csv(data_dir + "/METABRIC_gene_exp_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)

    edge_list  = [(i, i) for i in range(gene_expression.shape[0])]
    edge_array = np.array(edge_list).T
    edge_index = torch.tensor(edge_array, dtype=torch.long)
    print(edge_index.shape)

    Y = np.array(target)
    node_feats = np.array(gene_expression)
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

def load_metabrics_clinical_gene_exp_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer (Clinical data + Gene_Expression) data")
    clinical = pd.read_csv(data_dir + "/METABRIC_clinical_1980.txt", delimiter=' ', header=None)
    gene_expression = pd.read_csv(data_dir + "/METABRIC_gene_exp_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)
    merged_data = pd.concat([clinical, gene_expression], axis=1)

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

def load_metabrics_clinical_cnv_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer (Clinical data + CNV) data")
    cnv = pd.read_csv(data_dir + "/METABRIC_cnv_1980.txt", delimiter=' ', header=None)
    clinical = pd.read_csv(data_dir + "/METABRIC_clinical_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)
    merged_data = pd.concat([clinical, cnv], axis=1)

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

def load_metabrics_gene_exp_cnv_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer (Gene_Expression + CNV) data")
    cnv = pd.read_csv(data_dir + "/METABRIC_cnv_1980.txt", delimiter=' ', header=None)
    gene_expression = pd.read_csv(data_dir + "/METABRIC_gene_exp_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)
    merged_data = pd.concat([gene_expression, cnv], axis=1)

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
    

def load_metabrics_addition_left_padding_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer data padded with zeros on left (Addition of features)")
    cnv = pd.read_csv(data_dir + "/METABRIC_cnv_1980.txt", delimiter=' ', header=None)
    gene_expression = pd.read_csv(data_dir + "/METABRIC_gene_exp_1980.txt", delimiter=' ', header=None)
    clinical = pd.read_csv(data_dir + "/METABRIC_clinical_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)

    clinical_left_zeros_pad = pd.concat([pd.DataFrame(np.zeros((len(target), 375)), dtype=float), clinical], axis=1)
    clinical_left_zeros_pad.columns = range(clinical_left_zeros_pad.shape[1])

    cnv_left_zeros_pad = pd.concat([pd.DataFrame(np.zeros((len(target), 200)), dtype=float), cnv], axis=1)
    cnv_left_zeros_pad.columns = range(cnv_left_zeros_pad.shape[1])

    merged_data = clinical_left_zeros_pad + cnv_left_zeros_pad + gene_expression

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

def load_metabrics_addition_right_padding_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer data padded with zeros on right (Addition of features)")
    cnv = pd.read_csv(data_dir + "/METABRIC_cnv_1980.txt", delimiter=' ', header=None)
    gene_expression = pd.read_csv(data_dir + "/METABRIC_gene_exp_1980.txt", delimiter=' ', header=None)
    clinical = pd.read_csv(data_dir + "/METABRIC_clinical_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)

    clinical_right_zeros_pad = pd.concat([clinical, pd.DataFrame(np.zeros((len(target), 375)), dtype=float)], axis=1)
    clinical_right_zeros_pad.columns = range(clinical_right_zeros_pad.shape[1])

    cnv_right_zeros_pad = pd.concat([cnv, pd.DataFrame(np.zeros((len(target), 200)), dtype=float)], axis=1)
    cnv_right_zeros_pad.columns = range(cnv_right_zeros_pad.shape[1])
    
    merged_data = clinical_right_zeros_pad + cnv_right_zeros_pad + gene_expression

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

def load_metabrics_multiplication_left_padding_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer data padded with ones on left (Multiplication of features)")
    cnv = pd.read_csv(data_dir + "/METABRIC_cnv_1980.txt", delimiter=' ', header=None)
    gene_expression = pd.read_csv(data_dir + "/METABRIC_gene_exp_1980.txt", delimiter=' ', header=None)
    clinical = pd.read_csv(data_dir + "/METABRIC_clinical_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)
    
    clinical_left_ones_pad = pd.concat([pd.DataFrame(np.ones((len(target), 375)), dtype=float), clinical], axis=1)
    clinical_left_ones_pad.columns = range(clinical_left_ones_pad.shape[1])

    cnv_left_ones_pad = pd.concat([pd.DataFrame(np.ones((len(target), 200)), dtype=float), clinical], axis=1)
    cnv_left_ones_pad.columns = range(cnv_left_ones_pad.shape[1])

    merged_data = clinical_left_ones_pad * cnv_left_ones_pad * gene_expression

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

def load_metabrics_multiplication_right_padding_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer data padded with ones on right (Multiplication of features)")
    cnv = pd.read_csv(data_dir + "/METABRIC_cnv_1980.txt", delimiter=' ', header=None)
    gene_expression = pd.read_csv(data_dir + "/METABRIC_gene_exp_1980.txt", delimiter=' ', header=None)
    clinical = pd.read_csv(data_dir + "/METABRIC_clinical_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)
    
    clinical_right_ones_pad = pd.concat([clinical, pd.DataFrame(np.ones((len(target), 375)), dtype=float)], axis=1)
    clinical_right_ones_pad.columns = range(clinical_right_ones_pad.shape[1])

    cnv_right_ones_pad = pd.concat([cnv, pd.DataFrame(np.ones((len(target), 200)), dtype=float)], axis=1)
    cnv_right_ones_pad.columns = range(cnv_right_ones_pad.shape[1])

    merged_data = clinical_right_ones_pad * cnv_right_ones_pad * gene_expression

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

def load_metabrics_gaussian_addition_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer data after applying Gaussian Random Projection (Addition of features)")
    cnv = pd.read_csv(data_dir + "/METABRIC_cnv_1980.txt", delimiter=' ', header=None)
    gene_expression = pd.read_csv(data_dir + "/METABRIC_gene_exp_1980.txt", delimiter=' ', header=None)
    clinical = pd.read_csv(data_dir + "/METABRIC_clinical_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)
    
    grp = GaussianRandomProjection(n_components=400, random_state=42)
    clinical_grp  = grp.fit_transform(clinical)
    cnv_grp =  grp.fit_transform(cnv)  
    

    merged_data = clinical_grp + cnv_grp + gene_expression

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

def load_metabrics_gaussian_multiplication_data():
    src_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = src_dir+"/data/metabrics/"
    print(data_dir)
    print("loading Metabrics Breast Cancer data after applying Gaussian Random Projection (Multiplication of features)")
    cnv = pd.read_csv(data_dir + "/METABRIC_cnv_1980.txt", delimiter=' ', header=None)
    gene_expression = pd.read_csv(data_dir + "/METABRIC_gene_exp_1980.txt", delimiter=' ', header=None)
    clinical = pd.read_csv(data_dir + "/METABRIC_clinical_1980.txt", delimiter=' ', header=None)
    target = pd.read_csv(data_dir + "/METABRIC_label_5year_positive491.txt", header=None)
    
    grp = GaussianRandomProjection(n_components=400, random_state=42)
    clinical_grp  = grp.fit_transform(clinical)
    cnv_grp =  grp.fit_transform(cnv)  
    

    merged_data = clinical_grp * cnv_grp * gene_expression

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
