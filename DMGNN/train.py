import os
import sys
import torch
from torch_geometric.loader import NeighborSampler
import tqdm
import torch.nn as nn
import torch_sparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, matthews_corrcoef, recall_score
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_curve,auc, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from numpy import argmax, sqrt

import torch.nn.functional as F
import random
import copy
import argparse
src_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_dir)
from utils.data_loader import load_data
from utils.utils import seed_everything,create_otf_edges,get_feature_mask
from models.fognn import ScalableFOGNN as FOGNN

from feature_propagation import FeaturePropagation

seed_everything(0)
stratified_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def print_dataset_info(data):
    print("=== Dataset Information ===")
    print("Number of nodes:", data.num_nodes)
    print("Number of features per node:", data.num_features)
    # Check if 'num_classes' attribute is available
    if hasattr(data, 'num_classes'):
        print("Number of classes:", data.num_classes)
    else:
        # If not available, get the number of classes from the 'data.y' tensor
        num_classes = int(torch.max(data.y) + 1)
        print("Number of classes:", num_classes)
    print("Edge Index shape:", data.edge_index.shape)
    print("Training mask shape:", data.train_mask.shape)
    print("Testing mask shape:", data.test_mask.shape)
    print("Validation mask shape:", data.val_mask.shape)
    print("Training mask values:", data.train_mask)
    print("Testing mask values:", data.test_mask)
    print("Validation mask values:", data.val_mask)
    print("===========================")


   
def neighborhood_mean_filling(edge_index, X, feature_mask):
    n_nodes = X.shape[0]
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    edge_values = torch.ones(edge_index.shape[1]).to(edge_index.device)
    edge_index_mm = torch.stack([edge_index[1], edge_index[0]]).to(edge_index.device)

    D = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, feature_mask.float())
    mean_neighborhood_features = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, X_zero_filled) / D
    # If a feature is not present on any neighbor, set it to 0
    mean_neighborhood_features[mean_neighborhood_features.isnan()] = 0

    return mean_neighborhood_features

def feature_propagation(edge_index, X, feature_mask, num_iterations):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)
    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask)


parser = argparse.ArgumentParser()
parser.add_argument("--data", help="name of the dataset",
                    type=str)
parser.add_argument("--gpu",help="GPU no. to use, -1 in case of no gpu", type=int)
parser.add_argument("--missing_rate",help="% of features to be missed randomly", type=float)
parser.add_argument("--categorical",default=True,help="Make edges only when feature is present/categorical", type=bool)
parser.add_argument("--verbose",default=True,help="Print Model output during training", type=bool)
parser.add_argument("--num_epochs",default=200,help="Print Model output during training", type=int)
parser.add_argument("--num_layers",default=2,help="Num of layers (1,2)", type=int)
parser.add_argument("--bs_train_nbd",default=128,help="Num of nodes in training computation subgraph", type=int)
parser.add_argument("--bs_test_nbd",default=-1,help="Num of nodes in testing computation subgraph", type=int)
parser.add_argument("--drop_rate",default=0.2,help="Drop rate", type=float)
parser.add_argument("--result_file",type=str)
parser.add_argument("--edge_value_thresh",default=0.01,type=float)
parser.add_argument("--imputation",default='zero',type=str)
parser.add_argument("--heads",default=4,type=int)
parser.add_argument("--weight_decay",default=0,type=float)

args = parser.parse_args()
num_epochs = args.num_epochs
gpu = int(args.gpu)
dataset_name = args.data
missing_rate = args.missing_rate
categorical = args.categorical
verbose = args.verbose
num_layers = args.num_layers
bs_train_nbd = args.bs_train_nbd
bs_test_nbd = args.bs_test_nbd
drop_rate = args.drop_rate
result_file = args.result_file
edge_value_thresh = args.edge_value_thresh
imputation_method = args.imputation
heads = args.heads
weight_decay = args.weight_decay
print(args)

device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
data = load_data(dataset_name)
print_dataset_info(data)

if missing_rate >0 :
    print("missing rate,", missing_rate)
    feature_mask = get_feature_mask(missing_rate,data['x'].shape[0],data['x'].shape[1])
    data['x'][~feature_mask] = float('nan')  ### replaced values with nan
    if imputation_method=='zero':
        X_reconstructed = torch.zeros_like(data['x'])
    if imputation_method == 'nf':
        print("Neighbourhood mean")
        X_reconstructed = neighborhood_mean_filling(data.edge_index,data.x,feature_mask)
    if imputation_method == 'fp':
        print("Feature propogation")
        X_reconstructed = feature_propagation(data.edge_index,data.x,feature_mask,50)
    #X_reconstructed = feature_propagation(data.edge_index,data.x,feature_mask,50)#
    data['x'] = torch.where(feature_mask, data.x, X_reconstructed)
else:
    feature_mask = torch.ones_like(data['x']).bool()
print("Remaining edges ",feature_mask.sum(),data['x'].shape[0]*data['x'].shape[1])

num_samples = [20,15]

num_communities = int(torch.max(data.y) + 1)

# Function to compute metrics for a given fold
def compute_metrics(true_labels, predictions, predictions_prob):
    if num_communities == 2:
        # Calculate specificity
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    else:
        specificity = None

    accuracy = (tp+tn) / (tn+fp+fn+tp) if (tn+fp+fn+tp) != 0 else 0
    precision = tp / (tp+fp) if (tp+fp) != 0 else 0
    recall = tp / (tp+fn) if (tp+fn) != 0 else 0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) != 0 else 0
    auc = roc_auc_score(true_labels, predictions_prob) if len(set(true_labels)) == 2 else None
    confusion = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions)

    return accuracy, precision, recall, f1, specificity, auc, confusion, class_report
import gc

## head = 4
#print("number of heads,", heads)

Y = data.y.squeeze().to(device)
obs_features = torch.ones(data.x.shape[0],data.x.shape[1],dtype=torch.double).to(device) #torch.tensor(data.x,dtype=torch.double).to(device)
#obs_features = data.x.double().to(device)
print("obs_features.shape ", obs_features.shape)
feat_features = np.eye(data.x.shape[1])
feat_features = torch.tensor(feat_features,dtype=torch.double).to(device)
print("feat_features.shape: ", feat_features.shape)
feature_mask  = feature_mask.to(device)


fold_metrics = []

# Cross-validation loop
for fold, (train_index, val_index) in enumerate(stratified_kf.split(data.x, data.y)):
   
    # Beginning of the fold
    print(f"==================== Start of Fold {fold + 1} ====================")

    model = FOGNN(drop_rate=drop_rate, num_obs_node_features=data.num_node_features,
            num_feat_node_features=data.num_node_features,
            num_layers=2, hidden_size=256, out_channels=num_communities,heads=heads,categorical=categorical,device=device,feat_val_thresh=edge_value_thresh)
    model = model.to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay = weight_decay)
    
    rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                criterion='entropy'
           )
    # Update train_mask and val_mask for this fold
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_index] = True

    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[val_index] = True

    # Ensure the masks have the correct shape
    data.train_mask = data.train_mask.view(-1)  # Reshape to torch.Size([num_of_nodes])
    data.val_mask = data.val_mask.view(-1)
    
    if bs_train_nbd == -1:
        bs_train_nbd = len(train_index)

    if bs_test_nbd == -1:
        bs_test_nbd = len(val_index)

    print("bs_train_nbd and bs_test_nbd", bs_train_nbd,bs_test_nbd)    
    
    # Initialize NeighborSampler for training
    train_neigh_sampler = NeighborSampler(
        data.edge_index, node_idx=data.train_mask,
        sizes=num_samples, batch_size=bs_train_nbd, shuffle=True, num_workers=0) 
    
    subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=data.val_mask,
    sizes= [-1,-1], batch_size=bs_test_nbd, shuffle=False, num_workers=0) 
    best_val_acc = 0
    best_epoch = 0
    
    # Training Loop
    for epoch in range(0, num_epochs):
        train_a, train_labels, test_a, test_labels  = [], [], [], []
        train_a_combined,train_labels_combined, test_a_combined, test_labels_combined = [], [], [], []
        model.train()

        # Loop over batches using NeighborSampler
        for batch_size, n_id, adjs in train_neigh_sampler:
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out,a, b = model(obs_features=obs_features[n_id], feature_mask=feature_mask[n_id],
                              feat_features=feat_features, obs_adjs=adjs, data_x=data.x[n_id], num_layers=num_layers)
            train_a.append(a.cpu().detach().numpy())
            train_labels.append(Y[n_id[:batch_size]].cpu().numpy())                  
            loss = F.nll_loss(out, Y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()
        
        train_a_combined = np.concatenate(train_a, axis=0)
        train_labels_combined = np.concatenate(train_labels, axis=0)
        rf.fit(train_a_combined, train_labels_combined)
      
        # Clear GPU memory
        del out,a,b
        torch.cuda.empty_cache()

        #validation loop
        with torch.no_grad():
            model.eval()
            for batch_size, n_id, adjs in subgraph_loader:
                adjs = [adj.to(device) for adj in adjs]
                out,a,b = model(obs_features=obs_features[n_id],feature_mask = feature_mask[n_id],
                    feat_features=feat_features,obs_adjs = adjs,data_x = data.x[n_id],num_layers=num_layers)
                test_a.append(a.cpu().detach().numpy())
                test_labels.append(Y[n_id[:batch_size]].cpu().numpy())      

            test_a_combined = np.concatenate(test_a, axis=0)
            test_labels_combined = np.concatenate(test_labels, axis=0) 
            test_y_pred = rf.predict(test_a_combined)
            test_y_pred_proba = rf.predict_proba(test_a_combined)[:, 1]
            val_acc  = accuracy_score(test_labels_combined, test_y_pred)
            re = recall_score(test_labels_combined, test_y_pred)
            ba  = balanced_accuracy_score(test_labels_combined, test_y_pred)
            
            if val_acc > best_val_acc:
                best_val_acc =  val_acc
                best_epoch = epoch
                accuracy, precision, recall, f1, specificity, auc, confusion, class_report = compute_metrics(test_labels_combined, test_y_pred, test_y_pred_proba) 
            if verbose:
                if (epoch % 10 == 0):
                    print(f'Epoch: {epoch}, Val_acc:{val_acc:.4f}')
                    
    print(f'Best Epoch: {best_epoch}  Best Validation accuracy: {accuracy}')

    # Print metrics for this fold
    print(f"Fold: {fold + 1} Metrics:")
    print(f"Accuracy: {accuracy * 100:.4f}")
    print(f"Precision: {precision * 100:.4f}")
    print(f"Recall: {recall * 100:.4f}")
    print(f"F1 Score: {f1 * 100:.4f}")
    print(f"Specificity: {specificity * 100:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(class_report)

    # Store metrics for this fold
    fold_metrics.append({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Specificity': specificity,
        'AUC Score': auc,
        'Confusion Matrix': confusion,
        'Classification Report': class_report
    })


    # Clear GPU memory
    import gc
    gc.collect()

    # End of the fold
    print(f"==================== End of Fold {fold + 1} ====================")

numerical_metrics = {metric: [fold[metric] for fold in fold_metrics] for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'AUC Score']}


# Calculate mean and standard deviation for numerical metrics
mean_numerical_metrics = {metric: np.mean(numerical_metrics[metric]) for metric in numerical_metrics}
std_numerical_metrics = {metric: np.std(numerical_metrics[metric]) for metric in numerical_metrics}

print("Mean Metrics Across Folds:")
# Print mean and standard deviation for all metrics
for metric, mean_value in mean_numerical_metrics.items():
    std_value = std_numerical_metrics[metric]
    print(f"{metric}: {mean_value * 100:.2f} ± {std_value:.2f}")
