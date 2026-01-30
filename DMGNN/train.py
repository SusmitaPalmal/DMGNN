import sys
import torch
from torch_geometric.loader import NeighborSampler
import tqdm
import torch.nn as nn
import torch_sparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, matthews_corrcoef, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

import torch.nn.functional as F
import random
import copy
import argparse
src_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(src_dir)
from utils.data_loader import load_data
from utils.utils import seed_everything, create_otf_edges, get_feature_mask
from models.dmgnn import ScalableDMGNN as DMGNN
from feature_propagation import FeaturePropagation

seed_everything(0)

# ========================= Helper Functions =========================

def print_dataset_info(data):
    print("=== Dataset Information ===")
    print("Number of nodes:", data.num_nodes)
    print("Number of features per node:", data.num_features)
    if hasattr(data, 'num_classes'):
        print("Number of classes:", data.num_classes)
    else:
        num_classes = int(torch.max(data.y) + 1)
        print("Number of classes:", num_classes)
    print("===========================")

def neighborhood_mean_filling(edge_index, X, feature_mask):
    n_nodes = X.shape[0]
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    edge_values = torch.ones(edge_index.shape[1]).to(edge_index.device)
    edge_index_mm = torch.stack([edge_index[1], edge_index[0]]).to(edge_index.device)
    D = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, feature_mask.float())
    mean_neighborhood_features = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, X_zero_filled) / D
    mean_neighborhood_features[mean_neighborhood_features.isnan()] = 0
    return mean_neighborhood_features

def feature_propagation(edge_index, X, feature_mask, num_iterations):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)
    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask)

# ========================= Argument Parsing =========================

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--gpu", type=int)
parser.add_argument("--missing_rate", type=float)
parser.add_argument("--categorical", default=True, type=bool)
parser.add_argument("--verbose", default=True, type=bool)
parser.add_argument("--num_epochs", default=200, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--bs_train_nbd", default=128, type=int)
parser.add_argument("--bs_test_nbd", default=-1, type=int)
parser.add_argument("--drop_rate", default=0.2, type=float)
parser.add_argument("--result_file", type=str)
parser.add_argument("--edge_value_thresh", default=0.01, type=float)
parser.add_argument("--imputation", default='zero', type=str)
parser.add_argument("--heads", default=4, type=int)
parser.add_argument("--weight_decay", default=0, type=float)
args = parser.parse_args()

# ========================= Data Preparation =========================

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
data = load_data(args.data)
print_dataset_info(data)

if args.missing_rate > 0:
    feature_mask = get_feature_mask(args.missing_rate, data['x'].shape[0], data['x'].shape[1])
    data['x'][~feature_mask] = float('nan')
    if args.imputation == 'zero':
        X_reconstructed = torch.zeros_like(data['x'])
    elif args.imputation == 'nf':
        X_reconstructed = neighborhood_mean_filling(data.edge_index, data.x, feature_mask)
    elif args.imputation == 'fp':
        X_reconstructed = feature_propagation(data.edge_index, data.x, feature_mask, 50)
    data['x'] = torch.where(feature_mask, data.x, X_reconstructed)
else:
    feature_mask = torch.ones_like(data['x']).bool()

feature_mask = feature_mask.to(device)

# ========================= 80-20 Train Test Split =========================

train_idx, test_idx = train_test_split(
    np.arange(data.num_nodes),
    test_size=0.2,
    stratify=data.y.cpu().numpy(),
    random_state=42
)

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_idx] = True
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask[test_idx] = True

# ========================= Model and Training Setup =========================

obs_features = torch.ones(data.x.shape[0], data.x.shape[1], dtype=torch.double).to(device)
feat_features = torch.eye(data.x.shape[1], dtype=torch.double).to(device)
Y = data.y.squeeze().to(device)

num_communities = int(torch.max(Y) + 1)

model = DMGNN(drop_rate=args.drop_rate,
              num_obs_node_features=data.num_node_features,
              num_feat_node_features=data.num_node_features,
              num_layers=args.num_layers,
              hidden_size=256,
              out_channels=num_communities,
              heads=args.heads,
              categorical=args.categorical,
              device=device,
              feat_val_thresh=args.edge_value_thresh).to(device).double()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=args.weight_decay)

train_sampler = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                 sizes=[20, 15], batch_size=args.bs_train_nbd,
                                 shuffle=True, num_workers=0)

test_sampler = NeighborSampler(data.edge_index, node_idx=data.val_mask,
                                sizes=[-1, -1], batch_size=args.bs_test_nbd,
                                shuffle=False, num_workers=0)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    criterion='entropy'
)

# ========================= Training Loop =========================

for epoch in range(args.num_epochs):
    model.train()
    train_a, train_labels = [], []

    for batch_size, n_id, adjs in train_sampler:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out, a, _ = model(obs_features=obs_features[n_id], feature_mask=feature_mask[n_id],
                          feat_features=feat_features, obs_adjs=adjs,
                          data_x=data.x[n_id], num_layers=args.num_layers)
        loss = F.nll_loss(out, Y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        train_a.append(a.cpu().detach().numpy())
        train_labels.append(Y[n_id[:batch_size]].cpu().numpy())

    train_a_combined = np.concatenate(train_a, axis=0)
    train_labels_combined = np.concatenate(train_labels, axis=0)
    rf.fit(train_a_combined, train_labels_combined)

    if args.verbose and epoch % 10 == 0:
        print(f"Epoch {epoch} complete.")

# ========================= Evaluation Function =========================

def evaluate(model, rf_classifier, data_x_modified, feature_mask_modified):
    model.eval()
    preds_a = []
    labels = []

    masked_loader = NeighborSampler(data.edge_index, node_idx=data.val_mask,
                                    sizes=[-1, -1], batch_size=args.bs_test_nbd,
                                    shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch_size, n_id, adjs in masked_loader:
            adjs = [adj.to(device) for adj in adjs]
            out, a, _ = model(obs_features=obs_features[n_id], feature_mask=feature_mask_modified[n_id],
                              feat_features=feat_features, obs_adjs=adjs,
                              data_x=data_x_modified[n_id], num_layers=args.num_layers)
            preds_a.append(a.cpu().numpy())
            labels.append(Y[n_id[:batch_size]].cpu().numpy())

    test_a_combined = np.concatenate(preds_a, axis=0)
    test_labels_combined = np.concatenate(labels, axis=0)
    y_pred = rf_classifier.predict(test_a_combined)
    return accuracy_score(test_labels_combined, y_pred)

# ========================= Baseline Accuracy =========================

baseline_acc = evaluate(model, rf, data.x, feature_mask)
print(f"Baseline Accuracy: {baseline_acc:.4f}")

# ========================= Feature Importance =========================

# feature_importance = []

# for i in range(data.x.shape[1]):
#     print(f"Omitting feature {i}...")
#     modified_data_x = data.x.clone()
#     modified_data_x[:, i] = 0
#     modified_feature_mask = feature_mask.clone()
#     modified_feature_mask[:, i] = False

#     masked_acc = evaluate(model, rf, modified_data_x, modified_feature_mask)
#     importance = baseline_acc - masked_acc
#     feature_importance.append(importance)

# print("Feature Importance (accuracy drop for each feature):")
# print(feature_importance)

# ========================= Feature Importance =========================

feature_importance = []

for i in range(data.x.shape[1]):
    print(f"Omitting feature {i}...")
    modified_data_x = data.x.clone()
    modified_data_x[:, i] = 0
    modified_feature_mask = feature_mask.clone()
    modified_feature_mask[:, i] = False

    masked_acc = evaluate(model, rf, modified_data_x, modified_feature_mask)
    importance = baseline_acc - masked_acc
    feature_importance.append((i, masked_acc, importance))

    print(f"After omitting feature {i}= Accuracy = {masked_acc:.4f}= Importance (drop) = {importance:.4f}")

print("\nSummary of Feature Importance (Feature ID, Accuracy, Importance):")
for feat_id, acc, imp in feature_importance:
    # print(f"Feature {feat_id} = Accuracy = {acc:.4f}= Importance (drop) = {imp:.4f}")
    print(f"Feature {feat_id}  =  {imp:.4f}")


