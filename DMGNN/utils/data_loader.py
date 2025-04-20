
import os
import random
import torch
from torch_geometric.datasets import Reddit, Coauthor, Amazon, Planetoid, Actor, WikipediaNetwork, WebKB
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
import numpy as np


src_dir = os.path.dirname(__file__)
import sys
sys.path.append(src_dir)

#src_dir = '/content/drive/MyDrive/archive/Grafenne/utils'



from metabrics_data_loader import (
    load_metabrics_data, load_metabrics_clinical_data, load_metabrics_cnv_data,
    load_metabrics_gene_exp_data, load_metabrics_clinical_gene_exp_data,
    load_metabrics_clinical_cnv_data, load_metabrics_gene_exp_cnv_data,
    load_metabrics_addition_left_padding_data, load_metabrics_addition_right_padding_data,
    load_metabrics_multiplication_left_padding_data, load_metabrics_multiplication_right_padding_data,
    load_metabrics_gaussian_addition_data, load_metabrics_gaussian_multiplication_data
)

from tcga_data_loader import load_tcga_data
from seed import seed

def get_node_mapper(lcc: np.ndarray) -> dict:
    return {node: i for i, node in enumerate(lcc)}

def remap_edges(edges: list, mapper: dict) -> list:
    row, col = edges
    row = [mapper[x] for x in row]
    col = [mapper[x] for x in col]
    return [row, col]

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = {start}
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.add(current_node)
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = {n for n in neighbors if n not in visited_nodes and n not in queued_nodes}
        queued_nodes.update(neighbors)
    return visited_nodes

def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes -= comp
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))

def keep_only_largest_connected_component(dataset):
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
        x=x_new,
        edge_index=torch.LongTensor(edges),
        y=y_new,
        train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
        val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    dataset.data = data

    return dataset

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_data(dataset_name, musae_dataset_path=None, train_ratio=0.4, val_ratio=0.3):
    if dataset_name == 'Cora':
        path = os.path.join(os.getcwd(), 'data', 'Cora')
        dataset = Planetoid(path, name='Cora')
        data = dataset[0]
    elif dataset_name == 'CiteSeer':
        path = os.path.join(os.getcwd(), 'data', 'CiteSeer')
        dataset = Planetoid(path, name='CiteSeer')
        data = dataset[0]
    elif dataset_name in ['metabrics', 'metabrics_clinical', 'metabrics_gene_exp', 'metabrics_cnv',
                          'metabrics_clinical_gene_exp', 'metabrics_clinical_cnv', 'metabrics_gene_exp_cnv',
                          'metabrics_addition_left_padding', 'metabrics_addition_right_padding',
                          'metabrics_multiplication_left_padding', 'metabrics_multiplication_right_padding',
                          'metabrics_gaussian_addition', 'metabrics_gaussian_multiplication']:
        data_loader = globals()[f'load_{dataset_name}_data']
        data = data_loader()
    elif dataset_name == 'tcga':
        data = load_tcga_data()
    else:
        print(f'Dataset {dataset_name} is not defined')
        return None

    list_all_ids = list(range(len(data.train_mask))) if hasattr(data, 'train_mask') else list(range(len(data.x)))
    random.shuffle(list_all_ids)
    val_ratio = train_ratio + val_ratio
    split_train = list_all_ids[:int(train_ratio * len(list_all_ids))]
    split_val = list_all_ids[int(train_ratio * len(list_all_ids)):int(val_ratio * len(list_all_ids))]
    split_test = list_all_ids[int(val_ratio * len(list_all_ids)):]

    data.train_mask = torch.zeros(len(list_all_ids), dtype=torch.bool)
    data.val_mask = torch.zeros(len(list_all_ids), dtype=torch.bool)
    data.test_mask = torch.zeros(len(list_all_ids), dtype=torch.bool)

    data.train_mask[split_train] = True
    data.val_mask[split_val] = True
    data.test_mask[split_test] = True

    return data

seed_everything(seed)

