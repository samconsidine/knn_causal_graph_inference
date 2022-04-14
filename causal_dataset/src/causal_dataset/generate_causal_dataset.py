import torch
from torch.utils.data import Dataset
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

import networkx as nx
from causalgraphicalmodels import StructuralCausalModel
import numpy as np


def adj_mat_from_scm(scm: StructuralCausalModel) -> Tensor:
    dag = scm.cgm.dag
    adj_mat = nx.adjacency_matrix(dag).todense()
    edge_index = dense_to_sparse(torch.tensor(adj_mat))
    return edge_index[0]


def scm_to_dataloader(scm: StructuralCausalModel, n_samp: int, batch_size: int) -> Data:
    df = scm.sample(n_samp)
    n_nodes = len(df.columns)
    data_vec = torch.tensor(df.values).flatten().unsqueeze(0).T

    # Need to get the DAG
    dag = adj_mat_from_scm(scm)
    data = []

    assert len(data_vec) % n_nodes == 0
    for i in range(0, len(data_vec) // n_nodes, n_nodes):
        data.append(Data(x=data_vec[i:(i+n_nodes)], y=torch.tensor([1.0]).float(), edge_index=dag))

    return DataLoader(data, batch_size=1)

def scm_to_dataset(scm: StructuralCausalModel, n_samp: int, batch_size: int):
    df = scm.sample(n_samp)
    n_nodes = len(df.columns)
    data_vec = torch.tensor(df.values).flatten().unsqueeze(0).T.float()

    # Need to get the DAG
    dag = adj_mat_from_scm(scm)
    data = []
    labels = []

    assert len(data_vec) % n_nodes == 0
    for i in range(0, len(data_vec) // n_nodes, n_nodes):
        data.append(data_vec[i:(i+n_nodes)])
        labels.append(torch.tensor([1]))

    return data, labels

def generate_test_dataset_dgcnn():
    scm = StructuralCausalModel({
        "x1": lambda         n_samples: np.random.binomial(n=1,p=0.7,size=n_samples),
        "x2": lambda x1,     n_samples: np.random.normal(loc=x1, scale=0.1),
        "x3": lambda x2,     n_samples: x2 ** 2,
        "x4": lambda x1, x2, n_samples: x1 + x2
    })

    return scm_to_dataset_dgcnn(scm, 100, 16)


def scm_to_dataset_dgcnn(scm: StructuralCausalModel, n_samp: int, batch_size: int) -> Data:
    df = scm.sample(n_samp)
    return DataLoader(torch.tensor(df.values).float(), torch.tensor([1.0]).float(), batch_size=batch_size)



def generate_test_dataset_dgn():
    """TODO: Save these to disk"""
    scm = StructuralCausalModel({
        "x1": lambda         n_samples: np.random.binomial(n=1,p=0.7,size=n_samples),
        "x2": lambda x1,     n_samples: np.random.normal(loc=x1, scale=0.1),
        "x3": lambda x2,     n_samples: x2 ** 2,
        "x4": lambda x1, x2, n_samples: x1 + x2
    })

    return CausalDatasetDGM(scm)


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 


class CausalDatasetDGM(torch.utils.data.Dataset):
    def __init__(self, scm, split='train', samples_per_epoch=100, name='causal', device='cpu'):
        dataset = scm_to_dataset(scm, 100, 16)
        self.X = dataset[0].x.float().to(device)  # What's with the [0] index here?
        self.y = one_hot_embedding(dataset[0].y,dataset.num_classes).float().to(device)
        self.edge_index = dataset[0].edge_index.to(device)
        self.n_features = dataset[0].num_node_features
        self.num_classes = dataset.num_classes
        
        if split=='train':
            self.mask = dataset[0].train_mask.to(device)
        if split=='val':
            self.mask = dataset[0].val_mask.to(device)
        if split=='test':
            self.mask = dataset[0].test_mask.to(device)
         
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask,self.edge_index


def sample_test_scm(n_samp):
    scm = StructuralCausalModel({
        "x1": lambda     n_samples: np.random.binomial(n=1,p=0.7,size=n_samples),
        "x2": lambda x1, n_samples: np.random.normal(loc=x1, scale=0.1),
        "x3": lambda x2, n_samples: x2 ** 2,
        "x4": lambda x1, x2, n_samples: x1 + x2
    })
    return scm_to_dataset(scm, n_samp, 1)

class CausalDatasetDGCNN(Dataset):
    """DGCNN Needs __getitem__ to return the points and labels for the classification only"""
    def __init__(self, n_samp):
        self.data, self.labels = sample_test_scm(n_samp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    scm = StructuralCausalModel({
        "x1": lambda     n_samples: np.random.binomial(n=1,p=0.7,size=n_samples),
        "x2": lambda x1, n_samples: np.random.normal(loc=x1, scale=0.1),
        "x3": lambda x2, n_samples: x2 ** 2,
        "x4": lambda x1, x2, n_samples: x1 + x2
    })

    for item in scm_to_dataset(scm, 100, 3):
        print(item)
