import torch
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


def scm_to_dataset(scm: StructuralCausalModel, n_samp: int, batch_size: int) -> Data:
    df = scm.sample(n_samp)
    n_nodes = len(df.columns)
    data_vec = torch.tensor(df.values).flatten().unsqueeze(0).T

    # Need to get the DAG
    dag = adj_mat_from_scm(scm)
    data = []

    assert len(data_vec) % n_nodes == 0
    for i in range(0, len(data_vec) // n_nodes, n_nodes):
        data.append(Data(x=data_vec[i:(i+n_nodes)], edge_index=dag))

    return DataLoader(data, batch_size=batch_size)


if __name__ == "__main__":
    scm = StructuralCausalModel({
        "x1": lambda     n_samples: np.random.binomial(n=1,p=0.7,size=n_samples),
        "x2": lambda x1, n_samples: np.random.normal(loc=x1, scale=0.1),
        "x3": lambda x2, n_samples: x2 ** 2,
        "x4": lambda x1, x2, n_samples: x1 + x2
    })

    for item in scm_to_dataset(scm, 100, 3):
        print(item)
