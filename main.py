from models.dgcnn_model import DGCNN
from data.generate_causal_dataset import generate_test_dataset
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from itertools import repeat
import networkx as nx
import matplotlib.pyplot as plt

# Next up... Get model to train? No... Then I'll have to decide on a target value. Better leave that for tomorrow
# Next up.. Get other models to work

k = 3

model = DGCNN(k=k, emb_dims=64, dropout=0.5)
data = generate_test_dataset()

for batch in data:
    output, graphs = model(batch.unsqueeze(1), True)
    idxs = torch.arange(len(graphs[0]) // k).repeat_interleave(k)
    edge_index = torch.stack([idxs, graphs[0]])
    d = Data(x=batch, edge_index=edge_index)
    nxd = to_networkx(d)
    nx.draw(nxd)
    plt.show()
