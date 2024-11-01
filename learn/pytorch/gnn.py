# Install required packages.
# !pip install -q torch-scatter -f \
# https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# !pip install -q torch-sparse -f \
# https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# !pip install -q git+https://github.com/rusty1s/pytorch_geometric.git

import time

import matplotlib.pyplot as plt
import networkx as nx
import torch
from IPython.display import Javascript  # Restrict height of output cell.
from torch.nn import Linear
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx

dataset = KarateClub()
print(f"Dataset: {dataset}:")
print("======================")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

data = dataset[0]
print(data)
print("==============================================================")

# Gather some statistics about the graph.
print(f"Number of nodes: {data.num_nodes}")
print(f"data.x.shape: {data.x.shape}")
print(f"data.x: {data.x}")  # diag
print(f"data.y.shape: {data.y.shape}")
print(f"data.y: {data.y}")  # classes
print(f"Number of edges: {data.num_edges}")
print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
print(f"Number of training nodes: {data.train_mask.sum()}")
label_rate = int(data.train_mask.sum()) / data.num_nodes
print(f"Training node label rate: {label_rate:.2f}")
print(f"Contains isolated nodes: {data.contains_isolated_nodes()}")
print(f"Contains self-loops: {data.contains_self_loops()}")
print(f"Is undirected: {data.is_undirected()}")

edge_index = data.edge_index
print(edge_index.t()[:20])

# looking at the graph with nx ------------------------------------------------


def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f"Epoch: {epoch}, Loss: {loss.item():.4f}", fontsize=16)
    else:
        nx.draw_networkx(
            G,
            pos=nx.spring_layout(G, seed=42),
            with_labels=False,
            node_color=color,
            cmap="Set2",
        )


G = to_networkx(data, to_undirected=True)
visualize(G, color=data.y)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h


model = GCN()
print(model)

_, h = model(data.x, data.edge_index)
print(f"Embedding shape: {list(h.shape)}")

visualize(h, color=data.y)

model = GCN()
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.


def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    # Compute the loss solely based on the training nodes.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h


for epoch in range(401):
    loss, h = train(data)
    if epoch % 100 == 0:
        visualize(h, color=data.y, epoch=epoch, loss=loss)

plt.show()
