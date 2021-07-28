#!/usr/bin/env python3
# Node Classification on large Knowledge Graphs
# Based on [this exercise](https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing#scrollTo=paMH3_7ejSg4).

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from IPython.display import Image
from moviepy.editor import ImageSequenceClip
from sklearn.manifold import TSNE
from torch.nn import Linear
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv  # GATConv
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/ ', name='Cora',
                    transform=NormalizeFeatures())

# PyTorch Geometric provides different functions to investigate the dataset (e.g. node degrees, self-loops ect.) - You can find more of them in the documentation or in [this notebook](https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing#scrollTo=eqWR0j_kIx67).

# Get some basic info about the dataset
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(50*'=')

# There is only one graph in the dataset, use it as new data object
data = dataset[0]

# Gather some statistics about the graph.
print(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(
    f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Is undirected: {data.is_undirected()}')

# Observations:
# - We only have a relatively small set of training nodes (20 nodes per class)
# - There are binary test, train and validation masks of the size #nodes (they tell use which node can be used for which task)

print(f'data.x.shape: {data.x.shape} [No. Nodes x Features]')

# Print some of the normalized word counts of the first datapoint
data.x[0][:50]

# Why do we even use the graph structure - aren't the features enough?

# - Apparently, simple MLP models perform a lot worse than GNNs on this type of task, as the citation information is crucial for a correct classification

# How do the labels look like?
# - They are encoded as numeric value between 0-6

# data.y

# Example for the binary masks

print(len(data.test_mask) == data.num_nodes)
data.test_mask

# Example for the edge connections

data.edge_index.t()

# Graph Neural Network for Node Classification


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer
        x = F.softmax(self.out(x), dim=1)
        return x


model = GCN(hidden_channels=16)
print("model:")
print(model)

# Observations:
# - Dropout is only applied in the training step, but not for predictions
# - We have 2 Message Passing Layers and one Linear output layer
# - We use the softmax function for the classification problem
# - The output of the model are 7 probabilities, one for each class

# Training and Evaluation

# Initialize model
model = GCN(hidden_channels=16)

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=decay)
# Define loss function (CrossEntropyLoss for Classification Problems with
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    # Use all data as input, because all nodes have node features
    out = model(data.x, data.edge_index)
    # Only use nodes with labels available for loss calculation --> mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


losses = []
for epoch in range(0, 1001):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Visualize the training loss
losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
loss_indices = [i for i, l in enumerate(losses_float)]
sns.lineplot(loss_indices, losses_float)
plt.title("loss")
plt.show()

# Calculate test metric
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')


sample = 9
plt.figure()
plt.title(f'prediction for sample = {sample}')
sns.set_theme(style="whitegrid")
print(
    f'model(data.x, data.edge_index).shape: {model(data.x, data.edge_index).shape}')
pred = model(data.x, data.edge_index)
sns.barplot(x=np.array(range(7)), y=pred[sample].detach().cpu().numpy())
plt.show()

# Visualizing the embeddings


def plt2arr(fig):
    rgb_str = fig.canvas.tostring_rgb()
    (w, h) = fig.canvas.get_width_height()
    rgba_arr = np.fromstring(rgb_str, dtype=np.uint8,
                             sep='').reshape((w, h, -1))
    return rgba_arr


def visualize(h, color, epoch):
    fig = plt.figure(figsize=(5, 5), frameon=False)
    fig.suptitle(f'Epoch = {epoch}')
    # Fit TSNE with 2 components
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

    # Create scatterplot from embeddings
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0],
                z[:, 1],
                s=70,
                c=color.detach().cpu().numpy(),
                cmap="Set2")
    fig.canvas.draw()

    # Convert to numpy
    return plt2arr(fig)


# Reset the previously trained model weights
for layer in model.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

# Ignore deprecation warnings here

warnings.filterwarnings('ignore')

# Train the model and save visualizations
images = []
for epoch in range(0, 2000):
    loss = train()
    if epoch % 50 == 0:
        out = model(data.x, data.edge_index)
        images.append(visualize(out, color=data.y, epoch=epoch))
print("TSNE Visualization finished.")

# Building a GIF from this

# !pip install moviepy


fps = 1
filename = "embeddings.gif"
clip = ImageSequenceClip(images, fps=fps)
clip.write_gif(filename, fps=fps)


with open('embeddings.gif', 'rb') as f:
    display(Image(data=f.read(), format='png'))

# # Improving the model
# - Cross-Validation
# - Hyperparameter Optimization
# - Different layer types GCN, GAT... ([here](hhttps: // pytorch-geometric.readthedocs.io/en/latest/modules/nn.html))
# - Including edge features([here](https: // wiki.tum.de/display/dlma/Exploiting+Edge+Features+in+Graph+Neural+Networks))
# - -> The best performance is currently[at around 0.9](https: // paperswithcode.com/sota/node-classification-on-cora)