import torch
import torch.nn as nn
import torch_geometric


class EdgePolicy(nn.Module):
    def __init__(self, num_node_features, conv_channels):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(
            num_node_features, conv_channels)
        self.conv2 = torch_geometric.nn.GCNConv(
            conv_channels, conv_channels)
        self.readout = torch.nn.Linear(conv_channels, 1)

    def forward(self, x, edge_index, pos, node, batch):
        # Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        # Relate to edges
        # take only edges that start or end at node
        relevant_edge_index = edge_index[:,
                                         torch.bitwise_or(edge_index[0] == node,
                                                          edge_index[1] == node)]
        # make index of targets
        targets = torch.zeros(relevant_edge_index.shape[1], dtype=torch.long)
        for i, (s, d) in enumerate(relevant_edge_index.t()):
            if s == node and d == node:
                targets[i] = node
            elif s == node and d != node:
                targets[i] = d
            elif s != node and d == node:
                targets[i] = s
            else:
                raise ValueError("Edge not found")

        score = self.readout(x[targets])
        score = torch.sigmoid(score)
        return score[:, 0], targets


if __name__ == "__main__":
    n_nodes = 6
    num_node_features = 4
    conv_channels = 4
    policy = EdgePolicy(num_node_features, conv_channels)
    x = torch.randn(n_nodes, num_node_features)
    edge_index = torch.tensor([
        [0, 0, 0, 0, 1, 0, 3, 5],
        [1, 2, 3, 4, 4, 0, 1, 2]
    ])
    pos = torch.tensor([
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1]
    ])
    node = 0
    batch = torch.tensor([0, 0, 0, 0, 0])
    score, targets = policy(x, edge_index, pos, node, batch)

    # learning to always use self edge
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for i in range(30000):
        policy.train()
        node = 0
        score, targets = policy(x, edge_index, pos, node, batch)
        score_optimal = torch.zeros(score.shape)
        score_optimal[targets == node] = 1
        # bce loss
        loss = torch.nn.functional.binary_cross_entropy(
            score, score_optimal)
        if i % 3000 == 0:
            print(" ".join([f"{s:.3f}" for s in score.tolist()]))
            print(f"loss: {loss.item():.3f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
