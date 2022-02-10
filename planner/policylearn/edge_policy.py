import logging
from pickletools import optimize
from random import Random
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import scenarios.evaluators
import scenarios.solvers
import torch
import torch.nn as nn
import torch_geometric
from torch.nn import Dropout2d
from torch.nn.modules.module import T
from torch_geometric.data import Data

MODEL_INPUT = Tuple[
    torch.Tensor, torch.Tensor]


class EdgePolicyModel(nn.Module):
    def __init__(self, num_node_features=4, conv_channels=4, gpu=torch.device("cpu")):
        super().__init__()
        self.num_node_features = num_node_features
        self.conv_channels = conv_channels
        self.conv1 = torch_geometric.nn.ChebConv(
            num_node_features, conv_channels, K=2)
        self.conv2 = torch_geometric.nn.ChebConv(
            conv_channels, conv_channels, K=2)
        self.dropout = Dropout2d(p=.3)
        self.readout = torch.nn.Linear(conv_channels, 1)
        self.gpu = gpu  # type: torch.device

    def forward(self, x, edge_index):
        # Agents position is where x[0] is 1
        node = torch.nonzero(x[:, 0] == 1.).item()

        # Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        # x = self.dropout(x)

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

        # read values at potential targets as score
        score = self.readout(x[targets])[:, 0]
        score = torch.softmax(score, dim=0)
        return score, targets

    def predict(self, x, edge_index, big_from_small):
        self.eval()
        score, targets = self.forward(x, edge_index)
        return big_from_small[targets[torch.argmax(score)].item()]

    def accuracy(self, datas: List[Data], big_from_smalls):
        assert len(datas) == len(big_from_smalls)
        results = torch.zeros(len(datas))
        for i, (data, bfs) in enumerate(zip(datas, big_from_smalls)):
            pred = self.predict(data.x, data.edge_index, bfs)
            results[i] = int(pred == data.y)
        return torch.mean(results)

    def learn(self, datas: List[Data], optimizer):
        self.train()
        scores = torch.tensor([], device=self.gpu)
        targets = torch.tensor([], device=self.gpu)
        y_goals = torch.tensor([], device=self.gpu)
        for d in datas:
            d.to(self.gpu)
            score, targets = self.forward(d.x, d.edge_index)
            y_goal = torch.zeros(
                score.shape[0], dtype=torch.float, device=self.gpu)
            y_goal[(targets == d.y).nonzero()] = 1
            scores = torch.cat((scores, score))
            y_goals = torch.cat((y_goals, y_goal))
        loss = torch.nn.functional.binary_cross_entropy(
            scores, y_goals)
        try:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError:
            logging.warning(f"Could not train with: " +
                            f"y_goals {y_goals}, scores {scores}, " +
                            f"datas {datas}, targets {targets}")
            return None
        return float(loss)

    def train(self: T, mode: bool = True) -> T:
        if mode:  # train
            self.to(self.gpu)  # type: ignore
            for p in self.parameters():
                p.to(self.gpu)  # type: ignore
        else:  # eval
            self.to("cpu")
            for p in self.parameters():
                p.to("cpu")
        return super().train(mode)  # type: ignore
