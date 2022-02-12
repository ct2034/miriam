import logging
import os
import pickle
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch_geometric
from torch.nn.modules.module import T
from torch_geometric.data import Dataset
from torch_geometric.data.batch import Batch

MODEL_INPUT = Tuple[
    torch.Tensor, torch.Tensor]


class EdgePolicyDataset(Dataset):
    def __init__(self, path, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(transform, pre_transform, pre_filter)
        self.path = path
        self.loaded_fname = None
        self.loaded_data = None
        self.fnames = os.listdir(path)
        self.lookup: List[Tuple[str, int]] = []  # (fname, i)
        for fname in self.fnames:
            len_this_file = len(
                self.load_file(fname))
            lookup_section = [
                (fname, i_d) for i_d in range(len_this_file)]
            self.lookup.extend(lookup_section)

    def len(self):
        return len(self.lookup)

    def load_file(self, fname):
        if self.loaded_fname == fname:
            data = self.loaded_data
        else:
            fpath = self.path + "/" + fname
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            self.loaded_fname = fname
            self.loaded_data = data
        return data

    def get(self, idx):
        fname, i_d = self.lookup[idx]
        return self.load_file(fname)[i_d]


class EdgePolicyModel(nn.Module):
    def __init__(self, num_node_features=4, conv_channels=4, gpu=torch.device("cpu")):
        super().__init__()
        self.num_node_features = num_node_features
        self.conv_channels = conv_channels
        self.conv1 = torch_geometric.nn.ChebConv(
            num_node_features, conv_channels, K=2)
        self.conv2 = torch_geometric.nn.ChebConv(
            conv_channels, conv_channels, K=2)
        self.readout = torch.nn.Linear(conv_channels, 1)
        self.gpu = gpu  # type: torch.device

    def forward(self, x, edge_index, batch):
        # Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        # read values at potential targets as score
        x = self.readout(x)[:, 0]
        y_out_batched = torch.zeros_like(x)
        for i_b in torch.unique(batch):
            y_out_batched[batch == i_b] = torch.softmax(
                x[batch == i_b], dim=0)
        return y_out_batched

    def predict(self, x, edge_index):
        self.eval()
        n_nodes = x.shape[0]
        node = torch.nonzero(x[:, 0] == 1.).item()
        relevant_edge_index = edge_index[:,
                                         torch.bitwise_or(edge_index[0] == node,
                                                          edge_index[1] == node)]
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
        targets = torch.unique(targets)

        # read values at potential targets as score
        score = self.forward(x, edge_index, torch.tensor([0]*n_nodes))
        score_potential_targets = score[targets]
        return targets[torch.argmax(score_potential_targets).item()]

    def accuracy(self, databatch: Batch):
        datas = databatch.to_data_list()
        results = torch.zeros(len(datas))
        for i, data in enumerate(datas):
            pred = self.predict(data.x, data.edge_index)
            results[i] = int(pred == torch.argmax(data.y).item())
        return torch.mean(results)

    def learn(self, databatch: Batch, optimizer):
        self.train()
        databatch.to(self.gpu)
        y_out_batched = self.forward(
            databatch.x, databatch.edge_index, databatch.batch)

        loss = torch.nn.functional.binary_cross_entropy(
            y_out_batched, databatch.y)
        try:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError:
            logging.warning(f"Could not train with: " +
                            f"y_goals {y_goals}, scores {scores}, " +
                            f"datas {databatch}, targets {targets}")
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
