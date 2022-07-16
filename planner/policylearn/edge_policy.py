import logging
import os
import pickle
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from planner.policylearn.edge_policy_graph_utils import BFS_TYPE
from torch.nn.modules.module import T
from torch_geometric.data import Data, Dataset
from torch_geometric.data.batch import Batch

MODEL_INPUT = Tuple[
    torch.Tensor, torch.Tensor]
EVAL_LIST = List[Tuple[Data, BFS_TYPE]]


class EdgePolicyDataset(Dataset):
    def __init__(self, path, transform=None, pre_transform=None, pre_filter=None, gpu=None):
        super().__init__(transform, pre_transform, pre_filter)
        self.path = path
        self.lookup: List[Tuple[str, int]] = []  # (fname, i)
        self.gpu = gpu
        self.data_store = {}
        self.added_fnames: List[str] = []
        if os.path.isdir(path):
            fname_to_init = [self.path + "/" +
                             fname for fname in os.listdir(path)]
            for fname in fname_to_init:
                self.add_file(fname)

    def len(self):
        return len(self.lookup)

    def _load_file(self, fname):
        with open(fname, "rb") as f:
            data = pickle.load(f)
        return data

    def add_file(self, fname):
        if fname in self.added_fnames:
            # don't add the same file twice
            return
        self.added_fnames.append(fname)
        data = self._load_file(fname)
        len_this_file = len(data)
        self.data_store[fname] = data
        lookup_section = [
            (fname, i_d) for i_d in range(len_this_file)]
        self.lookup.extend(lookup_section)

    def get(self, idx):
        fname, i_d = self.lookup[idx]
        return self.data_store[fname][i_d].to(self.gpu)


class EdgePolicyModel(nn.Module):
    def __init__(
            self,
            num_node_features=4,  # type: int
            num_conv_channels=128,  # type: int
            num_conv_layers=4,  # type: int
            num_readout_layers=2,  # type: int
            cheb_filter_size=6,  # type: int
            dropout_p=.2,  # type: float
            gpu=torch.device("cpu")  # type: torch.device
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_conv_channels = num_conv_channels
        self.gpu = gpu  # type: torch.device

        # creating needed layers
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            channels_in = self.num_node_features if i == 0 else num_conv_channels
            self.conv_layers.append(
                torch_geometric.nn.ChebConv(channels_in, num_conv_channels,
                                            K=cheb_filter_size))
        self.dropout = nn.Dropout(dropout_p)
        self.readout_layers = torch.nn.ModuleList()
        for i in range(num_readout_layers):
            channels_out = 1 if i == num_readout_layers-1 else num_conv_channels
            self.readout_layers.append(
                torch.nn.Linear(num_conv_channels, channels_out)
            )

    def __hash__(self) -> int:
        rng = torch.Generator()
        rng.manual_seed(0)
        n_nodes = 10
        d = Data(
            x=torch.randn(n_nodes, self.num_node_features, generator=rng),
            edge_index=torch.tensor([range(n_nodes), [0]*n_nodes]),
            batch=torch.tensor([0]*n_nodes)
        )
        self.eval()
        return hash(str(self.forward(d.x, d.edge_index, d.batch)))

    def forward(self, x, edge_index, batch):
        x = x.to(self.gpu)
        edge_index = edge_index.to(self.gpu)

        # Convolution layer(s)
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index)
            x = self.dropout(x)
            x = x.relu()

        # read values at potential targets as score
        for readout_layer in self.readout_layers:
            x = readout_layer(x)
            x = x.relu()

        # flatten and softmax data respecting batches
        x = x[:, 0]
        y_out_batched = torch.zeros_like(x)
        for i_b in torch.unique(batch):
            y_out_batched[batch == i_b] = torch.softmax(
                x[batch == i_b], dim=0)
        return y_out_batched

    def predict_probablilistic(self, x, edge_index, big_from_small,
                               rng=random.Random(0)):
        targets, score = self.predict_scores_and_targets(x, edge_index)
        score_potential_targets = score[targets]
        # if self.training:
        node_small = rng.choices(
            targets,
            weights=score_potential_targets.tolist(),
            k=1)[0].item()
        # else:  # eval
        #     node_small = targets[torch.argmax(
        #         score_potential_targets).item()].item()
        return big_from_small[node_small]

    def predict_scores_and_targets(self, x, edge_index):
        self.eval()
        n_nodes = x.shape[0]
        try:
            node = torch.nonzero(x[:, 0] == 1.).item()
        except ValueError:
            raise RuntimeError("Unable to find own position in x")
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
        return targets, score

    def accuracy(self, eval_list: EVAL_LIST) -> float:
        self.eval()
        results = torch.zeros(len(eval_list))
        for i, (data, bfs) in enumerate(eval_list):
            pred = self.predict_probablilistic(data.x, data.edge_index, bfs)
            optimal_small = torch.argmax(data.y).item()
            if isinstance(optimal_small, int):
                results[i] = int(pred == bfs[optimal_small])
            else:
                results[i] = 0
        return torch.mean(results).item()

    def learn(self, databatch: Data, optimizer):
        databatch.to(self.gpu)  # type: ignore
        self.train()
        y_out_batched = self.forward(
            databatch.x, databatch.edge_index, databatch.batch)

        loss = F.binary_cross_entropy(
            y_out_batched, databatch.y)
        try:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError:
            logging.warning(f"Could not train with: " +
                            f"y_goals {databatch.y} " +
                            f"y_out_batched {y_out_batched} " +
                            f"loss {loss}")
            return None
        return float(loss)
