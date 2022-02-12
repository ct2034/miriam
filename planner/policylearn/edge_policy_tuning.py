import logging
import os
import pickle
from random import Random
from typing import Any, List, Tuple

import torch
from definitions import INVALID, POS
from matplotlib import pyplot as plt
from planner.policylearn.edge_policy import EdgePolicyDataset, EdgePolicyModel
from planner.policylearn.edge_policy_graph_utils import agents_to_data
from planner.policylearn.generate_data_demo import plot_graph_wo_pos_data
from scenarios.generators import arena_with_crossing
from scenarios.graph_converter import gridmap_to_nx, starts_or_goals_to_nodes
from scenarios.visualization import plot_with_paths
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import PolicyType
from sim.decentralized.runner import run_a_scenario
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


def learning():
    rng = Random(0)

    # run to learn from
    run_prefix: str = "tiny"
    batch_size = 32
    lr = 1E-2
    n_test = 100

    # load previously trained model
    model = EdgePolicyModel(gpu=torch.device("cpu"))
    model.load_state_dict(torch.load(
        f"multi_optim/results/{run_prefix}_policy_model.pt"))

    # load dataset from previous multi_optim_run
    dataset = EdgePolicyDataset(f"multi_optim/results/{run_prefix}_data")
    test_set_i_s = rng.sample(range(len(dataset)), n_test)
    test_set = dataset[test_set_i_s]
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, exclude_keys=test_set_i_s)

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for i_b, batch in enumerate(loader):
        loss = model.learn(batch, optimizer)
        if i_b % 10 == 0:
            accuracy = model.accuracy(test_set)
            print(f"accuracy: {accuracy}")
            print(f"loss: {loss}")


if __name__ == "__main__":
    learning()
