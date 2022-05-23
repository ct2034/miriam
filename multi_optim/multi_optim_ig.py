import os
import pickle
from itertools import product

import networkx as nx
import torch
import torch_geometric
from captum.attr import IntegratedGradients
from definitions import POS
from matplotlib import pyplot as plt
from planner.policylearn.edge_policy import EdgePolicyModel
from planner.policylearn.generate_data_demo import plot_graph_wo_pos_data

if __name__ == "__main__":
    # Definitions
    prefix: str = "medium_r64_e256"
    save_folder: str = "multi_optim/results"

    # Load the model
    model = EdgePolicyModel()
    model.load_state_dict(torch.load(
        f"{save_folder}/{prefix}_policy_model.pt",
        map_location=torch.device('cpu')))
    model.eval()

    # Load the dataset
    first_file = os.listdir(f"{save_folder}/{prefix}_data")[0]
    data = None
    with open(f"{save_folder}/{prefix}_data/{first_file}", "rb") as f:
        data = pickle.load(f)
    assert data is not None

    # Load Graph
    graph = nx.read_gpickle(f"{save_folder}/{prefix}_graph.gpickle")
    pos = nx.get_node_attributes(graph, POS)
    edge_index = torch.tensor(list(graph.edges.keys())).T

    # Find good and bad examples
    ex_good = None
    ex_bad = None
    for d in data:
        out = model(d.x, d.edge_index, torch.tensor([0, ]*d.num_nodes))
        y_out = out.argmax()
        y_target = d.y.argmax()
        if y_out == y_target:
            ex_good = d
        else:
            ex_bad = d
            y_out_bad = y_out
        if ex_good is not None and ex_bad is not None:
            break
    assert isinstance(ex_good, torch_geometric.data.Data)

    print(f"Good example: {ex_good.x}")
    print(f"Bad example: {ex_bad.x}")

    # Run IG
    ig = IntegratedGradients(model)
    ig_good = ig.attribute(
        ex_good.x,
        baselines=torch.zeros_like(ex_good.x),
        additional_forward_args=(
            ex_good.edge_index,
            torch.tensor([0, ]*ex_good.num_nodes)
        ),
        internal_batch_size=1)
    ig_bad = ig.attribute(
        ex_bad.x,
        baselines=torch.zeros_like(ex_bad.x),
        additional_forward_args=(
            ex_bad.edge_index,
            torch.tensor([0, ]*ex_bad.num_nodes)
        ),
        internal_batch_size=1)

    print(f"IG good: {ig_good}")
    print(f"IG bad: {ig_bad}")

    highlight_nodes_s = [
        {
            (d.x[:, 0] == 1.).nonzero(): "yellow",  # pose self
            (d.x[:, 1] == 1.).nonzero(): "cyan",  # pose others
            d.y.argmax(): "lime"  # correct next node
        } for d in [ex_good, ex_bad]
    ]
    highlight_nodes_s[1][y_out_bad] = "red"  # wrong next node
    print(f"Highlight nodes: {highlight_nodes_s}")

    f, axs = plt.subplots(2, 2, figsize=(9, 9))
    for i_smpl, d in enumerate([ex_good, ex_bad]):
        ax = axs[0, i_smpl]
        plot_graph_wo_pos_data(
            ax,
            edge_index,
            pos,
            d.x,
            highlight_nodes=highlight_nodes_s[i_smpl])
    axs[0, 0].set_title("Sample good")
    axs[0, 1].set_title("Sample bad")
    for i_smpl, ig in enumerate([ig_good, ig_bad]):
        ax = axs[1, i_smpl]
        plot_graph_wo_pos_data(
            ax,
            edge_index,
            pos,
            ig,
            highlight_nodes=highlight_nodes_s[i_smpl])
    axs[1, 0].set_title("IG good")
    axs[1, 1].set_title("IG bad")

    for x, y in product(range(2), repeat=2):
        axs[x, y].set_aspect('equal')
    plt.show()
