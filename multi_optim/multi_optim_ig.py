import os
import pickle

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
    prefix: str = "tiny_r256_e64"
    save_folder: str = "multi_optim/results"

    # Load the model
    model = EdgePolicyModel()
    model.load_state_dict(torch.load(
        f"{save_folder}/{prefix}_policy_model.pt"))
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
        if ex_good is not None and ex_bad is not None:
            break
    assert isinstance(ex_good, torch_geometric.data.Data)

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

    f, axs = plt.subplots(2, 2)
    for i_smpl, d in enumerate([ex_good, ex_bad]):
        ax = axs[0, i_smpl]
        plot_graph_wo_pos_data(
            ax,
            d.edge_index,
            pos,
            d.x)
    axs[0, 0].set_title("Sample good")
    axs[0, 1].set_title("Sample bad")
    for i_smpl, ig in enumerate([ig_good, ig_bad]):
        ax = axs[1, i_smpl]
        plot_graph_wo_pos_data(
            ax,
            ex_good.edge_index,
            pos,
            ig)
    axs[1, 0].set_title("IG good")
    axs[1, 1].set_title("IG bad")
    plt.show()
