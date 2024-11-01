import os
import pickle
from itertools import product

import networkx as nx
import torch
import torch_geometric
from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt

from definitions import POS
from planner.policylearn.edge_policy import EdgePolicyModel
from planner.policylearn.generate_data_demo import plot_graph_wo_pos_data


def print_matrix_with_row_numbers(X, prec=2):
    for i, row in enumerate(X):
        print(f"{i}:", end="\t")
        for j, col in enumerate(row):
            print(f"{col:.2E}", end="\t")
        print()


if __name__ == "__main__":
    # Definitions
    prefix: str = "default_seed_2"
    save_folder: str = "multi_optim/results/tuning"

    # Load the model
    model = EdgePolicyModel()
    model.load_state_dict(
        torch.load(
            f"{save_folder}/{prefix}_policy_model.pt", map_location=torch.device("cpu")
        )
    )
    model.eval()

    ex_good = None  # example of data which is correctly classified
    ex_bad = None  # example of data which is incorrectly classified
    i_f = 0  # index of the file
    while ex_good is None or ex_bad is None:
        # Load the dataset
        first_file = os.listdir(f"{save_folder}/{prefix}_data")[i_f]
        data = None
        with open(f"{save_folder}/{prefix}_data/{first_file}", "rb") as f:
            data = pickle.load(f)
        assert data is not None

        # Find good and bad examples
        for d in data:
            out = model(
                d.x,
                d.edge_index,
                torch.tensor(
                    [
                        0,
                    ]
                    * d.num_nodes
                ),
            )
            y_out = out.argmax()
            y_target = d.y.argmax()
            if y_out == y_target:
                ex_good = d
            else:
                ex_bad = d
                y_out_bad = y_out
            if ex_good is not None and ex_bad is not None:
                break

        i_f += 1  # Next file

    assert isinstance(ex_good, torch_geometric.data.Data)

    print(f"Good example")
    print_matrix_with_row_numbers(ex_good.x)
    if ex_good.pos is None:  # Read graph from file
        graph = nx.read_gpickle(f"{save_folder}/{prefix}_graph.gpickle")
        ex_good.pos = nx.get_node_attributes(graph, POS)
        ex_bad.pos = nx.get_node_attributes(graph, POS)
        ex_good.edge_index = torch.tensor(list(graph.edges.keys())).T
        ex_bad.edge_index = torch.tensor(list(graph.edges.keys())).T

    print(f"Bad example:")
    print_matrix_with_row_numbers(ex_bad.x)
    if ex_bad.pos is not None:
        pos_bad = ex_bad.pos
        edge_index_bad = ex_bad.edge_index
    else:  # Read graph from file
        graph = nx.read_gpickle(f"{save_folder}/{prefix}_graph.gpickle")
        pos_bad = nx.get_node_attributes(graph, POS)
        edge_index_bad = torch.tensor(list(graph.edges.keys())).T

    # Run IG
    ig = IntegratedGradients(model)
    ig_good = ig.attribute(
        ex_good.x,
        baselines=torch.ones_like(ex_good.x) * -1,
        additional_forward_args=(
            ex_good.edge_index,
            torch.tensor(
                [
                    0,
                ]
                * ex_good.num_nodes
            ),
        ),
        internal_batch_size=1,
    )
    ig_bad = ig.attribute(
        ex_bad.x,
        baselines=torch.ones_like(ex_bad.x) * -1,
        additional_forward_args=(
            ex_bad.edge_index,
            torch.tensor(
                [
                    0,
                ]
                * ex_bad.num_nodes
            ),
        ),
        internal_batch_size=1,
    )

    print(f"IG good:")
    print_matrix_with_row_numbers(ig_good)
    print(f"IG bad:")
    print_matrix_with_row_numbers(ig_bad)

    highlight_nodes_s = [
        {
            (d.x[:, 0] == 1.0).nonzero(): "yellow",  # pose self
            (d.x[:, 1] == 1.0).nonzero(): "cyan",  # pose others
            d.y.argmax(): "lime",  # correct next node
        }
        for d in [ex_good, ex_bad]
    ]
    highlight_nodes_s[1][y_out_bad] = "red"  # wrong next node
    print(f"Highlight nodes good:\n{highlight_nodes_s[0]}")
    print(f"Highlight nodes bad:\n{highlight_nodes_s[1]}")

    f, axs = plt.subplots(2, 2, figsize=(9, 9))
    for i_smpl, d in enumerate([(ex_good, ig_good), (ex_bad, ig_bad)]):
        ax = axs[0, i_smpl]
        data, ig = d
        plot_graph_wo_pos_data(
            ax,
            data.edge_index,
            data.pos,
            data.x,
            highlight_nodes=highlight_nodes_s[i_smpl],
        )
        ax = axs[1, i_smpl]
        plot_graph_wo_pos_data(
            ax, data.edge_index, data.pos, ig, highlight_nodes=highlight_nodes_s[i_smpl]
        )
    axs[1, 0].set_title("IG good")
    axs[1, 1].set_title("IG bad")
    axs[0, 0].set_title("Sample good")
    axs[0, 1].set_title("Sample bad")

    for x, y in product(range(2), repeat=2):
        axs[x, y].set_aspect("equal")
    plt.show()
