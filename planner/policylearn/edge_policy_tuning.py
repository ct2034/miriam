import json
import os
from random import Random
from typing import Dict, List, Union

import torch
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from planner.policylearn.edge_policy import EdgePolicyDataset, EdgePolicyModel
from tools import ProgressBar
from torch_geometric.loader import DataLoader


def learning(
    name: str,
    lr: float,
    batch_size: int,
    conv_channels: int,
    conv_layers: int,
    readout_layers: int,
    cheb_filter_size: int
):
    rng = Random(0)
    torch.manual_seed(0)

    # run to learn from
    run_prefix_data: str = "tiny"
    n_test = 100
    n_epochs = 20  # on hal = ~10h

    # load previously trained model
    model = EdgePolicyModel(
        gpu=torch.device("cpu"),
        num_conv_channels=conv_channels,
        num_conv_layers=conv_layers,
        num_readout_layers=readout_layers,
        cheb_filter_size=cheb_filter_size)
    # model.load_state_dict(torch.load(
    #     f"multi_optim/results/{run_prefix_data}_policy_model.pt"))

    # load dataset from previous multi_optim_run
    dataset = EdgePolicyDataset(f"multi_optim/results/{run_prefix_data}_data")
    test_set_i_s = range(len(dataset) - n_test, len(dataset))
    test_set = dataset[test_set_i_s]
    test_set = [(d, {n: n for n in range(d.num_nodes)}) for d in test_set]
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, exclude_keys=test_set_i_s)

    # stat collection
    stats = {
        "accuracy": [],
        "loss": [],
    }
    n_stat_points_desired = 200
    n_stat_points_per_epoch = n_stat_points_desired // n_epochs
    stats_every_x_batch = len(loader) // n_stat_points_per_epoch

    # training
    pb = ProgressBar(name, len(loader) * n_epochs, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for i_e in range(n_epochs):
        for i_b, batch in enumerate(loader):
            loss = model.learn(batch, optimizer)
            pb.progress()
            if i_b % stats_every_x_batch == 0:
                accuracy = model.accuracy(test_set)
                print(f"{name} accuracy: {accuracy}")
                print(f"{name} loss: {loss}")
                stats["accuracy"].append(accuracy)
                stats["loss"].append(loss)
    pb.end()

    # save model
    torch.save(model.state_dict(),
               f"planner/policylearn/results/edge_policy_{name}.pt")

    # save stats
    with open(f"planner/policylearn/results/edge_policy_{name}.json", "w") as f:
        json.dump(stats, f)
    plt.figure(figsize=(20, 20), dpi=500)
    plt.plot(stats["accuracy"], label="accuracy")
    plt.plot(stats["loss"], label="loss")
    plt.legend()
    plt.savefig(f"planner/policylearn/results/edge_policy_{name}.png")


def learning_proxy(kwargs):
    learning(**kwargs)


def tuning():
    lr_s = [3E-3, 1E-2]
    batch_size_s = [64, 32, 128]
    conv_channels_s = [128, 64]
    conv_layers_s = [3, 4]
    readout_layers_s = [2, 1]
    cheb_filter_size_s = [2, 1, 3, 4]
    parameter_experiments = {
        "lr": lr_s,
        "batch_size": batch_size_s,
        "conv_channels": conv_channels_s,
        "conv_layers": conv_layers_s,
        "readout_layers": readout_layers_s,
        "cheb_filter_size": cheb_filter_size_s
    }  # type: Dict[str, Union[str, List[Union[float, int]]]]

    # prepare multithreading
    params_to_run = []

    # default run
    kwargs = {k: v[0] for k, v in parameter_experiments.items()}
    kwargs["name"] = f"default"
    params_to_run.append(kwargs.copy())

    # experimental runs
    for name, values in parameter_experiments.items():
        for value in values[1:]:
            kwargs = {k: v[0] for k, v in parameter_experiments.items()}
            kwargs[name] = value
            kwargs["name"] = f"{name}_{value}"
            params_to_run.append(kwargs.copy())

    p = mp.Pool(len(params_to_run))
    p.map(learning_proxy, params_to_run)


def rolling_average(data: List[float], n: int = 10) -> List[float]:
    return [sum(data[i:i + n]) / n for i in range(len(data) - n)]


def plot_results():
    fnames = os.listdir("planner/policylearn/results")
    fnames_json = [f for f in fnames if f.endswith(".json")]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(20, 20),
        dpi=500,
        sharex=True)
    for fname in fnames_json:
        with open(f"planner/policylearn/results/{fname}", "r") as f:
            stats = json.load(f)
        label = fname.replace(".json", "").replace("edge_policy_", "")
        ax1.plot(rolling_average(stats["accuracy"]), label=label)
        ax2.plot(rolling_average(stats["loss"]), label=label)
    ax1.legend()
    ax1.set_ylabel("accuracy")
    ax2.legend()
    ax2.set_ylabel("loss")
    plt.savefig("planner/policylearn/results/edge_policy_results.png")


if __name__ == "__main__":
    tuning()
    plot_results()
