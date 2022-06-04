import json
import os
from random import Random
from typing import Dict, List, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from cuda_util import pick_gpu_low_memory
from matplotlib import pyplot as plt
from planner.policylearn.edge_policy import EdgePolicyDataset, EdgePolicyModel
from scenarios.visualization import get_colors
from tools import ProgressBar
from torch_geometric.loader import DataLoader


def learning(
    name: str,
    lr: float,
    batch_size: int,
    conv_channels: int,
    conv_layers: int,
    readout_layers: int,
    cheb_filter_size: int,
    dropout_p: float,
    seed: int,
    base_folder: str,
):
    rng = Random(seed)
    torch.manual_seed(seed)

    # run to learn from
    run_prefix_data: str = "large_r64_e256"
    n_test = 100
    n_epochs = 100

    if torch.cuda.is_available():
        gpu = torch.device(pick_gpu_low_memory())
        torch.cuda.set_device(gpu)
    else:
        gpu = torch.device("cpu")
    print(f"Using GPU {gpu}")

    # load previously trained model
    model = EdgePolicyModel(
        gpu=gpu,
        num_conv_channels=conv_channels,
        num_conv_layers=conv_layers,
        num_readout_layers=readout_layers,
        cheb_filter_size=cheb_filter_size,
        dropout_p=dropout_p)
    model.to(gpu)
    # model.load_state_dict(torch.load(
    #     f"multi_optim/results/{run_prefix_data}_policy_model.pt"))

    # load dataset from previous multi_optim_run
    dataset = EdgePolicyDataset(f"multi_optim/results/{run_prefix_data}_data")
    test_set_i_s = range(len(dataset) - n_test, len(dataset))
    test_set = dataset[test_set_i_s]
    test_set = [(d, {n: n for n in range(d.num_nodes)})
                for d in test_set]  # type: ignore
    exclude_keys = list(map(str, test_set_i_s))
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, exclude_keys=exclude_keys)

    # stat collection
    stats = {
        "accuracy": [],
        "loss": [],
    }  # type: Dict[str, List[float]]
    n_stat_points_desired = 200
    n_stat_points_per_epoch = n_stat_points_desired // n_epochs
    stats_every_x_batch = len(loader) // n_stat_points_per_epoch

    # training
    pb = ProgressBar(name, len(loader) * n_epochs, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for i_e in range(n_epochs):
        for i_b, batch in enumerate(loader):
            loss = model.learn(batch, optimizer)
            if loss is None:
                loss = 0.  # TODO: fix this
            pb.progress()
            if i_b % stats_every_x_batch == 0:
                accuracy = model.accuracy(test_set)
                print(f"{accuracy:.3f} (accuracy)@{name}")
                print(f"{loss:.3f} (loss)@{name}")
                stats["accuracy"].append(accuracy)
                stats["loss"].append(loss)
    pb.end()

    # save model
    torch.save(model.state_dict(),
               f"{base_folder}/edge_policy_{name}.pt")

    # save stats
    with open(f"{base_folder}/edge_policy_{name}.json", "w") as f:
        json.dump(stats, f)
    plt.figure(figsize=(20, 20), dpi=500)
    plt.plot(stats["accuracy"], label="accuracy")
    plt.plot(stats["loss"], label="loss")
    plt.legend()
    plt.savefig(f"{base_folder}/edge_policy_{name}.png")


def learning_proxy(kwargs):
    learning(**kwargs)


def tuning(base_folder):
    lr_s = [1E-4, 3E-4, 3E-5]
    batch_size_s = [64]
    conv_channels_s = [128, 256]
    conv_layers_s = [4, 5]
    readout_layers_s = [2, 3]
    cheb_filter_size_s = [5, 6]
    dropout_p_s = [0.2, 0.3]
    parameter_experiments = {
        "lr": lr_s,
        "batch_size": batch_size_s,
        "conv_channels": conv_channels_s,
        "conv_layers": conv_layers_s,
        "readout_layers": readout_layers_s,
        "cheb_filter_size": cheb_filter_size_s,
        "dropout_p": dropout_p_s
    }  # type: Dict[str, Union[str, List[float], List[int]]]

    seed_s = range(6)

    # prepare multithreading
    params_to_run = []

    # default run
    for seed in seed_s:
        kwargs = {k: v[0] for k, v in parameter_experiments.items()}
        kwargs["name"] = f"default_seed_{seed}"
        kwargs["seed"] = seed
        kwargs["base_folder"] = base_folder
        params_to_run.append(kwargs.copy())

    # experimental runs
    for name, values in parameter_experiments.items():
        for value in values[1:]:
            for seed in seed_s:
                kwargs = {k: v[0] for k, v in parameter_experiments.items()}
                kwargs[name] = value
                kwargs["name"] = f"{name}_{value}_seed_{seed}"
                kwargs["seed"] = seed
                kwargs["base_folder"] = base_folder
                params_to_run.append(kwargs.copy())

    mp.set_start_method("spawn")
    p = mp.Pool(8)
    p.map(learning_proxy, params_to_run)
    # for p in params_to_run:
    #     learning_proxy(p)


def rolling_average(data: List[float], n: int = 10) -> List[float]:
    return [sum(data[i:i + n]) / n for i in range(len(data) - n)]


def plot_results(base_folder):
    fnames = os.listdir(f"{base_folder}")
    fnames_json = [f for f in fnames if f.endswith(".json")]

    fnames_json = sorted(fnames_json)

    data = {}

    # stats over seeds
    max_seed = 0  # type: int

    for i_f, fname in enumerate(fnames_json):
        with open(f"{base_folder}/{fname}", "r") as f:
            stats = json.load(f)
        label = fname.split("edge_policy_")[1].split("_seed")[0]
        seed = int(fname.split("edge_policy_")[
                   1].split("seed_")[1].split(".")[0])
        max_seed = max(max_seed, seed)
        if label not in data:
            data[label] = {}
        data[label][seed] = stats

    n_seeds = max_seed + 1

    n_labels = len(data)
    experiments = sorted(data.keys())
    fig, (axs) = plt.subplots(
        3, n_labels,
        figsize=(5*n_labels, 10),
        dpi=500,
        sharex=True)
    # sharey=True)
    assert isinstance(axs, np.ndarray)

    maxloss = 0.  # type: float

    for i_l, label in enumerate(experiments):
        # collect data over seeds
        existing_seeds = 0  # type: int
        d_acc = []
        d_loss = []
        for i_s, seed in enumerate(sorted(data[label].keys())):
            max_seed = max(max_seed, seed)
            existing_seeds += 1
            d_acc.append(data[label][seed]["accuracy"])
            d_loss.append(data[label][seed]["loss"])

        # building statistics
        np_acc = np.array(d_acc)
        np_loss = np.array(d_loss)
        acc_mean = np.mean(np_acc, axis=0)
        loss_mean = np.mean(np_loss, axis=0)
        acc_std = np.std(np_acc, axis=0)
        loss_std = np.std(np_loss, axis=0)
        maxloss = max(maxloss, np.max(loss_mean) + np.max(loss_std))

        # barplots
        n_epochs = len(acc_mean)
        axs[0, i_l].bar(0, existing_seeds / n_seeds,
                        width=n_epochs/8., label="existing seeds")
        axs[0, i_l].set_yticks([0, 1])
        axs[0, i_l].set_yticklabels(["0%", "100%"])

        # plotting statistics
        axs[1, i_l].plot(acc_mean, label=label)
        axs[1, i_l].fill_between(
            range(len(acc_mean)), acc_mean - acc_std, acc_mean + acc_std,
            alpha=0.3)
        axs[2, i_l].plot(loss_mean, label=label)
        axs[2, i_l].fill_between(
            range(len(loss_mean)), loss_mean - loss_std, loss_mean + loss_std,
            alpha=0.3)

        # labels with means
        axs[1, i_l].text(len(acc_mean), acc_mean[-1], f"{acc_mean[-1]:.3f}")
        axs[2, i_l].text(len(loss_mean), loss_mean[-1], f"{loss_mean[-1]:.3f}")

        # additional text
        axs[1, i_l].set_title(label)
        axs[2, i_l].set_xlabel("epoch")

        axs[1, i_l].set_ylim(0, 1)
        axs[1, i_l].set_ylabel("accuracy")
        axs[2, i_l].set_ylabel("loss")

        # legend
        for ax in axs[:, i_l]:
            ax.legend()

    for ax in axs[2, :]:
        ax.set_ylim(0, maxloss)

    plt.savefig(f"{base_folder}/_edge_policy_results.png")


if __name__ == "__main__":
    base_folder = "planner/policylearn/results"
    # tuning(base_folder=base_folder)
    plot_results(base_folder)
