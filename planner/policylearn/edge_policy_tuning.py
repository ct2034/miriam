import json
from random import Random

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
    # conv_layers: int,
):
    rng = Random(0)
    torch.manual_seed(0)

    # run to learn from
    run_prefix_data: str = "tiny"
    n_test = 50
    n_epochs = 10  # on hal = ~5h

    # load previously trained model
    model = EdgePolicyModel(
        gpu=torch.device("cpu"),
        conv_channels=conv_channels)
    # model.load_state_dict(torch.load(
    #     f"multi_optim/results/{run_prefix_data}_policy_model.pt"))

    # load dataset from previous multi_optim_run
    dataset = EdgePolicyDataset(f"multi_optim/results/{run_prefix_data}_data")
    test_set_i_s = range(len(dataset) - n_test, len(dataset))
    test_set = dataset[test_set_i_s]
    test_set = [(d, {n: n for n in range(d.num_nodes)}) for d in test_set]
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, exclude_keys=test_set_i_s)

    stats = {
        "accuracy": [],
        "loss": [],
    }

    # training
    pb = ProgressBar(name, len(loader) * n_epochs, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for i_e in range(n_epochs):
        for i_b, batch in enumerate(loader):
            loss = model.learn(batch, optimizer)
            pb.progress()
        accuracy = model.accuracy(test_set)
        print(f"accuracy: {accuracy}")
        print(f"loss: {loss}")
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


if __name__ == "__main__":
    lr_s = [3E-2, 1E-2, 3E-3, 1E-3]
    batch_size_s = [64, 128]
    conv_channels_s = [32, 64]
    # conv_layers_s = [1, 2, 3]
    parameter_experiments = {
        "lr": lr_s,
        "batch_size": batch_size_s,
        "conv_channels": conv_channels_s,
        # "conv_layers": conv_layers_s,
    }

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

    p = mp.Pool(16)
    p.map(learning_proxy, params_to_run)
