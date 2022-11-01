import os
from typing import Dict

import latextable
import matplotlib
import numpy as np
import yaml
from matplotlib import pyplot as plt
from texttable import Texttable

matplotlib.use('cairo')
plt.style.use('bmh')


def get_runtime_data_from_stats_file(folder: str, prefix: str) -> Dict:
    """
    Reads the runtime data from the stats file.
    :param folder: folder where the stats file is located
    :param prefix: prefix of the stats file
    :return: dictionary with the runtime data
    """
    out = {}
    stats_file = os.path.join(folder, prefix + "_stats.yaml")
    with open(stats_file, "r") as f:
        stats = yaml.load(f, Loader=yaml.SafeLoader)
    for k, v in stats.items():
        if k.startswith("runtime") or k == "static":
            out[k] = v
    return out


if __name__ == "__main__":
    data = {}
    folder = "multi_optim/results"
    n_runtimes = 4
    prefixes = [
        # "debug",
        "tiny",
        # "tiny_plain",
        "small",
        "medium",
        "large",
        # "large_plain",
    ]
    for prefix in prefixes:
        this_data = get_runtime_data_from_stats_file(folder, prefix)
        n_nodes = this_data["static"]["n_nodes"]
        data[n_nodes] = this_data
    runtime_keys = [
        "runtime_generation_all",
        "runtime_optim_policy",
        "runtime_optim_poses"
    ]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # type: ignore

    # Plot the runtime data
    f, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    assert isinstance(ax, plt.Axes)

    n_nodes_s = sorted(list(data.keys()))
    width = 1 / (n_runtimes + .5)
    for n_nodes in n_nodes_s:
        this_data = data[n_nodes]
        runtime_ts = list(np.linspace(
            0, len(this_data[runtime_keys[0]]['x'])-1, n_runtimes, dtype=int))
        x = float(n_nodes_s.index(n_nodes)) - (n_runtimes + 1) / 2 * width
        for t in runtime_ts:
            bottom = 0
            x += width
            for runtime_key in runtime_keys:
                this_runtime = this_data[runtime_key]['x'][t]
                if n_nodes == n_nodes_s[0] and t == runtime_ts[0]:
                    ax.bar(x, this_runtime, width, bottom=bottom,
                           label=runtime_key, color=colors[runtime_keys.index(runtime_key)])
                else:  # label only on first n_nodes and first t
                    ax.bar(x, this_runtime, width, bottom=bottom,
                           color=colors[runtime_keys.index(runtime_key)])
                ax.text(x - width/2.1, bottom + this_runtime /
                        2, f"{this_runtime:.1f}s",
                        fontsize='small', va='center')
                bottom += this_runtime
    ax.set_xlabel("Number of Vertices")
    ax.set_ylabel("Runtime per Iteration [s]")
    ax.set_xticks(range(len(n_nodes_s)))
    ax.set_xticklabels(list(map(str, n_nodes_s)))
    ax.legend()
    f.tight_layout()
    f.savefig(os.path.join(folder, "runtime.pdf"))

    # make latex table
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.header(["Number of Vertices", "At iteration"] +
                 runtime_keys + ["Total"])
    table.set_cols_dtype(['t', 't', 't', 't', 't', 't'])
    table.set_cols_align(['c', 'c', 'r', 'r', 'r', 'r'])
    for n_nodes in n_nodes_s:
        this_data = data[n_nodes]
        for t in runtime_ts:
            row = [n_nodes, t]
            total = 0
            for runtime_key in runtime_keys:
                value = this_data[runtime_key]['x'][t]
                row.append(f"{value:.1f}s")
                total += value
            row.append(f"{total:.1f}s")
            table.add_row(row)
    print(latextable.draw_latex(
        table, caption="Runtime per Iteration", label="tab:runtime"))
