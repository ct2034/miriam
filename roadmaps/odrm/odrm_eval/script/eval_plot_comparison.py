#!/usr/bin/env python2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from odrm_eval.filename_verification import is_result_file


def save_violin_plot(res):
    data = []
    i = 0
    for k_agents in res.keys():
        for k_diff in ["undir", "rand"]:
            data.append(np.array(res[k_agents][k_diff]))
            i += 1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    violin_parts = ax.violinplot(data, showmeans=True, showextrema=False)
    violin_parts["cmeans"].set_color("C1")
    ax.set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])
    ax.set_xticklabels(
        [
            "undirected",
            "\n10 Agents",
            "random",
            "",
            "undirected",
            "\n20 Agents",
            "random",
            "",
            "undirected",
            "\n50 Agents",
            "random",
        ]
    )
    ax.set_ylabel("Cost difference [%]")
    plt.tight_layout()
    fig.savefig("eval_quality-" + fname.replace(".", "_") + ".png")


if __name__ == "__main__":
    plt.style.use("bmh")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["savefig.dpi"] = 500
    fname = sys.argv[1]
    with open(fname, "rb") as f:
        assert is_result_file(fname), "Call this on a *.pkl.eval file."
        res = pickle.load(f)
    print(res)
    save_violin_plot(res)
