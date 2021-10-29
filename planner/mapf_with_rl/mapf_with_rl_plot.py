#!/usr/bin/env python3
import json
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt


def plot_this_experiment(data: Dict[str, Tuple[List[int], List[float]]]):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    for name, (x, y) in data.items():
        if "success" in name:
            ax = ax2
        elif "regret" in name:
            ax = ax3
        else:
            ax = ax1
            ax.set_ylim(bottom=-1)
            ax.spines.bottom.set_position('zero')
            ax.spines.top.set_color('none')
        ax.plot(x, y, label=name, linewidth=.5)
    for ax in [ax1, ax2, ax3]:
        ax.legend(loc='lower left')


def make_plot_from_json(name: str):
    with open(f'planner/mapf_with_rl/results/{name}.json') as f:
        data = json.load(f)
    plot_this_experiment(data)
    plt.savefig(f'planner/mapf_with_rl/results/{name}.png', dpi=300)
