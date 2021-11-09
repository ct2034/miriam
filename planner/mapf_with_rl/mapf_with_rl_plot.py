#!/usr/bin/env python3
import json
import os
import sys
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
from tools import ProgressBar


def plot_this_experiment(data: Dict[str, Tuple[List[int], List[float]]]):
    plt.style.use('bmh')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14.4, 10.8))
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
        ax.plot(x, y, label=name, linewidth=1)
    for ax in [ax1, ax2, ax3]:
        ax.legend(loc='lower left')


def make_plot_from_json(name: str):
    with open(f'planner/mapf_with_rl/results/{name}.json') as f:
        data = json.load(f)
    plot_this_experiment(data)
    plt.savefig(f'planner/mapf_with_rl/results/{name}.png', dpi=300)


def make_plots_for_all_files_in_results_dir():
    files = os.listdir('planner/mapf_with_rl/results')
    pb = ProgressBar("Plots", len(files), 10)
    for filename in files:
        if filename.endswith('.json'):
            make_plot_from_json(filename[:-5])
        pb.progress()
    pb.end()


if __name__ == '__main__':
    make_plots_for_all_files_in_results_dir()
