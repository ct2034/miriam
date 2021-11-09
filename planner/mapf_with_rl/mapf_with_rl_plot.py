#!/usr/bin/env python3
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
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


def get_bare_eval_name(key):
    a_key = key\
        .replace('eval_', '')\
        .replace('regret_', '')\
        .replace('success_', '')\
        .replace('inv_', '')

    return a_key


def make_summary_plot_for_files(files: List[str], name_out: str):
    data = {}
    for name in files:
        with open(f'planner/mapf_with_rl/results/{name}.json') as f:
            data[name] = json.load(f)
    p_num = re.compile('[0-9]+')
    number_m = p_num.search(files[0])
    assert number_m is not None, "Number must be in the name of the file"
    prefix = files[0][:number_m.start()]
    assert prefix == "run", "Prefix must be 'run'"
    suffix = files[0][number_m.end():]
    data_keys = data[files[0]].keys()
    n_evaluations = np.count_nonzero(
        list(map(lambda x: x.startswith('eval_'), data_keys)))
    fig, axs = plt.subplots(int(n_evaluations/4) + 1, 1,
                            figsize=(28.8, 21.6), sharex=True)
    unique_evaluations = set()
    for key in data_keys:
        if key.startswith('eval_'):
            a_key = get_bare_eval_name(key)
            unique_evaluations.add(a_key)
        evaluations = sorted(list(unique_evaluations))
    for key in data_keys:
        if not key.startswith('eval_'):
            ev = axs[0]
            label = key
        else:
            ev = axs[evaluations.index(get_bare_eval_name(key)) + 1]
            label = key.replace("_"+get_bare_eval_name(key), '')
        x = data[files[0]][key][0]
        all_data = np.zeros((len(files), len(x)))
        for i, name in enumerate(files):
            all_data[i] = data[name][key][1]
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        ev.plot(x, mean, label=label, linewidth=1)
        ev.fill_between(x, mean - std, mean + std, alpha=0.2)
    for ev in axs:
        ev.legend(loc='lower left')
    axs[0].set_title(f'{name_out}')
    # set evaluation titles
    for i, ev in enumerate(evaluations):
        axs[i + 1].set_title(ev)
    fig.tight_layout()
    fig.savefig(f'planner/mapf_with_rl/results/{name_out}.png', dpi=300)


def make_summary_plot_for_all_files():
    files = os.listdir('planner/mapf_with_rl/results')
    json_files_by_suffix = {}
    for filename in files:
        if (
                filename.endswith('.json') and
                'runs' not in filename and  # not already a summary
                filename.startswith('run')):  # not a debug file or something else
            name = filename[:-5]
            p_num = re.compile('[0-9]+')
            number_m = p_num.search(name)
            assert number_m is not None, "Number must be in the name of the file"
            suffix = name[number_m.end():]
            if suffix not in json_files_by_suffix.keys():
                json_files_by_suffix[suffix] = []
            json_files_by_suffix[suffix].append(name)
    for suffix, names in json_files_by_suffix.items():
        make_summary_plot_for_files(names, f'runs{suffix}')


if __name__ == '__main__':
    make_summary_plot_for_all_files()
    make_plots_for_all_files_in_results_dir()
