import logging
import os
import queue
import subprocess
import time
from cmath import e
from ntpath import join
from typing import Dict, List, Union

import numpy as np
import yaml
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

TUNING_RES_FOLDER = "multi_optim/results/tuning"


def params_debug():
    n_nodes_s = [4]
    parameter_experiments = {
        "n_nodes": n_nodes_s,
        "n_runs_pose": [2],
        "n_runs_policy": [2, 4],
        "n_epochs_per_run_policy": [2],
        "batch_size_policy": [2],
        "stats_and_eval_every": [1],
        "lr_pos": [1E-4],
        "lr_policy": [1E-3],
        "n_agents": [2],
        "map_fname": ["roadmaps/odrm/odrm_eval/maps/x.png"],
        "save_images": [False],
        "save_folder": [TUNING_RES_FOLDER]
    }  # type: Dict[str, Union[List[int], List[float], List[str]]]
    n_runs = 2
    return parameter_experiments, n_runs


def params_run():
    n_nodes_s = [8]
    parameter_experiments = {
        "n_nodes": n_nodes_s,
        "n_runs_pose": [64],
        "n_runs_policy": [128],
        "n_epochs_per_run_policy":  [128],
        "batch_size_policy":  [128],
        "stats_and_eval_every": [1],
        "lr_pos": [1E-4],
        "lr_policy": [1E-3],
        "n_agents": [4],
        "map_fname": ["roadmaps/odrm/odrm_eval/maps/x.png"],
        "save_images": [False],
        "save_folder": [TUNING_RES_FOLDER]
    }  # type: Dict[str, Union[List[int], List[float], List[str]]]
    n_runs = 8
    return parameter_experiments, n_runs


def make_kwargs(parameter_experiments, n_runs):
    seed_s = range(n_runs)

    # prepare multithreading
    params_to_run = []

    # default run
    for seed in seed_s:
        kwargs = {k: v[0] for k, v in parameter_experiments.items()}
        kwargs["prefix"] = f"default_seed_{seed}"
        kwargs["seed"] = seed
        params_to_run.append(kwargs.copy())

    # experimental runs
    for name, values in parameter_experiments.items():
        for value in values[1:]:
            for seed in seed_s:
                kwargs = {k: v[0] for k, v in parameter_experiments.items()}
                kwargs[name] = value
                kwargs["prefix"] = f"{name}_{value}_seed_{seed}"
                kwargs["seed"] = seed
                params_to_run.append(kwargs.copy())
    return params_to_run


def start_process(kwargs):
    import_str = "from multi_optim.multi_optim_run import run_optimization"
    kwargs_str = repr(kwargs)
    process = subprocess.Popen(
        ["/usr/bin/python3",
         "-c",
         f"{import_str}; kwargs = {kwargs_str};"
         + "run_optimization(**kwargs)"],
        cwd=str(os.getcwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    return process


def clean_str(inp: str) -> str:
    return inp.replace("[1m", "").replace("[0m", "").replace("[31m", "")


def run(params_to_run):
    logger.info("Creating processes")
    cpus = os.cpu_count()
    assert isinstance(cpus, int)
    logger.info(f"{cpus=}")
    max_active_processes: int = min(cpus, 12)
    logger.info(f"{max_active_processes=}")
    active_processes = set()

    while len(params_to_run) > 0 or len(active_processes) > 0:
        if (len(active_processes) < max_active_processes
                and len(params_to_run) > 0):
            if len(params_to_run) > 0:
                kwargs = params_to_run.pop()
                prefix = kwargs["prefix"]
                process = start_process(kwargs)
                active_processes.add(process)
                logger.info("Started process "
                            + f"\"{prefix}\" @ [{process.pid}]")
            logger.debug(f"active_processes: {active_processes}")
        else:
            to_remove = set()
            for process in active_processes:
                if process.poll() is not None:
                    outs, errs = process.communicate()
                    logger.info("Finished process "
                                + f"[{process.pid}]")
                    logger.info(f"outs: [{process.pid}] >>>>>"
                                + f"{clean_str(outs.decode('utf-8'))}"
                                + f"<<<<< [{process.pid}]")
                    logger.error(f"errs: [{process.pid}] >>>>>"
                                 + f"{clean_str(errs.decode('utf-8'))}"
                                 + f"<<<<< [{process.pid}]")
                    to_remove.add(process)
            active_processes -= to_remove
        time.sleep(0.1)
    logger.info("Done")


def plot_data():
    fnames = os.listdir(TUNING_RES_FOLDER)
    fnames_json = [f for f in fnames if f.endswith(".yaml")]

    fnames_json = sorted(fnames_json)

    data = {}

    for i_f, fname in enumerate(fnames_json):
        with open(os.path.join(TUNING_RES_FOLDER, fname), "r") as f:
            stats = yaml.load(f, Loader=yaml.SafeLoader)
        exp = fname.split("_seed")[0]
        seed = int(fname.split("seed_")[1].split("_stats")[0])
        if exp not in data:
            data[exp] = {}
        data[exp][seed] = stats

    exps = list(data.keys())
    exps.sort()
    n_exps = len(exps)
    params = list(data[list(data.keys())[0]][0].keys())
    params.remove("static")
    params.sort()
    n_params = len(params)
    fig, (axs) = plt.subplots(
        n_params,
        n_exps,
        figsize=(5*n_exps, 5*n_params),
        dpi=500)

    top_lim_per_param = {}

    for i_exp, exp in enumerate(exps):
        for i_param, param in enumerate(params):
            ax = axs[i_param, i_exp]

            # consolidate data
            n_seeds = len(data[exp])
            t = data[exp][0][param]['t']
            n_t = len(t)
            this_data = np.zeros((n_seeds, n_t))
            for i_seed, seed in enumerate(data[exp].keys()):
                this_data[i_seed, :] = data[exp][seed][param]['x']
            mean = np.mean(this_data, axis=0)
            std = np.std(this_data, axis=0)

            # plot
            ax.plot(t, mean, label=exp)
            ax.fill_between(t, mean - std, mean + std, alpha=0.2)

            # labels
            if i_exp == 0:
                ax.set_ylabel(param)
            if i_param == 0:
                ax.set_title(exp)
            if i_param == len(params) - 1:
                ax.set_xlabel("epoch")
            ax.set_xlim(0, t[-1])
            ax.grid()

            # find toplim
            top_lim = max(np.max(mean + std), 1)
            if param not in top_lim_per_param:
                top_lim_per_param[param] = top_lim
            top_lim_per_param[param] = max(top_lim, top_lim_per_param[param])

    # cosmetics
    plt.tight_layout()

    for i_exp, exp in enumerate(exps):
        for i_param, param in enumerate(params):
            ax = axs[i_param, i_exp]
            ax.set_ylim(0, top_lim_per_param[param])

    plt.savefig(os.path.join(TUNING_RES_FOLDER, "_tuning_stats.png"))


if __name__ == "__main__":
    logging.basicConfig(filename="multi_optim/multi_optim_tuning.log",
                        filemode='w',
                        level=logging.DEBUG)
    # parameter_experiments, n_runs = params_debug()
    parameter_experiments, n_runs = params_run()
    params_to_run = make_kwargs(parameter_experiments, n_runs)
    run(params_to_run)
    plot_data()
