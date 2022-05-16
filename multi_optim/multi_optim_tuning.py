#!/usr/bin/env python3
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import yaml
from matplotlib import pyplot as plt
from tools import ProgressBar

logger = logging.getLogger(__name__)

TUNING_RES_FOLDER = "multi_optim/results/tuning"
ABLATION_RES_FOLDER = "multi_optim/results/ablation"


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
    n_agents_s = [4, 3, 5]
    parameter_experiments = {
        "n_nodes": n_nodes_s,
        "n_runs_pose": [64],
        "n_runs_policy": [64],
        "n_epochs_per_run_policy":  [256],
        "batch_size_policy":  [128],
        "stats_and_eval_every": [16],
        "lr_pos": [1E-3, 3E-3, 3E-4],
        "lr_policy": [1E-3],
        "n_agents": n_agents_s,
        "map_fname": ["roadmaps/odrm/odrm_eval/maps/x.png"],
        "save_images": [False],
        "save_folder": [TUNING_RES_FOLDER]
    }  # type: Dict[str, Union[List[int], List[float], List[str]]]
    n_runs = 8
    return parameter_experiments, n_runs


def params_ablation():
    parameter_experiments = {
        "n_nodes": [16, 8, 32],
        "n_runs_pose": [64, 1],
        "n_runs_policy": [128, 1],
        "n_epochs_per_run_policy": [128],
        "batch_size_policy": [128],
        "stats_and_eval_every": [8],
        "lr_pos": [1E-2],
        "lr_policy": [1E-3],
        "n_agents": [4, 3, 5, 6],
        "map_fname": ["roadmaps/odrm/odrm_eval/maps/c.png"],
        "save_images": [True],
        "save_folder": [ABLATION_RES_FOLDER]
    }  # type: Dict[str, Union[List[int], List[float], List[str]]]
    n_runs = 4
    return parameter_experiments, n_runs


def make_kwargs_for_tuning(parameter_experiments, n_runs):
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
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "-1"
    process = subprocess.Popen(
        ["/usr/bin/python3",
         "-c",
         f"{import_str}; kwargs = {kwargs_str};"
         + "run_optimization(**kwargs)"],
        cwd=str(os.getcwd()),
        env=my_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    return process


def clean_str(inp: str) -> str:
    return inp.replace("[1m", "").replace("[0m", "").replace("[31m", "")


def run(params_to_run):
    logger.info("Creating processes")
    logger.debug(f"{len(params_to_run)=}")
    cpus = os.cpu_count()
    assert isinstance(cpus, int)
    logger.info(f"{cpus=}")
    max_active_processes: int = 1  # min(cpus, 8)
    logger.info(f"{max_active_processes=}")
    active_processes = set()

    # finish out what ran before
    ran_prefixes = set()
    suffix = "_stats.yaml"
    for f in os.listdir(params_to_run[0]["save_folder"]):
        if f.endswith(suffix):
            prefix = f.replace(suffix, "")
            ran_prefixes.add(prefix)

    # not running them again
    logger.info(f"(all) {len(params_to_run)=}")
    logger.info(f"{len(ran_prefixes)=}")
    params_to_run = [
        p for p in params_to_run if p["prefix"] not in ran_prefixes]
    logger.info(f"(remaining) {len(params_to_run)=}")

    n_initial = len(params_to_run)
    pb = ProgressBar("Running", n_initial, 5, logger.info)

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
            logger.debug(f"{len(active_processes)=}")
            logger.debug(f"{len(params_to_run)=}")
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
            pb.progress(n_initial - len(params_to_run) - len(active_processes))
            active_processes -= to_remove
        time.sleep(0.1)
    pb.end()
    logger.info("Done")


def plot_data(path: str):
    fnames = os.listdir(path)
    fnames_json = [f for f in fnames if f.endswith(".yaml")]

    fnames_json = sorted(fnames_json)

    data = {}  # type: Dict[str, Dict[int, Any]]

    for i_f, fname in enumerate(fnames_json):
        with open(os.path.join(path, fname), "r") as f:
            stats = yaml.load(f, Loader=yaml.SafeLoader)
        exp = fname.split("_seed")[0]
        seed = int(fname.split("seed_")[1].split("_stats")[0])
        if exp not in data:
            data[exp] = {}
        data[exp][seed] = stats

    exps = list(data.keys())
    exps.sort()
    first_available_seed = min(data[exps[0]].keys())
    n_exps = len(exps)
    params = list(data[exps[0]][first_available_seed].keys())
    params.remove("static")
    params.sort()
    n_params = len(params)
    fig, (axs) = plt.subplots(
        n_params,
        n_exps,
        figsize=(4*n_exps, 4*n_params),
        dpi=200)

    lims_per_param = {}  # type: Dict[str, List[float]]

    for i_exp, exp in enumerate(exps):
        for i_param, param in enumerate(params):
            ax = axs[i_param, i_exp]  # type: ignore

            # consolidate data
            n_seeds = len(data[exp])
            first_available_seed = min(data[exp].keys())
            t_raw = data[exp][first_available_seed][param]['t']
            n_t = len(t_raw)
            this_data_raw = np.zeros((n_seeds, n_t))
            for i_seed, seed in enumerate(data[exp].keys()):
                this_data_raw[i_seed, :] = data[exp][seed][param]['x']

            means = np.empty((0,))
            stds = np.empty((0,))
            mins = np.empty((0,))
            maxs = np.empty((0,))
            ts = np.empty((0,))
            for i_t, t in enumerate(t_raw):
                t_slice = this_data_raw[:, i_t]
                if not np.isnan(t_slice).all():
                    mean = np.mean(
                        t_slice[np.logical_not(np.isnan(t_slice))])
                    means = np.append(means, mean)
                    std = np.std(
                        t_slice[np.logical_not(np.isnan(t_slice))])
                    stds = np.append(stds, std)
                    min_ = np.min(t_slice[np.logical_not(np.isnan(t_slice))])
                    mins = np.append(mins, min_)
                    max_ = np.max(t_slice[np.logical_not(np.isnan(t_slice))])
                    maxs = np.append(maxs, max_)
                    ts = np.append(ts, t)

            # plot
            ax.plot(ts, means, label=exp, color='black')
            ax.fill_between(ts, means - stds, means +
                            stds, alpha=0.2, color='grey')
            ax.plot(ts, mins, label=exp, color='green')
            ax.plot(ts, maxs, label=exp, color='red')

            # labels
            if i_exp == 0:
                ax.set_ylabel(param)
            if i_param == 0:
                ax.set_title(exp)
            if i_param == len(params) - 1:
                ax.set_xlabel("epoch")
            ax.set_xlim(0, ts[-1])
            ax.grid()

            # find limits
            bottom_lim = float(np.min(mins))
            top_lim = float(np.max(maxs))
            if param not in lims_per_param:
                lims_per_param[param] = [0., 0.]
                lims_per_param[param][0] = bottom_lim
                lims_per_param[param][1] = top_lim
            else:
                lims_per_param[param][0] = min(lims_per_param[param][0],
                                               bottom_lim)
                lims_per_param[param][1] = max(lims_per_param[param][1],
                                               top_lim)

    # cosmetics
    plt.tight_layout()

    for i_exp, exp in enumerate(exps):
        for i_param, param in enumerate(params):
            ax = axs[i_param, i_exp]  # type: ignore
            ax.set_ylim(lims_per_param[param][0], lims_per_param[param][1])

    plt.savefig(os.path.join(path, "_tuning_stats.png"))


if __name__ == "__main__":
    tuning = True
    ablation = False

    if tuning:
        folder = TUNING_RES_FOLDER
        # parameter_experiments, n_runs = params_debug()
        parameter_experiments, n_runs = params_run()
    elif ablation:
        folder = ABLATION_RES_FOLDER
        parameter_experiments, n_runs = params_ablation()
    else:
        raise ValueError("No tuning or ablation")

    if not os.path.exists(folder):
        os.makedirs(folder)

    logging.basicConfig(
        filename=os.path.join(
            folder, "_tuning.log"),
        filemode='w',
        level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    params_to_run = make_kwargs_for_tuning(parameter_experiments, n_runs)
    run(params_to_run)
    plot_data(folder)
