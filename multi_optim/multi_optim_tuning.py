import logging
import os
import queue
import subprocess
import time
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


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
        "save_folder": ["multi_optim/results/tuning"]
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
        "save_folder": ["multi_optim/results/tuning"]
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


if __name__ == "__main__":
    logging.basicConfig(filename="multi_optim/multi_optim_tuning.log",
                        filemode='w',
                        level=logging.DEBUG)
    parameter_experiments, n_runs = params_debug()
    # parameter_experiments, n_runs = params_run()
    params_to_run = make_kwargs(parameter_experiments, n_runs)
    run(params_to_run)
