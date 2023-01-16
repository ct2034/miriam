from typing import Any, Dict, List, Union

CONFIGS_TYPE = Dict[str, Dict[str, Union[int, float, str, bool]]]
configs: CONFIGS_TYPE = {
    "debug": {
        "n_nodes": 32,
        "n_runs_pose": 8,
        "n_runs_policy": 8,
        "n_episodes_per_run_policy": 2,
        "n_epochs_per_run_policy": 2,
        "batch_size_policy": 16,
        "stats_and_eval_every": 4,
        "lr_pos": 1e-3,
        "lr_policy": 1e-3,
        "max_n_agents": 2,
        "map_name": "x.png",
        "seed": 0,
        "prefix": "debug"
    },
    "debug_no_rd": {
        "n_nodes": 32,
        "n_runs_pose": 8,
        "n_runs_policy": 8,
        "n_episodes_per_run_policy": 2,
        "n_epochs_per_run_policy": 2,
        "batch_size_policy": 16,
        "stats_and_eval_every": 4,
        "lr_pos": 1e-3,
        "lr_policy": 1e-3,
        "max_n_agents": 2,
        "map_name": "x.png",
        "seed": 0,
        "use_reaction_diffusion": False,
        "prefix": "debug_no_rd"
    },
    "tiny": {
        "n_nodes": 16,
        "n_runs_pose": 64,
        "n_runs_policy": 64,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 1e-3,
        "lr_policy": 1e-3,
        "max_n_agents": 4,
        "map_name": "x.png",
        "seed": 0,
        "prefix": "tiny"
    },
    "small": {
        "n_nodes": 32,
        "n_runs_pose": 64,
        "n_runs_policy": 64,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 1e-3,
        "lr_policy": 3e-4,
        "max_n_agents": 6,
        "map_name": "x.png",
        "seed": 0,
        "prefix": "small"
    },
    "medium": {
        "n_nodes": 64,
        "n_runs_pose": 64,
        "n_runs_policy": 64,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 1e-3,
        "lr_policy": 1e-4,
        "max_n_agents": 8,
        "map_name": "x.png",
        "seed": 0,
        "prefix": "medium"
    },
    "large": {
        "n_nodes": 128,
        "n_runs_pose": 128,
        "n_runs_policy": 64,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 1e-5,  # old value 1e-3
        "lr_policy": 1e-4,
        "max_n_agents": 10,
        "map_name": "x.png",
        "seed": 0,
        "prefix": "large"
    },
    "large_no_rd": {
        "n_nodes": 128,
        "n_runs_pose": 128,
        "n_runs_policy": 64,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 1e-3,  # old value when not using RD
        "lr_policy": 1e-4,
        "max_n_agents": 10,
        "map_name": "x.png",
        "seed": 0,
        "use_reaction_diffusion": False,
        "prefix": "large_no_rd"
    },
    "large_plain": {
        "n_nodes": 128,
        "n_runs_pose": 64,
        "n_runs_policy": 64,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 1e-3,
        "lr_policy": 1e-4,
        "max_n_agents": 10,
        "map_name": "plain.png",
        "seed": 0,
        "prefix": "large_plain"
    },
    "mapf_benchm_random-32-32-10_debug": {
        "n_nodes": 50,
        "n_runs_pose": 4,
        "n_runs_policy": 4,
        "n_episodes_per_run_policy": 2,
        "n_epochs_per_run_policy": 2,
        "batch_size_policy": 16,
        "stats_and_eval_every": 1,
        "lr_pos": 1e-3,
        "lr_policy": 1e-4,
        "max_n_agents": 4,
        "map_name": "random-32-32-10.map",
        "seed": 0,
        "radius": 0.25 * (1. / 32),
        "prefix": "mapf_benchm_random-32-32-10_debug"
    },
    "mapf_benchm_random-32-32-10": {
        "n_nodes": 200,
        "n_runs_pose": 64,
        "n_runs_policy": 64,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 3e-4,
        "lr_policy": 1e-4,
        "max_n_agents": 8,
        "map_name": "random-32-32-10.map",
        "seed": 0,
        "radius": 0.25 * (1. / 32),
        "prefix": "mapf_benchm_random-32-32-10"
    },
    "mapf_benchm_random-32-32-10_no_rd": {
        "n_nodes": 200,
        "n_runs_pose": 64,
        "n_runs_policy": 64,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 3e-4,
        "lr_policy": 1e-4,
        "max_n_agents": 8,
        "map_name": "random-32-32-10.map",
        "seed": 0,
        "use_reaction_diffusion": False,
        "radius": 0.25 * (1. / 32),
        "prefix": "mapf_benchm_random-32-32-10_no_rd"
    },
    "mapf_benchm_random-32-32-10_300": {
        "n_nodes": 300,
        "n_runs_pose": 64,
        "n_runs_policy": 64,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 3e-4,
        "lr_policy": 1e-4,
        "max_n_agents": 8,
        "map_name": "random-32-32-10.map",
        "seed": 0,
        "radius": 0.25 * (1. / 32),
        "prefix": "mapf_benchm_random-32-32-10_300"
    },
    "mapf_benchm_random-32-32-10_400": {
        "n_nodes": 400,
        "n_runs_pose": 64,
        "n_runs_policy": 64,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 3e-4,
        "lr_policy": 1e-4,
        "max_n_agents": 8,
        "map_name": "random-32-32-10.map",
        "seed": 0,
        "radius": 0.25 * (1. / 32),
        "prefix": "mapf_benchm_random-32-32-10_400"
    }
}


def augment_config_by(
        configs: CONFIGS_TYPE,
        key: str,
        values: List[Any]) -> CONFIGS_TYPE:
    augmented_configs = configs.copy()
    for name, config in configs.items():
        for value in values:
            prefix: str = f"{name}_{key}_{value}"
            augmented_configs[prefix] = {
                **config,
                key: value,
                "prefix": prefix}
    return augmented_configs


configs_all_maps = augment_config_by(configs, "map_name", [
    "c.png", "z.png"
])
