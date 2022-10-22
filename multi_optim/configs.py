from typing import Any, Dict, List

configs = {
    "debug": {
        "n_nodes": 8,
        "n_runs_pose": 8,
        "n_runs_policy": 8,
        "n_episodes_per_run_policy": 2,
        "n_epochs_per_run_policy": 2,
        "batch_size_policy": 16,
        "stats_and_eval_every": 4,
        "lr_pos": 1e-2,
        "lr_policy": 1e-3,
        "max_n_agents": 2,
        "map_name": "x",
        "seed": 0,
        "prefix": "debug"
    },
    "tiny": {
        "n_nodes": 16,
        "n_runs_pose": 128,
        "n_runs_policy": 128,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 1e-3,
        "lr_policy": 1e-3,
        "max_n_agents": 4,
        "map_name": "x",
        "seed": 0,
        "prefix": "tiny"
    },
    "tiny_plain": {
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
        "map_name": "plain",
        "seed": 0,
        "prefix": "tiny_plain"
    },
    "small": {
        "n_nodes": 32,
        "n_runs_pose": 128,
        "n_runs_policy": 128,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 1e-3,
        "lr_policy": 3e-4,
        "max_n_agents": 6,
        "map_name": "x",
        "seed": 0,
        "prefix": "small"
    },
    "medium": {
        "n_nodes": 64,
        "n_runs_pose": 128,
        "n_runs_policy": 128,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 1e-3,
        "lr_policy": 1e-4,
        "max_n_agents": 8,
        "map_name": "x",
        "seed": 0,
        "prefix": "medium"
    },
    "large": {
        "n_nodes": 128,
        "n_runs_pose": 128,
        "n_runs_policy": 128,
        "n_episodes_per_run_policy": 256,
        "n_epochs_per_run_policy": 4,
        "batch_size_policy": 128,
        "stats_and_eval_every": 2,
        "lr_pos": 1e-3,
        "lr_policy": 1e-4,
        "max_n_agents": 10,
        "map_name": "x",
        "seed": 0,
        "prefix": "large"
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
        "map_name": "plain",
        "seed": 0,
        "prefix": "large_plain"
    }
}


def augment_config_by(
        configs: Dict[Any, Any],
        key: str,
        values: List[Any]) -> Dict[Any, Any]:
    augmented_configs = configs.copy()
    for name, config in configs.items():
        for value in values:
            prefix: str = f"{name}_{key}_{value}"
            augmented_configs[prefix] = {
                **config,
                key: value,
                "prefix": prefix}
    return augmented_configs
