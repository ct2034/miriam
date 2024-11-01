import json
import logging
import os
import subprocess
from functools import reduce
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from definitions import INVALID, MAP_IMG
from tools import hasher, run_command

SCENARIOS_FOLDER = ".scenarios_cache"
JSON_TEMPLATE_FOLDER = "json_templates"
ROBOT_X_FNAME = "robot_x.json"
TWOD_CONFIG_FNAME = "2d_config.json"
STATISTICS_FNAME = "statistics.json"
STR_GOAL_POS = "goal_position"
STR_RADIUS = "radius"
STR_START_POS = "start_position"
STR_BASE_PATH = "base_path"
STR_ROBOTS = "robots"
STR_FRAMES = "frames"
STR_ROBOT_POSITIONS = "robot_positions"
STR_ROBOT_ID = "robot_id"
STR_POSITION = "position"
STR_PLANNING_FAIL = "planning_fail"
STR_OBSTACLES = "obstacles"
STR_RESOLUTION = "resolution"
STR_GOAL_REACH_DISTANCE = "goal_reach_distance"

logger = logging.getLogger(__name__)


def get_scenario_folder(hash: str = ""):
    """
    Get the folder where the scenarios are stored.
    """
    return os.path.join(os.path.dirname(__file__), SCENARIOS_FOLDER, hash)


def return_succesful_path(
    paths_s: List[npt.NDArray[np.float64]], goals, goal_reach_distance
) -> Union[np.ndarray, Literal[-1]]:
    """
    Return the first set of paths that actually reaches the goals and INVALID
    if none of them do.
    """
    n_agents = len(goals)
    for paths in paths_s:
        assert len(paths.shape) == 3, "Paths must be 3D."

        # Check if successful.
        dists_from_goals: List[float] = list(
            map(
                lambda i_a: float(np.linalg.norm(paths[i_a, -1, :] - goals[i_a])),
                range(n_agents),
            )
        )
        if max(dists_from_goals) <= goal_reach_distance * 10:
            return paths
    return INVALID  # type: ignore # (because is -1)


def get_average_path_length(paths: npt.NDArray[np.float64]) -> float:
    """
    Get the average path length of a set of paths.
    """
    return np.mean(
        np.sum(np.linalg.norm(paths[:, 1:, :] - paths[:, :-1, :], axis=2), axis=1)
    )


def cleanup(scenario_folder):
    """Remove the `scenario_folder`."""
    if os.path.exists(scenario_folder):
        os.system(f"rm -rf {scenario_folder}")


def plan(map_img_big: MAP_IMG, starts, goals, radius: float):
    """
    Plan a path from start to goal.
    """
    # Scale down the map
    map_img_np: np.ndarray = np.array(map_img_big)
    assert len(map_img_np.shape) == 2, "Map image must be 2D."
    width: int = map_img_np.shape[1]
    print(f"{width=}")
    desired_width = width // 32
    print(f"{desired_width=}")
    bin_size = width // desired_width
    map_img_np = (
        map_img_np.reshape((desired_width, bin_size, desired_width, bin_size))
        .max(3)
        .max(1)
    )
    map_img_sml = tuple(
        [tuple(map_img_np[i, :].tolist()) for i in range(map_img_np.shape[0])]
    )

    # How many agents
    n_agents = len(starts)
    assert len(goals) == n_agents

    # Create the scenario folder.
    hash: str = hasher([map_img_sml, starts, goals])
    scenario_folder: str = get_scenario_folder(hash)
    robots_folder: str = os.path.join(scenario_folder, STR_ROBOTS)
    if not os.path.exists(robots_folder):
        os.makedirs(robots_folder)

    # Create the robot files.
    for i_a in range(n_agents):
        content = None
        with open(
            os.path.join(
                os.path.dirname(__file__), JSON_TEMPLATE_FOLDER, ROBOT_X_FNAME
            ),
            "r",
        ) as f:
            content = json.load(f)
        assert content is not None
        content[STR_START_POS] = starts[i_a]
        content[STR_GOAL_POS] = goals[i_a]
        content[STR_RADIUS] = radius
        with open(os.path.join(robots_folder, f"robot_{i_a}.json"), "w") as f:
            json.dump(content, f, indent=2)

    # Create the 2d config file.
    content = None
    with open(
        os.path.join(
            os.path.dirname(__file__), JSON_TEMPLATE_FOLDER, TWOD_CONFIG_FNAME
        ),
        "r",
    ) as f:
        content = json.load(f)
    assert content is not None
    content[STR_BASE_PATH] = scenario_folder
    content[STR_ROBOTS] = f"/{STR_ROBOTS}"
    goal_reach_distance = content[STR_GOAL_REACH_DISTANCE]
    # Obstacles
    width = len(map_img_sml)
    content[STR_RESOLUTION] = 1.0 / width
    content[STR_OBSTACLES] = []
    for i_x in range(width):
        assert len(map_img_sml[i_x]) == width
        for i_y in range(width):
            if map_img_sml[i_x][i_y] != 255:
                content[STR_OBSTACLES].append([i_y, i_x])
    with open(os.path.join(scenario_folder, "2d_config.json"), "w") as f:
        json.dump(content, f, indent=2)

    # Run the planner.
    timeout_s = 120
    try:
        stdout, stderr, retcode = run_command(
            "./../../mr-nav-stack/lib/bvc/build/examples/bvc_2d_sim --config 2d_config.json",
            timeout=timeout_s,
            cwd=scenario_folder,
        )
        logger.debug(f"{retcode=}")
        logger.debug(f"{stdout=}")
        logger.debug(f"{stderr=}")
    except subprocess.TimeoutExpired:
        logger.warn(f"Timeout ({timeout_s}s)")
        cleanup(scenario_folder)
        return INVALID

    if retcode != 0:
        # Cleanup.
        cleanup(scenario_folder)
        return INVALID

    # Read the paths.
    paths_s = []
    file: Optional[str] = None
    for file in os.listdir(scenario_folder):
        if file.startswith("vis") and file.endswith(".json"):
            try:
                paths: List[List[Tuple[float, float]]] = [
                    list() for _ in range(n_agents)
                ]
                with open(os.path.join(scenario_folder, file), "r") as f:
                    content = json.load(f)
                    for frame in content[STR_FRAMES]:
                        for position in frame[STR_ROBOT_POSITIONS]:
                            paths[position[STR_ROBOT_ID]].append(position[STR_POSITION])
                paths_np = np.array(paths)
                paths_s.append(paths_np)
            except KeyError:
                logger.warn(f"KeyError: {file}")

    if len(paths_s) == 0:
        cleanup(scenario_folder)
        return INVALID
    final_paths = return_succesful_path(paths_s, goals, goal_reach_distance)

    cleanup(scenario_folder)
    return final_paths
