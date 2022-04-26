import json
import os

import numpy as np
from definitions import MAP_IMG
from matplotlib import pyplot as plt
from tools import hasher, run_command

SCENARIOS_FOLDER = '.scenarios_cache'
JSON_TEMPLATE_FOLDER = 'json_templates'
ROBOT_X_FNAME = 'robot_x.json'
TWOD_CONFIG_FNAME = '2d_config.json'
STR_GOAL_POS = "goal_position"
STR_RADIUS = "radius"
STR_START_POS = "start_position"
STR_BASE_PATH = "base_path"
STR_ROBOTS = "robots"
STR_FRAMES = "frames"
STR_ROBOT_POSITIONS = "robot_positions"
STR_ROBOT_ID = "robot_id"
STR_POSITION = "position"


def get_scenario_folder(hash: str = ""):
    """
    Get the folder where the scenarios are stored.
    """
    return os.path.join(os.path.dirname(__file__), SCENARIOS_FOLDER, hash)


def plan(map_img: MAP_IMG, starts, goals, radius: float):
    """
    Plan a path from start to goal.
    """
    # How many agents
    n_agents = len(starts)
    assert len(goals) == n_agents

    # Create the scenario folder.
    hash: str = hasher([map_img, starts, goals])
    scenario_folder: str = get_scenario_folder(hash)
    robots_folder: str = os.path.join(scenario_folder, STR_ROBOTS)
    if not os.path.exists(robots_folder):
        os.makedirs(robots_folder)

    # Create the robot files.
    for i_a in range(n_agents):
        content = None
        with open(os.path.join(
                os.path.dirname(__file__),
                JSON_TEMPLATE_FOLDER,
                ROBOT_X_FNAME
        ), 'r') as f:
            content = json.load(f)
        assert content is not None
        content[STR_START_POS] = starts[i_a]
        content[STR_GOAL_POS] = goals[i_a]
        content[STR_RADIUS] = radius
        with open(os.path.join(robots_folder, f"robot_{i_a}.json"), 'w') as f:
            json.dump(content, f, indent=2)

    # Create the 2d config file.
    content = None
    with open(os.path.join(
            os.path.dirname(__file__),
            JSON_TEMPLATE_FOLDER,
            TWOD_CONFIG_FNAME
    ), 'r') as f:
        content = json.load(f)
    assert content is not None
    content[STR_BASE_PATH] = scenario_folder
    content[STR_ROBOTS] = f"/{STR_ROBOTS}"
    with open(os.path.join(scenario_folder, "2d_config.json"), 'w') as f:
        json.dump(content, f, indent=2)

    # Run the planner.
    stdout, stderr, retcode = run_command(
        "./../../mr-nav-stack/lib/bvc/build/examples/bvc_2d_sim --config 2d_config.json",
        timeout=60,
        cwd=scenario_folder)
    print(f"{retcode=}")
    print(f"{stdout=}")
    print(f"{stderr=}")

    # Read the paths.
    f, ax = plt.subplots()
    for file in os.listdir(scenario_folder):
        if file.startswith("vis") and file.endswith(".json"):
            paths = [list() for _ in range(n_agents)]
            with open(os.path.join(scenario_folder, file), 'r') as f:
                content = json.load(f)
                for frame in content[STR_FRAMES]:
                    for position in frame[STR_ROBOT_POSITIONS]:
                        paths[position[STR_ROBOT_ID]].append(
                            position[STR_POSITION])
            os.remove(os.path.join(scenario_folder, file))
            paths_np = np.array(paths)
            for i_a in range(n_agents):
                ax.plot(
                    paths_np[i_a, :, 0],
                    paths_np[i_a, :, 1],
                    label=f"robot_{i_a} {file}")
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

    # TODO: Check if successful.
