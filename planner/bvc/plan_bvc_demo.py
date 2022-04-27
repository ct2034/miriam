import numpy as np
from planner.bvc.plan_bvc import get_average_path_length, plan
from scenarios.visualization import plot_with_paths

if __name__ == "__main__":
    map_img = ((0,),)
    starts = [[0.3, 0.2], [0.6, 0.2], [0.4, 0.8]]
    goals = [[0.6, 0.8], [0.3, 0.8], [0.8, 0.3]]

    paths = plan(map_img, starts, goals, radius=0.05)
    assert isinstance(paths, np.ndarray)
    avg_len = get_average_path_length(paths)
    print(f"Average path length: {avg_len}")

    print(paths)
    plot_with_paths(map_img, paths)
