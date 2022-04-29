import random
from turtle import width

import numpy as np
from definitions import INVALID, MAP_IMG
from matplotlib import pyplot as plt
from planner.bvc.plan_bvc import get_average_path_length, plan
from roadmaps.var_odrm_torch.var_odrm_torch import read_map

if __name__ == "__main__":
    random.seed(0)
    map_fname: str = "roadmaps/odrm/odrm_eval/maps/x.png"
    map_img_np: np.ndarray = np.array(read_map(map_fname))
    width = map_img_np.shape[1]
    print(f"{width=}")
    desired_width = width // 32
    print(f"{desired_width=}")
    bin_size = width // desired_width
    map_img_np = map_img_np.reshape((desired_width, bin_size,
                                     desired_width, bin_size)).max(3).max(1)

    # starts = [[.1, .2], [.3, .4]]
    # goals = [[.9, .9], [.8, .8]]

    # map_img_np = np.full((10, 10), 255)
    # map_img_np[4, 2:7] = 0
    map_img = tuple([tuple(map_img_np[i, :].tolist())
                     for i in range(map_img_np.shape[0])])

    n_agents = 4
    paths = INVALID
    for _ in range(20):
        starts = [[random.random(), random.random()] for _ in range(n_agents)]
        goals = [[random.random(), random.random()] for _ in range(n_agents)]
        paths = plan(map_img, starts, goals, radius=0.01)
        if isinstance(paths, np.ndarray):
            break
    assert isinstance(paths, np.ndarray)
    avg_len = get_average_path_length(paths)
    print(f"Average path length: {avg_len}")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # show map
    try:
        xx, yy = np.meshgrid(np.linspace(0, 1, len(map_img)),
                             np.linspace(0, 1, len(map_img)))
        z = np.zeros_like(xx)
        img_data = np.array(map_img) / -255. + 1
        ax.contourf(xx, yy, img_data, len(map_img),
                    zdir='z', offset=0.5, cmap=plt.cm.Greys, zorder=200)
    except Exception as e:
        pass

    # show paths
    for i_a in range(n_agents):
        ax.plot(
            paths[i_a, :, 0],
            paths[i_a, :, 1],
            range(paths.shape[1]),
            label=f"robot_{i_a}",
            zorder=100+i_a)
    ax.set_zlim((0., paths.shape[1]))
    ax.legend()

    plt.show()
