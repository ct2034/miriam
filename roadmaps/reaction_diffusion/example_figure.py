from pprint import pprint

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from definitions import MAP_IMG
from roadmaps.reaction_diffusion.rd import (get_experiments,
                                            get_initial_configuration,
                                            gray_scott_update,
                                            plot_bitmap_and_poses)
from roadmaps.var_odrm_torch.var_odrm_torch import read_map

if __name__ == "__main__":
    # load map
    map_name = "x.png"
    map_fname = f"roadmaps/odrm/odrm_eval/maps/{map_name}"
    map_img = cv2.imread(map_fname)
    RESIZE_FACTOR = 4
    N = len(map_img) // RESIZE_FACTOR
    map_img_resized = cv2.resize(map_img, (N, N))
    mask = map_img_resized.max(axis=2) < 255
    print(f"{mask.shape=}")

    # configure reaction diffusion
    A, B = get_initial_configuration(N)
    A_bg = 0.0
    B_bg = 0.0
    delta_t, experiment, N_simulation_steps, _ = get_experiments(0.5)
    pprint(experiment)

    # run reaction diffusion
    for i in tqdm(range(N_simulation_steps)):
        A, B = gray_scott_update(
            A, B, A_bg, B_bg, mask, **experiment, delta_t=delta_t)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.imshow(map_img, cmap="gray")
    masked = np.ma.masked_where(B <= 0.001, B)
    masked_bigger = np.repeat(masked, RESIZE_FACTOR,
                              axis=0).repeat(RESIZE_FACTOR, axis=1)
    img = ax.imshow(masked_bigger, cmap="jet")
    fig.colorbar(img, ax=ax)
    ax.set_title("Value of V")

    # beautify
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()

    # fin
    fig.savefig(f"roadmaps/reaction_diffusion/example_figure.pdf")
    # fig.show()
