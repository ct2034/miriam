"""
Script for making example plots of roadmaps
- PRM
- PRM*
- RRT
- Sparse2
"""
import os
import shutil
from random import Random

import matplotlib.pyplot as plt
import networkx as nx

from definitions import DISTANCE, MAP_IMG, POS
from roadmaps.benchmark import PRM
from roadmaps.var_odrm_torch.var_odrm_torch import read_map
from roadmaps.ompl.build.libomplpy import Ompl

DPI = 500

MAP_PATH = './roadmaps/odrm/odrm_eval/maps/Berlin_1_256.png'
MAP_INFLATED_PATH = './roadmaps/benchmark_examples/Berlin_1_256_inflated_1000.png'
OUTPUT_PATH = 'roadmaps/plots/'

N_SAMPLES = 1000


def make_plot(graph, map_fname, path, _):
    assert graph is not None, 'Roadmap must be built.'

    fig = plt.figure(frameon=False, figsize=(4, 4), dpi=DPI)
    ax = fig.add_axes([-.05, -.05, 1.1, 1.1])
    ax.axis('off')
    ax.set_aspect('equal')

    map_img = read_map(map_fname)
    ax.imshow(map_img, cmap='gray')

    # plot graph
    pos = nx.get_node_attributes(graph, POS)
    pos = {k: (v[0] * len(map_img), v[1] * len(map_img))
           for k, v in pos.items()}
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=1)
    # exclude self edges
    edges = [(u, v) for u, v in graph.edges if u != v]
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edges,
                           width=0.5, edge_color='dimgrey')

    with open(path, 'wb') as outfile:
        fig.canvas.print_png(outfile)


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    rng = Random(0)

    # pseudo inflate
    shutil.copy(MAP_PATH, MAP_INFLATED_PATH)

    # PRM
    # prm = PRM(
    #     map_fname=MAP_PATH,
    #     rng=rng,
    #     roadmap_specific_kwargs={
    #         'target_n': N_SAMPLES,
    #         'ideal_n': N_SAMPLES,
    #         'n_edges': N_SAMPLES * 2,
    #         'start_radius': .033,
    #     }
    # )
    # make_plot(prm.g, MAP_PATH, os.path.join(OUTPUT_PATH, 'PRM.png'), 'PRM')

    # SPARS2
    ompl = Ompl()
    edges, duration_ms = ompl.runSparsTwo(
        MAP_INFLATED_PATH, #mapFile
        1, #seed
        2.5, #denseDelta
        10., #sparseDelta
        5., #stretchFactor
        1000, #maxFailures
        4., #maxTime
        40000, #maxIter
    )
    print(edges)

    # cleanup
    os.remove(MAP_INFLATED_PATH)
