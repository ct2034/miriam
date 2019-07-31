#!/usr/bin/env python3
from adamsmap import graphs_from_posar, make_edges, plot_graph
from adamsmap_filename_verification import is_result_file, resolve_mapname
import imageio
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

plt.style.use('bmh')
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 120
figsize = [8, 8]  # = 960px / 120dpi

if __name__ == '__main__':
    fname = sys.argv[1]
    assert is_result_file(fname), "Please call with result file (*.pkl)"
    with open(fname, "rb") as f:
        res = pickle.load(f)
        # res.keys() =
        # ['undir', 'rand', 'paths_ev', 'paths_undirected', 'paths_random']

    posar = res['posar']
    edgew = res['edgew']
    graph_fig = plt.figure(figsize=figsize)
    ax = graph_fig.add_subplot(111)
    map_im = imageio.imread(resolve_mapname(fname))
    N = posar.shape[0]
    g, _, pos = graphs_from_posar(posar.shape[0], posar)
    make_edges(N, g, _, posar, edgew, map_im)
    plt.gca().set_position([0, 0, 1, 1])
    plot_graph(graph_fig, ax, g, pos, edgew, map_im,
               fname=fname+"_final_sq.png")
    plt.show()
