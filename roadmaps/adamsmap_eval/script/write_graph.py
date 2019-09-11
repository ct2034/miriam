#!/usr/bin/env python2
from adamsmap.adamsmap import graphs_from_posar, make_edges, plot_graph
from adamsmap_eval.filename_verification import (
    get_graph_csvs,
    is_result_file,
    resolve_mapname,
    get_basename_wo_extension
)
import csv
import imageio
from matplotlib import animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import sys

plt.style.use('bmh')
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 120
figsize = [8, 8]  # = 960px / 120dpi

if __name__ == '__main__':
    fname = sys.argv[2]
    assert is_result_file(fname), "Please call with result file (*.pkl)"
    basename = get_basename_wo_extension(fname)
    with open(fname, "rb") as f:
        res = pickle.load(f)
    posar = res['posar']
    edgew = res['edgew']
    map_im = imageio.imread(resolve_mapname(fname))
    N = posar.shape[0]
    g, _, pos = graphs_from_posar(posar.shape[0], posar)
    if sys.argv[1] == "fig":
        graph_fig = plt.figure(figsize=figsize)
        ax = graph_fig.add_subplot(111)
        make_edges(N, g, _, posar, edgew, map_im)
        plt.gca().set_position([0, 0, 1, 1])
        plot_graph(graph_fig, ax, g, pos, edgew, map_im,
                   fname="res/" + basename + ".final_sq.png")
        plt.show()
    elif sys.argv[1] == "csv":
        fname_adjlist, fname_pos = get_graph_csvs(fname)
        nx.write_adjlist(g, fname_adjlist)                   #
        with open(fname_pos, 'w') as f_csv:                       #
            writer = csv.writer(f_csv, delimiter=' ')
            for i_a in range(N):
                writer.writerow(posar[i_a])
    else:
        assert False, "commands are `csv` and `fig`"
