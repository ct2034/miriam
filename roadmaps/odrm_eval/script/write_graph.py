#!/usr/bin/env python2
import csv
import matplotlib.pyplot as plt
import os
import pickle
import sys

import imageio
import networkx as nx
from odrm.odrm import graphs_from_posar, make_edges, plot_graph
from odrm_eval.filename_verification import (
    get_graph_csvs,
    get_graph_undir_csv,
    is_result_file,
    resolve_mapname,
    get_basename_wo_extension
)

plt.style.use('bmh')
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 120
figsize = [8, 8]  # = 960px / 120dpi


def remove_comment_lines(fname):
    tmp_fname = fname + "TMP"
    os.rename(fname, tmp_fname)
    with open(tmp_fname, 'r') as f_tmp:
        with open(fname, 'w') as f:
            for line in f_tmp.readlines():
                if not line.startswith("#"):
                    f.write(line)
    os.remove(tmp_fname)


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
    _, g, pos = graphs_from_posar(posar.shape[0], posar)
    make_edges(N, _, g, posar, edgew, map_im)
    if sys.argv[1] == "fig":
        graph_fig = plt.figure(figsize=figsize)
        ax = graph_fig.add_subplot(111)
        plt.gca().set_position([0, 0, 1, 1])
        plot_graph(graph_fig, ax, g, pos, edgew, map_im,
                   fname="res/" + basename + ".final_sq.png")
        plt.show()
    elif sys.argv[1] == "csv":
        fname_adjlist, fname_pos = get_graph_csvs(fname)
        fname_adjlist_undir = get_graph_undir_csv(fname)
        nx.write_adjlist(g, fname_adjlist)
        remove_comment_lines(fname_adjlist)
        g_undir = g.copy()
        for e in g.edges:
            g_undir.add_edge(e[1], e[0])
        nx.write_adjlist(g_undir, fname_adjlist_undir)
        remove_comment_lines(fname_adjlist_undir)
        with open(fname_pos, 'w') as f_csv:
            writer = csv.writer(f_csv, delimiter=' ')
            for i_a in range(N):
                writer.writerow(posar[i_a])
    else:
        assert False, "commands are `csv` and `fig`"
