#!/usr/bin/env python3
from adamsmap import graphs_from_posar, make_edges, plot_graph
from adamsmap_filename_verification import is_eval_file, resolve
import imageio
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import csv


if __name__ == '__main__':
    fname = sys.argv[1]
    assert is_eval_file(fname), "call with *.pkl.eval file"
    with open(fname, "rb") as f:
        res = pickle.load(f)
        # res[20].keys() =
        # ['undir', 'rand', 'paths_ev', 'paths_undirected', 'paths_random']
    n_agents = list(res.keys())[2]
    paths_type = list(res[n_agents].keys())[2]
    n_trial = 0
    paths = np.array(res[n_agents][paths_type][n_trial])
    print(paths.shape)
    fname_csv = ("_".join(resolve(fname))
                 + "n_agents" + str(n_agents)
                 + "_paths_type" + str(paths_type)
                 + "_n_trial" + str(n_trial)
                 + ".csv")
    with open(fname_csv, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for t in range(paths.shape[0]):
            line = []
            for agent in range(paths.shape[1]):
                for i in [0, 1]:
                    line.append(paths[t, agent, i])
            writer.writerow(line)
