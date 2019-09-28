#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
from adamsmap_eval.filename_verification import (
    resolve_number_of_iterations,
is_result_file,
resolve_number_of_nodes,
resolve)



plt.style.use('bmh')
# plt.rcParams["font.family"] = "serif"
plt.rcParams["savefig.dpi"] = 500

def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    folder = sys.argv[1]  # type: str
    assert folder.endswith("/"), "Please call with folder"

    dat = []
    legends = []
    dummy_lines = []
    for fname in sorted(os.listdir(folder)):
        fname = folder + fname
        if is_result_file(fname):
            nit = resolve_number_of_iterations(fname)
            scen = resolve(fname)[0]  # type: str
            scen = scen.upper()
            N = resolve_number_of_nodes(fname)
            if nit == 4096 and scen != "C" and N != 1000:
                print("reading " + fname)
                legends.append("{} Nodes, Scenario {}".format(
                    N,
                    scen
                ))
                with open(fname, "r") as f:
                    this_dat = pickle.load(f)
                    print(this_dat.keys())
                    d = moving_average(this_dat['batchcost'])
                    dat.append(d)

    lines_n = len(legends)
    colors = map(lambda i: "C{}".format(i), range(lines_n))
    f, ax = plt.subplots()
    for i in range(lines_n):
        dummy_lines.append(ax.plot([], [], color=colors[i]))
    for i in range(lines_n):
        ax.plot(dat[i], linewidth=.3, color=colors[i])
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Batch Cost")
    legend = plt.legend(legends)
    # ax.add_artist(legend)
    plt.tight_layout()

    plt.savefig('res/convergence.png')
