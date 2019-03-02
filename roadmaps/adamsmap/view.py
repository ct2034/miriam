#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

plt.style.use('bmh')
plt.rcParams["font.family"] = "serif"
plt.rcParams["savefig.dpi"] = 500

if __name__ == '__main__':
    res = np.zeros([3, 4096])
    i = 0
    for fname in ["z_200_4096.pkl",
                  "z_500_4096.pkl",
                  "z_1000_4096.pkl"]:
        with open(fname, "rb") as f:
            store = pickle.load(f)
        res[i, :] = store['batchcost']
        i += 1

    f, ax = plt.subplots()
    legends = ['200 Vertices', '500 Vertices', '1000 Vertices']
    ax.plot(np.transpose(res), linewidth=.3)
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Batch Cost")
    plt.legend(legends)
    plt.tight_layout()

    plt.savefig('convergence.png')
