#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

plt.style.use('bmh')
plt.rcParams["font.family"] = "serif"
plt.rcParams["savefig.dpi"] = 500

if __name__ == '__main__':
    with open(sys.argv[1], "rb") as f:
        res = pickle.load(f)

    data = []
    i = 0
    for k, v in res.items():
        for k2, v2 in res[k].items():
            data.append(np.array(v2))
            i += 1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    violin_parts = ax.violinplot(data, showmeans=True, showextrema=False)
    violin_parts['cmeans'].set_color("C1")
    ax.set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])
    ax.set_xticklabels(['undirected', '\n10 Agents', 'random', '',
                        'undirected', '\n20 Agents', 'random', '',
                        'undirected', '\n50 Agents', 'random'])
    ax.set_ylabel("Cost difference [%]")
    plt.tight_layout()
    fig.savefig('eval_quality-'
                + sys.argv[1].replace('.','_')
                + '.png')
