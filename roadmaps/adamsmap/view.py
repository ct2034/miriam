#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

if __name__ == '__main__':
    with open(sys.argv[1], "rb") as f:
        store = pickle.load(f)

    f, ax = plt.subplots()
    legends = []
    for k, v in store.items():
        ax.plot(v)
        legends.append(k)
    plt.legend(legends)
    plt.show()
