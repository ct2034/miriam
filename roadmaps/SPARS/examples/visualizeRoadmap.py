# pyline: disable=no-member
import numpy as np
import argparse

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.image as mpimg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input",  help="input file (yaml)")
    parser.add_argument("environment",  help="environment file (png)")
    args = parser.parse_args()

    data = np.loadtxt(args.input, delimiter=',')
    data = np.reshape(data, (data.shape[0], 2, 2))

    lc = mc.LineCollection(data)
    fig, ax = plt.subplots()

    img = mpimg.imread(args.environment)
    ax.imshow(img)

    ax.add_collection(lc)
    # ax.autoscale()
    # ax.margins(0.1)
    plt.show()
