#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np

from scenarios.generators import tracing_pathes_in_the_dark

if __name__ == "__main__":
    for i in range(10):
        size = 10
        env = tracing_pathes_in_the_dark(size, .2, 10, i)
        fill_actual = np.sum(env == 1) / size / size
        print(fill_actual)
        plt.imshow(env, cmap='Greys')
        plt.show()
