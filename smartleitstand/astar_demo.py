import astar

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("main ...")

    map = np.zeros([10, 10])
    map[:8, 2] = -1
    map[8, 2:6] = -1
    map[2, 4:8] = -1
    map[2:, 8] = -1

    path = astar.astar((1, 1), (9, 9), map)
    print("length: ", astar.path_length(path))  # 8.24 optimal for example (1,1) -> (6,6) with wall

    fig, ax = plt.subplots()

    ax.imshow(map.T, cmap='Greys', interpolation='nearest')
    ax.set_title('astar path')
    ax.axis([0, map.shape[0], 0, map.shape[1]])
    ax.plot(
        np.array(np.matrix(path)[:, 0]).flatten(),
        np.array(np.matrix(path)[:, 1]).flatten(),
        c='b',
        lw=2
    )

    plt.show()
