import matplotlib.pyplot as plt
import numpy as np

from astar import astar, path_length

if __name__ == "__main__":
    print("main ...")

    # map = abs(np.random.rand(100, 100)*0.01+1)
    map = abs(np.ones([100, 100])*0.5)
    map[:80,20] = -1
    map[80,20:60] = -1
    map[20,40:80] = -1
    map[20:,80] = -1

    path = astar((10, 10), (90, 90), map)
    print("length: ", path_length(path))  # 8.24 optimal for example (1,1) -> (6,6) with wall

    fig, ax = plt.subplots()

    ax.imshow(map.T, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('astar path')
    ax.axis([0, map.shape[0], 0, map.shape[1]])
    ax.plot(
        np.array(np.matrix(path)[:,0]).flatten(),
        np.array(np.matrix(path)[:,1]).flatten(),
        c='b',
        lw=2
    )

    plt.show()
