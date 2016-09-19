import matplotlib.pyplot as plt
import numpy as np

from astar import astar, path_length

if __name__ == "__main__":
    print("main ...")

    map = abs(np.random.rand(8, 8)*.5)
    map[2,4] = -1
    map[3,4] = -1
    map[4,4] = -1
    map[5,4] = -1

    #  ........ 0
    #  .s......
    #  ........ 2
    #  ........
    #  ..####.. 4
    #  ........
    #  ......g.
    #  ........ 7
    #  0      7

    path = astar((1, 1), (6, 6), map)
    print("length: ", path_length(path))  # 8.24 optimal for example (1,1) -> (6,6) with wall

    fig, ax = plt.subplots()

    ax.imshow(map.T, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('astar path')
    ax.plot(
        np.array(np.matrix(path)[:,0]).flatten(),
        np.array(np.matrix(path)[:,1]).flatten(),
        'r'
    )

    plt.show()
