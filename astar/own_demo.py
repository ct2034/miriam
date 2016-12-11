import datetime
import numpy as np

from astar import astar_grid48con

"""Scale of map (min: 1)"""
s = 2
"""Do you want to see something?"""
plot = False

map = np.zeros([s * 10, s * 10, s * 50])
map[:s * 8, s * 2, :] = -1
map[s * 8, s * 2:s * 6, :] = -1
map[s * 2, s * 4:s * 8, :] = -1
map[s * 2:, s * 8, :] = -1

startt = datetime.datetime.now()

path = astar_grid48con.astar_grid8con((s * 1, s * 1, 0), (s * 9, s * 9, s * 49), map)

print("computation time:", (datetime.datetime.now() - startt).total_seconds(), "s")

print("length: ", astar_grid48con.path_length(path))

if plot:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.imshow(map[:, :, 0].T, cmap='Greys', interpolation='nearest')
    #
    # X = np.arange(10)
    # Y = np.arange(10)
    # XY = np.meshgrid(X, Y)
    # Z = np.zeros(shape=XY[0].shape)  # TODO: include map in plot
    #
    # ax.plot_surface(X,
    #                 Y,
    #                 Z,
    #                 rstride=1,
    #                 cstride=1,
    #                 facecolors=plt.cm.BrBG(map[:, :, 0]),
    #                 shade=False)
    # plt.title('astar path')
    # plt.axis([0, map.shape[0], 0, map.shape[1]])

    patharray = np.array(path)
    ax.plot(
        xs=patharray[:, 0],
        ys=patharray[:, 1],
        zs=patharray[:, 2],
        c='b',
        lw=2
    )

    plt.show()
