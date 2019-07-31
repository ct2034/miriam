#!/usr/bin/env python3
from adamsmap import graphs_from_posar, make_edges, plot_graph
import imageio
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

plt.style.use('bmh')
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 120
figsize = [16, 9]
n_agents = 20

if __name__ == '__main__':
    fname = sys.argv[1]
    with open(fname, "rb") as f:
        res = pickle.load(f)
    paths_ev = res[n_agents]["paths_ev"][0]
    print(paths_ev.shape)
    print(np.max(paths_ev))
    T = int(paths_ev.shape[0] * .1)
    agents = paths_ev.shape[1]

    # get out the graph
    with open(fname.replace(".eval", ""), "rb") as f:
        res_w_graph = pickle.load(f)
    posar = res_w_graph['posar']
    edgew = res_w_graph['edgew']
    graph_fig = plt.figure(figsize=figsize)
    ax = graph_fig.add_subplot(111)
    map_im = imageio.imread("maps/"+fname.split("_")[0].split("/")[-1]+".png")
    N = posar.shape[0]
    g, _, pos = graphs_from_posar(posar.shape[0], posar)
    make_edges(N, g, _, posar, edgew, map_im)
    plot_graph(graph_fig, ax, g, pos, edgew, map_im, fname=fname+"_final.png")

    # First set up the figure, the axis, and the plot element we want to anim
    fig = plt.figure(figsize=figsize)
    ax_graph = plt.axes([0, 0, 1, 1])
    fig.add_axes(ax_graph)
    graph_im = imageio.imread(fname+"_final.png")
    ax_graph.imshow(graph_im, origin='lower')
    ax = plt.axes(xlim=(0, 1000), ylim=(0, 1000),
                  xticks=[], yticks=[],
                  aspect='equal',
                  alpha=.5)
    fig.add_axes(ax)
    ax.patch.set_visible(False)
    lines = []
    for _ in range(agents):
        line, = ax.plot([], [], 'o')
        lines.append(line)

    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    # animation function.  This is called sequentially
    def animate(t):
        for lnum, line in enumerate(lines):
            line.set_data(paths_ev[t, lnum, 0],
                          paths_ev[t, lnum, 1])
        return lines

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=range(T),
                                   interval=20)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('iros19adamsmap.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
