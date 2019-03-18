#!/usr/bin/env python3
import imageio
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

plt.style.use('bmh')
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 500

if __name__ == '__main__':
    with open(sys.argv[1], "rb") as f:
        res = pickle.load(f)
    im = imageio.imread("maps/"+sys.argv[1].split("_")[0].split("/")[-1]+".png")
    paths_ev = res[20]["paths_ev"][0]
    print(paths_ev.shape)
    print(np.max(paths_ev))
    T = 1000  # int(paths_ev.shape[0] * .1)
    agents = paths_ev.shape[1]

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=[3.84, 2.16])
    ax = plt.axes(xlim=(0, 1000), ylim=(0, 1000),
                  xticks=[], yticks=[],
                  aspect='equal')
    ax.imshow(im)
    lines = []
    for _ in range(agents):
        line, = ax.plot([], [], 'o')
        lines.append(line)

    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([],[])
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
