import numpy as np
import matplotlib.pyplot as plt
import png
import itertools


def get_img_coord(x):
    return tuple(np.array([
        x[0] * 20 + 160,
        x[1] * -20 + 100
    ]))


def shift(x, d):
    return (
        x[0] + d,
        x[1] + d
    )


def dist(a, b):  # Manhattan Distance
    return abs(a[0] - b[0] + a[1] - b[1])


def is_obstacle(amap, x):
    try:
        color = amap[int(x[1]), int(x[0]), :]
    except:
        return True
    return not (color == [1, 1, 1, 1]).all()


def check_path(amap, a, b):
    d = np.array(a) - np.array(b)
    l = dist(a, b)
    for p in [np.array(b) + i / l * d for i in np.arange(l)]:
        if is_obstacle(amap, p):
            return False
    return True


def make_graph(amap, ltop, rbot):
    graph = {}
    for point in itertools.product(
            np.arange(ltop[0], rbot[0] + 1),
            np.arange(ltop[1], rbot[1] + 1)
    ):
        graph[point] = []
        for d in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            goal = tuple(np.array(point) + d)
            imga = get_img_coord(point)
            imgb = get_img_coord(goal)
            if check_path(amap, imga, imgb):
                graph[point].append(goal)
    return graph


amap = plt.imread('multi_robot_nav/map/world.png')
graph = make_graph(amap, (-6, -4), (6, 4))
print(graph[(2, 0)])

if __name__ == '__main__':
    plt.imshow(amap)
    for o in graph.keys():
        for e in graph[o]:
            assert dist(o, e) == 1, "All edges should have dist == 1"
            orig = get_img_coord(o)
            end = get_img_coord(e)
            if orig > end:
                orig = shift(orig, 1)
                end = shift(end, 1)
                color = 'r'
            else:
                orig = shift(orig, -1)
                end = shift(end, -1)
                color = 'b'
            plt.plot([orig[0], end[0]],
                     [orig[1], end[1]],
                     color)
    plt.show()
