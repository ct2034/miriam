import matplotlib.pyplot as plt
import numpy as np

show_animation = True

KP = 5.0
KR = 100.0
AREA_WIDTH = 50.0


def calculate_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calculate_repulsive_potential(x, y, ox, oy, robot_rad):
    # find a better/concise way to find this index
    minind = -1
    dmin = float('inf')
    for i in range(len(ox)):
        d = np.hypot(x - ox[i], y - oy[i])
        if d <= dmin:
            dmin = d
            minind = i

    dq = np.hypot(x - ox[minind], y - oy[minind])

    if dq <= robot_rad:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * KR * (1.0 / dq - 1.0 / robot_rad) ** 2
    else:
        return 0.0


def calculate_potential_field(gx, gy, ox, oy, res, robot_rad):
    minx = min(ox) - AREA_WIDTH
    miny = min(oy) - AREA_WIDTH
    maxx = max(ox) + AREA_WIDTH
    maxy = max(oy) + AREA_WIDTH
    xw = int(round((maxx - minx) / res))
    yw = int(round((maxy - miny) / res))

    pmap = [[0.0 for i in range(yw)] for j in range(xw)]

    for i in range(xw):
        x = minx + res * i
        for j in range(yw):
            y = miny + j * res
            ug = calculate_attractive_potential(x, y, gx, gy)
            uo = calculate_repulsive_potential(x, y, ox, oy, robot_rad)
            uf = ug + uo
            pmap[i][j] = uf
    return pmap, minx, miny


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def potential_field_planner(sx, sy, gx, gy, ox, oy, robot_rad, res):

    potential_map, minx, miny = calculate_potential_field(
        gx, gy, ox, oy, res, robot_rad)
    # search path

    d = np.hypot(sx - gx, sy - gy)
    ix = round((sx - minx) / res)
    iy = round((sy - miny) / res)
    gix = round((gx - minx) / res)
    giy = round((gy - miny) / res)

    if show_animation:
        draw_heatmap(potential_map)
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")

    rx = [sx]
    ry = [sy]
    motion = get_motion_model()

    while d >= res:
        min_potential = float('inf')
        minix, miniy = -1, -1
        for i in range(len(motion)):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(potential_map) or iny >= len(potential_map[0]):
                p = float('inf')  # outside area
            else:
                p = potential_map[inx][iny]
            if p < min_potential:
                min_potential = p
                minix = inx
                miniy = iny
        ix = minix
        iy = miniy
        xp = ix * res + minx
        yp = iy * res + miny
        d = np.hypot(gx - xp, gy - yp)
        print(xp, yp)
        rx.append(xp)
        ry.append(yp)

        if show_animation:
            plt.plot(ix, iy, ".k")
            plt.pause(0.001)

    print("Reached Goal")

    return rx, ry


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.coolwarm)


def main():
    print("potential_field_planning start")

    sx = 10.  # start x position [m]
    sy = 5.  # start y positon [m]
    gx = 40.  # goal x position [m]
    gy = 45.  # goal y position [m]

    grid_size = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]

    n_obstacles = 50
    ox = (np.random.rand(n_obstacles) * AREA_WIDTH).tolist()
    oy = (np.random.rand(n_obstacles) * AREA_WIDTH).tolist()

    if show_animation:
        plt.axis("equal")

    # path generation
    rx, ry = potential_field_planner(
        sx, sy, gx, gy, ox, oy, robot_radius, grid_size)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
