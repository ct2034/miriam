import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.shape_base import _apply_along_axis_dispatcher


def min_dist(l1, l2):
    """
    Get the minimum distance between two lines.
    with `l1 = [[x1, y1], [x2, y2]]` and `l2 = [[x1, y1], [x2, y2]]`
    returns the distance and the two point `p1` and `p2` on the two lines
    """
    # Define points
    p1 = cp.Variable(2)  # point on l1
    p2 = cp.Variable(2)  # point on l2

    # Time
    t = cp.Variable(1)

    # Define objective
    obj = cp.Minimize(cp.norm(p1 - p2))

    # Define constraints
    constraints = [
        t >= 0,
        t <= 1,
        p1[0] == l1[0, 0] + t * (l1[1, 0] - l1[0, 0]),  # x
        p1[1] == l1[0, 1] + t * (l1[1, 1] - l1[0, 1]),  # y
        p2[0] == l2[0, 0] + t * (l2[1, 0] - l2[0, 0]),  # x
        p2[1] == l2[0, 1] + t * (l2[1, 1] - l2[0, 1]),  # y
    ]

    # Form and solve problem
    prob = cp.Problem(obj, constraints)
    prob.solve()

    return prob.value, p1.value, p2.value


if __name__ == "__main__":
    # Define line segments
    l1 = np.array([[1, 1], [3, 1]])  # Start point  # End point
    l2 = np.array([[3, 3], [1, 2]])  # Start point  # End point

    # Plot lines
    plt.plot(l1[:, 0], l1[:, 1], "b-")
    plt.plot(l2[:, 0], l2[:, 1], "g-")
    plt.axis("equal")

    # Plot potentioal distance lines
    n_lines = 20
    for t in np.linspace(0, 1, n_lines):
        x = l1[0] + t * (l1[1] - l1[0])
        y = l2[0] + t * (l2[1] - l2[0])
        plt.plot([x[0], y[0]], [x[1], y[1]], "grey", alpha=0.5)

    # Find minimum distance
    dist, p1, p2 = min_dist(l1, l2)
    print("Minimum distance:", dist)
    print("Point on l1:", p1)
    print("Point on l2:", p2)

    # Plot result
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "r--o")
    plt.show()
