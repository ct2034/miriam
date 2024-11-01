#!/usr/bin/env python3


from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi


def load_map(filename: str) -> np.ndarray:
    """
    Load a png file as map.
    """
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    # Convert to binary image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
    return img


def initialize_points(img: np.ndarray, k: int) -> np.ndarray:
    """
    Initialize the k points.
    """
    points = []
    for i in range(k):
        x = np.random.randint(0, img.shape[0])
        y = np.random.randint(0, img.shape[1])
        while img[x, y] == 0:
            x = np.random.randint(0, img.shape[0])
            y = np.random.randint(0, img.shape[1])
        points.append((x, y))
    return np.array(points)


def compute_centroids(vor: Voronoi, img: np.ndarray) -> np.ndarray:
    """
    Compute the centroids of the Voronoi cells.
    """
    centroids = []
    for region in vor.regions:
        if not region or -1 in region:
            continue
        polygon = [vor.vertices[i] for i in region]
        polygon = np.array(polygon)
        # Compute the centroid of the polygon
        centroid = np.mean(polygon, axis=0)
        x, y = int(centroid[0]), int(centroid[1])
        # Ensure the centroid is within the image bounds and not in an obstacle
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and img[y, x] != 0:
            centroids.append([x, y])
    return np.array(centroids)


def one_iter(points: np.ndarray, img: np.ndarray, k: int) -> np.ndarray:
    """
    Perform Lloyd's algorithm to refine the points.
    """
    vor = Voronoi(points)
    centroids = compute_centroids(vor, img)
    if len(centroids) == k:
        points = centroids
    return points


def lloyds_algorithm(points: np.ndarray, img: np.ndarray, k: int) -> np.ndarray:
    """
    Perform Lloyd's algorithm.
    """
    for _ in range(100):
        points = one_iter(points, img, k)
    return points


def make_colors(n: int) -> List[Tuple[float, float, float]]:
    """
    Make n colors.
    """
    colors = []
    for i in range(n):
        r = np.random.rand()
        g = np.random.rand()
        b = np.random.rand()
        colors.append((r, g, b))
    return colors


def show_img(
    img: np.ndarray,
    colors: List[Tuple[float, float, float]],
    points: np.ndarray,
    _cells: Dict[int, List[Tuple[int, int]]],
):
    """
    Show the image. With points.
    """
    # image
    plt.imshow(img, cmap="gray")

    # points
    plt.scatter(points[:, 1], points[:, 0], c="r", marker="+")

    # cells
    print(f"{len(_cells)=}")
    if _cells is not None:
        for i_p, cell in enumerate(_cells.values()):
            print(cell)
            if len(cell) == 0:
                continue
            npcell = np.array(cell)
            plt.fill(npcell[:, 1], npcell[:, 0], color=colors[i_p], alpha=0.5)

    # simplify plot
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    img = load_map("map.png")
    n_nodes = 10
    n_iterations = 100
    points = initialize_points(img, n_nodes)
    points = lloyds_algorithm(points, img, n_nodes)
    colors = make_colors(n_nodes)
    show_img(img, colors, points, {})
