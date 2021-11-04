#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

import networkx as nx


def make_random_maze(m, n, p):
    """
    Make a random maze.
    """
    maze = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if np.random.random() < p:
                maze[i, j] = 1
    return maze


def find_random_free_pose(maze):
    """
    Find a free pose in a maze.
    """
    m, n = maze.shape
    while True:
        i = np.random.randint(m)
        j = np.random.randint(n)
        if maze[i, j] == 0:
            return (i, j)


def get_neighbors(current, maze):
    """
    Get the neighbors of a position.
    """
    neighbors = []
    m, n = maze.shape
    if current[0] > 0 and maze[current[0] - 1, current[1]] == 0:
        neighbors.append((current[0] - 1, current[1]))
    if current[0] < m - 1 and maze[current[0] + 1, current[1]] == 0:
        neighbors.append((current[0] + 1, current[1]))
    if current[1] > 0 and maze[current[0], current[1] - 1] == 0:
        neighbors.append((current[0], current[1] - 1))
    if current[1] < n - 1 and maze[current[0], current[1] + 1] == 0:
        neighbors.append((current[0], current[1] + 1))
    return neighbors


def plot_maze_with_path(maze, path, start, goal):
    """
    Plot a maze with a path.
    """
    plt.imshow(maze, cmap='binary', interpolation='nearest')
    plt.plot(start[1], start[0], 'ro')
    plt.plot(goal[1], goal[0], 'go')
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    plt.plot(path_x, path_y, 'b-')
    plt.xticks(np.arange(0, maze.shape[1], 1.0))
    plt.yticks(np.arange(0, maze.shape[0], 1.0))
    plt.grid(False)
    plt.show()


def find_path(maze, start, goal):
    """
    Find a path in a maze.
    """
    # convert to networkx graph
    graph = nx.Graph()
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 0:
                graph.add_node((i, j))
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i, j] == 0:
                neighbors = get_neighbors(
                    [i, j], maze)
                for neighbor in neighbors:
                    graph.add_edge((i, j), neighbor)
    # find path
    path = nx.shortest_path(graph, tuple(start), tuple(goal))
    return path


if __name__ == '__main__':
    np.random.seed(0)
    maze = make_random_maze(100, 100, 0.3)
    start = find_random_free_pose(maze)
    goal = find_random_free_pose(maze)
    # plot_maze(maze, start, goal)
    path = find_path(maze, start, goal)
    plot_maze_with_path(maze, path, start, goal)
