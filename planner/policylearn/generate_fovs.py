import numpy as np


def add_padding_to_gridmap(gridmap, radius):
    """add a border of blocks around the map of given radius.
    (The new size will be old size + 2 * radius in both directions)"""
    size = gridmap.shape
    padded_gridmap = np.ones([
        size[0] + 2 * radius,
        size[1] + 2 * radius],
        dtype=np.int8)
    padded_gridmap[
        radius:size[0]+radius,
        radius:size[1]+radius] = gridmap
    return padded_gridmap


def init_empty_fov(radius, t):
    return np.zeros([
        1 + 2 * radius,
        1 + 2 * radius,
        t
    ])


def make_obstacle_fovs(padded_gridmap, path, t, radius):
    """create for all agents a set of FOVS of radius containing positions of
    obstacles in gridmap."""
    obstacle_fovs = []
    for i_t in range(t+1):
        pos = path[i_t]
        obstacle_fovs.append(
            padded_gridmap[
                int(pos[0]):int(pos[0]) + 1 + 2 * radius,
                int(pos[1]):int(pos[1]) + 1 + 2 * radius
            ]
        )
    obstacle_fovs_np = np.stack(obstacle_fovs, axis=2)
    return obstacle_fovs_np


def make_all_agents_fovs(paths, agent, other_agent, radius):
    """create for the agent a set of FOVS of radius containing positions of
    other agents."""
    t = paths[0].shape[0]
    other_agent_fovs = init_empty_fov(radius, t)
    for i_t in range(t):
        pos = paths[agent][i_t]
        for i_a in [i for i in range(len(paths)) if i != agent] + [other_agent, ]:
            d = paths[i_a][i_t] - pos
            if (abs(d[0]) <= radius and
                    abs(d[1]) <= radius):
                other_agent_fovs[
                    int(d[0]) + radius,
                    int(d[1]) + radius,
                    i_t
                ] = 1. if i_a == other_agent else .5
    return other_agent_fovs


def make_path_fovs(paths, paths_until_col, agent, other_agent,
                   t_until_col, radius):
    """create for the agent a set of layers indicating their single-agent
    paths."""
    lengths = list(map(lambda x: x.shape[0], paths))
    path_fovs = init_empty_fov(radius, t_until_col + 1)
    paths_other_agent_fovs = init_empty_fov(radius, t_until_col + 1)
    paths_other_agents_fovs = init_empty_fov(radius, t_until_col + 1)
    for i_t_steps in range(t_until_col + 1):
        for i_a in range(len(paths)):
            if i_a == agent:
                fov_to_write = path_fovs
            elif i_a == other_agent:
                fov_to_write = paths_other_agent_fovs
            else:
                fov_to_write = paths_other_agents_fovs
            pos = paths_until_col[agent][i_t_steps]
            for i_t_path in range(paths[i_a].shape[0]):
                d = paths[i_a][i_t_path] - pos
                if (abs(d[0]) <= radius and
                        abs(d[1]) <= radius):
                    fov_to_write[
                        int(d[0]) + radius,
                        int(d[1]) + radius,
                        i_t_steps
                    ] += i_t_path / lengths[i_a]
    return path_fovs, paths_other_agent_fovs, paths_other_agents_fovs


def extract_all_fovs(t, paths_until_col, paths_full, padded_gridmap, i_a, i_oa, radius):
    obstacle_fovs = make_obstacle_fovs(
        padded_gridmap, paths_until_col[i_a], t, radius)
    pos_other_agent_fovs = make_all_agents_fovs(
        paths_until_col, i_a, i_oa, radius)
    (path_fovs, paths_other_agent_fovs,
     paths_other_agents_fovs) = make_path_fovs(
        paths_full, paths_until_col, i_a, i_oa, t,
        radius)
    x = np.stack([obstacle_fovs, pos_other_agent_fovs,
                  path_fovs, paths_other_agent_fovs,
                  paths_other_agents_fovs], axis=3)
    return x
