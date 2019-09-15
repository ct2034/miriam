#!/usr/bin/env python2
import csv
import logging
import pickle
import random
import sys
from functools import reduce
from itertools import combinations, product

import imageio
import networkx as nx
import numpy as np
from adamsmap import (
    dist,
    get_edge_statistics,
    get_random_pos,
    graphs_from_posar,
    is_pixel_free,
    make_edges,
    vertex_path
)
from bresenham import bresenham
from scipy.spatial import Delaunay

from adamsmap_eval.filename_verification import (
    is_result_file,
    is_eval_file,
    resolve_mapname,
    resolve
)

logging.basicConfig(level=logging.INFO)

# how bigger than its size should the robot sense?
SENSE_FACTOR = 0.


def eval_disc(batch_, g, posar_, agent_diameter_, v_):
    """
    Evaluating a given graph by simulating disc-shaped robots to travel through
    it. The robots move to their individual goals and whenever they would
    collide, they stop based on an arbitrary priority.
    :param batch_: a batch of start / goal pairs for agents
    :param g: the graph to plan on (undirected)
    :param posar_: poses of the graph nodes
    :param agent_diameter_: how big is the agents disc
    :param v_: speed of travel
    :return: sum of costs, paths
    """
    sim_paths = simulate_paths_synch(batch_, g, posar_, v_)
    t_end, sim_paths_coll = simulate_paths_and_waiting(sim_paths, agent_diameter_)
    return float(sum(t_end)) / batch_.shape[0], sim_paths_coll


def eval_graph(batch_, g, posar_):
    """
    Evaluating a given graph by planning graph based motions
    :param batch_: a batch of start / goal pairs for agents
    :param g: the graph to plan on (undirected)
    :param posar_: poses of the graph nodes
    :return: sum of costs, paths
    """
    vertex_paths = []
    n_agents = batch_.shape[0]
    for i_a in range(batch_.shape[0]):
        p = vertex_path(g, batch_[i_a, 0], batch_[i_a, 1], posar_)
        vertex_paths.append(p)
    vertex_paths_synced = synchronize_paths(vertex_paths)
    for i_a, p in enumerate(vertex_paths_synced):
        if p is None:
            vertex_paths_synced[i_a] = []
    return sum(map(len, vertex_paths_synced)) / n_agents, vertex_paths_synced


def get_collisions(positions):
    """
    find agents that are colliding
    :param positions: the positions per agent
    :return: a dict with agents per colliding vertex
    """
    colls = {}
    for i_a, vertex in enumerate(positions):
        if vertex != -1:  # -1 if there is no position for different reasons
            if vertex in colls.keys():
                colls[vertex].append(i_a)
            else:
                colls[vertex] = [i_a]
    to_delete = list(filter(lambda x: len(colls[x]) == 1, colls.keys()))
    for k in to_delete:
        colls.pop(k)
    all_colls = sorted(list(reduce(lambda x, y: x + y, colls.values(), [])))
    return colls, all_colls


def synchronize_paths(vertex_paths):
    """
    make sure no two agents are at the same vertex at the same time by making them wait
    :param vertex_paths: the paths to check
    :return: the paths with waiting
    """
    n_agents = len(vertex_paths)
    i_per_agent = [-1] * n_agents
    finished = [False] * n_agents
    out_paths = [[] for _ in range(n_agents)]
    prev_i_per_agent = [-2] * n_agents
    next_poss = [-1] * n_agents
    for i_a in range(n_agents):
        if vertex_paths[i_a] is None:
            finished[i_a] = True
            out_paths[i_a] = None
    while not all(finished):
        assert len(get_collisions(next_poss)[1]) == 0, str(next_poss)
        assert prev_i_per_agent != i_per_agent
        prev_i_per_agent = list(i_per_agent)
        # trying to just increment
        blocked = [False] * n_agents
        prev_blocked = [False] * n_agents
        next_poss, i_per_agent, finished = make_next_poss(prev_i_per_agent, blocked, vertex_paths)
        next_coll, all_coll = get_collisions(next_poss)
        i = 1
        while len(next_coll) or all(np.logical_or(blocked, finished)) and not all(finished):
            # okay, solve the collisions
            if all(np.logical_or(blocked, finished)):
                blocked = to_block(n_agents, next_coll, all_coll)
            else:
                blocked = to_block(n_agents, next_coll, all_coll)
                blocked = list(np.logical_or(blocked, prev_blocked))
            prev_blocked = list(blocked)
            next_poss, i_per_agent, finished = make_next_poss(prev_i_per_agent, blocked, vertex_paths)
            next_coll, all_coll = get_collisions(next_poss)
            i += 1
        # collisions are ok
        for i_a in range(n_agents):
            if not finished[i_a]:
                out_paths[i_a].append(next_poss[i_a])
        logging.debug("i_per_agent:" + str(i_per_agent))
        logging.debug("finished:" + str(finished))
        logging.debug("=" * 10)
    for i_a in range(n_agents):
        if out_paths[i_a] is not None and vertex_paths[i_a] is not None:
            assert len(out_paths[i_a]) >= len(vertex_paths[i_a])  # after sync it may be longer but not shorter
        else:  # either none or both ar None
            assert out_paths[i_a] is None
            assert vertex_paths[i_a] is None
    return out_paths


def to_block(n_agents, next_coll, all_coll):
    n_to_block = len(all_coll) - len(next_coll)
    to_block_of_all_coll = [True] * n_to_block + [False] * (len(all_coll) - n_to_block)
    random.shuffle(to_block_of_all_coll)
    blocked = [False] * n_agents
    for i_b, i_a in enumerate(all_coll):
        blocked[i_a] = to_block_of_all_coll[i_b]
    return blocked


def make_next_poss(prev_i_per_agent, blocked, vertex_paths):
    n_agents = len(prev_i_per_agent)
    next_poss = [-1] * n_agents
    i_per_agent = [-1] * n_agents
    finished = [False] * n_agents
    for i_a in range(n_agents):
        if vertex_paths[i_a] is not None:
            if prev_i_per_agent[i_a] + 1 < len(vertex_paths[i_a]) and not blocked[i_a]:
                i_per_agent[i_a] = prev_i_per_agent[i_a] + 1
                next_poss[i_a] = vertex_paths[i_a][i_per_agent[i_a]]
            elif prev_i_per_agent[i_a] + 1 >= len(vertex_paths[i_a]):
                i_per_agent[i_a] = prev_i_per_agent[i_a]
                next_poss[i_a] = -1
                finished[i_a] = True
            elif blocked[i_a]:
                i_per_agent[i_a] = prev_i_per_agent[i_a]
                next_poss[i_a] = vertex_paths[i_a][i_per_agent[i_a]]
            else:
                assert False
        else:  # vertex_paths[i_a] is None:
            finished[i_a] = True
    return next_poss, i_per_agent, finished


def iterate_poss(current_poss, finished, i_per_agent, n_agents, vertex_paths):
    """
    Check if agents ar finished or assign position based on paths and indices
    :param current_poss: the current pos to be set
    :param finished: to set which one is finished
    :param i_per_agent: index per agent in the paths
    :param n_agents: how many agents are there?
    :param vertex_paths: the independent paths
    """
    for i_a in range(n_agents):
        if vertex_paths[i_a] is not None:
            if i_per_agent[i_a] >= len(vertex_paths[i_a]):
                finished[i_a] = True
            else:
                current_poss[i_a] = vertex_paths[i_a][i_per_agent[i_a]]


def simulate_paths_synch(batch_, g, posar_, v_):
    """
    simulate the paths
    :param batch_: a batch of start / goal pairs for agents
    :param g: the graph to plan on (undirected)
    :param posar_: poses of the graph nodes
    :param v_: speed of travel
    :return: simulated paths
    """
    sim_paths = []
    vertex_paths = []
    for i_a in range(batch_.shape[0]):
        p = vertex_path(g, batch_[i_a, 0], batch_[i_a, 1], posar_)
        vertex_paths.append(p)
    vertex_paths_synced = synchronize_paths(vertex_paths)
    for i_a, p in enumerate(vertex_paths_synced):
        if p is not None:
            coord_p = np.array([posar_[i_p] for i_p in p])
            goal = batch_[i_a, 1]
            assert goal == p[-1], str(p) + str(batch_[i_a])
            mean_edge_length, _ = get_edge_statistics(g, posar_)
            sim_path = simulate_one_path(coord_p, v_, mean_edge_length)
            sim_paths.append(sim_path)
        else:
            logging.warn("Path failed !!")
            sim_paths.append(np.array([]))
    return sim_paths


def simulate_paths_and_waiting(sim_paths, agent_diameter_):
    """
    Simulate paths over time and let robots stop if required.
    :param sim_paths: the coordinate based paths
    :param agent_diameter_: how big is the agents disc
    :return: times when agents finished, actual paths
    """
    n_agents_ = len(sim_paths)
    sim_paths_coll = None
    ended = [False for _ in range(n_agents_)]
    waiting = [False for _ in range(n_agents_)]
    i_per_agent = [-1 for _ in range(n_agents_)]
    t_end = [0 for _ in range(n_agents_)]
    prev_i_per_agent = [0 for _ in range(n_agents_)]
    while not all(ended):
        if prev_i_per_agent == i_per_agent:
            logging.debug("e:" + str(ended))
            logging.debug("ipa:" + str(i_per_agent))
            logging.debug("pipa:" + str(prev_i_per_agent))
            logging.debug("w:" + str(waiting))
            logging.error("deadlock")
            raise Exception("deadlock")
        prev_i_per_agent = list(i_per_agent)
        sim_paths_coll, ended, t_end, waiting, i_per_agent = iterate_sim(
            t_end, i_per_agent, sim_paths, sim_paths_coll, agent_diameter_
        )
        logging.debug(i_per_agent)
    return t_end, sim_paths_coll


def iterate_sim(t_end, i_per_agent, sim_paths, sim_paths_coll, agent_diameter_):
    """
    iterate a cycle of the simulation
    :param t_end: save time when agents ended
    :param i_per_agent: index per agent in the paths
    :param sim_paths: the coordinate based paths
    :param sim_paths_coll: the paths considering collisions
    :param agent_diameter_: how big should the agents be?
    :return: sim_paths_coll, ended, t_end, waiting, i_per_agent
    """
    n_agents_ = len(sim_paths)
    ended = [sim_paths[i].shape[0] - 1 == i_per_agent[i]
             for i in range(n_agents_)]
    time_slice = np.zeros([n_agents_, 2])
    for i_a in range(n_agents_):
        if not ended[i_a]:
            time_slice[i_a, :] = sim_paths[i_a][i_per_agent[i_a]]
        else:
            t_end[i_a] = i_per_agent[i_a]
    if sim_paths_coll is None:
        sim_paths_coll = np.array([time_slice, ])
    else:
        sim_paths_coll = np.append(sim_paths_coll,
                                   np.array([time_slice, ]),
                                   axis=0)
    waiting = [False for _ in range(n_agents_)]
    i_per_agent = [i_per_agent[i_a] + (1 if (not waiting[i_a]
                                             and not ended[i_a])
                                       else 0)
                   for i_a in range(n_agents_)]
    return sim_paths_coll, ended, t_end, waiting, i_per_agent


def write_csv(n_agents, paths, paths_type, n_trial, fname_):
    """
    Write one set of paths as csv file to be read in ROS
    :param n_agents: How many agents are simulated
    :param paths: The simulated paths
    :param paths_type: The type of algorithm {ev, random, undirected}
    :param n_trial: number of the trial {0 ...}
    :param fname_: filename of the underlying result file
    """
    logging.debug(paths.shape)
    assert is_result_file(fname_)
    fname_csv = ("res/"
                 + "_".join(resolve(fname_))
                 + "n_agents" + str(n_agents)
                 + "_paths_type" + str(paths_type)
                 + "_n_trial" + str(n_trial)
                 + ".csv")
    with open(fname_csv, 'w') as f_csv:
        writer = csv.writer(f_csv, delimiter=' ')
        for t in range(paths.shape[0]):
            line_ = []
            for agent in range(paths.shape[1]):
                for i in [0, 1]:
                    line_.append(paths[t, agent, i])
            writer.writerow(line_)


def simulate_one_path(coord_p, v_, mean_edge_length):
    """
    Simulate one agent path through coordinates.
    :param coord_p: the coordinates for the path to be followed
    :param v_: speed of travel
    :return: the path in coordinates
    """
    sim_path = []
    goal = list(coord_p[-1])
    n_per_edge = int(mean_edge_length / v_)
    for i_v in range(len(coord_p)-1):
        last = coord_p[i_v]
        delta = coord_p[i_v + 1] - last
        for i_step in range(n_per_edge):
            sim_path.append(last + delta * float(i_step) / n_per_edge)
    sim_path.append(goal)
    assert len(sim_path) == n_per_edge * (len(coord_p) - 1) + 1
    for i in range(len(sim_path)-1):
        assert np.linalg.norm(sim_path[i] - sim_path[i+1]) < 10 * v_
    return np.array(sim_path)


def get_unique_batch(N, n_agents):
    assert n_agents <= N, "Can only have as much agents as vertices"
    used_starts = set([-1])
    used_goals = set([-1])
    batch = []
    s = -1
    g = -1
    for i_a in range(n_agents):
        while s in used_starts:
            s = random.randint(0, N-1)
        used_starts.add(s)
        while g in used_goals:
            g = random.randint(0, N-1)
        used_goals.add(g)
        batch.append([s, g])
    return np.array(batch)
