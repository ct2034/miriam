import logging
import random
from itertools import product
from typing import *

import networkx as nx
import numpy as np
import torch
from definitions import (BLOCKED_EDGES_TYPE, BLOCKED_NODES_TYPE, EDGE_TYPE,
                         FREE, INVALID, PATH, C, C_grid, N)
from planner.astar_boost.build.libastar_graph import AstarSolver
from scenarios.types import (COORD_TO_NODE_TYPE, POTENTIAL_ENV_TYPE,
                             is_gridmap, is_roadmap)
from sim.decentralized.policy import Policy, PolicyType
from tools import hasher

logging.basicConfig()
logger = logging.getLogger(__name__)
COST = "cost"
POS = "pos"


def gridmap_to_graph(gridmap: np.ndarray) -> Tuple[
        nx.Graph, COORD_TO_NODE_TYPE]:
    w, h = gridmap.shape

    # transformation between coordinates and node numbers
    coord_to_node: COORD_TO_NODE_TYPE = {}
    node_to_coord: Dict[int, Tuple[float, float]] = {}
    for x, y in product(range(w), range(h)):
        node = x + y * w
        coord_to_node[(x, y)] = node
        node_to_coord[node] = (float(x), float(y))

    # create graph
    g = nx.Graph()
    for x, y in product(range(w), range(h)):
        if gridmap[x, y] == FREE:
            g.add_node(coord_to_node[(x, y)])
            if x > 0 and gridmap[x - 1, y] == FREE:
                g.add_edge(coord_to_node[(x, y)], coord_to_node[(x - 1, y)])
            if y > 0 and gridmap[x, y - 1] == FREE:
                g.add_edge(coord_to_node[(x, y)], coord_to_node[(x, y - 1)])

    nx.set_node_attributes(g, POS, node_to_coord)
    return g, coord_to_node


class Agent(object):
    def __init__(
        self, env: POTENTIAL_ENV_TYPE, pos: Union[C, C_grid],
        policy: PolicyType = PolicyType.LEARNED,
        radius: Optional[float] = None,
        rng: random.Random = random.Random()
    ):
        """Initialize a new agent at a given postion `pos` using a given
        `policy` for resolution of errors."""
        if isinstance(pos, np.ndarray):
            pos = tuple(pos)  # type: ignore  # dirty fix for some tests
        if is_gridmap(env):
            self.has_roadmap: bool = False
            self.has_gridmap: bool = True
            assert isinstance(env, np.ndarray), "Env must be numpy array"
            (self.env, self.coord_to_node
             ) = gridmap_to_graph(env)
            assert isinstance(pos, tuple)
            self.pos = self.coord_to_node[pos]
            assert len(pos) == 2  # (x, y)self.pos: C = pos
            assert radius is None, "Radius not supported for gridmap"
            self.radius: Optional[float] = None
        elif is_roadmap(env):
            self.has_roadmap = True
            self.has_gridmap = False
            assert isinstance(env, nx.Graph)
            self.env = env
            self.n_nodes = env.number_of_nodes()
            assert isinstance(pos, int)  # (node)
            assert pos < self.n_nodes, "Position must be a node index"
            self.pos = pos
            self.radius = radius
        self.astar_solver = AstarSolver(self.env)
        self.start: C = self.pos
        self.goal: Optional[C] = None
        self.path: Union[PATH, None] = None
        self.path_i: Union[int, None] = None
        self.policy: Policy = Policy.construct_by_type(policy, self)
        self.rng = rng
        self.id: int = self.rng.randint(0, int(2E14))
        self.blocked_edges: BLOCKED_EDGES_TYPE = set()
        self.blocked_nodes: BLOCKED_NODES_TYPE = set()

    def __hash__(self):
        return hash(hasher([
            f"start: {self.start}\n",
            f"goal: {self.goal}\n",
            f"policy: {self.policy}\n",
            f"hash(env): {hasher(self.env)}"
        ]))  # should not change over time

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return (
            f"id: {self.id}\n" +
            f"start: {self.start}\n" +
            f"goal: {self.goal}\n" +
            f"pos: {self.pos}\n" +
            f"policy: {self.policy}\n" +
            f"hash(env): {hasher(self.env)}"
        )

    def give_a_goal(self, goal: C) -> bool:
        """Set a new goal for the agent, this will calculate the path,
        if the goal is new."""
        if isinstance(goal, np.ndarray):
            goal = tuple(goal)  # type: ignore  # dirty fix for some tests
        if self.has_gridmap:
            assert isinstance(goal, tuple)
            assert len(goal) == 2  # (x, y)
        elif self.has_roadmap:
            assert isinstance(goal, int)  # (node)
        path = self.plan_path(
            start=self.pos,
            goal=goal
        )
        if path is not None:
            self.goal = goal
            self.path = path
            self.path_i = 0
            return True
        else:
            return False  # there is no path to this goal

    def plan_path(self, start: C, goal: C) -> Optional[PATH]:
        """Plan a path from `start` to `goal`."""
        p = self.astar_solver(start, goal)
        if len(p) == 0:
            return None
        return p

    def is_at_goal(self, dt: Optional[int] = None):
        """returns true iff the agent is at its goal at delta time `dt` from now."""
        if self.pos == self.goal:
            return True
        if self.path_i is None or self.path is None:
            return False
        if dt is None:
            dt = 0
        i = self.path_i + dt
        if i >= len(self.path):
            return True
        else:
            return self.path[i] == self.goal

    def what_is_next_step(self) -> C:
        """Return the position where this agent would like to go next."""
        if self.is_at_goal():
            return self.pos  # stay at final pose
        else:
            assert self.path is not None, "Should have a path by now"
            assert self.path_i is not None, "Should have a path index by now"
            if self.has_gridmap:
                return self.path[self.path_i + 1][:-1]  # type: ignore
            elif self.has_roadmap:
                return int(self.path[self.path_i + 1][0])  # type: ignore
            else:
                raise RuntimeError

    def replan(self):
        """Replan the path to the current goal."""
        assert self.goal is not None, "Should have a goal to replan"
        self.path = self.plan_path(
            start=self.pos,
            goal=self.goal
        )
        self.path_i = 0
        assert self.path is not None, "We must be successful with no blocks"

    def back_to_the_start(self):
        """Reset current progress and place agent at its start as if nothing ever 
        happened."""
        self.pos = self.start
        self.replan()

    def make_next_step(self, next_pos_to_check: C):
        """Move agent to its next step, pass that pose for clarification."""
        potential_next_pos = self.what_is_next_step()
        if self.has_gridmap:
            assert (potential_next_pos == next_pos_to_check
                    ), "Our next position has to be correct."
        if self.is_at_goal():
            pass
        else:  # not at goal yet
            assert self.path_i is not None, "Should have a path_i by now"
            self.path_i += 1
            self.pos = potential_next_pos

    def make_this_step(self, pos_to_go_to: C):
        """Move agent to the given position. (Ignoring the path)
        Motion must be possible by the environment."""
        if self.has_gridmap:
            assert isinstance(pos_to_go_to, tuple), "Should be a tuple"
            assert len(pos_to_go_to) == 2, "Should have 2 dims"
            raise NotImplementedError
        elif self.has_roadmap:
            assert isinstance(pos_to_go_to, int), "Should be an int"
            assert isinstance(self.env, nx.Graph), "Should be a nx graph"
            if self.pos != pos_to_go_to:
                if not self.env.has_edge(
                        self.pos, pos_to_go_to):
                    raise RuntimeError(
                        "Should have edge from current pos to pos_to_go_to")
            assert self.path_i is not None, "Should have a path_i by now"
            self.path_i += 1
        self.pos = pos_to_go_to

    def get_path_i_not_none(self) -> int:
        assert self.path_i is not None
        return self.path_i
