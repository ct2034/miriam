import logging
import random
import uuid
from enum import Enum
from typing import List, Tuple

import networkx as nx
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)

def gridmap_to_nx(env: np.ndarray) -> nx.Graph:
    """convert numpy gridmap into networkx graph."""
    g = nx.grid_graph(dim=list(env.shape))
    obstacles = np.where(env == 1)
    for i_o in range(len(obstacles[0])):
        g.remove_node(
            (obstacles[0][i_o],
                obstacles[1][i_o])
        )
    return g


class Policy(Enum):
    RANDOM = 0
    CLOSEST = 1
    FILL = 2


class Agent():
    def __init__(
        self, env: np.ndarray, pos: np.ndarray,
        policy: Policy = Policy.RANDOM
    ):
        """Initialize a new agent at a given postion `pos` using a given
        `policy` for resolution of errors."""
        self.env = env
        self.env_nx = None
        self.pos = pos
        self.goal = None
        self.policy = policy
        self.path = None
        self.path_i = None
        self.uuid = uuid.uuid1()

    def __hash__(self):
        return hash(
            hash(str(self.pos) + "pos") +
            hash(str(self.goal) + "goal") +
            hash(str(self.policy) + "policy") +
            hash(str(self.uuid) + "uuid")
        )

    def __eq__(self, other):
        return hash(self) == hash(other)

    def give_a_goal(self, goal: np.ndarray) -> bool:
        """Set a new goal for the agent, this will calculate the path,
        if the goal is new."""
        if (self.goal != goal).any():  # goal is new
            self.goal = goal
            self.env_nx = gridmap_to_nx(self.env)
            path = self.plan_path()
            if path is not None:
                self.path = path
                self.path_i = 0
                return True
            else:
                return False  # there is no path to this goal
        else:  # still have old goal and path
            return True

    def plan_path(self, env_nx: nx.Graph = None) -> bool:
        """Plan path from currently set `pos` to current `goal` and save it
        in `path`."""
        if env_nx is None:
            env_nx = self.env_nx
        try:
            tuple_path = nx.shortest_path(
                env_nx, tuple(self.pos), tuple(self.goal))
        except nx.exception.NetworkXNoPath as e:
            logger.warning(e)
            return None
        except nx.exception.NodeNotFound as e:
            logger.warning(e)
            return None
        return np.array(tuple_path)

    def is_there_path_with_node_blocks(self, blocks: List[Tuple[int]]) -> bool:
        """check if the agent can find a path to his goal with given n blocks [2, n]"""
        tmp_env = self.env_nx.copy()
        tmp_env.remove_nodes_from(blocks)
        path = self.plan_path(tmp_env)
        return path is not None

    def block_edge(self, a: tuple, b: tuple) -> bool:
        """this will make the agent block this edge. It will return `Treu`
        if there still is a path to the current goal. `False` otherwise."""
        old_graph = self.env_nx.copy()
        try:
            self.env_nx.remove_edge(a, b)
        except nx.exception.NetworkXError:
            logger.warning("Edge already removed")
        path = self.plan_path()
        if path is not None:
            # all good, and we have a new path now
            self.path = path
            self.path_i = 0
            return True
        else:
            # undo changes
            self.env_nx = old_graph.copy()
            return False

    def is_at_goal(self):
        """returns true iff the agent is at its goal."""
        return all(self.pos == self.goal) or self.path is None

    def get_priority(self):
        """Based on the selected policy, this will give the priority of this
        agent."""
        if self.policy == Policy.RANDOM:
            # simply returning a random number on every call
            return random.random()
        elif self.policy == Policy.CLOSEST:
            # a policy that prefers the agent that is currently closest to its
            # goal.
            return 1. / np.linalg.norm(self.goal - self.pos)
        elif self.policy == Policy.FILL:
            # an agent will get a higher prio if the map around it is fuller of
            # obstacles.
            FILL_RADIUS = 2
            n_total = ((FILL_RADIUS * 2 + 1)**2)
            n_free = 0
            for x in range(max(0, self.pos[0] - FILL_RADIUS),
                           min(self.env.shape[0],
                               self.pos[0] + FILL_RADIUS + 1)):
                for y in range(max(0, self.pos[1] - FILL_RADIUS),
                               min(self.env.shape[1],
                                   self.pos[1] + FILL_RADIUS + 1)):
                    if self.env[x, y] == 0:
                        n_free += 1
            return float(n_total - n_free) / n_total

    def what_is_next_step(self) -> np.ndarray:
        """Return the position where this agent would like to go next."""
        if self.is_at_goal():
            return self.pos  # stay at final pose
        else:
            return self.path[self.path_i + 1]

    def make_next_step(self, next_pos_to_check: np.ndarray):
        """Move agent to its next step, pass that pose for clarification."""
        potential_next_pos = self.what_is_next_step()
        assert (potential_next_pos == next_pos_to_check).all(
        ), "Our next position has to be corect."
        if self.is_at_goal():
            pass  # at goal already
        else:
            self.path_i += 1
            self.pos = potential_next_pos
