import logging
import random
from enum import Enum
from typing import *

import networkx as nx
import numpy as np
from cachier import cachier

import tools

logging.basicConfig()
logger = logging.getLogger(__name__)

BLOCKED_EDGES_TYPE = Set[Tuple[Tuple[int, int], Tuple[int, int]]]


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
        self.env: np.ndarray = env
        self.env_nx: Union[nx.Graph, None] = None
        self.pos: np.ndarray = np.array(pos)
        self.goal: Union[np.ndarray, None] = None
        self.policy: Policy = policy
        self.path = None
        self.path_i: Union[int, None] = None
        self.id: int = random.randint(0, int(2E14))
        self.blocked_edges: BLOCKED_EDGES_TYPE = set()

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return "".join(map(str, [
            self.env,
            self.pos,
            self.goal,
            self.policy,
            self.id]))

    def filter_node(self, n):
        """node filter for gridmap_to_nx"""
        return self.env[n] == 0

    def filter_edge(self, a, b):
        """edge filter for gridmap_to_nx"""
        return (a, b) not in self.blocked_edges

    def gridmap_to_nx(self, env: np.ndarray) -> nx.Graph:
        """convert numpy gridmap into networkx graph."""
        nx_base_graph = nx.grid_graph(dim=list(env.shape))
        assert len(env.shape) == 2
        assert nx_base_graph.number_of_nodes() == env.shape[0] * env.shape[1]
        view = nx.subgraph_view(
            nx_base_graph, filter_node=self.filter_node, filter_edge=self.filter_edge)
        return view

    def give_a_goal(self, goal: np.ndarray) -> bool:
        """Set a new goal for the agent, this will calculate the path,
        if the goal is new."""
        if (self.goal != goal).any():  # goal is new
            self.goal = goal
            self.env_nx = self.gridmap_to_nx(self.env)
            path = self.plan_path()
            if path is not None:
                self.path = path
                self.path_i = 0
                return True
            else:
                return False  # there is no path to this goal
        else:  # still have old goal and path
            return True

    def plan_path(self, env_nx: Union[nx.Graph, None] = None
                  ) -> Union[np.ndarray, None]:
        """Plan path from currently set `pos` to current `goal` and save it
        in `path`."""
        if env_nx is None:
            env_nx = self.env_nx
        try:
            assert self.goal is not None, "Should have a goal"
            tuple_path = nx.shortest_path(
                env_nx, tuple(self.pos), tuple(self.goal))
        except nx.exception.NetworkXNoPath as e:
            logger.warning(e)
            return None
        except nx.exception.NodeNotFound as e:
            logger.warning(e)
            return None
        return np.array(tuple_path)

    def is_there_path_with_node_blocks(self, blocks: List[Tuple[Any, ...]]
                                       ) -> bool:
        """check if the agent can find a path to his goal with given
        n blocks [2, n]"""
        assert self.env_nx is not None, "Should have a env_nx"
        tmp_env = self.env_nx.copy()
        tmp_env.remove_nodes_from(blocks)
        path = self.plan_path(tmp_env)
        return path is not None

    def block_edge(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """this will make the agent block this edge. It will return `True`
        if there still is a path to the current goal. `False` otherwise."""
        assert self.env_nx is not None, "Should have a env_nx"
        tmp_edge = (a, b)
        save_blocked_edges = self.blocked_edges.copy()
        self.blocked_edges.add(tmp_edge)
        tmp_env_nx = self.gridmap_to_nx(
            self.env)

        path = self.plan_path(tmp_env_nx)
        if path is not None:
            # all good, and we have a new path now
            self.path = path
            self.path_i = 0
            self.env_nx = self.gridmap_to_nx(self.env)
            return True
        else:
            # forget changes
            self.blocked_edges = save_blocked_edges
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
            assert self.path is not None, "Should have a path by now"
            return self.path[self.path_i + 1]

    def make_next_step(self, next_pos_to_check: np.ndarray):
        """Move agent to its next step, pass that pose for clarification."""
        potential_next_pos = self.what_is_next_step()
        assert (potential_next_pos == next_pos_to_check).all(
        ), "Our next position has to be correct."
        if self.is_at_goal():
            pass  # at goal already
        else:
            assert self.path_i is not None, "Should have a path_i by now"
            self.path_i += 1
            self.pos = potential_next_pos
