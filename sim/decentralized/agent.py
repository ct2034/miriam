import logging
import random
from itertools import product
from typing import *

import networkx as nx
import numpy as np
from sim.decentralized.policy import Policy, PolicyType
from tools import hasher

logging.basicConfig()
logger = logging.getLogger(__name__)

EDGE_TYPE = Tuple[Tuple[int, int], Tuple[int, int], int]
BLOCKED_EDGES_TYPE = Set[EDGE_TYPE]
NODE_TYPE = Tuple[int, int, int]
BLOCKED_NODES_TYPE = Set[NODE_TYPE]


class Agent():
    def __init__(
        self, env: np.ndarray, pos: np.ndarray,
        policy: PolicyType = PolicyType.RANDOM
    ):
        """Initialize a new agent at a given postion `pos` using a given
        `policy` for resolution of errors."""
        self.env: np.ndarray = env
        self.env_nx: nx.Graph = self.gridmap_to_nx(self.env)
        assert isinstance(pos, np.ndarray), "Position must be numpy array"
        self.pos: np.ndarray = pos
        self.start: np.ndarray = pos
        self.goal: Union[np.ndarray, None] = None
        self.path: Union[np.ndarray, None] = None
        self.path_i: Union[int, None] = None
        self.policy: Policy = Policy.construct_by_type(policy, self)
        self.id: int = random.randint(0, int(2E14))
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

    def gridmap_to_nx(self, env: np.ndarray) -> nx.Graph:
        """convert numpy gridmap into networkx graph."""
        t = env.shape[0] * env.shape[1]
        dim = (t,) + env.shape
        g = nx.DiGraph(nx.grid_graph(dim, periodic=False))
        free = np.min(env)

        for i_t in range(t-1):
            t_from = i_t
            t_to = i_t + 1
            for x, y in product(range(env.shape[0]), range(env.shape[1])):
                for x_to in [x-1, x+1]:
                    n_to = (x_to, y, t_to)
                    if n_to in g.nodes():
                        g.add_edge((x, y, t_from), n_to)
                for y_to in [y-1, y+1]:
                    n_to = (x, y_to, t_to)
                    if n_to in g.nodes():
                        g.add_edge((x, y, t_from), n_to)

        def cost(e):
            if (
                e[0][0] == e[1][0] and
                e[0][1] == e[1][1]
            ):
                # waiting generally is a little cheaper
                return 1. - 1E-9
            else:
                # normal cost
                return 1

        nx.set_edge_attributes(g, {e: cost(e) for e in g.edges()}, "cost")

        def filter_node(n):
            return env[n[0], n[1]] == free

        def filter_edge(n1, n2):
            return n2[2] > n1[2]

        return nx.DiGraph(
            nx.subgraph_view(g, filter_node, filter_edge))

    def give_a_goal(self, goal: np.ndarray) -> bool:
        """Set a new goal for the agent, this will calculate the path,
        if the goal is new."""
        path = self.plan_timed_path(
            start=(self.pos[0], self.pos[1]),
            goal=(goal[0], goal[1])
        )
        if path is not None:
            self.goal = goal
            self.path = path
            self.path_i = 0
            return True
        else:
            return False  # there is no path to this goal

    def plan_timed_path(self,
                        start: Tuple[int, int],
                        goal: Tuple[int, int],
                        _blocked_nodes: Set[Tuple[int, int, int]] = None,
                        _blocked_edges: Set[
                            Tuple[Tuple[int, int],
                                  Tuple[int, int],
                                  int]] = None):
        if _blocked_edges is None:  # give these values
            blocked_edges = self.blocked_edges
        else:
            blocked_edges = _blocked_edges
        if _blocked_nodes is None:
            blocked_nodes = self.blocked_nodes
        else:
            blocked_nodes = _blocked_nodes

        g = self.env_nx
        t_max = np.max(np.array(g.nodes())[:, 2])

        logger.debug(f"start: {start}")
        logger.debug(f"goal: {goal}")
        logger.debug(f"blocked_nodes: {blocked_nodes}")
        logger.debug(f"blocked_edges: {blocked_edges}")

        goal_waiting_edges = [
            ((goal[0], goal[1], i), (goal[0], goal[1], i+1)) for i in range(t_max-1)]

        nx.set_edge_attributes(g, {e: 0 for e in goal_waiting_edges}, "cost")

        def filter_node(n):
            return n not in blocked_nodes

        def filter_edge(n1, n2):
            return (
                (n1[:2], n2[:2], n1[2]) not in blocked_edges and
                (n2[:2], n1[:2], n1[2]) not in blocked_edges
            )

        g_blocks = nx.subgraph_view(
            g, filter_node=filter_node, filter_edge=filter_edge)

        def dist(a, b):
            (x1, y1, _) = a
            (x2, y2, _) = b
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        try:
            p = np.array(nx.astar_path(
                g_blocks,
                start + (0,),
                goal + (t_max,),
                heuristic=dist,
                weight="cost"))
        except nx.NetworkXNoPath as e:
            logger.warning(e)
            return None
        except nx.NodeNotFound as e:
            logger.warning(e)
            return None

        # check end to only return useful part of path
        end = None
        assert all(p[-1][:2] == goal)
        i = len(p) - 1
        while i > 0 or end is None:
            if all(p[i][:2] == goal):
                end = i+1
            i -= 1
        assert end is not None
        return p[0:end]

    def is_there_path_with_node_blocks(self, blocks: List[Tuple[Any, ...]]
                                       ) -> bool:
        """check if the agent can find a path to his goal with given
        n blocks [2, n]"""
        assert self.env_nx is not None, "Should have a env_nx"
        for b in blocks:
            self.env[b] = 1
        self.env_nx = self.gridmap_to_nx(self.env)
        assert self.goal is not None
        path = self.plan_timed_path(
            start=(self.pos[0], self.pos[1]),
            goal=(self.goal[0], self.goal[1]))
        return path is not None

    def block_edge(self, e: EDGE_TYPE) -> bool:
        """this will make the agent block this edge. It will return `True`
        if there still is a path to the current goal. `False` otherwise."""
        assert self.env_nx is not None, "Should have a env_nx"
        tmp_blocked_edges = self.blocked_edges.union({e})

        assert self.goal is not None
        path = self.plan_timed_path(
            start=(self.pos[0], self.pos[1]),
            goal=(self.goal[0], self.goal[1]),
            _blocked_nodes=None,
            _blocked_edges=tmp_blocked_edges)
        if path is not None:
            # all good, and we have a new path now
            self.path = path
            self.path_i = 0
            self.blocked_edges.add(e)
            return True
        else:
            return False

    def block_node(self, n: NODE_TYPE) -> bool:
        """this will make the agent block this node. It will return `True`
        if there still is a path to the current goal. `False` otherwise."""
        assert self.env_nx is not None, "Should have a env_nx"
        assert len(n) == 3
        tmp_blocked_nodes = self.blocked_nodes.union({n})
        assert self.goal is not None
        path = self.plan_timed_path(
            start=(self.pos[0], self.pos[1]),
            goal=(self.goal[0], self.goal[1]),
            _blocked_nodes=tmp_blocked_nodes,
            _blocked_edges=None)
        if path is not None:
            # all good, and we have a new path now
            self.path = path
            self.path_i = 0
            self.blocked_nodes.add(n)
            return True
        else:
            return False

    def is_at_goal(self, dt: Optional[int] = None):
        """returns true iff the agent is at its goal at delta time `dt` from now."""
        if all(self.pos == self.goal):
            return True
        if self.path_i is None or self.path is None:
            return False
        if dt is None:
            dt = 0
        i = self.path_i + dt
        if i >= len(self.path):
            return True
        else:
            return all(self.path[i, :2] == self.goal)

    def get_priority(self, other_id) -> float:
        """Based on the selected policy, this will give the priority of this
        agent."""
        return self.policy.get_priority(other_id)

    def what_is_next_step(self) -> np.ndarray:
        """Return the position where this agent would like to go next."""
        if self.is_at_goal():
            return self.pos  # stay at final pose
        else:
            assert self.path is not None, "Should have a path by now"
            assert self.path_i is not None, "Should have a path index by now"
            return self.path[self.path_i + 1, :2]

    def remove_all_blocks_and_replan(self):
        # resetting blocks now
        self.blocked_nodes = set()
        self.blocked_edges = set()
        self.path = self.plan_timed_path(  # TODO: only replan if we actually cleared lists
            start=(self.pos[0], self.pos[1]),
            goal=(self.goal[0], self.goal[1])
        )
        self.path_i = 0
        assert self.path is not None, "We must be successful with no blocks"

    def make_next_step(self, next_pos_to_check: np.ndarray):
        """Move agent to its next step, pass that pose for clarification."""
        potential_next_pos = self.what_is_next_step()
        assert (potential_next_pos == next_pos_to_check).all(
        ), "Our next position has to be correct."
        if self.is_at_goal():
            pass
        else:  # not at goal yet
            assert self.path_i is not None, "Should have a path_i by now"
            self.path_i += 1
            self.pos = potential_next_pos

    def get_path_i_not_none(self) -> int:
        assert self.path_i is not None
        return self.path_i
