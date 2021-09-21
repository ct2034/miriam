import logging
import random
from itertools import product
from typing import *

import networkx as nx
import numpy as np
import torch
from definitions import (BLOCKED_EDGES_TYPE, BLOCKED_NODES_TYPE, EDGE_TYPE,
                         PATH, C, N)
from roadmaps.var_odrm_torch.var_odrm_torch import make_graph
from scenarios.types import POTENTIAL_ENV_TYPE, is_gridmap, is_roadmap
from sim.decentralized.policy import Policy, PolicyType
from tools import hasher

logging.basicConfig()
logger = logging.getLogger(__name__)
COST = "cost"


class Agent(Generic[C, N]):
    def __init__(
        self, env: POTENTIAL_ENV_TYPE, pos: C,
        policy: PolicyType = PolicyType.RANDOM, env_nx: Optional[nx.Graph] = None
    ):
        """Initialize a new agent at a given postion `pos` using a given
        `policy` for resolution of errors."""
        self.env: POTENTIAL_ENV_TYPE = env
        if isinstance(pos, np.ndarray):
            pos = tuple(pos)
        if is_gridmap(env):
            self.has_roadmap: bool = False
            self.has_gridmap: bool = True
            assert len(pos) == 2  # (x, y)
            assert isinstance(self.env, np.ndarray), "Env must be numpy array"
            self.pos: C = pos
        elif is_roadmap(env):
            self.has_roadmap = True
            self.has_gridmap = False
            self.n_nodes = env.shape[0]
            assert len(pos) == 1  # (node)
            assert pos[0] < self.n_nodes, "Position must be a node index"
            self.pos = pos
        if env_nx is None:
            self.env_nx: nx.Graph = self.env_to_nx(self.env)
        else:
            self.env_nx = env_nx
        self.start: C = pos
        self.goal: Optional[C] = None
        self.path: Union[List[N], None] = None
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

    def env_to_nx(self, env: POTENTIAL_ENV_TYPE) -> nx.Graph:
        """convert numpy gridmap into networkx graph."""
        if self.has_gridmap:
            t = env.shape[0] * env.shape[1]
            dim = env.shape
            flat_graph = nx.grid_graph(dim, periodic=False)
            free = np.min(env)

            def filter_node(n):
                return env[n[0], n[1]] == free
            flat_graph = nx.subgraph_view(flat_graph, filter_node=filter_node)
        elif self.has_roadmap:
            flat_graph = make_graph(env)
            t = flat_graph.number_of_nodes()

        # add timed edges
        timed_graph = nx.DiGraph()
        for i_t in range(t-1):
            t_from = i_t
            t_to = i_t + 1
            for e in flat_graph.edges:
                x, y = e
                if self.has_roadmap:
                    x = (x,)
                    y = (y,)
                timed_graph.add_edge(
                    x + (t_from,),
                    y + (t_to,)
                )
                timed_graph.add_edge(
                    y + (t_from,),
                    x + (t_to,)
                )
            for n in flat_graph.nodes:
                if self.has_roadmap:
                    n = (n,)
                timed_graph.add_edge(
                    n + (t_from,),
                    n + (t_to,)
                )
        if self.has_gridmap:
            def move_cost(e):
                if (
                    e[0][:-1] == e[1][:-1]
                ):
                    # waiting generally is a little cheaper
                    return 1. - 1E-9
                else:
                    # unit cost
                    return 1
            nx.set_edge_attributes(
                timed_graph, {e: move_cost(e) for e in timed_graph.edges()}, COST)
        elif self.has_roadmap:
            HIGH_COST = 99

            def move_cost(e):
                a, b = e
                if a[:-1] != b[:-1]:  # moving
                    # geometric distance
                    return torch.linalg.vector_norm(
                        self.env[a[:-1]] - self.env[b[:-1]]
                    )
                else:
                    return HIGH_COST
            nx.set_edge_attributes(
                timed_graph, {e: move_cost(e) for e in timed_graph.edges()}, COST)
            edge_costs = list(nx.get_edge_attributes(
                timed_graph, COST).values())
            min_edge_cost = torch.min(torch.Tensor(edge_costs))
            for e in timed_graph.edges:
                a, b = e
                if a[:-1] == b[:-1]:  # waiting
                    timed_graph.edges[e][COST] = min_edge_cost*.9
            edge_costs = list(nx.get_edge_attributes(
                timed_graph, COST).values())
            max_edge_cost = torch.max(torch.Tensor(edge_costs))
            assert max_edge_cost < HIGH_COST
        return timed_graph

    def give_a_goal(self, goal: C) -> bool:
        """Set a new goal for the agent, this will calculate the path,
        if the goal is new."""
        if isinstance(goal, np.ndarray):
            goal = tuple(goal)
        if self.has_gridmap:
            assert len(goal) == 2  # (x, y)
        elif self.has_roadmap:
            assert len(goal) == 1  # (node)
        path = self.plan_timed_path(
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

    def plan_timed_path(self,
                        start: C,
                        goal: C,
                        _blocked_nodes: BLOCKED_NODES_TYPE = None,
                        _blocked_edges: BLOCKED_EDGES_TYPE = None
                        ) -> Optional[PATH]:
        if _blocked_edges is None:  # give these values
            blocked_edges = self.blocked_edges
        else:
            blocked_edges = _blocked_edges
        if _blocked_nodes is None:
            blocked_nodes = self.blocked_nodes
        else:
            blocked_nodes = _blocked_nodes

        g = nx.subgraph_view(self.env_nx)
        t_max = np.max(np.array(g.nodes())[:, -1])

        logger.debug(f"start: {start}")
        logger.debug(f"goal: {goal}")
        logger.debug(f"blocked_nodes: {blocked_nodes}")
        logger.debug(f"blocked_edges: {blocked_edges}")

        goal_waiting_edges = [
            (goal + (i,), goal + (i+1,)) for i in range(t_max-1)]
        for e in goal_waiting_edges:
            g.edges[e][COST] = 0.

        def filter_node(n):
            return n not in blocked_nodes

        def filter_edge(n1, n2):
            return (
                (n1[:-1], n2[:-1], n1[-1]) not in blocked_edges and
                (n2[:-1], n1[:-1], n1[-1]) not in blocked_edges
            )

        g_blocks = nx.subgraph_view(
            g, filter_node=filter_node, filter_edge=filter_edge)

        if self.has_gridmap:
            def dist(a, b):
                (x1, y1, _) = a
                (x2, y2, _) = b
                return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        elif self.has_roadmap:
            def dist(a, b):
                # geometric distance
                return torch.linalg.vector_norm(
                    self.env[a[:-1]] - self.env[b[:-1]]
                )

        try:
            p = list(nx.astar_path(
                g_blocks,
                start + (0,),
                goal + (t_max,),
                heuristic=dist,
                weight=COST))
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.warning(e)
            return None

        # check end to only return useful part of path
        end = None
        assert p[-1][:-1] == goal
        i = len(p) - 1
        while i > 0 or end is None:
            if p[i][:-1] == goal:
                end = i+1
            i -= 1
        assert end is not None
        return p[0:end]

    def is_there_path_with_node_blocks(self, blocks: BLOCKED_NODES_TYPE
                                       ) -> bool:
        """check if the agent can find a path to his goal with given
        n blocks [2, n]"""
        assert self.env_nx is not None, "Should have a env_nx"
        for b in blocks:
            self.env[b] = 1
        assert isinstance(self.env, np.ndarray), "Env must be numpy array"
        self.env_nx = self.env_to_nx(self.env)
        assert self.goal is not None
        path = self.plan_timed_path(
            start=self.pos,
            goal=self.goal)
        return path is not None

    def block_edge(self, e: EDGE_TYPE) -> bool:
        """this will make the agent block this edge. It will return `True`
        if there still is a path to the current goal. `False` otherwise."""
        assert self.env_nx is not None, "Should have a env_nx"
        tmp_blocked_edges = self.blocked_edges.union({e})

        assert self.goal is not None
        path = self.plan_timed_path(
            start=self.pos,
            goal=self.goal,
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

    def block_node(self, n: N) -> bool:
        """this will make the agent block this node. It will return `True`
        if there still is a path to the current goal. `False` otherwise."""
        assert self.env_nx is not None, "Should have a env_nx"
        if self.has_gridmap:
            assert len(n) == 3
        elif self.has_roadmap:
            assert len(n) == 2
        tmp_blocked_nodes = self.blocked_nodes.union({n})
        assert self.goal is not None
        path = self.plan_timed_path(
            start=self.pos,
            goal=self.goal,
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
            return self.path[i][:-1] == self.goal

    def get_priority(self, other_id: int) -> float:
        """Based on the selected policy, this will give the priority of this
        agent."""
        return self.policy.get_priority(other_id)

    def what_is_next_step(self) -> C:
        """Return the position where this agent would like to go next."""
        if self.is_at_goal():
            return self.pos  # stay at final pose
        else:
            assert self.path is not None, "Should have a path by now"
            assert self.path_i is not None, "Should have a path index by now"
            return self.path[self.path_i + 1][:-1]  # type: ignore

    def remove_all_blocks_and_replan(self):
        if (  # there were blocks
            len(self.blocked_edges) > 0 or
            len(self.blocked_nodes) > 0
        ):
            # resetting blocks now
            self.blocked_nodes = set()
            self.blocked_edges = set()
            self.path = self.plan_timed_path(
                start=self.goal,
                goal=self.goal
            )
            self.path_i = 0
            assert self.path is not None, "We must be successful with no blocks"

    def make_next_step(self, next_pos_to_check: C):
        """Move agent to its next step, pass that pose for clarification."""
        potential_next_pos = self.what_is_next_step()
        assert (potential_next_pos == next_pos_to_check
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
