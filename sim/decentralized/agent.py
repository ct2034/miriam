import logging
import random
from typing import *

import networkx as nx
import numpy as np
import torch
from definitions import (BLOCKED_EDGES_TYPE, BLOCKED_NODES_TYPE, EDGE_TYPE,
                         PATH, C, N)
from scenarios.types import POTENTIAL_ENV_TYPE, is_gridmap, is_roadmap
from sim.decentralized.policy import Policy, PolicyType
from tools import hasher

logging.basicConfig()
logger = logging.getLogger(__name__)
COST = "cost"
POS = "pos"
WAITING_COST = 1. - 1E-9
MOVING_COST = 1


def get_t_from_env(env: POTENTIAL_ENV_TYPE) -> int:
    if is_gridmap(env):
        return (env.shape[0] + env.shape[1]) * 3
    elif is_roadmap(env):
        assert isinstance(env, nx.Graph)
        return env.number_of_nodes()
    else:
        raise ValueError("Env must be gridmap or roadmap")


def env_to_nx(env: POTENTIAL_ENV_TYPE) -> nx.Graph:
    """convert numpy gridmap into networkx graph."""
    if is_gridmap(env):
        has_gridmap = True
        has_roadmap = False
    elif is_roadmap(env):
        has_gridmap = False
        has_roadmap = True
    else:
        raise ValueError("Env must be gridmap or roadmap")
    t = get_t_from_env(env)
    if has_gridmap:
        dim = env.shape
        flat_graph = nx.grid_graph(dim, periodic=False)
        free = np.min(env)

        def filter_node(n):
            return env[n[0], n[1]] == free
        flat_graph = nx.subgraph_view(flat_graph, filter_node=filter_node)
    elif has_roadmap:
        assert isinstance(env, nx.Graph)
        flat_graph = env  # is already a networkx graph

    # add timed edges
    timed_graph = nx.DiGraph()
    for i_t in range(t-1):
        t_from = i_t
        t_to = i_t + 1
        for e in flat_graph.edges:
            x, y = e
            if has_roadmap:
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
            if has_roadmap:
                n = (n,)
            timed_graph.add_edge(
                n + (t_from,),
                n + (t_to,)
            )
    if has_gridmap:
        def move_cost(e):
            if (
                e[0][:-1] == e[1][:-1]
            ):
                # waiting generally is a little cheaper
                return WAITING_COST
            else:
                # unit cost
                return MOVING_COST
        nx.set_edge_attributes(
            timed_graph, {e: move_cost(e) for e in timed_graph.edges()}, COST)
    elif has_roadmap:
        HIGH_COST = 99
        pos = nx.get_node_attributes(flat_graph, "pos")

        def move_cost(e):
            a, b = e
            if a[0] != b[0]:  # moving
                # geometric distance
                return torch.linalg.vector_norm(
                    torch.tensor(pos[a[0]]) - torch.tensor(pos[b[0]])
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
            if a[0] == b[0]:  # waiting
                timed_graph.edges[e][COST] = min_edge_cost*.99
        edge_costs = list(nx.get_edge_attributes(
            timed_graph, COST).values())
        max_edge_cost = torch.max(torch.Tensor(edge_costs))
        assert max_edge_cost < HIGH_COST
    return timed_graph


class Agent(Generic[C, N]):
    def __init__(
        self, env: POTENTIAL_ENV_TYPE, pos: C,
        policy: PolicyType = PolicyType.RANDOM,
        env_nx: Optional[nx.Graph] = None,
        radius: Optional[float] = None,
        rng: random.Random = random.Random()
    ):
        """Initialize a new agent at a given postion `pos` using a given
        `policy` for resolution of errors."""
        self.env: POTENTIAL_ENV_TYPE = env
        if isinstance(pos, np.ndarray):
            pos = tuple(pos)  # type: ignore  # dirty fix for some tests
        if is_gridmap(env):
            self.has_roadmap: bool = False
            self.has_gridmap: bool = True
            assert isinstance(pos, tuple)
            assert len(pos) == 2  # (x, y)
            assert isinstance(self.env, np.ndarray), "Env must be numpy array"
            self.pos: C = pos
            assert radius is None, "Radius not supported for gridmap"
            self.radius: Optional[float] = None
        elif is_roadmap(env):
            self.has_roadmap = True
            self.has_gridmap = False
            assert isinstance(env, nx.Graph)
            self.n_nodes = env.number_of_nodes()
            assert isinstance(pos, int)  # (node)
            assert pos < self.n_nodes, "Position must be a node index"
            self.pos = pos
            self.radius = radius
        if env_nx is None:
            self.env_nx: nx.Graph = env_to_nx(self.env)
        else:
            self.env_nx = env_nx
        self.t_max = np.max(np.array(self.env_nx.nodes())[:, -1])
        self.start: C = pos
        self.goal: Optional[C] = None
        self.path: Union[List[N], None] = None
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

        logger.debug(f"start: {start}")
        logger.debug(f"goal: {goal}")
        logger.debug(f"blocked_nodes: {blocked_nodes}")
        logger.debug(f"blocked_edges: {blocked_edges}")

        # create filters
        def filter_node(n):
            return n not in blocked_nodes

        if self.has_gridmap:
            def filter_edge(n1, n2):
                return (
                    (n1[:-1], n2[:-1], n1[-1]) not in blocked_edges and
                    (n2[:-1], n1[:-1], n1[-1]) not in blocked_edges
                )
        elif self.has_roadmap:
            def filter_edge(n1, n2):
                return (
                    ((n1[0], n1[1]), (n2[0], n2[1])) not in blocked_edges and
                    ((n2[0], n1[1]), (n1[0], n2[1])) not in blocked_edges
                )

        g = nx.subgraph_view(
            self.env_nx, filter_node=filter_node, filter_edge=filter_edge)

        # define distance function and goal edges
        if self.has_gridmap:
            def dist(a, b):
                (x1, y1, _) = a
                (x2, y2, _) = b
                return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            assert isinstance(goal, tuple)
            goal_waiting_edges = [
                (goal + (i,), goal + (i+1,)) for i in range(self.t_max-1)]

        elif self.has_roadmap:
            pos_s = nx.get_node_attributes(self.env, POS)

            def dist(a, b):
                # geometric distance
                return torch.linalg.vector_norm(
                    torch.tensor(pos_s[a[0]]) - torch.tensor(pos_s[b[0]])
                )
            assert isinstance(goal, int)
            goal_waiting_edges = [
                ((goal, i), (goal, i+1)) for i in range(self.t_max)]

        # make goal waiting edges free
        any_goal_edge_existed = False
        for e in goal_waiting_edges:
            try:
                g.edges[e][COST] = 0.
                any_goal_edge_existed = True
            except KeyError:
                pass
        if not any_goal_edge_existed:
            return None

        # define start and goal for timed graph
        if self.has_gridmap:
            assert isinstance(start, tuple)
            assert isinstance(goal, tuple)
            start_t = start + (0,)  # type: ignore
            goal_t = goal + (self.t_max,)  # type: ignore
        elif self.has_roadmap:
            assert isinstance(start, int)
            assert isinstance(goal, int)
            start_t = (start, 0)  # type: ignore
            goal_t = (goal, self.t_max)  # type: ignore

        # plan path
        try:
            p = list(nx.astar_path(
                g,
                start_t,
                goal_t,
                heuristic=dist,
                weight=COST))
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.warning(e)
            return None
        finally:
            # restore goal waiting edges
            for e in goal_waiting_edges:  # type: ignore
                if e in self.env_nx.edges:
                    self.env_nx.edges[e][COST] = WAITING_COST

        # check end to only return useful part of path
        end = None
        if self.has_gridmap:
            assert p[-1][: -1] == goal
            i = len(p) - 1
            while i >= 0 or end is None:
                if p[i][: -1] == goal:
                    end = i+1
                i -= 1
        elif self.has_roadmap:
            assert p[-1][0] == goal
            i = len(p) - 1
            while i >= 0 or end is None:
                if p[i][0] == goal:
                    end = i+1
                i -= 1
        assert end is not None
        return p[0: end]

    def is_there_path_with_node_blocks(self, blocks: BLOCKED_NODES_TYPE
                                       ) -> bool:
        """check if the agent can find a path to his goal with given
        n blocks [2, n]"""
        assert self.env_nx is not None, "Should have a env_nx"
        for b in blocks:
            self.env[b] = 1
        assert isinstance(self.env, np.ndarray), "Env must be numpy array"
        self.env_nx = env_to_nx(self.env)
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
        # agent at goal can not block this node:
        if self.is_at_goal():
            if self.goal == n[:-1]:
                return False
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
            if self.has_gridmap:
                return self.path[self.path_i + 1][:-1]  # type: ignore
            elif self.has_roadmap:
                return int(self.path[self.path_i + 1][0])  # type: ignore
            else:
                raise RuntimeError

    def remove_all_blocks_and_replan(self):
        if (  # there were blocks
            len(self.blocked_edges) > 0 or
            len(self.blocked_nodes) > 0
        ) or (  # at a position that is not in path
            self.path_i is not None and
            self.path is not None and
            (self.path_i >= len(self.path) or
             self.path[self.path_i] != (self.pos, self.path_i))
        ):
            # resetting blocks now
            self.blocked_nodes = set()
            self.blocked_edges = set()
            self.path = self.plan_timed_path(
                start=self.pos,
                goal=self.goal
            )
            self.path_i = 0
            assert self.path is not None, "We must be successful with no blocks"

    def back_to_the_start(self):
        """Reset current progress and place agent at its start as if nothing ever 
        happened."""
        self.pos = self.start
        self.blocked_nodes = set()
        self.blocked_edges = set()
        self.path = self.plan_timed_path(
            start=self.pos,
            goal=self.goal
        )
        self.path_i = 0

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
