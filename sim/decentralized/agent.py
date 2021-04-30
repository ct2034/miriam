import logging
import random
from typing import *

import networkx as nx
import numpy as np
from sim.decentralized.policy import Policy, PolicyType

logging.basicConfig()
logger = logging.getLogger(__name__)

BLOCKED_EDGES_TYPE = Set[Tuple[Tuple[int, int], Tuple[int, int]]]
BLOCKED_NODES_TYPE = Set[Tuple[int, int]]


class Agent():
    def __init__(
        self, env: np.ndarray, pos: np.ndarray,
        policy: PolicyType = PolicyType.RANDOM
    ):
        """Initialize a new agent at a given postion `pos` using a given
        `policy` for resolution of errors."""
        self.env: np.ndarray = env
        self.env_nx: Union[nx.Graph, None] = None
        assert isinstance(pos, np.ndarray), "Position must be numpy array"
        self.pos: np.ndarray = pos
        self.goal: Union[np.ndarray, None] = None
        self.path: Union[np.ndarray, None] = None
        self.path_i: Union[int, None] = None
        self.policy: Policy = Policy.construct_by_type(policy, self)
        self.id: int = random.randint(0, int(2E14))
        self.blocked_edges: BLOCKED_EDGES_TYPE = set()
        self.filter_blocked_edges: BLOCKED_EDGES_TYPE = set()
        self.blocked_nodes: BLOCKED_NODES_TYPE = set()
        self.filter_blocked_nodes: BLOCKED_NODES_TYPE = set()

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
        return self.env[n] == 0 and n not in self.filter_blocked_nodes

    def filter_edge(self, a, b):
        """edge filter for gridmap_to_nx"""
        return (a, b) not in self.filter_blocked_edges

    def gridmap_to_nx(self, env: np.ndarray,
                      blocked_edges: Union[None, BLOCKED_EDGES_TYPE] = None,
                      blocked_nodes: Union[None, BLOCKED_NODES_TYPE] = None) -> nx.Graph:
        """convert numpy gridmap into networkx graph."""
        if blocked_edges is None:
            self.filter_blocked_edges = self.blocked_edges
        else:
            self.filter_blocked_edges = blocked_edges
        if blocked_nodes is None:
            self.filter_blocked_nodes = self.blocked_nodes
        else:
            self.filter_blocked_nodes = blocked_nodes
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
            tuple_path = nx.astar_path(
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
        self.filter_blocked_edges = self.blocked_edges.union({tmp_edge})
        tmp_env_nx = self.gridmap_to_nx(
            self.env, self.blocked_edges.union({tmp_edge}), None)

        assert not tmp_env_nx.has_edge(a, b)
        path = self.plan_path(tmp_env_nx)
        if path is not None:
            # all good, and we have a new path now
            self.path = path
            self.path_i = 0
            self.blocked_edges.add(tmp_edge)
            self.env_nx = self.gridmap_to_nx(self.env)
            return True
        else:
            # forget changes
            self.filter_blocked_edges = self.blocked_edges
            return False

    def block_node(self, n: Tuple[int, int]) -> bool:
        """this will make the agent block this node. It will return `True`
        if there still is a path to the current goal. `False` otherwise."""
        assert self.env_nx is not None, "Should have a env_nx"
        self.filter_blocked_nodes = self.blocked_nodes.union({n})
        tmp_env_nx = self.gridmap_to_nx(
            self.env, None, self.blocked_nodes.union({n}))

        assert not tmp_env_nx.has_node(n)
        path = self.plan_path(tmp_env_nx)
        if path is not None:
            # all good, and we have a new path now
            self.path = path
            self.path_i = 0
            self.blocked_nodes.add(n)
            self.env_nx = self.gridmap_to_nx(self.env)
            return True
        else:
            # forget changes
            self.filter_blocked_nodes = self.blocked_nodes
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
            return all(self.path[i] == self.goal)

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
            return self.path[self.path_i + 1]

    def remove_all_blocks_and_replan(self):
        # resetting blocks now
        self.blocked_nodes = set()
        self.filter_blocked_nodes = set()
        self.blocked_edges = set()
        self.filter_blocked_edges = set()
        path = self.plan_path()
        assert path is not None, "We must be successful with no blocks"

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
            self.remove_all_blocks_and_replan()
