import random
from enum import Enum

import networkx as nx
import numpy as np


class Policy(Enum):
    RANDOM = 0
    CLOSEST = 1


class Agent():
    def __init__(
        self, env: np.ndarray, env_nx: nx.Graph, pos: np.ndarray,
        policy: Policy
    ):
        """Initialize a new agent at a given postion `pos` using a given
        `policy` for resolution of errors."""
        self.env = env
        self.env_nx = env_nx
        self.pos = pos
        self.goal = None
        self.policy = policy
        self.path = None
        self.path_i = None

    def give_a_goal(self, goal: np.ndarray):
        """Set a new goal for the agent, this will calculate the path,
        if the goal is new."""
        if (self.goal != goal).all():
            self.goal = goal
            self.plan_path()
            self.path_i = 0

    def is_at_goal(self):
        """returns true iff the agent is at its goal."""
        return all(self.pos == self.goal)

    def get_priority(self):
        """Based on the selected policy, this will give the priority of this
        agent."""
        if self.policy == Policy.RANDOM:
            return random.random()
        elif self.policy == Policy.CLOSEST:
            raise NotImplementedError

    def plan_path(self):
        """Plan path from currently set `pos` to current `goal` and save it
        in `path`."""
        tuple_path = nx.shortest_path(
            self.env_nx, tuple(self.pos), tuple(self.goal))
        self.path = np.array(tuple_path)

    def what_is_next_step(self) -> np.ndarray:
        """Return the position where this agent would like to go next."""
        assert self.path_i is not None, "We need to have a current path_i"
        if self.path_i + 1 == len(self.path):
            return self.path[-1]  # stay at final pose
        else:
            return self.path[self.path_i + 1]

    def make_next_step(self, next_pos_to_check: np.ndarray):
        """Move agent to its next step, pass that pose for clarification."""
        potential_next_pos = self.what_is_next_step()
        assert (potential_next_pos == next_pos_to_check).all(
        ), "Our next position has to be corect."
        if self.path_i + 1 == len(self.path):
            pass  # at goal already
        else:
            self.path_i += 1
            self.pos = potential_next_pos
