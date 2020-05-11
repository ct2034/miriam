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

    def give_a_goal(self, goal: np.ndarray):
        """Set a new goal for the agent, this will calculate the path,
        if the goal is new."""
        if (self.goal != goal).all():
            self.goal = goal
            self.plan_path()

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
        tuple_path = nx.shortest_path(self.env_nx, tuple(self.pos), tuple(self.goal))
        self.path = np.array(tuple_path)
