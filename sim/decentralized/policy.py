import logging
import random
from collections import OrderedDict
from enum import Enum, auto
from typing import Optional, Set

import numpy as np
import torch
from planner.policylearn.edge_policy import EdgePolicyModel
from planner.policylearn.edge_policy_graph_utils import (agents_to_data,
                                                         get_optimal_edge)
from planner.policylearn.generate_fovs import (add_padding_to_gridmap,
                                               extract_all_fovs)
from planner.policylearn.generate_graph import (get_agent_path_layer,
                                                get_agent_pos_layer,
                                                gridmap_to_graph)
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class PolicyCalledException(Exception):
    def __init__(self, policy, id_coll: int,
                 agents_with_colissions: Optional[Set[int]] = None) -> None:
        super().__init__()
        self.policy: RaisingPolicy = policy
        self.agents_with_colissions = agents_with_colissions
        self.id_coll = id_coll


class PolicyType(Enum):
    RANDOM = auto()
    LEARNED = auto()
    OPTIMAL = auto()
    RAISING = auto()
    VALUE_THEN_RAISING = auto()


class Policy(object):
    @classmethod
    def construct_by_type(cls, type: PolicyType, agent, **kwargs):
        if type == PolicyType.RANDOM:
            return RandomPolicy(agent)
        elif type == PolicyType.LEARNED:
            nn = EdgePolicyModel(gpu=torch.device("cpu"))
            nn.load_state_dict(torch.load(
                "sim/decentralized/policy_model.pt", map_location="cpu"))
            return LearnedPolicy(agent, nn, **kwargs)
        elif type == PolicyType.OPTIMAL:
            return OptimalPolicy(agent, 0, **kwargs)
        elif type == PolicyType.RAISING:
            return RaisingPolicy(agent, **kwargs)
        else:
            assert type == PolicyType.RAISING
            return RaisingPolicy(agent)

    def __init__(self, agent) -> None:
        super().__init__()
        self.a = agent  # type: ignore

    def __str__(self):
        return type(self).__name__

    def get_edge(self, agents, agents_with_colissions):
        raise NotImplementedError

    def step(self):
        pass


class RandomPolicy(Policy):
    def __init__(self, agent, _=None) -> None:
        super().__init__(agent)

    def get_edge(self, _, __):
        potential_next_nodes = list(self.a.env.adj[self.a.pos])
        if self.a.pos not in potential_next_nodes:
            potential_next_nodes.append(self.a.pos)
        return int(self.a.rng.choice(potential_next_nodes))


class LearnedPolicy(Policy):
    def __init__(self, agent, nn: EdgePolicyModel) -> None:
        super().__init__(agent)
        self.nn: EdgePolicyModel = nn
        self.nn.eval()

    def get_edge(self, agents, _):
        i_a_self = agents.index(self.a)
        data, big_from_small = agents_to_data(
            agents, i_a_self, with_optimal=False)
        node_to_go = self.nn.predict_probablilistic(
            data.x,
            data.edge_index,
            big_from_small,
            rng=self.a.rng)
        return node_to_go


class OptimalPolicy(Policy):
    def __init__(self, agent, _) -> None:
        super().__init__(agent)

    def get_edge(self, agents, _):
        return get_optimal_edge(agents, agents.index(self.a))


class RaisingPolicy(Policy):
    def __init__(self, agent) -> None:
        super().__init__(agent)

    def get_edge(self, _, agents_with_colissions):
        raise PolicyCalledException(self, 0, agents_with_colissions)


class ValueThenRaisingPolicy(Policy):
    def __init__(self, agent, first_val: int) -> None:
        super().__init__(agent)
        self.first_round = True
        self.first_val = first_val

    def get_edge(self, _, agents_with_colissions):
        if self.first_round:
            return self.first_val
        else:
            raise PolicyCalledException(self, 0, agents_with_colissions)

    def step(self):
        self.first_round = False
