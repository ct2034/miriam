import logging
from collections import OrderedDict
from enum import Enum, auto
from typing import Optional

import numpy as np
import torch
from definitions import C
from importtf import keras, tf
from planner.policylearn.generate_fovs import (add_padding_to_gridmap,
                                               extract_all_fovs)
from planner.policylearn.generate_graph import (get_agent_path_layer,
                                                get_agent_pos_layer,
                                                gridmap_to_graph)
from planner.policylearn.train_model import (CLASSIFICATION_STR, CONVRNN_STR,
                                             fix_data_convrnn)
from tensorflow.keras.models import load_model
from torch_geometric.data import Data

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = logging.getLogger(__name__)


class PolicyCalledException(Exception):
    def __init__(self, policy, id_coll: int) -> None:
        super().__init__()
        self.policy: LearnedRaisingPolicy = policy
        self.id_coll = id_coll

    def get_agent_state(self, hop_dist: int):
        return self.policy.get_state(self.id_coll, hop_dist)


class PolicyType(Enum):
    RANDOM = auto()
    CLOSEST = auto()
    FILL = auto()
    LEARNED = auto()
    INVERSE_LEARNED = auto()
    LEARNED_RAISING = auto()
    Q_LEARNING_POLICY = auto()
    INVERSE_Q_LEARNING_POLICY = auto()
    ONE_THEN_RANDOM = auto()
    ZERO_THEN_RANDOM = auto()
    FIRST_THEN_RAISING = auto()


class Policy(object):
    @classmethod
    def construct_by_type(cls, type: PolicyType, agent, **kwargs):
        if type == PolicyType.RANDOM:
            return RandomPolicy(agent)
        elif type == PolicyType.CLOSEST:
            return ClosestPolicy(agent)
        elif type == PolicyType.FILL:
            return FillPolicy(agent)
        elif type == PolicyType.LEARNED:
            return LearnedPolicy(agent)
        elif type == PolicyType.INVERSE_LEARNED:
            return InverseLearnedPolicy(agent)
        elif type == PolicyType.LEARNED_RAISING:
            return LearnedRaisingPolicy(agent)
        elif type == PolicyType.Q_LEARNING_POLICY:
            return QLearningPolicy(agent, **kwargs)
        elif type == PolicyType.INVERSE_Q_LEARNING_POLICY:
            return InverseQLearningPolicy(agent, **kwargs)
        elif type == PolicyType.ONE_THEN_RANDOM:
            return FirstThenRandomPolicy(agent, 1.)
        elif type == PolicyType.ZERO_THEN_RANDOM:
            return FirstThenRandomPolicy(agent, 0.)
        elif type == PolicyType.FIRST_THEN_RAISING:
            return FirstThenRaisingPolicy(agent, **kwargs)

    def __init__(self, agent) -> None:
        super().__init__()
        self.a = agent  # type: ignore

    def __str__(self):
        return type(self).__name__

    def register_observation(self, id, path, pos, path_i) -> None:
        """we have seen another agent"""
        pass

    def get_priority(self, id: int) -> float:
        raise NotImplementedError()


class RandomPolicy(Policy):
    # simply returning a random number on every call
    def __init__(self, agent) -> None:
        super().__init__(agent)

    def get_priority(self, _) -> float:
        return self.a.rng.random()


class ClosestPolicy(Policy):
    # a policy that prefers the agent that is currently closest to its
    # goal.
    def __init__(self, agent) -> None:
        super().__init__(agent)

    def get_priority(self, _) -> float:
        return 1. / np.linalg.norm(np.array(self.a.goal) - np.array(self.a.pos))


class FillPolicy(Policy):
    # an agent will get a higher prio if the map around it is fuller of
    # obstacles.
    def __init__(self, agent) -> None:
        super().__init__(agent)

    def get_priority(self, _) -> float:
        FILL_RADIUS = 2
        n_total = ((FILL_RADIUS * 2 + 1)**2)
        n_free = 0
        for x in range(max(0, self.a.pos[0] - FILL_RADIUS),
                       min(self.a.env.shape[0],
                           self.a.pos[0] + FILL_RADIUS + 1)):
            for y in range(max(0, self.a.pos[1] - FILL_RADIUS),
                           min(self.a.env.shape[1],
                               self.a.pos[1] + FILL_RADIUS + 1)):
                if self.a.env[x, y] == 0:
                    n_free += 1
        return float(n_total - n_free) / n_total


class LearnedPolicy(Policy):
    # using machine learning for a greater tomorrow
    def __init__(self, agent) -> None:
        super().__init__(agent)
        self.radius = 3  # how far to look in each direction
        self.ts = 3  # how long to collect data for
        self.padded_gridmap = add_padding_to_gridmap(self.a.env, self.radius)
        self.paths: OrderedDict = OrderedDict()
        self.poss: OrderedDict = OrderedDict()
        self.path_is: OrderedDict = OrderedDict()
        self.t = 0
        self.model: tf.keras.Model = load_model(
            "planner/policylearn/my_model.h5")
        model_input_shape = list(self.model.layers[0].input_spec[0].shape)
        self.model_type: Optional[str] = None
        if model_input_shape[-3:] == [7, 7, 5]:
            self.model_type = CONVRNN_STR
        else:
            logging.error("can not determine model type")
            raise NotImplementedError("see how this data looks")
        logging.info(f"model_type: {self.model_type}")

    def _path_until_coll(self, path, path_i, n_t):
        """Make subpath only until collision happens. n_t is the number of steps to produce, will give all if None"""
        path_until_pos = []
        if n_t is None:
            n_t = len(path)
        path = np.array(path)
        start = max(0, path_i - n_t)
        for t in range(start, path_i+1):
            path_until_pos.append(path[t, :])
        return path_until_pos

    def step(self):
        super().step()
        self.paths = OrderedDict()
        self.poss = OrderedDict()
        self.path_is = OrderedDict()

    def register_observation(self, id, path, pos, path_i) -> None:
        if path is None:
            self.paths[id] = np.array([pos] * self.ts)
            self.path_is[id] = 0
        else:
            path = np.array(path)
            self.paths[id] = path[:, :-1]
            self.path_is[id] = path_i
        self.poss[id] = pos

    def get_priority(self, id_coll: int) -> float:
        """[summary]

        :param id: which agent are we meeting
        :return: priority
        """
        logger.debug(
            f"get_priority, self_agent: {self.a}, other_id: {id_coll}")
        N_T = 3
        i_oa = None
        path = np.array(self.a.path)
        paths_full = [path[:, :-1]]  # self always first
        paths_until_col = [self._path_until_coll(
            path[:, :-1], self.a.path_i, N_T)]  # self always first
        ids = sorted(self.paths.keys())
        if id_coll not in ids:
            logger.warn(f"{id_coll} not in {ids}")
            return .5
        i_oa = ids.index(id_coll) + 1
        for i_id in ids:
            paths_full.append(self.paths[i_id])
            paths_until_col.append(self._path_until_coll(
                self.paths[i_id], self.path_is[i_id], N_T))
        assert i_oa is not None
        x = np.array(
            [extract_all_fovs(
                t=N_T-1,
                paths_until_col=np.array(paths_until_col),
                paths_full=paths_full,
                padded_gridmap=self.padded_gridmap,
                i_a=0,  # self always first
                i_oa=i_oa,
                radius=self.radius
            )]
        )
        if self.model_type == CONVRNN_STR:
            x = fix_data_convrnn(x)
        x_tensor = tf.constant(x)
        y = self.model.predict(x_tensor)[0][0]
        logger.debug(f"y: {y}")
        # from planner.policylearn.generate_data_demo import plot_fovs
        # plot_fovs(x, y)
        return y


class InverseLearnedPolicy(Policy):
    # For demonstration purposes: What if we do the exact opposite?
    def __init__(self, agent) -> None:
        self.po = LearnedPolicy(agent)

    def step(self):
        self.po.step()

    def register_observation(self, id, path, pos, path_i) -> None:
        self.po.register_observation(id, path, pos, path_i)

    def get_priority(self, id_coll: int) -> float:
        prio = 1 - self.po.get_priority(id_coll)
        assert prio >= 0
        assert prio <= 1
        return prio


class LearnedRaisingPolicy(LearnedPolicy):
    def __init__(self, agent) -> None:
        self.a = agent  # type: ignore
        self.paths: OrderedDict = OrderedDict()
        self.poss: OrderedDict = OrderedDict()
        self.path_is: OrderedDict = OrderedDict()

    def get_priority(self, id_coll: int) -> float:
        raise PolicyCalledException(self, id_coll)

    def get_state(self, id_coll: int, hop_dist: int) -> Data:
        """return a data object of the state when this collision happens
        :param id_coll: id of the agent in *self* perspective
        :param hop_dist: how big the view range is (in hop distance)
        :return: data object containing graph info for learning
        """
        # indices for relevant agents
        ids = ([self.a.id]+list(self.paths.keys()))
        i_a = 0
        i_ca = ids.index(id_coll)
        i_oas = [i for i in range(len(ids)) if (
            i != 0 and i != i_ca
        )]
        # paths
        self.path_is[self.a.id] = self.a.path_i
        own_path = np.array(self.a.path)[:, :-1]
        paths_full = ([own_path] +
                      list(self.paths.values()))
        paths_until_col = []
        for i_id, id in enumerate(ids):
            paths_until_col.append(self._path_until_coll(
                paths_full[i_id], self.path_is[id], None))
        # making basic graph info
        own_pos: C = tuple(paths_until_col[i_a][-1])  # type: ignore
        assert own_pos == self.a.pos
        if self.a.has_gridmap:
            data_edge_index, data_pos = gridmap_to_graph(
                self.a.env, hop_dist, own_pos)
        elif self.a.has_roadmap:
            raise NotImplementedError()
        # node features
        data_x = torch.cat((
            get_agent_pos_layer(data_pos, paths_until_col, [i_a]),
            get_agent_path_layer(data_pos, paths_until_col, [i_a]),
            get_agent_path_layer(data_pos, paths_full, [i_a]),
            get_agent_pos_layer(data_pos, paths_until_col, [i_ca]),
            get_agent_path_layer(data_pos, paths_until_col, [i_ca]),
            get_agent_path_layer(data_pos, paths_full, [i_ca]),
            get_agent_pos_layer(data_pos, paths_until_col, i_oas),
            get_agent_path_layer(data_pos, paths_until_col, i_oas),
            get_agent_path_layer(data_pos, paths_full, i_oas),
        ), 1)
        return Data(
            x=data_x,
            edge_index=data_edge_index,
            pos=data_pos
        )


class QLearningPolicy(LearnedRaisingPolicy):
    def __init__(self, agent, hop_dist: int) -> None:
        super().__init__(agent)
        self.model: Optional[torch.nn.Module] = None
        self.hop_dist = hop_dist

    def set_qfun(self, model):
        self.model = model

    def get_priority(self, id_coll: int) -> float:
        assert self.model is not None
        data: Data = self.get_state(id_coll, self.hop_dist)
        self.model.eval()
        # this will be two values for [0, 1]
        out = self.model(data)[0]
        # now values are between [0, 1]
        expit = torch.special.expit(out)
        prio = float(torch.mean(torch.tensor([
            1 - expit[0],  # would indicate 0 prio
            expit[1]  # for 1 prio
        ])))
        if out[0] > out[1]:
            if not prio < .5:
                print(out)
                print(expit)
                print(prio)
        else:
            if not prio > .5:
                print(out)
                print(expit)
                print(prio)
        return prio


class InverseQLearningPolicy(QLearningPolicy):
    # For demonstration purposes: What if we do the exact opposite?
    def get_priority(self, id_coll: int) -> float:
        prio = 1 - super().get_priority(id_coll)
        assert prio >= 0
        assert prio <= 1
        return prio


class FirstThenRaisingPolicy(LearnedRaisingPolicy):
    def __init__(self, agent, first_return_value) -> None:
        super().__init__(agent)
        self.first_call = True
        self.first_return_value = first_return_value

    def get_priority(self, id_coll: int) -> float:
        if self.first_call:
            self.first_call = False
            return self.first_return_value
        return super().get_priority(id_coll)


class FirstThenRandomPolicy(Policy):
    def __init__(self, agent, first_return_value) -> None:
        super().__init__(agent)
        self.first_call = True
        self.first_return_value = first_return_value

    def get_priority(self, _) -> float:
        if self.first_call:
            self.first_call = False
            return self.first_return_value
        return self.a.rng.random()
