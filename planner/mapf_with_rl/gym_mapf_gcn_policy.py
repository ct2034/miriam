import gym
import torch
import torch as tn
from planner.mapf_with_rl.mapf_with_rl import Qfunction
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import MlpPolicy


class GcnPolicy(MlpPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Schedule,
                 **kwargs):
        super(MlpPolicy, self).__init__(observation_space, action_space)
        print("GcnPolicy __init__")
        print(f"{kwargs}")
        self.q_net = Qfunction(9, 2, 16)
        self.q_net_target = Qfunction(9, 2, 16)
        assert self.q_net is not None
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=1e-3)  # default
        self.q_net._predict = lambda a: print(a)

    def forward(self, obs: tn.Tensor, deterministic: bool = True) -> tn.Tensor:
        return torch.zeros(2)

    def to(self, device):
        return self

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.train(mode)
