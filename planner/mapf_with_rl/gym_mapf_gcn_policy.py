import torch
import torch as tn
from stable_baselines3.dqn.policies import MlpPolicy


class GcnPolicy(MlpPolicy):
    def __init__(self):
        pass

    def forward(self, obs: tn.Tensor, deterministic: bool = True) -> tn.Tensor:
        return torch.zeros(2)
