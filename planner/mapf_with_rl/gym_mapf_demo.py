import gym
from gym.spaces import Discrete
from planner.mapf_with_rl.gym_mapf_gcn_policy import GcnPolicy
from planner.mapf_with_rl.mapf_with_rl import evaluate, make_useful_scenarios
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy

env = gym.make("gym_mapf:mapf-v0")

model = DQN(GcnPolicy, env, verbose=1)
model.learn(total_timesteps=10000, log_interval=100)

# testing this
test_scenarios = make_useful_scenarios(100, 0, True, 6, 4)
assert isinstance(model.policy, GcnPolicy)
qfun = model.policy.q_net
evaluate(test_scenarios, qfun, True, False)
evaluate(test_scenarios, qfun, True, True)
