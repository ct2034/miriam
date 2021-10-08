import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy

env = gym.make("gym_mapf:mapf-v0")

policy = MlpPolicy()
model = DQN(policy, env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
