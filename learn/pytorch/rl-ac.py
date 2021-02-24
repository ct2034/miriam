#!/usr/bin/env python3
"""
src:
https://github.com/bentrevett/pytorch-rl/blob/master/
2%20-%20Actor%20Critic%20%5BCartPole%5D.ipynb
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train_env = gym.make('CartPole-v1')
test_env = gym.make('CartPole-v1')

SEED = 0

train_env.seed(SEED)
test_env.seed(SEED+1)
np.random.seed(SEED)
torch.manual_seed(SEED)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred


INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = train_env.action_space.n
actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)
policy = ActorCritic(actor, critic)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


print(policy.apply(init_weights))
LEARNING_RATE = 0.01
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)


def train(env, policy, optimizer, discount_factor):
    policy.train()
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    state = env.reset()
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_pred, value_pred = policy(state)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        state, reward, done, _ = env.step(action.item())
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    returns = calculate_returns(rewards, discount_factor)
    policy_loss, value_loss = update_policy(
        returns, log_prob_actions, values, optimizer)
    return policy_loss, value_loss, episode_reward


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns


def update_policy(returns, log_prob_actions, values, optimizer):
    returns = returns.detach()
    policy_loss = - (returns * log_prob_actions).sum()
    value_loss = F.smooth_l1_loss(returns, values).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    value_loss.backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item()


def evaluate(env, policy):
    policy.eval()
    done = False
    episode_reward = 0
    state = env.reset()
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1)
        state, reward, done, _ = env.step(action.item())
        episode_reward += reward
    return episode_reward


def demo(policy):
    demo_env = gym.make('CartPole-v1')
    state = demo_env.reset()
    done = False
    while not done:
        demo_env.render()
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1)
        state, _, done, _ = demo_env.step(action.item())
    demo_env.close()


MAX_EPISODES = 500
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25
REWARD_THRESHOLD = 500
PRINT_EVERY = 10

train_rewards = []
test_rewards = []

for episode in range(1, MAX_EPISODES+1):
    policy_loss, critic_loss, train_reward = train(
        train_env, policy, optimizer, DISCOUNT_FACTOR)
    test_reward = evaluate(test_env, policy)
    train_rewards.append(train_reward)
    test_rewards.append(test_reward)
    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
    if episode % PRINT_EVERY == 0:
        print(
            f'| Episode: {episode:3} | Mean Train Rewards: '
            f'{mean_train_rewards:5.1f} | Mean Test Rewards: '
            f'{mean_test_rewards:5.1f} |')
        # demo(policy)
    if mean_test_rewards >= REWARD_THRESHOLD:
        print(f'Reached reward threshold in {episode} episodes')
        demo(policy)
        break

plt.figure(figsize=(12, 8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
plt.legend(loc='lower right')
plt.grid()
plt.show()
