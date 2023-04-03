import math
import numpy as np
import d3rlpy
import pickle
import gym
from pathlib import Path
from d3rlpy.dataset import MDPDataset
from gym_env.envs.offline_gym import get_reward
# prepare dataset

ENV_NAME = 'carla-v1'

dataset_from_td3 = pickle.load(open( '/home/haojiachen/桌面/offline_rl/dataset_from_td3', 'rb'))

rewards_ = dataset_from_td3['reward']
terminals_ = dataset_from_td3['done']
observations_ = dataset_from_td3['state']
actions_ = dataset_from_td3['action']

while True:
    if terminals_[-1] == 0:
        terminals_.pop()
        rewards_.pop()
        observations_.pop()
        actions_.pop()
    else:
        break

observations = np.array(np.squeeze(dataset_from_td3['state']))
actions = np.array(dataset_from_td3['action'])
rewards = np.array(dataset_from_td3['reward'])
terminals = np.array(dataset_from_td3['done'])
ego_v_state = []

obs = np.random.random((1000, 100))
rew = np.random.random(1000)
ter = np.random.randint(2, size=100)

# observations = np.array(observations, dtype='float32')
# actions = np.array(actions, dtype='float32')
# rewards = np.array(rewards, dtype='float32')
# terminals = np.array(terminals, dtype='float32')

dataset = MDPDataset(observations, actions, rewards, terminals)

a = dataset.episode_terminals
env = gym.make(ENV_NAME)
env = env.unwrapped

# dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-v0')

# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

# train
cql.fit(
    dataset,
    eval_episodes=dataset.episodes,
    tensorboard_dir='runs_with_td3_100',
    n_epochs=30,
    scorers={
        'environment': d3rlpy.metrics.evaluate_on_environment(env),
        'td_error': d3rlpy.metrics.td_error_scorer,
    },
)

# prepare algorithm
bcq = d3rlpy.algos.BCQ(use_gpu=True)

# train
bcq.fit(
    dataset,
    eval_episodes=dataset.episodes,
    tensorboard_dir='runs_with_td3_100',
    n_epochs=30,
    scorers={
        'environment': d3rlpy.metrics.evaluate_on_environment(env),
        'td_error': d3rlpy.metrics.td_error_scorer,
    },
)

# prepare algorithm
td3_bc = d3rlpy.algos.TD3PlusBC(use_gpu=True)

# train
td3_bc.fit(
    dataset,
    eval_episodes=dataset.episodes,
    tensorboard_dir='runs_with_td3_100',
    n_epochs=30,
    scorers={
        'environment': d3rlpy.metrics.evaluate_on_environment(env),
        'td_error': d3rlpy.metrics.td_error_scorer,
    },
)

