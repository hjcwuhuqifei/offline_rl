import math
import numpy as np
import d3rlpy
import pickle
import gym
from pathlib import Path
from d3rlpy.dataset import MDPDataset
from gym_env.envs.offline_gym import get_reward

argoverse_scenario_dir = Path(
        '/home/haojiachen/桌面/offline_rl/offline_/')
all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.pkl"))
scenario_file_lists = (all_scenario_files[:112])
scenarios = []
for scenario_file_list in scenario_file_lists:
    scenario = pickle.load(open(scenario_file_list, 'rb'))
    scenarios.append(scenario)

observations = []
next_observations = []
actions = []
rewards = []
terminals = []
ego_v_state = []

for scenario in scenarios:
    states_of_scenario = scenario['states']
    for i in range(len(states_of_scenario)):
        state = states_of_scenario[i]
        ego_v = math.sqrt(state[0][3] ** 2 + state[0][4] ** 2)
        object_front_v = math.sqrt(state[1][5] ** 2 + state[1][6] ** 2)
        object_behind_v = math.sqrt(state[2][5] ** 2 + state[2][6] ** 2)
        object_left_front_v = math.sqrt(state[3][5] ** 2 + state[3][6] ** 2)
        object_right_front_v = math.sqrt(state[4][5] ** 2 + state[4][6] ** 2)
        object_left_behind_v = math.sqrt(state[5][5] ** 2 + state[5][6] ** 2)
        object_right_behind_v = math.sqrt(state[6][5] ** 2 + state[6][6] ** 2)
        observation = np.array([ego_v, state[1][0], state[1][1], object_front_v,
                       state[2][0], state[2][1], object_behind_v,
                       state[3][0], state[3][1], object_left_front_v,
                       state[4][0], state[4][1], object_right_front_v,
                       state[5][0], state[5][1], object_left_behind_v,
                       state[6][0], state[6][1], object_right_behind_v])
        ego_v_state.append(ego_v)
        if i != len(states_of_scenario) - 1:
            next_state = states_of_scenario[i + 1]
            next_ego_v = math.sqrt(next_state[0][3] ** 2 + next_state[0][4] ** 2)
            action = next_ego_v

            reward =  0.5 * action/20

            terminal = 0
        else:
            action = actions[-1][0]

            terminal = 1
            reward = 0.5 * action/15 + 10

        rewards.append(reward)
        observations.append(observation)
        actions.append([action])
        terminals.append(terminal)

        # reward = get_reward(observation)
for i in range(1, len(observations)):
    next_observations.append(observations[i])
next_observations.append(observations[0])

observations = np.array(observations, dtype='float32')
next_observations = np.array(next_observations, dtype='float32')
actions = np.array(actions, dtype='float32')
rewards = np.array(rewards, dtype='float32')
terminals = np.array(terminals, dtype='float32')

b = rewards.shape[0]

dataset = {}
dataset['observations'] = observations
dataset['next_observations'] = next_observations
dataset['actions'] = actions
dataset['rewards'] = rewards
dataset['terminals'] = terminals

pickle.dump(dataset, open('dataset', 'wb'))

