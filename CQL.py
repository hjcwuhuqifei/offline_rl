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

argoverse_scenario_dir = Path(
        '/home/haojiachen/桌面/offline_rl/offline_/')
all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.pkl"))
scenario_file_lists = (all_scenario_files[:112])
scenarios = []
for scenario_file_list in scenario_file_lists:
    scenario = pickle.load(open(scenario_file_list, 'rb'))
    scenarios.append(scenario)

observations = []
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
        if abs(state[1][0]) > 100:
            state[1][0] = 0
        if abs(state[2][0]) > 100:
            state[2][0] = 0
        if abs(state[3][0]) > 100:
            state[3][0] = 0
        if abs(state[4][0]) > 100:
            state[4][0] = 0
        if abs(state[5][0]) > 100:
            state[5][0] = 0
        if abs(state[6][0]) > 100:
            state[6][0] = 0
        if abs(state[1][1]) > 100:
            state[1][1] = 0
        if abs(state[2][1]) > 100:
            state[2][1] = 0
        if abs(state[3][1]) > 100:
            state[3][1] = 0
        if abs(state[4][1]) > 100:
            state[4][1] = 0
        if abs(state[5][1]) > 100:
            state[5][1] = 0
        if abs(state[6][1]) > 100:
            state[6][1] = 0
        if object_front_v > 100:
            object_front_v = 0
        if object_behind_v > 100:
            object_behind_v = 0
        if object_left_front_v > 100:
            object_left_front_v = 0
        if object_right_front_v > 100:
            object_right_front_v = 0
        if object_left_behind_v > 100:
            object_left_behind_v = 0
        if object_right_behind_v > 100:
            object_right_behind_v = 0
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

            reward = 0.5 * action/2

            terminal = 0
        else:
            action = actions[-1][0]

            terminal = 1
            reward = 0.5 * action/2 + 10

        rewards.append(reward)
        observations.append(observation)
        actions.append([action])
        terminals.append(terminal)

        # reward = get_reward(observation)

speed = sum(ego_v_state) / len(ego_v_state)
obs = np.random.random((1000, 100))
rew = np.random.random(1000)
ter = np.random.randint(2, size=100)

observations = np.array(observations, dtype='float32')
actions = np.array(actions, dtype='float32')
rewards = np.array(rewards, dtype='float32')
terminals = np.array(terminals, dtype='float32')

dataset = MDPDataset(observations, actions, rewards, terminals)

# dataset_.episode_terminals
# episode = dataset_.episodes[0]
# print("************************")
# print(episode[0].observation, episode[0].action, episode[0].reward, episode[0].next_observation, episode[0].terminal)

env = gym.make(ENV_NAME)
env = env.unwrapped

# dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-v0')
# dataset.episode_terminals
# episode = dataset.episodes[0]
# print("************************")
# print(episode[0].observation, episode[0].action, episode[0].reward, episode[0].next_observation, episode[0].terminal)


d3rlpy.seed(100)

# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

# train
cql.fit(
    dataset,
    eval_episodes=dataset.episodes,
    tensorboard_dir='runs_only_expert_100_plustime',
    n_epochs=30,
    scorers={
        'environment': d3rlpy.metrics.evaluate_on_environment(env),
        'td_error': d3rlpy.metrics.td_error_scorer,
    },
)

# # prepare algorithm
# bc = d3rlpy.algos.BC(use_gpu=True)

# # train
# bc.fit(
#     dataset,
#     eval_episodes=dataset.episodes,
#     tensorboard_dir='runs_only_expert',
#     n_epochs=300,
#     scorers={
#         'environment': d3rlpy.metrics.evaluate_on_environment(env),
#         'td_error': d3rlpy.metrics.td_error_scorer,
#     },
# )

# prepare algorithm
bcq = d3rlpy.algos.BCQ(use_gpu=True)

# train
bcq.fit(
    dataset,
    eval_episodes=dataset.episodes,
    tensorboard_dir='runs_only_expert_100_plustime',
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
    tensorboard_dir='runs_only_expert_100_plustime',
    n_epochs=30,
    scorers={
        'environment': d3rlpy.metrics.evaluate_on_environment(env),
        'td_error': d3rlpy.metrics.td_error_scorer,
    },
)


