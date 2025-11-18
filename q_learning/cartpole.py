import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

from QLearning.CartPole.config import seed

env = gym.make('CartPole-v1', render_mode='human')

# simulate the environment
episodeNumber = 5
timeSteps = 100

for episodeIndex in range(episodeNumber):
    total_reward = 0
    initial_state = env.reset(seed=seed)
    env.render()
    appendedObservations = []
    for timeIndex in range(timeSteps):
        random_action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(random_action)
        total_reward += reward
        appendedObservations.append(observation)
        time.sleep(0.1)
        if (terminated):
            time.sleep(1)
            print(f'episode {episodeIndex}: {total_reward}')
            break
env.close()