
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

from qlearning import QLearning

# env=gym.make('CartPole-v1',render_mode='human')
env = gym.make('CartPole-v1')
(state, _) = env.reset()


# here define the parameters for state discretization
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin = -3
cartVelocityMax = 3
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]

# define the parameters
alpha = 0.1
gamma = 1
epsilon = 0.2
numberEpisodes = 10000

# create an object
Q1 = QLearning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)
# run the Q-Learning algorithm
Q1.simulate_episodes()
# simulate the learned strategy
(obtainedRewardsOptimal, env1) = Q1.simulate_learned_strategy()

plt.figure(figsize=(12, 5))
# plot the figure and adjust the plot parameters
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.show()

# close the environment
env1.close()
# get the sum of rewards
np.sum(obtainedRewardsOptimal)

# now simulate a random strategy
(obtainedRewardsRandom, env2) = Q1.simulate_random_strategy()
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.show()

# run this several times and compare with a random learning strategy
(obtainedRewardsOptimal, env1) = Q1.simulate_learned_strategy()