import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

from CartPole.config import seed



class QLearning:

    def __init__(self, env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actionNumber = env.action_space.n
        self.numberEpisodes = numberEpisodes
        self.numberOfBins = numberOfBins
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.sumRewardsEpisode = []
        self.QMatrix=np.random.uniform(low=0, high=1, size=(numberOfBins[0],numberOfBins[1],numberOfBins[2],numberOfBins[3],self.actionNumber))


    def returnIndexState(self, state):
        position= state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin=np.linspace(self.lowerBounds[0],self.upperBounds[0],self.numberOfBins[0])
        cartVelocityBin=np.linspace(self.lowerBounds[1],self.upperBounds[1],self.numberOfBins[1])
        poleAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2])
        poleAngleVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numberOfBins[3])

        indexPosition = np.maximum(np.digitize(position, cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(velocity, cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(angle, poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(angularVelocity, poleAngleVelocityBin) - 1, 0)

        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])


    def selectAction(self,state,index):












# env = gym.make('CartPole-v1', render_mode='human')
#
# # simulate the environment
# episodeNumber = 5
# timeSteps = 100
#
# for episodeIndex in range(episodeNumber):
#     total_reward = 0
#     initial_state = env.reset(seed=seed)
#     env.render()
#     appendedObservations = []
#     for timeIndex in range(timeSteps):
#         random_action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(random_action)
#         total_reward += reward
#         appendedObservations.append(observation)
#         time.sleep(0.1)
#         if (terminated):
#             time.sleep(1)
#             print(f'episode {episodeIndex}: {total_reward}')
#             break
# env.close()



