import numpy as np
import gymnasium as gym
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


    def return_index_state(self, state):
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



    def select_action(self,state,index):
        if index<500:
            return np.random.choice(self.actionNumber)
        randomNumber = np.random.random()
        if index>7000:
            self.epsilon = 0.999 * self.epsilon

        if randomNumber<self.epsilon:
            return  np.random.choice(self.actionNumber)
        else:
            return np.random.choice(np.where(self.QMatrix[self.return_index_state(state)]==np.max(self.QMatrix[self.return_index_state(state)]))[0])
            # here we need to return the minimum index since it can happen
            # that there are several identical maximal entries, for example
            # import numpy as np
            # a=[0,1,1,0]
            # np.where(a==np.max(a))
            # this will return [1,2], but we only need a single index
            # that is why we need to have np.random.choice(np.where(a==np.max(a))[0])
            # note that zero has to be added here since np.where() returns a tuple


    def simulate_episodes(self):
        for indexEpisode in range(self.numberEpisodes):
            rewardsEpisode = []
            (stateS,_) = self.env.reset()
            stateS = list(stateS)
            print("Simulating episode {}".format(indexEpisode))
            terminalState=False
            while not terminalState:
                stateSIndex=self.return_index_state(stateS)
                actionA = self.select_action(stateS,indexEpisode)
                (stateSprime, reward, terminalState,_,_) = self.env.step(actionA)
                rewardsEpisode.append(reward)
                stateSprime = list(stateSprime)
                stateSprimeIndex = self.return_index_state(stateSprime)
                QMaxPrime = np.max(self.QMatrix[stateSprimeIndex])
                if not terminalState:
                    error = reward + self.gamma * QMaxPrime - self.QMatrix[stateSIndex + (actionA,)]
                    self.QMatrix[stateSIndex + (actionA,)] = self.QMatrix[stateSIndex + (actionA,)] + self.alpha * error
                else:
                    # in the terminal state, we have QMatrix[stateSprime,actionAprime]=0
                    error = reward - self.QMatrix[stateSIndex + (actionA,)]
                    self.QMatrix[stateSIndex + (actionA,)] = self.QMatrix[stateSIndex + (actionA,)] + self.alpha * error

            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))



    def simulate_learned_strategy(self):
        env1 = gym.make('CartPole-v1', render_mode='human')
        (currentState, _) = env1.reset()
        env1.render()
        timeSteps = 1000
        # obtained rewards at every time step
        obtainedRewards = []

        for timeIndex in range(timeSteps):
            # select greedy actions
            actionInStateS = np.random.choice(np.where(self.QMatrix[self.return_index_state(currentState)] ==
                                                       np.max(self.QMatrix[self.return_index_state(currentState)]))[0])
            currentState, reward, terminated, truncated, info = env1.step(actionInStateS)
            obtainedRewards.append(reward)
            time.sleep(0.05)
            if terminated:
                time.sleep(1)
                break
        return obtainedRewards, env1

    def simulate_random_strategy(self):
        env2 = gym.make('CartPole-v1')
        (currentState, _) = env2.reset()
        env2.render()
        # number of simulation episodes
        episodeNumber = 100
        # time steps in every episode
        timeSteps = 1000
        # sum of rewards in each episode
        sumRewardsEpisodes = []

        for episodeIndex in range(episodeNumber):
            rewardsSingleEpisode = []
            initial_state = env2.reset()
            for timeIndex in range(timeSteps):
                random_action = env2.action_space.sample()
                observation, reward, terminated, truncated, info = env2.step(random_action)
                rewardsSingleEpisode.append(reward)
                if (terminated):
                    break
            sumRewardsEpisodes.append(np.sum(rewardsSingleEpisode))
        return sumRewardsEpisodes, env2











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



