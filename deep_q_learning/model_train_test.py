import warnings
import os
import gymnasium as gym
import numpy as np
import pygame
import torch
from matplotlib import pyplot as plt

from config import device, seed
from dqn_agent import DQNAgent
from step_wrapper import StepWrapper


class ModelTrainTest():
    def __init__(self, hyperparams):

        # Hyperparameters
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]

        self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        self.render = hyperparams["render"]

        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]

        self.memory_capacity = hyperparams["memory_capacity"]
        self.render_fps = hyperparams["render_fps"]

        # Base environment
        self.env = gym.make(
            'CartPole-v1',
            max_episode_steps=self.max_steps,
            render_mode="human" if self.render else None
        )
        self.env.metadata['render_fps'] = self.render_fps

        warnings.filterwarnings("ignore", category=UserWarning)

        # Apply wrapper pipeline
        self.env = StepWrapper(self.env)

        # Define agent
        self.agent = DQNAgent(
            env=self.env,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            clip_grad_norm=self.clip_grad_norm,
            learning_rate=self.learning_rate,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity
        )

        os.makedirs('plots', exist_ok=True)

    # ----------------------------------------------------
    # TRAINING
    # ----------------------------------------------------
    def train(self):

        total_steps = 0
        self.reward_history = []

        for episode in range(1, self.max_episodes + 1):

            render_this = (episode <= 5) or (episode > self.max_episodes - 10)

            # New env for rendering choice
            self.env = gym.make(
                'CartPole-v1',
                max_episode_steps=self.max_steps,
                render_mode="human" if render_this else None
            )
            self.env.metadata['render_fps'] = self.render_fps
            self.env = StepWrapper(self.env)

            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:

                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                # Store in replay buffer
                self.agent.replay_memory.store(state, action, next_state, reward, done, truncation)

                # Learn
                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (done or truncation))

                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()

                state = next_state
                episode_reward += reward
                step_size += 1

            self.reward_history.append(episode_reward)
            total_steps += step_size

            # Epsilon decay
            self.agent.update_epsilon()

            # Save model periodically
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode == self.max_episodes:
                    self.plot_training(episode)
                print("\n~~~~~~Interval Save: Model saved.\n")

            print(
                f"Episode: {episode}, "
                f"Total Steps: {total_steps}, "
                f"Ep Steps: {step_size}, "
                f"Reward: {episode_reward:.2f}, "
                f"Epsilon: {self.agent.epsilon_max:.2f}"
            )

        self.plot_training(episode)

    # ----------------------------------------------------
    # TESTING
    # ----------------------------------------------------
    def test(self, max_episodes):

        # Load trained model
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()

        # Test env with rendering
        self.env = gym.make(
            'CartPole-v1',
            max_episode_steps=self.max_steps,
            render_mode="human"
        )
        self.env = StepWrapper(self.env)

        for episode in range(1, max_episodes + 1):

            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                state = next_state
                episode_reward += reward
                step_size += 1

            print(
                f"Episode: {episode}, "
                f"Steps: {step_size}, "
                f"Reward: {episode_reward:.2f}"
            )

        pygame.quit()

    # ----------------------------------------------------
    # PLOTTING
    # ----------------------------------------------------
    def plot_training(self, episode):

        sma = np.convolve(self.reward_history, np.ones(50) / 50, mode='valid')

        reward_history = np.clip(self.reward_history, None, 500)
        sma = np.clip(sma, None, 500)

        plt.figure()
        plt.title("Episode Rewards")
        plt.plot(reward_history, label='Raw Reward', alpha=1)
        plt.plot(sma, label='SMA-50')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        if episode == self.max_episodes:
            plt.savefig('plots/reward_plot.png', dpi=600, bbox_inches='tight')

        plt.show()
        plt.close()

        # Loss
        plt.figure()
        plt.title("Network Loss")
        plt.plot(self.agent.loss_history, label='Loss')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.grid(True)

        if episode == self.max_episodes:
            plt.savefig('plots/loss_plot.png', dpi=600, bbox_inches='tight')

        plt.show()
        plt.close()
