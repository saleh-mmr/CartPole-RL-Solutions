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
    """
    Wrapper class responsible for:
    - Creating environment
    - Running training loop
    - Running evaluation loop
    - Plotting results

    Decouples higher-level RL process from agent logic.
    """

    def __init__(self, hyperparams):

        # -------------------------------
        # Load training/testing parameters
        # -------------------------------
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]     # path for loading weights in test mode
        self.save_path = hyperparams["save_path"]           # save checkpoint prefix
        self.save_interval = hyperparams["save_interval"]   # save model every X episodes

        self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]

        # How often to apply soft update during training
        self.update_frequency = hyperparams["update_frequency"]

        self.max_episodes = hyperparams["max_episodes"]     # training duration
        self.max_steps = hyperparams["max_steps"]           # max steps per episode in env
        self.render = hyperparams["render"]                 # whether to display environment
        self.tau = hyperparams["tau"]                       # SOFT UPDATE RATE → (NEW FEATURE)

        # Epsilon-greedy exploration setup
        self.epsilon_max = hyperparams["epsilon_max"]       # starting epsilon
        self.epsilon_min = hyperparams["epsilon_min"]       # lower bound on epsilon
        self.epsilon_decay = hyperparams["epsilon_decay"]   # multiplicative decay per episode

        self.memory_capacity = hyperparams["memory_capacity"]
        self.render_fps = hyperparams["render_fps"]

        # Logging for plots
        self.reward_history = []
        self.epsilon_history = []

        # -------------------------------
        # Create environment
        # -------------------------------
        self.env = gym.make(
            'CartPole-v1',
            max_episode_steps=self.max_steps,
            render_mode="human" if self.render else None
        )
        self.env.metadata['render_fps'] = self.render_fps

        warnings.filterwarnings("ignore", category=UserWarning)

        # Wrap environment — currently only identity step wrapper
        self.env = StepWrapper(self.env)

        # -------------------------------
        # Initialize DQN Agent
        # -------------------------------
        self.agent = DQNAgent(
            env=self.env,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            clip_grad_norm=self.clip_grad_norm,
            learning_rate=self.learning_rate,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
            tau=self.tau                      # pass soft update parameter
        )

        os.makedirs('plots', exist_ok=True)

    # ====================================================
    # TRAINING LOOP
    # ====================================================
    def train(self):

        total_steps = 0  # tracks total gradient steps for update scheduling

        for episode in range(1, self.max_episodes + 1):

            # Render only on first 5 episodes & last 10 → less GPU usage
            render_this = (episode <= 5) or (episode > self.max_episodes - 10)

            # Re-create environment with rendering toggle
            self.env = gym.make(
                'CartPole-v1',
                max_episode_steps=self.max_steps,
                render_mode="human" if render_this else None
            )
            self.env.metadata['render_fps'] = self.render_fps
            self.env = StepWrapper(self.env)

            # Reset episode state
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            # ------------------------------------------
            # STEP THROUGH ENVIRONMENT
            # ------------------------------------------
            while not done and not truncation:

                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                # Store transition into replay memory
                self.agent.replay_memory.store(state, action, next_state, reward, done, truncation)

                # Learning only starts when memory contains enough data
                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (done or truncation))

                    # SOFT TARGET UPDATE — provides stable Q-learning updates
                    if total_steps % self.update_frequency == 0:
                        self.agent.soft_update()

                state = next_state
                episode_reward += reward
                step_size += 1

            # ------------------------------------------
            # Episode end: log progress
            # ------------------------------------------
            self.reward_history.append(episode_reward)
            total_steps += step_size
            self.epsilon_history.append(self.agent.epsilon_max)

            # Decay exploration schedule
            self.agent.update_epsilon()

            # Periodic checkpoint saving
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode == self.max_episodes:
                    self.plot_training(episode)
                print("\n~~~~~~ Interval Save: Model saved.\n")

            print(
                f"Episode: {episode}, "
                f"Total Steps: {total_steps}, "
                f"Ep Steps: {step_size}, "
                f"Reward: {episode_reward:.2f}, "
                f"Epsilon: {self.agent.epsilon_max:.2f}"
            )

        self.plot_training(episode)

    # ====================================================
    # TEST / EVALUATION LOOP
    # ====================================================
    def test(self, max_episodes):

        # Load model weights trained earlier
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()  # disable gradients

        self.env = gym.make(
            'CartPole-v1',
            max_episode_steps=self.max_steps,
            render_mode="human"
        )
        self.env = StepWrapper(self.env)

        # Episodes run with deterministic policy (epsilon likely near 0)
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

    # ====================================================
    # TRAINING PERFORMANCE PLOTS
    # ====================================================
    def plot_training(self, episode):

        # Smooth reward for readability using 50-episode SMA
        sma = np.convolve(self.reward_history, np.ones(50) / 50, mode='valid')

        reward_history = np.clip(self.reward_history, None, 500)
        sma = np.clip(sma, None, 500)

        # ------- Reward plot -------
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

        # ------- Loss plot -------
        plt.figure()
        plt.title("Network Loss")
        plt.plot(self.agent.loss_history, label='Loss')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.grid(True)

        if episode == self.max_episodes:
            plt.savefig('plots/loss_plot.png', dpi=600, bbox_inches='tight')

        # ------- Exploration decay -------
        plt.figure()
        plt.title("Epsilon per Episode")
        plt.plot(self.epsilon_history, label='Epsilon')
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.grid(True)

        if episode == self.max_episodes:
            plt.savefig('plots/epsilon_plot.png', dpi=600, bbox_inches='tight')

        plt.show()
        plt.close()
