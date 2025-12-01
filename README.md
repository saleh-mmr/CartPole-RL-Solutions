# Double Deep Q-Network (Double DQN) for CartPole-v1

This project implements a Double Deep Q-Network (Double DQN) to solve the CartPole-v1 control problem using the Gymnasium reinforcement learning environment. The objective of the task is to balance an inverted pendulum on a moving cart by applying left or right force.

The implementation emphasizes:

* Stability in learning (soft target updates)
* Data efficiency (experience replay)
* Controlled and scaled inputs (observation normalization)
* Reproducibility and structured software engineering

The system is coded in Python using PyTorch for deep learning.

---

## 1. Algorithm Overview

### 1.1 Deep Q-Network (DQN)

DQN uses a neural network Q(s, a; θ) to estimate the optimal action-value function:

[
Q^*(s, a) = \max_\pi \mathbb{E}[,r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots \mid s_t=s, a_t=a, \pi,]
]

### 1.2 Double DQN Enhancement

Standard DQN tends to **overestimate** Q-values due to coupling of action selection and evaluation. Double DQN decouples them:

* The **main network** selects the next action:
  [
  a' = \arg\max_a Q_{\text{main}}(s', a)
  ]

* The **target network** evaluates it:
  [
  y = r + \gamma , Q_{\text{target}}(s', a')
  ]

This results in more stable and accurate value estimation.

### 1.3 Soft Target Network Updates (Polyak Averaging)

Instead of infrequent and abrupt hard updates, the target network parameters are updated gradually:

[
\theta_{\text{target}} \leftarrow \tau , \theta_{\text{main}} + (1-\tau), \theta_{\text{target}}
]

where ( \tau \in (0,1) ).
This smooth transition helps prevent training divergence.

---

## 2. Key Features

| Feature                       | Purpose                                        |
| ----------------------------- | ---------------------------------------------- |
| Double Q-learning             | Reduces action-value overestimation            |
| Soft target updates (τ=0.005) | Improves convergence stability                 |
| Replay memory                 | Breaks correlation and reuses past transitions |
| Epsilon-greedy exploration    | Balances exploration vs. exploitation          |
| Normalized observations       | Supports better gradient scaling               |
| Training diagnostics          | Reward, loss and epsilon plots                 |

This implementation follows best practices recommended in modern RL research.

---

## 3. Observation and Action Spaces

### Observation (input to neural network)

4 continuous features:

1. Cart position
2. Cart velocity
3. Pole angle
4. Pole angular velocity

These are normalized into the range [0, 1] to improve learning dynamics.

### Action Space

* Discrete(2): apply left or right force

---

## 4. Project Structure

```
DoubleDQN/
│
├── dqn_agent.py              # Double DQN algorithm (learning + soft updates)
├── dqn_network.py            # Fully connected neural network architecture
├── replay_memory.py          # Experience replay buffer
├── model_train_test.py       # Training loop, evaluation loop, plotting
├── step_wrapper.py           # Observation normalization wrapper
├── config.py                 # Device selection, random seeds
├── run.py                    # Entry point for training/testing
│
├── plots/                    # Output graphs for reward, loss, epsilon
└── weights/                  # Saved model states during training
```

Code is structured for readability, modularity, and future extensibility.

---

## 5. Training Configuration

| Parameter             | Default       | Importance                       |
| --------------------- | ------------- | -------------------------------- |
| Learning Rate         | 1e-4          | Stable Q-learning optimization   |
| Discount Factor (γ)   | 0.90          | Short-term reward prioritization |
| Replay Memory Size    | 150,000       | Large sample diversity           |
| Batch Size            | 128           | Balanced learning updates        |
| Max Steps per Episode | 500           | Standard CartPole-v1 limit       |
| Soft update rate (τ)  | 0.005         | Slow stable target transfer      |
| Exploration ε         | 1.0 → 0.02    | Gradually reduces randomness     |
| Epsilon decay         | 0.999/episode | Controls exploration schedule    |

The model is typically trained for up to **3000 episodes**, though convergence can occur sooner.

---

## 6. Running the Project

### Training

In `run.py`, set:

```python
train_mode = True
```

Then execute:

```bash
python run.py
```

### Testing (Evaluation)

To run the trained model with visual rendering:

```bash
python run.py
```

`train_mode = False` automatically loads the saved model and disables exploration.

---

## 7. Results and Output

After training completes, three performance plots are generated:

1. Reward achieved per episode (with 50-episode moving average)
2. Loss curve across training updates
3. Epsilon decay per episode

These are saved in:

```
plots/reward_plot.png
plots/loss_plot.png
plots/epsilon_plot.png
```

The environment is considered solved when:

* The agent achieves an average reward ≥ 475 over 100 episodes
* Maximum episode reward is 500 (no failure within time limit)

Soft update and normalized observations significantly enhance reliability of reaching solving criteria.

---

## 8. Future Work

Potential improvements include:

* Reward clipping for improved numerical stability
* Per-step epsilon decay rather than per-episode
* Prioritized Experience Replay (PER)
* Dueling DQN architecture
* Noisy networks for exploration enhancement
* Support for visualization video export

These extensions can further enhance learning performance and robustness.

---

## 9. License and Use

This implementation is open for academic, educational, and research purposes.
Users may modify and extend the system to investigate reinforcement learning behavior in control tasks.
