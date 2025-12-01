from model_train_test import ModelTrainTest
import os


if __name__ == '__main__':

    # ========================================================
    # 1️⃣ Train or Test mode toggle
    # ========================================================
    train_mode = False  # Set True → Training, False → Testing / Inference Only
    render = not train_mode  # Render only in test mode for efficiency

    # Ensure weights folder exists for saving/ loading trained models
    os.makedirs('./weights', exist_ok=True)

    # ========================================================
    # 2️⃣ All Reinforcement Learning hyperparameters in one dict
    # ========================================================
    RL_hyperparams = {

        # General run configuration
        "train_mode": train_mode,
        "RL_load_path": './weights/weights_3000.pth',  # Model to load for testing
        "save_path": './weights/weights',              # Prefix for saving model checkpoints
        "save_interval": 3000,                         # Save every X episodes

        # ---------------- RL Algorithm Config ----------------
        "clip_grad_norm": 5,                    # Prevent gradient explosion
        "learning_rate": 1e-4,                  # Adam learning rate
        "discount_factor": 0.9,                 # γ — Future reward discounting
                                                # NOTE: Recommended improvement → 0.99 for CartPole

        "batch_size": 128,                      # Replay batch size
        "update_frequency": 20,                 # Soft update frequency (every X gradient steps)
        "max_episodes": 3000 if train_mode else 5,  # Train long, test short
        "max_steps": 500,                       # Maximum environment steps per episode
        "render": render,                       # Real-time animation ON only in test
        "tau": 0.005,                            # Polyak soft update rate

        # ---------------- Epsilon-Greedy Exploration ---------
        "epsilon_max": 1.0 if train_mode else -1,  # -1 → Always choose best action in test mode
        "epsilon_min": 0.02,                       # Minimum exploration
        "epsilon_decay": 0.999,                    # Multiplicative per-episode decay

        # ---------------- Replay Buffer ----------------------
        "memory_capacity": 150_000,                # Large buffer → more training diversity

        # ---------------- Rendering Config -------------------
        "render_fps": 60,                          # Frame-rate for human mode
    }

    # ========================================================
    # 3️⃣ Create Trainer / Tester Object
    # ========================================================
    DRL = ModelTrainTest(RL_hyperparams)

    # ========================================================
    # 4️⃣ Training OR Testing execution
    # ========================================================
    if train_mode:
        DRL.train()  # Launch training loop
    else:
        DRL.test(max_episodes=RL_hyperparams["max_episodes"])  # Run evaluation episodes
