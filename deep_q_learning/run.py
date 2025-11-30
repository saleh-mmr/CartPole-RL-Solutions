from model_train_test import ModelTrainTest
import os


if __name__ == '__main__':

    # Train or test mode
    train_mode = False
    render = not train_mode

    # Make sure weight folder exists
    os.makedirs('./weights', exist_ok=True)

    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": './weights/weights_900.pth',
        "save_path": './weights/weights',
        "save_interval": 900,

        # RL config
        "clip_grad_norm": 5,
        "learning_rate": 1e-4,
        "discount_factor": 0.9,
        "batch_size": 128,
        "update_frequency": 20,
        "max_episodes": 2000 if train_mode else 5,
        "max_steps": 500,              # CartPole can go up to 500 steps
        "render": render,

        # Exploration
        "epsilon_max": 1.0 if train_mode else -1,  # -1 = eval mode
        "epsilon_min": 0.02,
        "epsilon_decay": 0.999,

        # Memory
        "memory_capacity": 150_000,

        # Visuals
        "render_fps": 60,
    }

    # Create trainer/tester
    DRL = ModelTrainTest(RL_hyperparams)

    # Train
    if train_mode:
        DRL.train()
    # Test
    else:
        DRL.test(max_episodes=RL_hyperparams["max_episodes"])
