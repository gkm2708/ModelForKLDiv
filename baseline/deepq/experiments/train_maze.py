import gym
import os

from baselines import deepq
from baselines.common import models
from baselines import logger

#import gym_lmaze
import gym_lmaze

def main():
    env = gym.make("lmaze-v1")

    HOSTNAME = os.uname()[1]

    # Çhange to log in a different directory
    logger.configure('./logs/'+HOSTNAME)

    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        lr=1e-4,
        total_timesteps=int(1e7),
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=1000,
        target_network_update_freq=1000,
        gamma=0.95,
        checkpoint_path='./logs',  # Change to save model in a different directory
        batch_size=64,
    )

    print("Saving model to maze_model.pkl")
    model.save("maze_model.pkl")


if __name__ == '__main__':
    main()
