import gym
import os

from baselines import deepq

from baselines import ddpg_unity_her_fixedT

from baselines import logger

from gym_unity.envs.unity_env import UnityEnv
from mlagents.envs import UnityEnvironment



def main():
    env = UnityEnvironment(file_name="/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget/Build/mazeBasic_fixedTarget_fullSpeed", worker_id=1)

    #eval_env = UnityEnvironment(file_name="/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget/Build/mazeBasic_fixedTarget", worker_id=1)

    print("Created Env")

    HOSTNAME = os.uname()[1]
    os.system('mkdir -p ./logs/' + HOSTNAME)
    logger.configure('./logs/'+HOSTNAME) # Çhange to log in a different directory

    act = ddpg_unity_her_fixedT.learn(
        "mlp", # conv_only is also a good choice for GridWorld
        env,
        total_timesteps=10000000)


    print("Saving model to unity_model.pkl")
    act.save("unity_model.pkl")

if __name__ == '__main__':
    main()