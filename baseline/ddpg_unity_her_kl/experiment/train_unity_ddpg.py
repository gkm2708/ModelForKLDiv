import gym
import os

from baselines import deepq

from baselines import ddpg_unity_her_kl

from baselines import logger

from gym_unity.envs.unity_env import UnityEnv
from mlagents.envs import UnityEnvironment



def main():


    #evaluate = True
    #train = True
    #test = False


    evaluate = False
    train = True
    test = False

    train_file_name = "/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget_fullMaze/Build/mazeBasic_Continuous_fixedGoal_test1_100X"
    test_file_name = "/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget_fullMaze/Build/mazeBasic_Continuous_fixedGoal_test1_realtime"
    evaluate_file_name = "/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget_fullMaze/Build/mazeBasic_Continuous_fixedGoal_test1_realtime"

    #train_file_name="/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget_1/Build/mazeBasic_fullDynamic_fullSpeed"
    #test_file_name="/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget_1/Build/mazeBasic_fullDynamic_fullSpeed_test"
    #evaluate_file_name="/homes/gkumar/Documents/UnityProjects/mazeContinuousTarget_1/Build/mazeBasic_fullDynamic_fullSpeed"

    if train:
        env = UnityEnvironment(
            file_name=train_file_name,
            worker_id=1)
    elif test:
        env = UnityEnvironment(
            file_name=test_file_name,
            worker_id=1)
    elif evaluate:
        env = UnityEnvironment(
            file_name=evaluate_file_name,
            worker_id=1)
    else:
        print("decide between test and train mode")
        exit(0)

    print("Created Env")

    host_name = os.uname()[1]
    os.system('mkdir -p ./logs/' + host_name)
    logger.configure('./logs/'+host_name)               # Çhange to log in a different directory

    if train:
        act = ddpg_unity_her_kl.learn(
            "mlp",                                      # conv_only is also a good choice for GridWorld
            env,
            nb_epochs=1000,
            nb_epoch_cycles=100,
            nb_rollout_steps=500,                       # total_timesteps=10000000,
            test=test,
            train=train)
    if test:
        act = ddpg_unity_her_kl.learn(
            "mlp",                                      # conv_only is also a good choice for GridWorld
            env,
            nb_epochs=1000,
            nb_epoch_cycles=100,
            nb_rollout_steps=500,                       # total_timesteps=10000000,
            test=test,
            train=train)
    elif evaluate:
        act = ddpg_unity_her_kl.evaluate(
            "mlp",                                      # conv_only is also a good choice for GridWorld
            env)


    print("Saving model to unity_model.pkl")
    act.save("unity_model.pkl")


if __name__ == '__main__':
    main()