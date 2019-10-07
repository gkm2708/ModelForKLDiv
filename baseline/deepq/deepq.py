import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq

from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session


from baselines.deepq.models import build_q_func
from baselines.deepq.models_plan import build_q_func_plan

import matplotlib.pyplot as plt
plt.ion()
plt.show(block=False)

VISUALIZE = True


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=1000000,
          train_freq=1,
          print_freq=100,
          checkpoint_freq=10000,
          target_network_update_freq=500,
          learning_starts=1000,
          train_freq_plan=10,
          print_freq_plan=10,
          checkpoint_freq_plan=1000,
          target_network_update_freq_plan=5000,
          learning_starts_plan=20000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          batch_size=32,
          checkpoint_path=None,
          gamma=1.0,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs
        ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    # +++++++++++++  Build Behavioural network
    q_func = build_q_func(network, **network_kwargs)
    # +++++++++++++  Build planning network
    plan_func = build_q_func_plan(network, **network_kwargs)

    observation_space = env.observation_space

    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    # +++++++++++++  build Behavioural network
    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
    )
    # +++++++++++++  build planning network with same param but different action space
    plan, train_plan, update_plan_target, debug_plan = deepq.build_train_plan(
        make_obs_ph=make_obs_ph,
        q_func=plan_func,
        num_actions=env.fovea * env.fovea,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
    )

    # +++++++++++++  act param
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }
    # +++++++++++++  plan param
    plan_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': plan_func,
        'num_actions': env.observation_space.shape[1]*env.observation_space.shape[2],
    }

    # +++++++++++++  ActWrapper for behavioural param
    act = ActWrapper(act, act_params)
    # +++++++++++++  ActWrapper for plan param
    plan = ActWrapper(plan, plan_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
        # +++++++++++++  build buffer
        replay_buffer_plan = ReplayBuffer(buffer_size)
        beta_schedule_plan = None

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()

    # +++++++++++++  update behavioural target
    update_target()
    # +++++++++++++  update planning target
    update_plan_target()

    # +++++++++++++  build reward variables
    plan_episode_rewards = [0.0]
    saved_mean_reward_plan = None

    # +++++++++++++  build reward variables
    episode_rewards = [0.0]
    saved_mean_reward = None
    global_obs = env.reset()
    reset = True






    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        t_fovea = 0
        globalGoalReachCount = 0
        localGoalReachCount = 0

        while t_fovea < total_timesteps:

            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t_fovea)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.

                update_param_noise_threshold = -np.log(1. - exploration.value(t_fovea) + exploration.value(t_fovea) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            goal = plan(np.array(global_obs)[None], update_eps=update_eps, **kwargs)[0]

            # set local goal and receive updated fovea
            obs = env.setFovealGoal(int(goal/5), int(goal%5))

            obs_fovea = obs
            obs_fovea_next = obs

            originalReward, fovealReward, cummulativeFovealReward = 0.0, 0.0, 0.0

            breaker = True
            while (breaker):
                t_fovea += 1
                if callback is not None:
                    if callback(locals(), globals()):
                        break

                action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
                reset = False

                new_obs, originalReward, fovealReward, local_done, global_done, _ = env.step(env_action)

                episode_rewards[-1] += fovealReward
                cummulativeFovealReward += originalReward

                replay_buffer.add(obs, action, fovealReward, new_obs, float(local_done))
                obs = new_obs

                if local_done:
                    episode_rewards.append(0.0)
                    reset = True
                    obs_fovea_next = new_obs

                if t_fovea > learning_starts and t_fovea % train_freq == 0:

                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if prioritized_replay:
                        experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t_fovea))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None

                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                    if prioritized_replay:
                        new_priorities = np.abs(td_errors) + prioritized_replay_eps
                        replay_buffer.update_priorities(batch_idxes, new_priorities)

                if t_fovea > learning_starts_plan and t_fovea % train_freq_plan == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    obses_t_plan, actions_plan, rewards_plan, obses_tp1_plan, dones_plan = replay_buffer_plan.sample(
                        batch_size)
                    weights_plan, batch_idxes_plan = np.ones_like(rewards_plan), None
                    td_errors_plan = train_plan(obses_t_plan, actions_plan, rewards_plan, obses_tp1_plan, dones_plan,
                                                weights_plan)
                if t_fovea > learning_starts and t_fovea % target_network_update_freq == 0:
                    update_target()

                if t_fovea > learning_starts_plan and t_fovea % target_network_update_freq_plan == 0:
                    update_plan_target()

                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                num_episodes = len(episode_rewards)

                if local_done and int(fovealReward) == 1:
                    localGoalReachCount += 1

                if local_done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                    logger.record_tabular("steps", t_fovea)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t_fovea)))
                    logger.record_tabular("Local Reached Total Count", localGoalReachCount)
                    logger.dump_tabular()

                if (checkpoint_freq is not None and t_fovea > learning_starts and
                        num_episodes > 100 and t_fovea % checkpoint_freq == 0):

                    if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:

                        if print_freq is not None:

                            logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))

                        save_variables(model_file)
                        model_saved = True
                        saved_mean_reward = mean_100ep_reward

                if local_done or global_done:
                    breaker = False

                if global_done and int(originalReward) == 1:
                    globalGoalReachCount += 1

                if t_fovea % 10000 == 0:
                    env.VISUALIZE = True

                if (t_fovea - 1000) % 10000 == 0:
                    env.VISUALIZE = False

            replay_buffer_plan.add(obs_fovea, goal, cummulativeFovealReward, obs_fovea_next, float(global_done))

            plan_episode_rewards[-1] += cummulativeFovealReward
            mean_100ep_reward_plan = round(np.mean(plan_episode_rewards[-101:-1]), 1)
            num_plan_episodes = len(plan_episode_rewards)

            if global_done:
                global_obs = env.reset()
                plan_episode_rewards.append(0.0)
                reset = True


            if global_done and print_freq_plan is not None and len(plan_episode_rewards) % print_freq_plan == 0:
                logger.record_tabular("plan steps", t_fovea)
                logger.record_tabular("plan episodes", num_plan_episodes)
                logger.record_tabular("plan mean 100 episode reward", mean_100ep_reward_plan)
                logger.record_tabular("% time spent exploring in plan", int(100 * exploration.value(t_fovea)))
                logger.record_tabular("Goal Reached Total Count", globalGoalReachCount)

                logger.dump_tabular()

            if (checkpoint_freq_plan is not None and t_fovea > learning_starts_plan and
                    num_plan_episodes > 100 and t_fovea % checkpoint_freq == 0):

                if saved_mean_reward_plan is None or mean_100ep_reward_plan > saved_mean_reward_plan:

                    if print_freq is not None:

                        logger.log("Saving plan model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward_plan, mean_100ep_reward_plan))

                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward_plan = mean_100ep_reward_plan



        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act
