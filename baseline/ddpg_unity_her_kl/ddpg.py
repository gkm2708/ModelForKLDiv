import os
import time
from collections import deque
import pickle

from baselines.ddpg_unity_her_kl.ddpg_learner import DDPG
from baselines.ddpg_unity_her_kl.models import Actor, Critic
from baselines.ddpg_unity_her_kl.memory import Memory
from baselines.ddpg_unity_her_kl.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
import baselines.common.tf_util as U

import matplotlib.pyplot as plt

from baselines import logger
import numpy as np
import cv2

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def learn(network,
          env,
          seed=None,
          total_timesteps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=100,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          test=True,
          train=False,
          **network_kwargs):

    set_global_seeds(seed)
    HOSTNAME = os.uname()[1]

    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    env_type = 'UnityEnvironment'

    if env_type == 'UnityEnvironment':
        default_brain = env.brain_names[0]
        brain = env.brains[default_brain]
        env_info = env.reset(train_mode=True)[default_brain]

        actions_template = np.zeros((brain.vector_action_space_size[0]), dtype=np.float32)

        observations_template = np.zeros((18 + int(env_info.visual_observations[0][0].shape[0]/7) *
                                          int(env_info.visual_observations[0][0].shape[1]/7)),
                                         dtype=np.float32)

        maze_dim = int(env_info.visual_observations[0][0].shape[0]/7)

        nb_actions = actions_template.size

        memory = Memory(limit=int(1e6), action_shape=actions_template.shape, observation_shape=observations_template.shape)

    critic = Critic(network=network, **network_kwargs)
    actor = Actor(nb_actions, network=network, **network_kwargs)

    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    if env_type == 'UnityEnvironment':
        max_action = 1

    logger.info('scaling actions by {} before executing in env'.format(max_action))

    if env_type == 'UnityEnvironment':

        # +++++++++++++ enc +++++++++++++++++ # pass encoder network here
        agent = DDPG(actor,
                     critic,
                     memory,
                     observations_template.shape,
                     actions_template.shape,
                     gamma=gamma,
                     tau=tau,
                     normalize_returns=normalize_returns,
                     normalize_observations=normalize_observations,
                     batch_size=batch_size,
                     action_noise=action_noise,
                     param_noise=param_noise,
                     critic_l2_reg=critic_l2_reg,
                     actor_lr=actor_lr,
                     critic_lr=critic_lr,
                     enable_popart=popart,
                     clip_norm=clip_norm,
                     reward_scale=reward_scale)

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    sess = U.get_session()

    if train:
        agent.initialize(sess)
    if test:
        #agent.load(sess, './logs/'+HOSTNAME+'/model.pkl')
        agent.load(sess, './logs/meppen/model.pkl')

    sess.graph.finalize()

    agent.reset()

    if env_type == 'UnityEnvironment':
        brain_obs = env.reset(train_mode=True)[default_brain]

        p_image = enqueue(brain_obs.visual_observations[0])

        obs = np.concatenate([np.reshape(p_image, (-1)),
                              np.reshape(brain_obs.vector_observations[0][0:4], (-1)),
                              np.reshape(brain_obs.vector_observations[0][6:10], (-1)),
                              np.reshape(brain_obs.vector_observations[0][12:16], (-1)),
                              np.reshape(brain_obs.vector_observations[0][18:24], (-1))], axis=0)

        obs = np.expand_dims(obs, axis=0)


    if eval_env is not None:
        eval_obs = eval_env.reset()

    if env_type == 'UnityEnvironment':
        nenvs = 1

    episode_reward = np.zeros(nenvs, dtype = np.float32)        # vector
    episode_step = np.zeros(nenvs, dtype = int)                 # vector
    episodes = 0                                                # scalar
    t = 0                                                       # scalar

    epoch = 0
    max_sRate = 0
    nenvs = 1
    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0

    for epoch in range(nb_epochs):
        episode_end_result = []
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.
            episode_data = []
            draw = 0
            win = 0
            loss = 0
            if nenvs > 1:
                # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
                # of the environments, so resetting here instead
                agent.reset()
            for t_rollout in range(nb_rollout_steps):

                # Manipulate observations

                # Predict next action.
                if env_type == "UnityEnvironment":

                    if train:
                        action, q, _, _ = agent.step(obs, apply_noise=True, compute_Q=True)
                    if test:
                        action, q, _, _ = agent.learnt_step(obs)

                    brain_obs_new = env.step(max_action * action)[default_brain]

                    p_image = enqueue(brain_obs_new.visual_observations[0])

                    new_obs = np.concatenate([np.reshape(p_image, (-1)),
                                            np.reshape(brain_obs_new.vector_observations[0][0:4], (-1)),
                                            np.reshape(brain_obs_new.vector_observations[0][6:10], (-1)),
                                            np.reshape(brain_obs_new.vector_observations[0][12:16], (-1)),
                                            np.reshape(brain_obs_new.vector_observations[0][18:24], (-1))], axis=0)

                    new_obs = np.expand_dims(new_obs, axis=0)

                    r = brain_obs_new.rewards[0]
                    done = brain_obs_new.local_done[0]
                    info = ""

                # note these outputs are batched from vecenv
                t += 1
                if rank == 0 and render:
                    env.render()
                episode_reward += r
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)

                if env_type == 'UnityEnvironment':
                    action = np.asarray(action)
                    r = np.expand_dims(np.expand_dims(np.asarray(r), axis=0), axis=0)
                    done = np.expand_dims(np.expand_dims(np.asarray(done), axis=0), axis=0)

                if train:
                    # +++++++++++++++++++++ SIMPLE EXPERIENCE REPLAY ++++++++++++++++++++++++++++++++
                    agent.store_transition(obs, action, r, new_obs, done)
                    #the batched data will be unrolled in memory.py's append.

                episode_data.append((obs, action, r, new_obs, done))

                obs = new_obs

                for d in range(len(done)):
                    if done[d]:
                        # Episode done.
                        # ++++++++++++++++++ Episode Summary ++++++++++++++++++
                        episode_end_result.append(r[0][0])

                        epoch_episode_rewards.append(episode_reward[d])
                        episode_rewards_history.append(episode_reward[d])
                        epoch_episode_steps.append(episode_step[d])
                        episode_reward[d] = 0.
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1
                        # ------------------ Episode Summary ------------------


                        # ++++++++++++++++++++++++++++++++++ HER EXPERIENCE REPLAY ++++++++++++++++++++++++++++++++++++
                        if train:
                            her(episode_data, agent, d, maze_dim)
                        # --------------------------------------- HER ---------------------------------------

                        if nenvs == 1:
                            brain_obs = env.reset(train_mode=True)[default_brain]
                            p_image = enqueue(brain_obs.visual_observations[0])

                            obs = np.concatenate([np.reshape(p_image, (-1)),
                                                  np.reshape(brain_obs.vector_observations[0][0:4], (-1)),
                                                  np.reshape(brain_obs.vector_observations[0][6:10], (-1)),
                                                  np.reshape(brain_obs.vector_observations[0][12:16], (-1)),
                                                  np.reshape(brain_obs.vector_observations[0][18:24], (-1))], axis=0)
                            obs = np.expand_dims(obs, axis=0)
                            agent.reset()

                            t_rollout = nb_rollout_steps

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []

            if train:
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

            # Evaluate.
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype = np.float32)
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
                    eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)
                    # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    if render_eval:
                        eval_env.render()
                    eval_episode_reward += eval_r

                    eval_qs.append(eval_q)
                    for d in range(len(eval_done)):
                        if eval_done[d]:
                            eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_rewards_history.append(eval_episode_reward[d])
                            eval_episode_reward[d] = 0.0

        if MPI is not None:
            mpi_size = MPI.COMM_WORLD.Get_size()
        else:
            mpi_size = 1

        print(episode_end_result)
        for x in range(len(episode_end_result)):
            if round(episode_end_result[x], 2) == 0.0:
                win += 1
            else:
                loss += 1

        sRate = (np.sum([1.0 if item == 0.0 else 0.0 for item in episode_end_result])/len(episode_end_result))*100

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/1_win'] = win
        combined_stats['rollout/2_draw'] = draw
        combined_stats['rollout/3_loss'] = loss
        combined_stats['rollout/4_count'] = len(episode_end_result)
        combined_stats['rollout/5_successRate'] = sRate
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_std'] = np.std(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/return_history_std'] = np.std(episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)
        # Evaluation statistics.
        if eval_env is not None:
            combined_stats['eval/return'] = eval_episode_rewards
            combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            combined_stats['eval/Q'] = eval_qs
            combined_stats['eval/episodes'] = len(eval_episode_rewards)

        def as_scalar(x):
            if isinstance(x, np.ndarray):
                assert x.size == 1
                return x[0]
            elif np.isscalar(x):
                return x
            else:
                raise ValueError('expected scalar, got %s'%x)

        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t

        if sRate > max_sRate:
            max_sRate = sRate
            agent.save('./logs/'+HOSTNAME+'/model.pkl')

        if train:
            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])

        if rank == 0:
            logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if rank == 0 and logdir:
            if hasattr(env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(env.get_state(), f)
            if eval_env and hasattr(eval_env, 'get_state'):
                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                    pickle.dump(eval_env.get_state(), f)

    return agent


def enqueue(cv_image):

    value = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    return value


def her(episode_data, agent, d, maze_dim):

    maze_block_size = maze_dim*maze_dim

    replacement_goal = episode_data[len(episode_data) - 1][0][d][maze_block_size + 16 : maze_block_size + 18]

    for x in range(0, len(episode_data) - 2):
        temp1 = np.asarray(episode_data[x][0][d][0:maze_block_size + 16])
        temp2 = np.asarray(episode_data[x][3][d][0:maze_block_size + 16])
        obs = np.concatenate((temp1, replacement_goal), axis=0)
        obs_new = np.concatenate((temp2, replacement_goal), axis=0)

        agent.store_transition(np.expand_dims(obs, axis=0),
                                   np.expand_dims(episode_data[x][1][d], axis=0),
                                   np.expand_dims(episode_data[x][2][d], axis=0),
                                   np.expand_dims(obs_new, axis=0),
                                   np.expand_dims(episode_data[x][4][d], axis=0))

    temp1 = np.asarray(episode_data[len(episode_data) - 1][0][d][0:maze_block_size + 16])
    temp2 = np.asarray(episode_data[len(episode_data) - 1][3][d][0:maze_block_size + 16])

    obs = np.concatenate((temp1, replacement_goal), axis=0)
    obs_new = np.concatenate((temp2, replacement_goal), axis=0)

    agent.store_transition(np.expand_dims(obs, axis=0),
                               np.expand_dims(episode_data[len(episode_data) - 1][1][d], axis=0),
                               np.expand_dims([0.0], axis=0),
                               np.expand_dims(obs_new, axis=0),
                               np.expand_dims(episode_data[len(episode_data) - 1][4][d], axis=0))



def evaluate(network,
          env,
          seed=None,
          reward_scale=1.0,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          batch_size=64, # per MPI worker
          tau=0.01,
          **network_kwargs):

    set_global_seeds(seed)

    env_type = 'UnityEnvironment'

    if env_type == 'UnityEnvironment':
        default_brain = env.brain_names[0]
        brain = env.brains[default_brain]
        env_info = env.reset(train_mode=True)[default_brain]

        actions_template = np.zeros((brain.vector_action_space_size[0]), dtype=np.float32)

        observations_template = np.zeros((18 + int(env_info.visual_observations[0][0].shape[0]/7) *
                                          int(env_info.visual_observations[0][0].shape[1]/7)),
                                         dtype=np.float32)

        nb_actions = actions_template.size

        # make all spaces here
        memory = Memory(limit=int(1e6), action_shape=actions_template.shape,
                        observation_shape=observations_template.shape)

    critic = Critic(network=network, **network_kwargs)
    actor = Actor(nb_actions, network=network, **network_kwargs)

    action_noise = None
    param_noise = None

    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    if env_type == 'UnityEnvironment':
        max_action = 1

    logger.info('scaling actions by {} before executing in env'.format(max_action))

    if env_type == 'UnityEnvironment':
        agent = DDPG(actor,
                     critic,
                     memory,
                     observations_template.shape,
                     actions_template.shape,
                     gamma=gamma,
                     tau=tau,
                     normalize_returns=normalize_returns,
                     normalize_observations=normalize_observations,
                     batch_size=batch_size,
                     action_noise=action_noise,
                     param_noise=param_noise,
                     critic_l2_reg=critic_l2_reg,
                     actor_lr=actor_lr,
                     critic_lr=critic_lr,
                     enable_popart=popart,
                     clip_norm=clip_norm,
                     reward_scale=reward_scale)

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    sess = U.get_session()

    #agent.load(sess, './logs/meppen/model.pkl')
    #agent.load(sess, './logs/lage/model.pkl')
    agent.load(sess, './logs/trier/model.pkl')
    #agent.load(sess, './logs/model.pkl')

    sess.graph.finalize()

    agent.reset()

    l = 100

    q_value_list = np.ndarray((l,l), dtype=np.float32)

    p_image = enqueue()

    for i in range(0,l):
        for j in range(0, l):

            tempf1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5*(i-l/2)/l, 2.5*(j-l/2)/l, 0.0, 0.0]

            obs = np.concatenate([np.reshape(p_image, (-1)),
                              np.reshape(tempf1, (-1))], axis=0)

            obs = np.expand_dims(obs, axis=0)

            _, q, _, _ = agent.learnt_step(obs)

            q_value_list[i][j] = q

    # Create heatmap
    extent = [-l/2, l/2, -l/2, l/2]

    # Plot heatmap
    plt.clf()
    plt.title('Q-Value Map')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.imshow(q_value_list, extent=extent)
    plt.show()

    return agent