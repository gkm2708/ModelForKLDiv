#!/usr/bin/env python

import tensorflow as tf
from lmazeInterface import LMazeInterface
from learner import Learner
from param import Param
import keras.backend as K
import numpy as np
#import rospy
#import os.path
import random                               # Handling random number generation
import os
import time



p = Param()

def main():
    
    ###########  Create environment, learner, action choices etc....##############

    # Build session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    K.set_session(sess)

    # Build Environment
    lmaze = LMazeInterface()

    # Instantiate the learner
    learner = Learner(sess)

    saver = tf.train.Saver()

    loaded = False

    # Placeholders and summaries for report

    #tb_step_loss = tf.placeholder(tf.float32, shape=[], name="1_loss")   
    #tf_step_loss_summary = tf.summary.scalar("1_Step_Loss", tb_step_loss)                   
    #tb_step_reward = tf.placeholder(tf.float32, shape=[], name="2_reward")
    #tf_step_reward_summary = tf.summary.scalar('2_Step_Reward', tb_step_reward)

    tb_loss0 = tf.placeholder(tf.float32, shape=[], name="3_loss0Per_n_TrainingStepOf_m_BatchSize")
    tb_loss1 = tf.placeholder(tf.float32, shape=[], name="3_loss1Per_n_TrainingStepOf_m_BatchSize")
    tf_loss0_summary = tf.summary.scalar("3_loss0Per_n_TrainingStepOf_m_BatchSize", tb_loss0)                   
    tf_loss1_summary = tf.summary.scalar("3_loss1Per_n_TrainingStepOf_m_BatchSize", tb_loss1)   

    tb_reward0 = tf.placeholder(tf.float32, shape=[], name="4_reward0PerRollout")
    tb_reward1 = tf.placeholder(tf.float32, shape=[], name="4_reward1PerRollout")
    tf_reward0_summary = tf.summary.scalar('4_reward0PerRollout', tb_reward0)
    tf_reward1_summary = tf.summary.scalar('4_reward1PerRollout', tb_reward1)

    tb_reward0Avg = tf.placeholder(tf.float32, shape=[], name="5_reward0AvgPer_n_Rollouts")
    tb_reward1Avg = tf.placeholder(tf.float32, shape=[], name="5_reward1AvgPer_n_Rollouts")
    tf_reward0Avg_summary = tf.summary.scalar('5_reward0AvgPer_n_Rollouts', tb_reward0Avg)
    tf_reward1Avg_summary = tf.summary.scalar('5_reward1AvgPer_n_Rollouts', tb_reward1Avg)

    tb_rewardEval = tf.placeholder(tf.float32, shape=[], name="6_n_intermediateEvalPer_m_Rollouts")
    tb_target_rewardEval = tf.placeholder(tf.float32, shape=[], name="7_n_intermediateTargetEvalPer_m_Rollouts")
    tf_rewardEval_summary = tf.summary.scalar('6_n_intermediateEvalPer_m_Rollouts', tb_rewardEval)
    tf_target_rewardEval_summary = tf.summary.scalar('7_n_intermediateTargetEvalPer_m_Rollouts', tb_target_rewardEval)

    # Initialize all the tensorflow variables
    sess.run(tf.global_variables_initializer())              

    # check if buffer exists then remove it as it is unusable if not closed properly
    if os.path.exists(p.BUFFER_FILE):
        os.system('rm '+p.BUFFER_FILE)

    # check and load model if it exists
    if os.path.exists("/homes/gkumar/models/"+p.RUN_ID+"/"+p.HOSTNAME+"/model.ckpt.index"):
        #rospy.loginfo("Staring with loading saved session")
        saver.restore(sess, "/homes/gkumar/models/"+p.RUN_ID+"/"+p.HOSTNAME+"/model.ckpt")
        loaded = True

    # get a writer for tensorboard data
    writer = tf.summary.FileWriter("/homes/gkumar/models/"+p.RUN_ID+"/"+p.HOSTNAME+"/tensorboard", sess.graph)        


    #rospy.loginfo("Created environment, learner, action choices etc....")






    







    """ ################################# Training ############################### """

    if p.TRAINING == True :

        """ ############################# Run pre-training ############################### """
        PreTrain(lmaze, learner, loaded)

        # Now training specific variables etc.    
        avgSession_rewards0 = []
        avgSession_rewards1 = []

        episodeNum, decay_step = p.DECAY_EPISODE_INIT, p.DECAY_STEP_INIT    

        avg_total_reward = -0.0
        train_step = 0
        BEST_REWARD = -0.0
           
        # Now run until the training steps are remaining    
        while decay_step < p.TRAIN_LENGTH:
        
            # Rollout some episodes
            step, episode_data, state, action_sequence = rollout(lmaze, learner, loaded, "train", decay_step, episodeNum)
            replay(episode_data, learner, state, step)

            decay_step += step
            # Train network if training point is reached
            #rospy.loginfo("First Model Update")

            if episodeNum % p.SWITCH_CYCLE >= 0 and episodeNum % p.SWITCH_CYCLE < p.SWITCH_CYCLE_HALF:
                #rospy.loginfo("First Model Update")
                for x in range(0,p.TRAIN_SIZE):
                
                    loss = learner.train1(train_step)             

                    summary0 = sess.run(tf_loss0_summary, feed_dict={tb_loss0: loss})
                    writer.add_summary(summary0, train_step)

                    train_step += 1

                total_reward0 = np.sum([tup[1] for tup in episode_data])
                avgSession_rewards0.append(total_reward0)

                summary = sess.run(tf_reward0_summary, feed_dict={tb_reward0: total_reward0})
                writer.add_summary(summary, episodeNum)


            else:
                #rospy.loginfo("Second Model Update")
                for x in range(0,p.TRAIN_SIZE):
                
                    loss = learner.train2(train_step)             

                    summary1 = sess.run(tf_loss1_summary, feed_dict={tb_loss1: loss})
                    writer.add_summary(summary1, train_step)

                    train_step += 1

                total_reward1 = np.sum([tup[1] for tup in episode_data])
                avgSession_rewards1.append(total_reward1)

                summary = sess.run(tf_reward1_summary, feed_dict={tb_reward1: total_reward1})
                writer.add_summary(summary, episodeNum)
        
            # increment episode count
            episodeNum += 1

            ################################# Reporting and Backup ###############################

            # save average stats
            if episodeNum % p.WHEN_SAVE_STATS == 0 and episodeNum != 0:
                avg_total_reward = np.sum(avgSession_rewards0[-p.WHEN_SAVE_STATS:])/float(p.WHEN_SAVE_STATS)
                summary2 = sess.run(tf_reward0Avg_summary, feed_dict={tb_reward0Avg: avg_total_reward})
                writer.add_summary(summary2, episodeNum)                

                avg_total_reward = np.sum(avgSession_rewards1[-p.WHEN_SAVE_STATS:])/float(p.WHEN_SAVE_STATS)
                summary2 = sess.run(tf_reward1Avg_summary, feed_dict={tb_reward1Avg: avg_total_reward})
                writer.add_summary(summary2, episodeNum)                

            # Test network if test point reached
            if (episodeNum % p.WHEN_EVALUATED == 0  ) and episodeNum != 0:
                BEST_REWARD = testNetRun(sess, writer, episodeNum, lmaze, learner, "train", tf_rewardEval_summary, tb_rewardEval, tf_target_rewardEval_summary, tb_target_rewardEval, loaded, BEST_REWARD, saver)















    elif p.TEST == True :
        episodeNum = 0
        _ = testNetRun(sess, writer, episodeNum, lmaze, learner, "test", tf_rewardEval_summary, tb_rewardEval, tf_target_rewardEval_summary, tb_target_rewardEval, loaded, 0.0, saver)






def PreTrain(lmaze, learner, loaded):

    trainStep = 0

    # if loaded then we need decay_step too
    decay_step = 0
    episodeNum = 0
    while trainStep < p.PRETRAIN_LENGTH:

        step, episode_data, state, action_sequence = rollout(lmaze, learner, loaded, "pretrain", decay_step, episodeNum)

        trainStep += step
        replay(episode_data, learner, state, step)
        episodeNum += 1

    #rospy.loginfo("Memory popup done")






def testNetRun(sess, writer, episode, lmaze, learner, mode, tf_rewardEval_summary, tb_rewardEval, tf_target_rewardEval_summary, tb_target_rewardEval, loaded, BEST_REWARD, saver):

    episodeNum  = 0
    
    session_data = []
    while episodeNum < p.HOW_MANY_EVALUATIONS:
        step, episode_data, state, action_sequence = rollout(lmaze, learner, loaded, "test", "0", episodeNum)
        episodeNum += 1
        session_data.append(np.sum([tup[1] for tup in episode_data]))
    avg_total_reward = np.sum(session_data[:])/float(p.HOW_MANY_EVALUATIONS)
    summary3 = sess.run(tf_rewardEval_summary, feed_dict={tb_rewardEval: avg_total_reward})
    writer.add_summary(summary3, episode)

    session_data = []
    while episodeNum < 2*p.HOW_MANY_EVALUATIONS:
        step, episode_data, state, action_sequence = rollout(lmaze, learner, loaded, "test", "0", episodeNum)
        episodeNum += 1
        session_data.append(np.sum([tup[1] for tup in episode_data]))
    avg_total_reward1 = np.sum(session_data[:])/float(p.HOW_MANY_EVALUATIONS)
    summary4 = sess.run(tf_target_rewardEval_summary, feed_dict={tb_target_rewardEval: avg_total_reward1})
    writer.add_summary(summary4, episode)

    # save model if previous best performance is supersided
    if (avg_total_reward > BEST_REWARD or avg_total_reward1 > BEST_REWARD) and mode == "train":
        if avg_total_reward > BEST_REWARD:
            BEST_REWARD = avg_total_reward
        if avg_total_reward1 > BEST_REWARD:
            BEST_REWARD = avg_total_reward1
        saver.save(sess, "/homes/gkumar/models/"+p.RUN_ID+"/"+p.HOSTNAME+"/model_Best_"+str(episode)+".ckpt") 
        os.system('mkdir -p /homes/gkumar/models/'+p.RUN_ID+'/'+p.HOSTNAME+'/best')
        os.system('mv /homes/gkumar/models/'+p.RUN_ID+'/'+p.HOSTNAME+'/model_Best_'+str(episode)+'.ckpt.* /homes/gkumar/models/'+p.RUN_ID+'/'+p.HOSTNAME+'/best/')

    elif ( episode % p.WHEN_BACKUP == 0 ) and episode != 0:
        saver.save(sess, "/homes/gkumar/models/"+p.RUN_ID+"/"+p.HOSTNAME+"/model_"+str(episode)+".ckpt") 
    return BEST_REWARD




def rollout(lmaze, learner, loaded, mode, decay_step, episodeNum):

    reward = -0.0
    step = 0        
    episode_data = []
    action_sequence = []
    probability = 0.0
        
    while step < p.MAX_STEPS:            

        if step == 0:
            lmaze.reset()

            action = random.choice(p.POSSIBLE_ACTIONS)
                
            state, reward, done, info = lmaze.step(p.POSSIBLE_ACTIONS.index(action))
            while not(info['newState']):
                state, reward, done, info = lmaze.step(p.POSSIBLE_ACTIONS.index(action))


        if mode == "train" or (mode == "pretrain" and loaded):                        # predict epsilon-greedy action
            if episodeNum % p.SWITCH_CYCLE >= 0 and episodeNum % p.SWITCH_CYCLE < p.SWITCH_CYCLE_HALF:
                action, probability = learner.predict_action1(decay_step+step, state)
            else:
                action, probability = learner.predict_action2(decay_step+step, state)
        elif mode == "test":                             # predict action
            if episodeNum % (2*p.HOW_MANY_EVALUATIONS) >= 0 and episodeNum % (2*p.HOW_MANY_EVALUATIONS) < p.HOW_MANY_EVALUATIONS :
                #time.sleep(0.01)
                action = learner.learned_action1(state)
            else :
                #time.sleep(0.01)
                action = learner.learned_action2(state)
            action_sequence.append(p.POSSIBLE_ACTIONS.index(action))
        elif mode == "target":                             # predict action
            action = learner.learned_target_action(state)
        else :                                              # random action
            action = random.choice(p.POSSIBLE_ACTIONS)

        action_sequence.append(p.POSSIBLE_ACTIONS.index(action))

        state, reward, done, info = lmaze.step(p.POSSIBLE_ACTIONS.index(action))
        while not(info['newState']):
            state, reward, done, info = lmaze.step(p.POSSIBLE_ACTIONS.index(action))

        if done or step == p.MAX_STEPS-1:
            episode_data.append((np.asarray(action),reward, None))                        
            step += 1
            """
            if p.STEP_PENALTY == True:
                if mode == "test":
                    rospy.loginfo("episodeNum : %s, decayStep : %s, step : %s, probability : %s, Reward : %s, stepPenaltyReward : %s, action sequence %s", episodeNum, decay_step, step, probability, np.sum([tup[1] for tup in episode_data]), np.float(np.sum([tup[1] for tup in episode_data]))/np.float(step), action_sequence)
                else:
                    rospy.loginfo("episodeNum : %s, decayStep : %s, step : %s, probability : %s, Reward : %s, stepPenaltyReward : %s", episodeNum, decay_step, step, probability, np.sum([tup[1] for tup in episode_data]), np.float(np.sum([tup[1] for tup in episode_data]))/np.float(step))
            else :
                if mode == "test":
                    rospy.loginfo("episodeNum : %s, decayStep : %s, step : %s, probability : %s, Reward : %s, action sequence %s", episodeNum, decay_step, step, probability, np.sum([tup[1] for tup in episode_data]), action_sequence)
                else:
                    rospy.loginfo("episodeNum : %s, decayStep : %s, step : %s, probability : %s, Reward : %s", episodeNum, decay_step, step, probability, np.sum([tup[1] for tup in episode_data]))
            """
            break
        else:
            episode_data.append((np.asarray(action), reward, state))
            step += 1
    return step, episode_data, state, action_sequence



def replay(episode_data, learner, next_state, step):
    
    if p.STEP_PENALTY == True:

        for x in range(0,step-2):
            learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                           episode_data[x][2])                
        learner.add_experience(episode_data[step-1][0], np.float(episode_data[step-1][1])/np.float(step), None)

        if p.ENDPOINT_AUGMENT == True :
            replacement_goal = np.array(next_state[p.GRID_SIZE*p.GRID_SIZE*0:p.GRID_SIZE*p.GRID_SIZE*1])           
            for x in range(0,step-2):
                temp1 = np.asarray(episode_data[x][2][p.GRID_SIZE*p.GRID_SIZE*0:p.GRID_SIZE*p.GRID_SIZE*4])                                                          
                if p.RANDOM_GOAL == True:
                    temp2 = np.asarray(episode_data[x][2][p.GRID_SIZE*p.GRID_SIZE*5:p.GRID_SIZE*p.GRID_SIZE*6])                                                          
                    learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                            np.concatenate((temp1, replacement_goal, temp2), axis=0))
                else :
                    learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                            np.concatenate((temp1, replacement_goal), axis=0))
            learner.add_experience(episode_data[step-1][0], 1.0/np.float(step), None)

        if p.EPISODE_AUGMENT == True:
            if step > p.HER_SAMPLE+2:
                sampl = np.random.random_integers(low=1, high=step-2, size=(p.HER_SAMPLE,))
                for x in range(0, p.HER_SAMPLE):
                    replacement_goal = np.asarray(episode_data[sampl[x]][2][p.GRID_SIZE*p.GRID_SIZE*0:p.GRID_SIZE*p.GRID_SIZE*1])                    
                    for y in range(1, sampl[x]-1):
                        temp1 = np.asarray(episode_data[y][2][p.GRID_SIZE*p.GRID_SIZE*0:p.GRID_SIZE*p.GRID_SIZE*4])                                                          
                        if p.RANDOM_GOAL == True:
                            temp2 = np.asarray(episode_data[x][2][p.GRID_SIZE*p.GRID_SIZE*5:p.GRID_SIZE*p.GRID_SIZE*6])                                                          
                            learner.add_experience(episode_data[y][0],
                                           episode_data[y][1],
                                            np.concatenate((temp1, replacement_goal, temp2), axis=0))
                        else:
                            learner.add_experience(episode_data[y][0],
                                           episode_data[y][1],
                                            np.concatenate((temp1, replacement_goal), axis=0))                                                           
                    learner.add_experience(episode_data[sampl[x]][0], 1.0/np.float(sampl[x]), None)

    else :
        for x in range(0,step-2):
            learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                           episode_data[x][2])                
        learner.add_experience(episode_data[step-1][0], episode_data[step-1][1], None)

        if p.ENDPOINT_AUGMENT == True :
            replacement_goal = np.array(next_state[p.GRID_SIZE*p.GRID_SIZE*0:p.GRID_SIZE*p.GRID_SIZE*1])           
            for x in range(0,step-2):
                temp1 = np.asarray(episode_data[x][2][p.GRID_SIZE*p.GRID_SIZE*0:p.GRID_SIZE*p.GRID_SIZE*4])                                                          
                if p.RANDOM_GOAL == True:
                    temp2 = np.asarray(episode_data[x][2][p.GRID_SIZE*p.GRID_SIZE*5:p.GRID_SIZE*p.GRID_SIZE*6])                                                          
                    learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                            np.concatenate((temp1, replacement_goal, temp2), axis=0))
                else:
                    learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                            np.concatenate((temp1, replacement_goal), axis=0))
            learner.add_experience(episode_data[step-1][0], 1.0, None)

        if p.EPISODE_AUGMENT == True:
            if step > p.HER_SAMPLE+2:
                sampl = np.random.random_integers(low=1, high=step-2, size=(p.HER_SAMPLE,))
                for x in range(0, p.HER_SAMPLE):
                    replacement_goal = np.asarray(episode_data[sampl[x]][2][p.GRID_SIZE*p.GRID_SIZE*0:p.GRID_SIZE*p.GRID_SIZE*1])                    
                    for y in range(0, sampl[x]-1):
                        temp1 = np.asarray(episode_data[y][2][p.GRID_SIZE*p.GRID_SIZE*0:p.GRID_SIZE*p.GRID_SIZE*4])                                                          
                        if p.RANDOM_GOAL == True:
                            temp2 = np.asarray(episode_data[x][2][p.GRID_SIZE*p.GRID_SIZE*5:p.GRID_SIZE*p.GRID_SIZE*6])                                                          
                            learner.add_experience(episode_data[y][0],
                                           episode_data[y][1],
                                            np.concatenate((temp1, replacement_goal, temp2), axis=0))
                        else:
                            learner.add_experience(episode_data[y][0],
                                           episode_data[y][1],
                                            np.concatenate((temp1, replacement_goal), axis=0))
                    learner.add_experience(episode_data[sampl[x]][0], 1.0, None)


if __name__ == "__main__":
    main()