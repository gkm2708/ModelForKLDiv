#!/usr/bin/env python

import tensorflow as tf
from lmazeInterface import LMazeInterface
from learner import Learner
import keras.backend as K
import numpy as np
import rospy
#import os.path
import random                               # Handling random number generation
import os
import time


#################################################################################################################
######################################### MODEL HYPERPARAMETERS##################################################
#################################################################################################################
outputDim = 7
imageDim = 84

poseDim = 0
dyna = False # Should not be true if gyro is not set

randomGoal = False
randomHole = False
randomBall = False

if randomHole == True:
    STATE_SIZE = [outputDim*outputDim*6 + poseDim]                      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
    STATE_SHAPE = np.zeros((outputDim*outputDim*6 + poseDim), dtype=np.float32)
else:
    STATE_SIZE = [outputDim*outputDim*5 + poseDim]                      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
    STATE_SHAPE = np.zeros((outputDim*outputDim*5 + poseDim), dtype=np.float32)


ACTION_SIZE = 4                             #
FRAME_SKIP = 3

DECAY_STEP_INIT = 0
DECAY_EPISODE_INIT = 0

LEARNING_RATE = 0.0001                    # Alpha 10^-7
DECAY_WINDOW = 20000
DECAY_BASE = 0.95
DECAY_UNTIL = 1000000

### TRAINING HYPERPARAMETERS
#MAX_EPISODE = 500000
MAX_STEPS = 20                             # maximum time step in one episode
BATCH_SIZE = 64             
TRAIN_SIZE = 102
# Exploration parameters for epsilon greedy strategy
EXPLORE_START = 1.0                         # exploration probability at start
EXPLORE_STOP = 0.1                         # minimum exploration probability 
DECAY_RATE = 0.000002                         # exponential decay rate for exploration prob

# Q learning hyperparameters
GAMMA = 0.99                                # reward discount in TD error
ALPHA = 0.01

### MEMORY HYPERPARAMETERS
PRETRAIN_LENGTH = 50000                    # Number of experiences stored in the Memory when initialized for the first time
MEMORY_SIZE     = 500000                   # Number of experiences the Memory can keep
TRAIN_LENGTH    = 10000000

WHEN_BACKUP = 5000
WHEN_SAVED = 5000

WHEN_EVALUATED = 1000
HOW_MANY_EVALUATIONS = 100

WHEN_SAVE_STATS = 100

WHEN_UPDATE_TARGET = 100
WHEN_TRAINED = 5

SWITCH_CYCLE = 10
SWITCH_CYCLE_HALF = 5


### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
TRAINING = True
PRE_TRAINING = TRAINING
TEST = True

ENDPOINT_AUGMENT = True
EPISODE_AUGMENT = True
HER_SAMPLE = 3
STEP_PENALTY = False

NUM_LAYERS=5

RUN_ID = "DDQN_GRID_04_July"


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
def create_environment():
    lmaze = LMazeInterface(FRAME_SKIP, outputDim, poseDim, dyna, randomGoal, randomHole, randomBall)
    #lmaze = LMazeInterface()
    xdotplus = [1, 0, 0, 0]
    xdotminus = [0, 1, 0, 0]
    ydotplus = [0, 0, 1, 0]
    ydotminus = [0, 0, 0, 1]
    possible_actions = [xdotplus, ydotplus, xdotminus, ydotminus]

    return lmaze, possible_actions
    #return possible_actions

def printParam(myhost):
    
    os.system('mkdir -p /homes/gkumar/models/'+RUN_ID+'/'+myhost)

    file = open("/homes/gkumar/models/"+RUN_ID+"/"+myhost+"/initParam.txt","w+") 

    file.write("\n outputDim = "+str(outputDim))
    file.write("\n imageDim = "+str(imageDim))
    file.write("\n poseDim = "+str(poseDim))
    file.write("\n dyna = "+str(dyna))
    file.write("\n randomGoal = "+str(randomGoal))
    file.write("\n randomHole = "+str(randomHole))    
    file.write("\n randomBall = "+str(randomBall))    

    file.write("\n STATE_SIZE = "+str(STATE_SIZE))
    file.write("\n STATE_SHAPE = "+str(STATE_SHAPE.shape))
    file.write("\n ACTION_SIZE = "+str(ACTION_SIZE))
    file.write("\n FRAME_SKIP = "+str(FRAME_SKIP))
    file.write("\n DECAY_STEP_INIT = "+str(DECAY_STEP_INIT))
    file.write("\n DECAY_EPISODE_INIT = "+str(DECAY_EPISODE_INIT))
    file.write("\n LEARNING_RATE = "+str(LEARNING_RATE))
    file.write("\n DECAY_WINDOW = "+str(DECAY_WINDOW))
    file.write("\n DECAY_BASE = "+str(DECAY_BASE))
    file.write("\n DECAY_UNTIL = "+str(DECAY_UNTIL))
    file.write("\n MAX_STEPS = "+str(MAX_STEPS))
    file.write("\n BATCH_SIZE = "+str(BATCH_SIZE))             
    file.write("\n TRAIN_SIZE = "+str(TRAIN_SIZE))
    file.write("\n EXPLORE_START = "+str(EXPLORE_START))
    file.write("\n EXPLORE_STOP = "+str(EXPLORE_STOP))
    file.write("\n DECAY_RATE = "+str(DECAY_RATE))
    file.write("\n GAMMA = "+str(GAMMA))
    file.write("\n ALPHA = "+str(ALPHA))
    file.write("\n PRETRAIN_LENGTH = "+str(PRETRAIN_LENGTH))
    file.write("\n MEMORY_SIZE = "+str(MEMORY_SIZE))
    file.write("\n TRAIN_LENGTH = "+str(TRAIN_LENGTH))
    file.write("\n WHEN_BACKUP = "+str(WHEN_BACKUP))
    file.write("\n WHEN_SAVED = "+str(WHEN_SAVED))
    file.write("\n WHEN_EVALUATED = "+str(WHEN_EVALUATED))
    file.write("\n HOW_MANY_EVALUATIONS = "+str(HOW_MANY_EVALUATIONS))
    file.write("\n WHEN_SAVE_STATS = "+str(WHEN_SAVE_STATS))
    file.write("\n WHEN_UPDATE_TARGET = "+str(WHEN_UPDATE_TARGET))
    file.write("\n WHEN_TRAINED = "+str(WHEN_TRAINED))
    file.write("\n SWITCH_CYCLE = "+str(SWITCH_CYCLE))
    file.write("\n SWITCH_CYCLE_HALF = "+str(SWITCH_CYCLE_HALF))
    file.write("\n TRAINING = "+str(TRAINING))
    file.write("\n PRE_TRAINING = "+str(PRE_TRAINING))
    file.write("\n TEST = "+str(TEST))
    file.write("\n ENDPOINT_AUGMENT = "+str(ENDPOINT_AUGMENT))
    file.write("\n EPISODE_AUGMENT = "+str(EPISODE_AUGMENT))
    file.write("\n HER_SAMPLE = "+str(HER_SAMPLE))
    file.write("\n STEP_PENALTY = "+str(STEP_PENALTY))

    file.write("\n NUM_LAYERS = "+str(NUM_LAYERS))
    file.write("\n RUN_ID = "+str(RUN_ID))

    file.close() 

def main():
    # Get hostname
    myhost = os.uname()[1]
    rospy.loginfo(myhost)
    
    # write param file here
    printParam(myhost)

    ###########  Create environment, learner, action choices etc....##############

    # Build session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    K.set_session(sess)

    # Build Environment and action choices
    lmaze, possible_actions = create_environment()
    #possible_actions = create_environment()

    # check if buffer exists then remove it as it is unusable if not closed properly
    if os.path.exists("/vol/speech/gaurav/datadrive/"+myhost+"/rbuffer.hdf"):
        os.system('rm /vol/speech/gaurav/datadrive/'+myhost+'/rbuffer.hdf')

    # Instantiate the learner
    learner = Learner(sess, 
                      GAMMA, 
                      ALPHA,
                      LEARNING_RATE, 
                      STATE_SIZE, 
                      ACTION_SIZE, 
                      EXPLORE_START, 
                      EXPLORE_STOP, 
                      DECAY_RATE, 
                      possible_actions, 
                      BATCH_SIZE,
                      MEMORY_SIZE,
                      PRETRAIN_LENGTH,
                      STATE_SHAPE,
                      DECAY_WINDOW,
                      DECAY_BASE,
                      DECAY_UNTIL,
                      outputDim)

    ##########################################################################
    rospy.loginfo("Created environment, learner, action choices etc....")
    ##########################################################################


    saver = tf.train.Saver()

    loaded = False

    # check and load model if it exists

    if os.path.exists("/homes/gkumar/models/"+RUN_ID+"/"+myhost+"/model.ckpt.index"):
        rospy.loginfo("Staring with loading saved session")
        saver.restore(sess, "/homes/gkumar/models/"+RUN_ID+"/"+myhost+"/model.ckpt")
        loaded = True

    # Placeholders and summaries for report

    #tb_step_loss = tf.placeholder(tf.float32, shape=[], name="1_loss")   
    #tf_step_loss_summary = tf.summary.scalar("1_Step_Loss", tb_step_loss)                   

    #tb_step_reward = tf.placeholder(tf.float32, shape=[], name="2_reward")
    #tf_step_reward_summary = tf.summary.scalar('2_Step_Reward', tb_step_reward)

    tb_loss0 = tf.placeholder(tf.float32, shape=[], name="3_loss0Per_n_TrainingStepOf_m_BatchSize")
    tb_loss1 = tf.placeholder(tf.float32, shape=[], name="3_loss1Per_n_TrainingStepOf_m_BatchSize")

    #tf_loss_summary = tf.summary.scalar("3_loss0Per_n_TrainingStepOf_m_BatchSize", tb_loss0)                   

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


    # get a writer for tensorboard data
    writer = tf.summary.FileWriter("/homes/gkumar/models/"+RUN_ID+"/"+myhost+"/tensorboard", sess.graph)        
    

    # Run pre-training only if memory cannot be loaded
    if learner.load == False and TRAINING == True:
        PreTrain(lmaze, learner, possible_actions, loaded)




    ################################# Training ###############################

    if TRAINING == True :

        # Now training specific variables etc.    
        avgSession_rewards0 = []
        avgSession_rewards1 = []

        episodeNum, decay_step = DECAY_EPISODE_INIT, DECAY_STEP_INIT    

        avg_total_reward = -0.0
        train_step = 0
        BEST_REWARD = -0.0
   
        
        # Now run until the training steps are remaining    
        while decay_step < TRAIN_LENGTH:
        
            # Rollout some episodes
            step, episode_data, state, action_sequence = rollout(lmaze, learner, possible_actions, loaded, "train", decay_step, episodeNum)
            replay(episode_data, learner, state, step)

            decay_step += step
            # Train network if training point is reached
            #rospy.loginfo("First Model Update")

            if episodeNum % SWITCH_CYCLE >= 0 and episodeNum % SWITCH_CYCLE < SWITCH_CYCLE_HALF:
                #rospy.loginfo("First Model Update")
                for x in range(0,TRAIN_SIZE):
                
                    loss = learner.train1(train_step)             

                    summary0 = sess.run(tf_loss0_summary, feed_dict={tb_loss0: loss})
                    writer.add_summary(summary0, train_step)

                    train_step += 1

                total_reward0 = np.sum(episode_data[-1][1])
                avgSession_rewards0.append(total_reward0)

                summary = sess.run(tf_reward0_summary, feed_dict={tb_reward0: total_reward0})
                writer.add_summary(summary, episodeNum)


            else:
                #rospy.loginfo("Second Model Update")
                for x in range(0,TRAIN_SIZE):
                
                    loss = learner.train2(train_step)             

                    summary1 = sess.run(tf_loss1_summary, feed_dict={tb_loss1: loss})
                    writer.add_summary(summary1, train_step)

                    train_step += 1

                total_reward1 = np.sum(episode_data[-1][1])
                avgSession_rewards1.append(total_reward1)

                summary = sess.run(tf_reward1_summary, feed_dict={tb_reward1: total_reward1})
                writer.add_summary(summary, episodeNum)
        
            # increment episode count
            episodeNum += 1

            ################################# Reporting and Backup ###############################

            # save average stats
            if episodeNum % WHEN_SAVE_STATS == 0 and episodeNum != 0:
                avg_total_reward = np.sum(avgSession_rewards0[-WHEN_SAVE_STATS:])/float(WHEN_SAVE_STATS)
                summary2 = sess.run(tf_reward0Avg_summary, feed_dict={tb_reward0Avg: avg_total_reward})
                writer.add_summary(summary2, episodeNum)                

                avg_total_reward = np.sum(avgSession_rewards1[-WHEN_SAVE_STATS:])/float(WHEN_SAVE_STATS)
                summary2 = sess.run(tf_reward1Avg_summary, feed_dict={tb_reward1Avg: avg_total_reward})
                writer.add_summary(summary2, episodeNum)                

            # Test network if test point reached
            if (episodeNum % WHEN_EVALUATED == 0  ) and episodeNum != 0:
                testNetRun(sess, writer, episodeNum, lmaze, learner, possible_actions, "train", tf_rewardEval_summary, tb_rewardEval, tf_target_rewardEval_summary, tb_target_rewardEval, loaded, BEST_REWARD, saver)

    elif TEST == True :
        episodeNum = 0
        testNetRun(sess, writer, episodeNum, lmaze, learner, possible_actions, "test", tf_rewardEval_summary, tb_rewardEval, tf_target_rewardEval_summary, tb_target_rewardEval, loaded, 0.0, saver)






def PreTrain(lmaze, learner, possible_actions, loaded):

    trainStep = 0

    # if loaded then we need decay_step too
    decay_step = 0
    episodeNum = 0
    while trainStep < PRETRAIN_LENGTH:

        step, episode_data, state, action_sequence = rollout(lmaze, learner, possible_actions, loaded, "pretrain", decay_step, episodeNum)

        trainStep += step
        replay(episode_data, learner, state, step)
        episodeNum += 1

    rospy.loginfo("Memory popup done")











def testNetRun(sess, writer, episode, lmaze, learner, possible_actions, mode, tf_rewardEval_summary, tb_rewardEval, tf_target_rewardEval_summary, tb_target_rewardEval, loaded, BEST_REWARD, saver):

    episodeNum  = 0
    #episode_data = []
    session_data = []
        
    while episodeNum < HOW_MANY_EVALUATIONS:


        step, episode_data, state, action_sequence = rollout(lmaze, learner, possible_actions, loaded, "test", "0", episodeNum)
        episodeNum += 1
        session_data.append(episode_data[-1][1])

        # print action_sequence if you like

    # use session_data to build stats   
    avg_total_reward = np.sum(session_data[:])/float(HOW_MANY_EVALUATIONS)
    summary3 = sess.run(tf_rewardEval_summary, feed_dict={tb_rewardEval: avg_total_reward})
    writer.add_summary(summary3, episode)


    session_data = []

    while episodeNum < 2*HOW_MANY_EVALUATIONS:

        step, episode_data, state, action_sequence = rollout(lmaze, learner, possible_actions, loaded, "test", "0", episodeNum)
        episodeNum += 1
        session_data.append(episode_data[-1][1])

        # print action_sequence if you like

    # use session_data to build stats
    avg_total_reward1 = np.sum(session_data[:])/float(HOW_MANY_EVALUATIONS)
    summary4 = sess.run(tf_target_rewardEval_summary, feed_dict={tb_target_rewardEval: avg_total_reward1})
    writer.add_summary(summary4, episode)



    # save model if previous best performance is supersided
    myhost = os.uname()[1]    
    if (avg_total_reward > BEST_REWARD or avg_total_reward1 > BEST_REWARD) and mode == "train":
        if avg_total_reward > BEST_REWARD:
            BEST_REWARD = avg_total_reward
        if avg_total_reward1 > BEST_REWARD:
            BEST_REWARD = avg_total_reward1
        saver.save(sess, "/homes/gkumar/models/"+RUN_ID+"/"+myhost+"/model_Best_"+str(episode)+".ckpt") 
        os.system('mkdir -p /homes/gkumar/models/'+RUN_ID+'/'+myhost+'/best')
        os.system('mv /homes/gkumar/models/'+RUN_ID+'/'+myhost+'/model_Best_'+str(episode)+'.ckpt.* /homes/gkumar/models/'+RUN_ID+'/'+myhost+'/best/')

    elif ( episode % WHEN_BACKUP == 0 ) and episode != 0:
        saver.save(sess, "/homes/gkumar/models/"+RUN_ID+"/"+myhost+"/model_"+str(episode)+".ckpt") 







def rollout(lmaze, learner, possible_actions, loaded, mode, decay_step, episodeNum):

    reward = -0.0
    step = 0        
    episode_data = []
    action_sequence = []
    probability = 0.0
    #if loaded == False and testOnly == True:
    #    rospy.loginfo("No Model to test")
        
    while step < MAX_STEPS:            

        if step == 0:
            lmaze.reset()

            action = random.choice(possible_actions)
                
            state, reward, done, info = lmaze.step(possible_actions.index(action))
            while not(info['newState']):
                state, reward, done, info = lmaze.step(possible_actions.index(action))


        if mode == "train" or (mode == "pretrain" and loaded):                        # predict epsilon-greedy action
            if episodeNum % SWITCH_CYCLE >= 0 and episodeNum % SWITCH_CYCLE < SWITCH_CYCLE_HALF:
                #rospy.loginfo("First Model Prediction")
                action, probability = learner.predict_action1(decay_step+step, state)
            else:
                #rospy.loginfo("Second Model Prediction")
                action, probability = learner.predict_action2(decay_step+step, state)
        elif mode == "test":                             # predict action
            if episodeNum % (2*HOW_MANY_EVALUATIONS) >= 0 and episodeNum % (2*HOW_MANY_EVALUATIONS) < HOW_MANY_EVALUATIONS :
                #rospy.loginfo("First Model Evaluation")
                #time.sleep(0.01)
                action = learner.learned_action1(state)
            else :
                #rospy.loginfo("Second Model Evaluation")
                action = learner.learned_action2(state)
            action_sequence.append(possible_actions.index(action))
        elif mode == "target":                             # predict action
            action = learner.learned_target_action(state)
        else :                                              # random action
            action = random.choice(possible_actions)


        action_sequence.append(possible_actions.index(action))

        #rospy.loginfo("action as predicted : %s ",possible_actions.index(action))

        state, reward, done, info = lmaze.step(possible_actions.index(action))
        while not(info['newState']):
            state, reward, done, info = lmaze.step(possible_actions.index(action))

        #rospy.loginfo(state)
        #rospy.loginfo(reward)
        #rospy.loginfo(done)
        #rospy.loginfo(info['newState'])

        if done or step == MAX_STEPS-1:
            episode_data.append((np.asarray(action),reward, None))                        
            step += 1
            if STEP_PENALTY == True:
                if mode == "test":
                    rospy.loginfo("episodeNum : %s, decayStep : %s, step : %s, probability : %s, Reward : %s, stepPenaltyReward : %s, action sequence %s", episodeNum, decay_step, step, probability, reward, np.float(reward)/np.float(step), action_sequence)
                else:
                    rospy.loginfo("episodeNum : %s, decayStep : %s, step : %s, probability : %s, Reward : %s, stepPenaltyReward : %s", episodeNum, decay_step, step, probability, reward, np.float(reward)/np.float(step))
            else :
                if mode == "test":
                    rospy.loginfo("episodeNum : %s, decayStep : %s, step : %s, probability : %s, Reward : %s, action sequence %s", episodeNum, decay_step, step, probability, reward, action_sequence)
                else:
                    rospy.loginfo("episodeNum : %s, decayStep : %s, step : %s, probability : %s, Reward : %s", episodeNum, decay_step, step, probability, reward)
            #step = MAX_STEPS
            break
        else:
            episode_data.append((np.asarray(action), reward, state))
            step += 1
        #rospy.loginfo(step)
    #rospy.loginfo(step)
    return step, episode_data, state, action_sequence










def replay(episode_data, learner, next_state, step):
    #rospy.loginfo("HER Episode Augment")
    #rospy.loginfo(episode_data[step])
    #rospy.loginfo(episode_data[step])
    #rospy.loginfo(episode_data[0][2])
    #rospy.loginfo(episode_data[0][2].shape)
    
    if STEP_PENALTY == True:

        for x in range(0,step-2):
            learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                           episode_data[x][2])                
        learner.add_experience(episode_data[step-1][0], np.float(episode_data[step-1][1])/np.float(step), None)
        #rospy.loginfo("saved original episode in replay buffer")


        if ENDPOINT_AUGMENT == True :
            replacement_goal = np.array(next_state[outputDim*outputDim*0:outputDim*outputDim*1])           
            for x in range(0,step-2):
                temp1 = np.asarray(episode_data[x][2][outputDim*outputDim*0:outputDim*outputDim*4])                                                          
                if randomGoal == True:
                    temp2 = np.asarray(episode_data[x][2][outputDim*outputDim*5:outputDim*outputDim*6])                                                          
                    learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                            np.concatenate((temp1, replacement_goal, temp2), axis=0))
                else :
                    learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                            np.concatenate((temp1, replacement_goal), axis=0))
            learner.add_experience(episode_data[step-1][0], 1.0/np.float(step), None)
            #rospy.loginfo("saved original episode with final sampling in replay buffer")

        if EPISODE_AUGMENT == True:
            if step > HER_SAMPLE+2:
                sampl = np.random.random_integers(low=1, high=step-2, size=(HER_SAMPLE,))
                for x in range(0, HER_SAMPLE):
                    replacement_goal = np.asarray(episode_data[sampl[x]][2][outputDim*outputDim*0:outputDim*outputDim*1])                    
                    for y in range(1, sampl[x]-1):
                        #temp2 = np.asarray(episode_data[y][2][0:144])                    
                        temp1 = np.asarray(episode_data[y][2][outputDim*outputDim*0:outputDim*outputDim*4])                                                          
                        if randomGoal == True:
                            temp2 = np.asarray(episode_data[x][2][outputDim*outputDim*5:outputDim*outputDim*6])                                                          
                            learner.add_experience(episode_data[y][0],
                                           episode_data[y][1],
                                            np.concatenate((temp1, replacement_goal, temp2), axis=0))
                        else:
                            learner.add_experience(episode_data[y][0],
                                           episode_data[y][1],
                                            np.concatenate((temp1, replacement_goal), axis=0))                                                           
                    learner.add_experience(episode_data[sampl[x]][0], 1.0/np.float(sampl[x]), None)
                #rospy.loginfo("saved episode with episode sampling in replay buffer")

    else :
        for x in range(0,step-2):
            learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                           episode_data[x][2])                
        learner.add_experience(episode_data[step-1][0], episode_data[step-1][1], None)
        #rospy.loginfo("saved original episode in replay buffer")



        if ENDPOINT_AUGMENT == True :
            replacement_goal = np.array(next_state[outputDim*outputDim*0:outputDim*outputDim*1])           
            for x in range(0,step-2):
                temp1 = np.asarray(episode_data[x][2][outputDim*outputDim*0:outputDim*outputDim*4])                                                          
                if randomGoal == True:
                    temp2 = np.asarray(episode_data[x][2][outputDim*outputDim*5:outputDim*outputDim*6])                                                          
                    learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                            np.concatenate((temp1, replacement_goal, temp2), axis=0))
                else:
                    learner.add_experience(episode_data[x][0],
                                           episode_data[x][1],
                                            np.concatenate((temp1, replacement_goal), axis=0))
            learner.add_experience(episode_data[step-1][0], 1.0, None)
            #rospy.loginfo("saved original episode with final sampling in replay buffer")



        if EPISODE_AUGMENT == True:
            if step > HER_SAMPLE+2:
                sampl = np.random.random_integers(low=1, high=step-2, size=(HER_SAMPLE,))
                for x in range(0, HER_SAMPLE):
                    replacement_goal = np.asarray(episode_data[sampl[x]][2][outputDim*outputDim*0:outputDim*outputDim*1])                    
                    for y in range(0, sampl[x]-1):
                        #temp2 = np.asarray(episode_data[y][2][0:144])                    
                        temp1 = np.asarray(episode_data[y][2][outputDim*outputDim*0:outputDim*outputDim*4])                                                          
                        if randomGoal == True:
                            temp2 = np.asarray(episode_data[x][2][outputDim*outputDim*5:outputDim*outputDim*6])                                                          
                            learner.add_experience(episode_data[y][0],
                                           episode_data[y][1],
                                            np.concatenate((temp1, replacement_goal, temp2), axis=0))
                        else:
                            learner.add_experience(episode_data[y][0],
                                           episode_data[y][1],
                                            np.concatenate((temp1, replacement_goal), axis=0))
                    learner.add_experience(episode_data[sampl[x]][0], 1.0, None)
                #rospy.loginfo("saved episode with episode sampling in replay buffer")








if __name__ == "__main__":
    main()