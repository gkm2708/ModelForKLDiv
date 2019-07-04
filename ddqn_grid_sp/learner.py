#!/usr/bin/env python
import random
import tensorflow as tf
from util import ReplayDataset
import numpy as np 
import rospy
from network import Network

#import os.path
import os
from sensor_msgs.msg import Image

class Learner:
    def __init__(self, 
                 SESS, 
                 GAMMA,
                 ALPHA,
                 LEARNING_RATE, 
                 STATE_SIZE, 
                 ACTION_SIZE, 
                 EXPLORE_START, 
                 EXPLORE_STOP, 
                 DECAY_RATE, 
                 POSSIBLE_ACTIONS,
                 BATCH_SIZE,
                 MEMORY_SIZE,
                 PRETRAIN_LENGTH,
                 STATE_SHAPE,
                 DECAY_WINDOW,
                 DECAY_BASE,
                 DECAY_UNTIL,
                 outputDim):

        self.SESS = SESS
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.LEARNING_RATE = LEARNING_RATE
        self.STATE_SIZE = STATE_SIZE
        self.ACTION_SIZE = ACTION_SIZE
        self.EXPLORE_START = EXPLORE_START
        self.EXPLORE_STOP = EXPLORE_STOP
        self.DECAY_RATE = DECAY_RATE
        self.POSSIBLE_ACTIONS = POSSIBLE_ACTIONS
        self.BATCH_SIZE = BATCH_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        self.PRETRAIN_LENGTH = PRETRAIN_LENGTH
        self.STATE_SHAPE = STATE_SHAPE
        self.DECAY_WINDOW = DECAY_WINDOW
        self.DECAY_BASE = DECAY_BASE
        self.DECAY_UNTIL = DECAY_UNTIL
        self.outputDim = outputDim
        
        tf.global_variables_initializer()

        # self.memory = deque(maxlen=self.MEMORY_SIZE)

        self.image_pub1 = rospy.Publisher('/interface/scene1', Image, queue_size=1)
        self.image_pub2 = rospy.Publisher('/interface/scene2', Image, queue_size=1)
        self.image_pub3 = rospy.Publisher('/interface/scene3', Image, queue_size=1)
        self.image_pub4 = rospy.Publisher('/interface/scene4', Image, queue_size=1)

        self.myhost = os.uname()[1] 
        rospy.loginfo(self.myhost)
        
        
        self.actor = Network()

        self.inputs_, self.model, self.Q, self.loss, self.optimizer = self.actor.createNet(self.STATE_SIZE, 
                                                                                           self.ACTION_SIZE, 
                                                                                           self.LEARNING_RATE,
                                                                                           self.DECAY_WINDOW,
                                                                                           self.DECAY_BASE)
        self.actor_double = Network()
        self.t_inputs_, self.t_model, self.t_Q, self.t_loss, self.t_optimizer = self.actor_double.createNet(self.STATE_SIZE, 
                                                                                                     self.ACTION_SIZE, 
                                                                                                     self.LEARNING_RATE,
                                                                                                     self.DECAY_WINDOW,
                                                                                                     self.DECAY_BASE)
        # check existing before load and set loaded or not variable
        if os.path.exists("/vol/speech/gaurav/datadrive/"+self.myhost+"/bkp_rbuffer.hdf"):
            rospy.loginfo("Staring with loading saved memory")
        else :
            self.rBuffer = ReplayDataset('/vol/speech/gaurav/datadrive/'+self.myhost+'/rbuffer.hdf', self.STATE_SHAPE, self.ACTION_SIZE, self.MEMORY_SIZE, False)
        self.load = self.rBuffer.load


    def train1(self, decay_step):
        self.decay_step = decay_step

        try:
            batch11, batch12, batch13, batch14, batch15 = self.rBuffer.sample(self.BATCH_SIZE)
        except :
            rospy.loginfo("batch cannot be sampled")
            return 0.0

        target_Qs_batch = []
        if self.DECAY_UNTIL > self.decay_step:
            self.actor.global_step = self.decay_step
        else :
            self.actor.global_step = self.DECAY_UNTIL
            
            
            
        states_mb = batch11.reshape((self.BATCH_SIZE, self.STATE_SIZE[0]))
        actions_mb = batch12
        rewards_mb = batch13
        next_states_mb = batch14.reshape((self.BATCH_SIZE, self.STATE_SIZE[0]))
        dones_mb = batch15


        Qs_next_state = self.t_model.predict(next_states_mb)

        for i in range(0, self.BATCH_SIZE):
            non_terminal = dones_mb[i]
            #terminal = dones_mb[i]
            # If we are in a terminal state, only equals reward
            if non_terminal:
            #if rewards_mb[i] == -0.1 or rewards_mb[i] == 1:
                target = rewards_mb[i] + self.GAMMA * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
            else:
                target_Qs_batch.append(rewards_mb[i])

                
        targets_mb = np.array([each for each in target_Qs_batch])




        loss, _ = self.SESS.run([self.actor.loss, self.actor.optimizer],
                                       feed_dict={
                                           self.actor.inputs_: states_mb, 
                                           self.actor.target_Q: targets_mb, 
                                           self.actor.actions_: actions_mb
                                           })

                                           
        return loss


    def train2(self, decay_step):
        self.decay_step = decay_step

        try:
            batch11, batch12, batch13, batch14, batch15 = self.rBuffer.sample(self.BATCH_SIZE)
        except :
            rospy.loginfo("batch cannot be sampled")
            return 0.0

        target_Qs_batch = []
        if self.DECAY_UNTIL > self.decay_step:
            self.actor_double.global_step = self.decay_step
        else :
            self.actor_double.global_step = self.DECAY_UNTIL
            
            
            
        states_mb = batch11.reshape((self.BATCH_SIZE, self.STATE_SIZE[0]))
        actions_mb = batch12
        rewards_mb = batch13
        next_states_mb = batch14.reshape((self.BATCH_SIZE, self.STATE_SIZE[0]))
        dones_mb = batch15

        Qs_next_state = self.model.predict(next_states_mb)

        for i in range(0, self.BATCH_SIZE):
            non_terminal = dones_mb[i]
            #terminal = dones_mb[i]
            # If we are in a terminal state, only equals reward
            if non_terminal:
            #if rewards_mb[i] == -0.1 or rewards_mb[i] == 1:
                target = rewards_mb[i] + self.GAMMA * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
            else:
                target_Qs_batch.append(rewards_mb[i])

                
        targets_mb = np.array([each for each in target_Qs_batch])




        loss, _ = self.SESS.run([self.actor_double.loss, self.actor_double.optimizer],
                                       feed_dict={
                                           self.actor_double.inputs_: states_mb, 
                                           self.actor_double.target_Q: targets_mb, 
                                           self.actor_double.actions_: actions_mb
                                           })

                                           
        return loss

    # ========================================================================= #
    #                              Replay Buffer                                #
    # ========================================================================= #

    def add_experience(self, action, reward, next_state):
        self.rBuffer.add_experience(np.asarray(action), reward, next_state)
        
    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #


    def predict_action1(self, DECAY_STEP, STATE):
        # exploration-exploitation tradeoff
        exp_exp_tradeoff = np.random.rand()
        self.DECAY_STEP = DECAY_STEP
        #explore_probability = self.EXPLORE_STOP + (self.EXPLORE_START - self.EXPLORE_STOP) * np.exp(-self.DECAY_RATE * self.DECAY_STEP)

        if self.DECAY_STEP * self.DECAY_RATE >= 1 :
            explore_probability = self.EXPLORE_STOP
        else :
            explore_probability = self.EXPLORE_START - (self.EXPLORE_START - self.EXPLORE_STOP) * (self.DECAY_RATE * self.DECAY_STEP)

        if (explore_probability > exp_exp_tradeoff):
            # rospy.loginfo (random.choice(self.POSSIBLE_ACTIONS))
            return random.choice(self.POSSIBLE_ACTIONS), explore_probability
        else:  # Get action from Q-network (exploitation) Estimate the Qs values state
            Qs = self.SESS.run(self.actor.output, feed_dict = {self.actor.inputs_: STATE.reshape((1, self.STATE_SIZE[0]))})       
            choice = np.argmax(Qs)
            # rospy.loginfo("planned action %s %s", choice, self.POSSIBLE_ACTIONS[choice])
            return self.POSSIBLE_ACTIONS[choice], explore_probability


    def predict_action2(self, DECAY_STEP, STATE):
        # exploration-exploitation tradeoff
        exp_exp_tradeoff = np.random.rand()
        self.DECAY_STEP = DECAY_STEP
        #explore_probability = self.EXPLORE_STOP + (self.EXPLORE_START - self.EXPLORE_STOP) * np.exp(-self.DECAY_RATE * self.DECAY_STEP)

        if self.DECAY_STEP * self.DECAY_RATE >= 1 :
            explore_probability = self.EXPLORE_STOP
        else :
            explore_probability = self.EXPLORE_START - (self.EXPLORE_START - self.EXPLORE_STOP) * (self.DECAY_RATE * self.DECAY_STEP)

        if (explore_probability > exp_exp_tradeoff):
            # rospy.loginfo (random.choice(self.POSSIBLE_ACTIONS))
            return random.choice(self.POSSIBLE_ACTIONS), explore_probability
        else:  # Get action from Q-network (exploitation) Estimate the Qs values state
            Qs = self.SESS.run(self.actor_double.output, feed_dict = {self.actor_double.inputs_: STATE.reshape((1, self.STATE_SIZE[0]))})       
            choice = np.argmax(Qs)
            # rospy.loginfo("planned action %s %s", choice, self.POSSIBLE_ACTIONS[choice])
            return self.POSSIBLE_ACTIONS[choice], explore_probability



    def learned_action1(self, STATE):
        #Qs = self.SESS.run(self.actor.output, feed_dict = {self.actor.inputs_: STATE.reshape((1, self.STATE_SIZE[0]))})       
        Qs = self.model.predict(STATE.reshape((1, self.STATE_SIZE[0])))
        choice = np.argmax(Qs)
        return self.POSSIBLE_ACTIONS[choice]

    def learned_action2(self, STATE):
        #Qs = self.SESS.run(self.actor_double.output, feed_dict = {self.actor_double.inputs_: STATE.reshape((1, self.STATE_SIZE[0]))})       
        Qs = self.t_model.predict(STATE.reshape((1, self.STATE_SIZE[0])))
        choice = np.argmax(Qs)
        return self.POSSIBLE_ACTIONS[choice]
