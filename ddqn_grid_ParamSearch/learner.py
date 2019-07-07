#!/usr/bin/env python
import random
import tensorflow as tf
import numpy as np 
#import rospy
from network import Network
from util import ReplayDataset

#import os.path
import os
#from sensor_msgs.msg import Image

class Learner:
    def __init__(self, SESS, p):
                     
        self.p = p
        self.SESS = SESS

        tf.global_variables_initializer()

        # self.memory = deque(maxlen=self.MEMORY_SIZE)

        #self.image_pub1 = rospy.Publisher('/interface/scene1', Image, queue_size=1)
        #self.image_pub2 = rospy.Publisher('/interface/scene2', Image, queue_size=1)
        #self.image_pub3 = rospy.Publisher('/interface/scene3', Image, queue_size=1)
        #self.image_pub4 = rospy.Publisher('/interface/scene4', Image, queue_size=1)
                
        self.actor = Network(self.p)
        self.inputs_, self.model, self.Q, self.loss, self.optimizer = self.actor.createNet()

        self.actor_double = Network(self.p)
        self.t_inputs_, self.t_model, self.t_Q, self.t_loss, self.t_optimizer = self.actor_double.createNet()

        # check existing before load and set loaded or not variable
        #if os.path.exists("/vol/speech/gaurav/datadrive/"+self.p.HOSTNAME+"/bkp_rbuffer.hdf"):
        #    rospy.loginfo("Staring with loading saved memory")
        #else :
        #    self.rBuffer = ReplayDataset()

        self.rBuffer = ReplayDataset(self.p)
        self.load = self.rBuffer.load


    def train1(self, decay_step):
        self.decay_step = decay_step

        try:
            batch11, batch12, batch13, batch14, batch15 = self.rBuffer.sample(self.p.BATCH_SIZE)
        except :
            #rospy.loginfo("batch cannot be sampled")
            return 0.0

        target_Qs_batch = []
        if self.p.DECAY_UNTIL > self.decay_step:
            self.actor.global_step = self.decay_step
        else :
            self.actor.global_step = self.p.DECAY_UNTIL
            
            
            
        states_mb = batch11.reshape((self.p.BATCH_SIZE, self.p.STATE_SIZE[0]))
        actions_mb = batch12
        rewards_mb = batch13
        next_states_mb = batch14.reshape((self.p.BATCH_SIZE, self.p.STATE_SIZE[0]))
        dones_mb = batch15


        Qs_next_state = self.t_model.predict(next_states_mb)

        for i in range(0, self.p.BATCH_SIZE):
            non_terminal = dones_mb[i]
            #terminal = dones_mb[i]
            # If we are in a terminal state, only equals reward
            if non_terminal:
            #if rewards_mb[i] == -0.1 or rewards_mb[i] == 1:
                target = rewards_mb[i] + self.p.GAMMA * np.max(Qs_next_state[i])
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
            batch11, batch12, batch13, batch14, batch15 = self.rBuffer.sample(self.p.BATCH_SIZE)
        except :
            #rospy.loginfo("batch cannot be sampled")
            return 0.0

        target_Qs_batch = []
        if self.p.DECAY_UNTIL > self.decay_step:
            self.actor_double.global_step = self.decay_step
        else :
            self.actor_double.global_step = self.p.DECAY_UNTIL
            
            
            
        states_mb = batch11.reshape((self.p.BATCH_SIZE, self.p.STATE_SIZE[0]))
        actions_mb = batch12
        rewards_mb = batch13
        next_states_mb = batch14.reshape((self.p.BATCH_SIZE, self.p.STATE_SIZE[0]))
        dones_mb = batch15

        Qs_next_state = self.model.predict(next_states_mb)

        for i in range(0, self.p.BATCH_SIZE):
            non_terminal = dones_mb[i]
            #terminal = dones_mb[i]
            # If we are in a terminal state, only equals reward
            if non_terminal:
            #if rewards_mb[i] == -0.1 or rewards_mb[i] == 1:
                target = rewards_mb[i] + self.p.GAMMA * np.max(Qs_next_state[i])
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


    def predict_action1(self, decay_step, state):
        # exploration-exploitation tradeoff
        exp_exp_tradeoff = np.random.rand()
        #decay_step = decay_step
        #explore_probability = self.EXPLORE_STOP + (self.EXPLORE_START - self.EXPLORE_STOP) * np.exp(-self.DECAY_RATE * self.DECAY_STEP)

        if decay_step * self.p.DECAY_RATE >= 1 :
            explore_probability = self.p.EXPLORE_STOP
        else :
            explore_probability = self.p.EXPLORE_START - (self.p.EXPLORE_START - self.p.EXPLORE_STOP) * (self.p.DECAY_RATE * decay_step)

        if (explore_probability > exp_exp_tradeoff):
            # rospy.loginfo (random.choice(self.POSSIBLE_ACTIONS))
            return random.choice(self.p.POSSIBLE_ACTIONS), explore_probability
        else:  # Get action from Q-network (exploitation) Estimate the Qs values state
            Qs = self.SESS.run(self.actor.output, feed_dict = {self.actor.inputs_: state.reshape((1, self.p.STATE_SIZE[0]))})       
            choice = np.argmax(Qs)
            # rospy.loginfo("planned action %s %s", choice, self.POSSIBLE_ACTIONS[choice])
            return self.p.POSSIBLE_ACTIONS[choice], explore_probability


    def predict_action2(self, decay_step, state):
        # exploration-exploitation tradeoff
        exp_exp_tradeoff = np.random.rand()
        #explore_probability = self.EXPLORE_STOP + (self.EXPLORE_START - self.EXPLORE_STOP) * np.exp(-self.DECAY_RATE * self.DECAY_STEP)
        #decay_step = decay_step

        if decay_step * self.p.DECAY_RATE >= 1 :
            explore_probability = self.p.EXPLORE_STOP
        else :
            explore_probability = self.p.EXPLORE_START - (self.p.EXPLORE_START - self.p.EXPLORE_STOP) * (self.p.DECAY_RATE * decay_step)

        if (explore_probability > exp_exp_tradeoff):
            # rospy.loginfo (random.choice(self.POSSIBLE_ACTIONS))
            return random.choice(self.p.POSSIBLE_ACTIONS), explore_probability
        else:  # Get action from Q-network (exploitation) Estimate the Qs values state
            Qs = self.SESS.run(self.actor_double.output, feed_dict = {self.actor_double.inputs_: state.reshape((1, self.p.STATE_SIZE[0]))})       
            choice = np.argmax(Qs)
            # rospy.loginfo("planned action %s %s", choice, self.POSSIBLE_ACTIONS[choice])
            return self.p.POSSIBLE_ACTIONS[choice], explore_probability



    def learned_action1(self, state):
        #Qs = self.SESS.run(self.actor.output, feed_dict = {self.actor.inputs_: STATE.reshape((1, self.STATE_SIZE[0]))})       
        Qs = self.model.predict(state.reshape((1, self.p.STATE_SIZE[0])))
        choice = np.argmax(Qs)
        return self.p.POSSIBLE_ACTIONS[choice]

    def learned_action2(self, state):
        #Qs = self.SESS.run(self.actor_double.output, feed_dict = {self.actor_double.inputs_: STATE.reshape((1, self.STATE_SIZE[0]))})       
        Qs = self.t_model.predict(state.reshape((1, self.p.STATE_SIZE[0])))
        choice = np.argmax(Qs)
        return self.p.POSSIBLE_ACTIONS[choice]
