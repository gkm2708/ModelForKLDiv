#!/usr/bin/env python
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from param import Param

import random
import h5py
#import rospy
import os
import cv2
###############################################################################

class ReplayDataset(object):

    def __init__(self, overwrite=False):  
        self.p = Param()        
        self.load = False
        
        self.fp = h5py.File(self.p.BUFFER_FILE, 'a')
        
        self.state = self.fp.create_dataset("state", (self.p.MEMORY_SIZE,) + self.p.STATE_SHAPE.shape , dtype=np.float32)

        self.fp.create_dataset("action", (self.p.MEMORY_SIZE, self.p.ACTION_SIZE, ), dtype='uint8')
        self.fp.create_dataset("reward", (self.p.MEMORY_SIZE,), dtype=np.float32)
        self.fp.create_dataset("non_terminal", (self.p.MEMORY_SIZE,), dtype=bool)
            
        self.action = np.empty((self.p.MEMORY_SIZE,self.p.ACTION_SIZE), dtype=np.uint8)
        self.reward = np.empty(self.p.MEMORY_SIZE, dtype=np.float32)
        self.non_terminal = np.empty(self.p.MEMORY_SIZE, dtype=bool)

        self.head = 0
        self.valid = 0



        #rospy.loginfo("Alternative Memory Model %s",self.p.STATE_SHAPE.shape)
        #rospy.loginfo("Alternative Memory Model : %s ", (self.p.MEMORY_SIZE,) + self.p.STATE_SHAPE.shape )
        #rospy.loginfo("Alternative Memory Model : %s ", self.state.shape)
        #rospy.loginfo("Alternative Memory Model : %s ", self.action.shape)
        #rospy.loginfo("Alternative Memory Model : %s ", self.reward.shape)
        #rospy.loginfo("Alternative Memory Model : %s ", self.non_terminal.shape)
        #rospy.loginfo("Alternative Memory Model : %s ",self.head)
        #rospy.loginfo("Alternative Memory Model : %s ",self.valid)


    def add_experience(self, action, reward, state):
        # Add the next step in a game sequence, i.e. a triple (action, reward, state) indicating that the agent took 'action',
        # received 'reward' and *then* ended up in 'state.' The original state is presumed to be the state at index (head - 1)
        # Args:
        #    action :  index of the action chosen
        #    reward :  integer value of reward, positive or negative
        #    state  :  a numpy array of shape NUM_FRAMES x WIDTH x HEIGHT or None if this action ended the game.

        self.action[self.head] = action.reshape(-1)
        self.reward[self.head] = reward
        if state is not None:
            try:
                self.state[self.head] = state.reshape(-1)
            except :
                raise ValueError("Can't add an experience of size %s in replay dataset of size %s" % (state.shape, self.state.size))
                #rospy.loginfo(state.shape)
                #rospy.loginfo(self.head)
                #rospy.loginfo(self.valid)
                #rospy.loginfo(self.state.len())
                #rospy.loginfo(self.state.size)

            self.non_terminal[self.head] = True
        else:
            self.non_terminal[self.head] = False
        self.head = (self.head + 1) % self.p.MEMORY_SIZE
        self.valid = min(self.p.MEMORY_SIZE, self.valid + 1)
        

    def sample(self, sample_size):
        # Uniformly sample (s,a,r,s) experiences from the replay dataset.
        # Args:
        #    sample_size: (self explanatory)
        # Returns:
        #    A tuple of numpy arrays for the |sample_size| experiences.
        #        (state, action, reward, next_state)
        #    The zeroth dimension of each array corresponds to the experience index. The i_th experience is given by:
        #        (state[i], action[i], reward[i], next_state[i])
        if sample_size >= self.valid:
            raise ValueError("Can't draw sample of size %d from replay dataset of size %d" % (sample_size, self.valid))
        idx = random.sample(xrange(0, self.valid), sample_size)
        while (self.head - 1) in idx:
            idx = random.sample(xrange(0, self.valid), sample_size)
        idx.sort()  # Slicing for hdf5 must be in increasing order
        next_idx = [x + 1 for x in idx]

        # next_state might wrap around end of dataset
        if next_idx[-1] == self.p.MEMORY_SIZE:
            next_idx[-1] = 0
            shape = (sample_size,)+self.state[0].shape
            next_states = np.empty(shape, dtype=np.uint8)
            next_states[0:-1] = self.state[next_idx[0:-1]]
            next_states[-1] = self.state[0]
        else:
            next_states = self.state[next_idx]
            
        return (self.state[idx], self.action[next_idx], self.reward[next_idx], next_states, self.non_terminal[next_idx])

    def __del__(self):
        self.fp['non_terminal'][:] = self.non_terminal
        self.fp['action'][:] = self.action
        self.fp['reward'][:] = self.reward
        self.state.attrs['head'] = self.head
        self.state.attrs['valid'] = self.valid

        self.fp.close()

###############################################################################