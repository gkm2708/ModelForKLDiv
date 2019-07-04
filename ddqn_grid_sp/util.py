#!/usr/bin/env python
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import random
import h5py
import rospy
import os
import cv2
###############################################################################


class imageQueue:

    def __init__(self, window, stride, outputDim):
        self.width = 84
        self.height = 84
        self.window = window
        self.reset()
        self.firstFrame = True
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('/interface/scene', Image, queue_size=1)
        self.image_pub1 = rospy.Publisher('/interface/scene1', Image, queue_size=1)
        self.image_pub2 = rospy.Publisher('/interface/scene2', Image, queue_size=1)
        self.image_pub3 = rospy.Publisher('/interface/scene3', Image, queue_size=1)
        self.image_pub4 = rospy.Publisher('/interface/scene4', Image, queue_size=1)
        self.image_pubBit = rospy.Publisher('/interface/sceneBit', Image, queue_size=1)
        
        self.imagered_pub = rospy.Publisher('/interface/ballscene', Image, queue_size=1)
        self.imageyellow_pub = rospy.Publisher('/interface/holescene', Image, queue_size=1)
        self.imagegreen_pub = rospy.Publisher('/interface/wallscene', Image, queue_size=1)
        self.imageblue_pub = rospy.Publisher('/interface/goalscene', Image, queue_size=1)

        #self.image_pubRecons = rospy.Publisher('/interface/sceneReconstructed', Image, queue_size=1)
        #self.ball_sub = rospy.Subscriber('/BC/pose', PoseStamped, self.ballState)
        self.items = np.zeros((self.width, self.height,self.window), dtype=np.float32)
        self.m = np.array([0.114, 0.587, 0.299]).reshape((1,3))           # Luminance Method

        self.outputDim = outputDim
        self.step = self.width / self.outputDim
        # New Logic
        self.valueHoles = np.zeros((self.outputDim,self.outputDim),dtype=np.float)
        self.valueGoal = np.zeros((self.outputDim,self.outputDim),dtype=np.float)
        self.valueBallF1 = np.zeros((self.outputDim,self.outputDim),dtype=np.float)
        self.valueBallF2 = np.zeros((self.outputDim,self.outputDim),dtype=np.float)
        self.valueBallF3 = np.zeros((self.outputDim,self.outputDim),dtype=np.float)
        self.valueBallF4 = np.zeros((self.outputDim,self.outputDim),dtype=np.float)
        self.valueWall = np.zeros((self.outputDim,self.outputDim),dtype=np.float)



    def ballState(self, msg):
        rospy.loginfo("%s, %s", msg.pose.position.x,msg.pose.position.y)


    def emit(self):
        herInput = np.concatenate([self.valueGoal.reshape(-1),self.valueBallF1.reshape(-1),self.valueBallF2.reshape(-1),self.valueBallF3.reshape(-1),self.valueBallF4.reshape(-1)])
        return herInput
        
    def reset(self):
        self.firstFrame = True

    def enqueue(self, cv_image):
            
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        """ Holes 
        value = np.zeros((self.outputDim,self.outputDim),dtype=np.float)
        value1 = np.zeros((self.outputDim,self.outputDim),dtype=np.uint8)
        value2 = np.zeros((self.outputDim,self.outputDim),dtype=np.float)

        lower = np.array((23, 100, 100))
        upper = np.array((40, 255, 255))
        mask = cv2.inRange(hsv_image, lower, upper)
        output = cv2.bitwise_and(cv_image, cv_image, mask = mask)
            
        grayFiltered = cv2.transform(output, self.m)
        #grayFiltered = cv2.GaussianBlur(grayFiltered,(5,5),0)
        #_, grayFiltered = cv2.threshold(grayFiltered,20,255,cv2.THRESH_BINARY)
            

        for x in range(0,12) :
            for y in range(0,12) :
                value[x,y] = np.sum(grayFiltered[x*7:x*7+7, y*7:y*7+7])/(7*7*255)
            
        for x in range(0,12) :
            for y in range(0,12) :
                if value[x,y] > 0.05 :
                    value1[x,y] = 255
                    value2[x,y] = 1
                elif value[x,y] <= 0.05 :                        
                    value1[x,y] = 0
                    value2[x,y] = 0
        #rospy.loginfo(" Holes %s ", value2)
            
        #self.imageyellow_pub.publish(self.bridge.cv2_to_imgmsg(output,encoding="bgr8"))
        self.imageyellow_pub.publish(self.bridge.cv2_to_imgmsg(value1,encoding="mono8"))
        #self.imageyellow_pub.publish(self.bridge.cv2_to_imgmsg(grayFiltered,encoding="mono8"))

        self.valueHoles = value2
        """


        """ Goals """
        value = np.zeros((self.outputDim,self.outputDim),dtype=np.float)
        value1 = np.zeros((self.outputDim,self.outputDim),dtype=np.uint8)
        value2 = np.zeros((self.outputDim,self.outputDim),dtype=np.float)

        lower = np.array((90, 100, 100))
        upper = np.array((120, 255, 255))
        mask = cv2.inRange(hsv_image, lower, upper)
        output = cv2.bitwise_and(cv_image, cv_image, mask = mask)

        grayFiltered = cv2.transform(output, self.m)
        #grayFiltered = cv2.GaussianBlur(grayFiltered,(5,5),0)
        #_, grayFiltered = cv2.threshold(grayFiltered,100,255,cv2.THRESH_BINARY)


        for x in range(0,self.outputDim) :
            for y in range(0,self.outputDim) :
                value[x,y] = np.sum(grayFiltered[x*self.step:x*self.step+self.step, y*self.step:y*self.step+self.step])/(self.step*self.step*255)


        for x in range(0,self.outputDim) :
            for y in range(0,self.outputDim) :
                if value[x,y] > 0.05 :
                    value1[x,y] = 255
                    value2[x,y] = 1
                elif value[x,y] <= 0.05 :                        
                    value1[x,y] = 0
                    value2[x,y] = 0
        #rospy.loginfo(" Goal %s ",value2)
        #self.imageblue_pub.publish(self.bridge.cv2_to_imgmsg(output,encoding="bgr8"))
                    #rospy.loginfo(value1.shape)
        self.imageblue_pub.publish(self.bridge.cv2_to_imgmsg(value1,encoding="mono8"))
        #self.imageblue_pub.publish(self.bridge.cv2_to_imgmsg(grayFiltered,encoding="mono8"))

        self.valueGoal = value2
            
        """ Walls 
        value = np.zeros((self.outputDim,self.outputDim),dtype=np.float)
        value1 = np.zeros((self.outputDim,self.outputDim),dtype=np.uint8)
        value2 = np.zeros((self.outputDim,self.outputDim),dtype=np.float)

        lower = np.array((55, 100, 100))
        upper = np.array((90, 255, 255))
        mask = cv2.inRange(hsv_image, lower, upper)
        output = cv2.bitwise_and(cv_image, cv_image, mask = mask)

            

        #grayFiltered = cv2.transform(output, self.m)
        # blur = cv2.GaussianBlur(img,(5,5),0)
        # median = cv2.medianBlur(img,5)
        # blur = cv2.bilateralFilter(img,9,75,75)
        #grayFiltered = cv2.transform(cv2.cvtColor(output, cv2.COLOR_HSV2BGR), self.m)
        grayFiltered = cv2.transform(output, self.m)
        #grayFiltered = cv2.GaussianBlur(grayFiltered,(5,5),0)
        _, grayFiltered = cv2.threshold(grayFiltered,100,255,cv2.THRESH_BINARY)

            
            

        for x in range(0,12) :
            for y in range(0,12) :
                value[x,y] = np.sum(grayFiltered[x*7:x*7+7, y*7:y*7+7])/(7*7*255)
        #rospy.loginfo(" Walls real value %s ",value)

        #threshold = np.amax(value) / 49
            
        for x in range(0,12) :
            for y in range(0,12) :
                if value[x,y] >= 0.7 :
                    value1[x,y] = 255
                    value2[x,y] = 1
                elif value[x,y] < 0.7 :                        
                    value1[x,y] = 0
                    value2[x,y] = 0
                        
        #rospy.loginfo(" Walls after threshold %s ",value2)
            
        self.valueWall = value2
            
        #self.imagegreen_pub.publish(self.bridge.cv2_to_imgmsg(output,encoding="bgr8"))
        self.imagegreen_pub.publish(self.bridge.cv2_to_imgmsg(value1,encoding="mono8"))


        """

        """ Ball """
        
        value = np.zeros((self.outputDim,self.outputDim),dtype=np.float)
        value1 = np.zeros((self.outputDim,self.outputDim),dtype=np.uint8)
        value2 = np.zeros((self.outputDim,self.outputDim),dtype=np.float)

        boundaries = [([0, 120, 70], [10, 255, 255]),
                          ([170, 120, 70], [180, 255, 255])]
             
        # loop over the boundaries
        for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
 
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(hsv_image, lower, upper)
            output = cv2.bitwise_and(cv_image, cv_image, mask = mask)

            grayFiltered = cv2.transform(output, self.m)
            #grayFiltered = cv2.GaussianBlur(grayFiltered,(5,5),0)
            #_, grayFiltered = cv2.threshold(grayFiltered,100,255,cv2.THRESH_BINARY)

            for x in range(0,self.outputDim) :
                for y in range(0,self.outputDim) :
                    if np.sum(grayFiltered[x*self.step:x*self.step+self.step, y*self.step:y*self.step+self.step]) > value[x,y]:                        
                        value[x,y] = np.sum(grayFiltered[x*self.step:x*self.step+self.step, y*self.step:y*self.step+self.step])/(self.step*self.step*255)

        for x in range(0,self.outputDim) :
            for y in range(0,self.outputDim) :
                if value[x,y] >= 0.01 :
                    value1[x,y] = 255
                    value2[x,y] = 1
                elif value[x,y] < 0.01 :                        
                    value1[x,y] = 0
                    value2[x,y] = 0

        self.imagered_pub.publish(self.bridge.cv2_to_imgmsg(value1,encoding="mono8"))

        self.valueBallF1 = self.valueBallF2
        self.valueBallF2 = self.valueBallF3
        self.valueBallF3 = self.valueBallF4
        self.valueBallF4 = value

        #rospy.loginfo(" Ball %s ",value2)


class ReplayDataset(object):

    def __init__(self, filename, state_shape, action_size, dset_size, overwrite):  
        self._filename = filename
        self._state_shape = state_shape
        self._dset_size = dset_size
        self.action_shape = action_size
        self.load = False
        self.init(filename, state_shape, dset_size, overwrite)
        self.myhost = os.uname()[1] 
        rospy.loginfo(self.myhost)

         
    def init(self, filename, state_shape, dset_size, overwrite):
        self.fp = h5py.File(filename, 'a')
        
        rospy.loginfo("Alternative Memory Model %s",state_shape.shape)

        self.state = self.fp.create_dataset("state", (dset_size,) + state_shape.shape , dtype=np.float32)

        self.fp.create_dataset("action", (dset_size, self.action_shape, ), dtype='uint8')
        self.fp.create_dataset("reward", (dset_size,), dtype=np.float32)
        self.fp.create_dataset("non_terminal", (dset_size,), dtype=bool)
            
        self.action = np.empty((dset_size,self.action_shape), dtype=np.uint8)
        self.reward = np.empty(dset_size, dtype=np.float32)
        self.non_terminal = np.empty(dset_size, dtype=bool)

        self.dset_size = dset_size            

        self.head = 0
        self.valid = 0

        rospy.loginfo("Alternative Memory Model : %s ", (dset_size,) + state_shape.shape )
        rospy.loginfo("Alternative Memory Model : %s ", self.state.shape)
        rospy.loginfo("Alternative Memory Model : %s ", self.action.shape)
        rospy.loginfo("Alternative Memory Model : %s ", self.reward.shape)
        rospy.loginfo("Alternative Memory Model : %s ", self.non_terminal.shape)
        rospy.loginfo("Alternative Memory Model : %s ",self.head)
        rospy.loginfo("Alternative Memory Model : %s ",self.valid)


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
                rospy.loginfo(state.shape)
                rospy.loginfo(self.head)
                rospy.loginfo(self.valid)
                rospy.loginfo(self.state.len())
                rospy.loginfo(self.state.size)

            self.non_terminal[self.head] = True
        else:
            self.non_terminal[self.head] = False
        # Update head pointer and valid pointer
        self.head = (self.head + 1) % self.dset_size
        self.valid = min(self.dset_size, self.valid + 1)
        
        #rospy.loginfo("Memory Status %s %s",self.head, self.valid)

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
        # We can't include head - 1 in sample because we don't know the next state, so simply resample (very rare if dataset is large)
        # rospy.loginfo("Sample size ")
        # rospy.loginfo( idx)
        while (self.head - 1) in idx:
            idx = random.sample(xrange(0, self.valid), sample_size)
        #rospy.loginfo( idx)
        idx.sort()  # Slicing for hdf5 must be in increasing order
        #rospy.loginfo( idx)
        next_idx = [x + 1 for x in idx]

        #rospy.loginfo( next_idx)
        # next_state might wrap around end of dataset
        if next_idx[-1] == self.dset_size:
            #rospy.loginfo(" If ")
            next_idx[-1] = 0
            shape = (sample_size,)+self.state[0].shape
            #rospy.loginfo( shape )
            next_states = np.empty(shape, dtype=np.uint8)
            next_states[0:-1] = self.state[next_idx[0:-1]]
            next_states[-1] = self.state[0]
            #rospy.loginfo( next_states )
        else:
            #rospy.loginfo(" Else ")
            next_states = self.state[next_idx]
            #rospy.loginfo( next_states.shape )
            #rospy.loginfo( next_states )
            
        #is_non_terminal = np.array([not self.is_terminal(idx) for idx in next_idx], dtype=bool)
        return (self.state[idx], self.action[next_idx], self.reward[next_idx], next_states, self.non_terminal[next_idx])

    def __del__(self):
        self.fp['non_terminal'][:] = self.non_terminal
        self.fp['action'][:] = self.action
        self.fp['reward'][:] = self.reward
        self.state.attrs['head'] = self.head
        self.state.attrs['valid'] = self.valid

        self.fp.close()

###############################################################################