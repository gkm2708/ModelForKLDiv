#!/usr/bin/env python

#import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from random import randint
from param import Param

class LMazeInterface():

    def __init__(self):

        #rospy.init_node("lmaze_interface")

        self.p = Param()
        self.bridge = CvBridge()

        self.reset()

        #self.imagered1_pub = rospy.Publisher('/interface/ballscene1', Image, queue_size=1)
        #self.imageblue1_pub = rospy.Publisher('/interface/holescene1', Image, queue_size=1)
        #self.imageyellow1_pub = rospy.Publisher('/interface/goalscene1', Image, queue_size=1)

    """
        L   ->      0
        R   ->      1
        U   ->      2
        D   ->      3
    """
        
    """             Reset-Initialize Handle           """
    """    
    """
    def reset(self):                        # just reset

        self.o_x = 0
        self.o_y = 0
        
        self.buildRandomMaze()
        self.reward = -0.0

    def buildRandomMaze(self):                        # just reset
        rand = randint(0,3)
        if self.p.RANDOM_GOAL == True:
            if rand == 0:
                self.goal_x = 1
                self.goal_y = 1
            elif rand == 1:
                self.goal_x = self.p.GRID_SIZE-2
                self.goal_y = self.p.GRID_SIZE-2
            elif rand == 2:
                self.goal_x = 1
                self.goal_y = self.p.GRID_SIZE-2
            elif rand == 3:
                self.goal_x = self.p.GRID_SIZE-2
                self.goal_y = 1
        else :
            self.goal_x = self.p.static_goal_x
            self.goal_y = self.p.static_goal_y

        if self.p.RANDOM_HOLE == True:
            if rand == 0:
                self.hole_x1 = self.p.GRID_SIZE-2
                self.hole_y1 = self.p.GRID_SIZE-2
                self.hole_x2 = 1
                self.hole_y2 = self.p.GRID_SIZE-2
                self.hole_x3 = self.p.GRID_SIZE-2
                self.hole_y3 = 1
            elif rand == 1:
                self.hole_x1 = 1
                self.hole_y1 = 1
                self.hole_x2 = 1
                self.hole_y2 = self.p.GRID_SIZE-2
                self.hole_x3 = self.p.GRID_SIZE-2
                self.hole_y3 = 1
            elif rand == 2:
                self.hole_x1 = 1
                self.hole_y1 = 1
                self.hole_x2 = self.p.GRID_SIZE-2
                self.hole_y2 = self.p.GRID_SIZE-2
                self.hole_x3 = self.p.GRID_SIZE-2
                self.hole_y3 = 1
            elif rand == 3:
                self.hole_x1 = 1
                self.hole_y1 = 1
                self.hole_x2 = self.p.GRID_SIZE-2
                self.hole_y2 = self.p.GRID_SIZE-2
                self.hole_x3 = 1
                self.hole_y3 = self.p.GRID_SIZE-2
        else :
            self.hole_x1 = -1
            self.hole_y1 = -1
            self.hole_x2 = -1
            self.hole_y2 = -1
            self.hole_x3 = -1
            self.hole_y3 = -1

            
        if self.p.RANDOM_BALL == True:
            self.ball_x0 = randint(1, self.p.GRID_SIZE-2)
            self.ball_y0 = randint(1, self.p.GRID_SIZE-2)
            self.ball_x1 = self.ball_x0
            self.ball_y1 = self.ball_y0
            self.ball_x2 = self.ball_x0
            self.ball_y2 = self.ball_y0
            self.ball_x3 = self.ball_x0
            self.ball_y3 = self.ball_y0
        else:
            self.ball_x0 = self.p.static_start_x
            self.ball_y0 = self.p.static_start_y
            self.ball_x1 = self.ball_x0
            self.ball_y1 = self.ball_y0
            self.ball_x2 = self.ball_x0
            self.ball_y2 = self.ball_y0
            self.ball_x3 = self.ball_x0
            self.ball_y3 = self.ball_y0

    """    
    """                        
    def step(self,msg):   

        """ Build Action Value to be used for ball position update """        
        if self.p.DYNAMICS == True:
            if msg == 0:
                self.o_x += -1
            elif msg == 1:
                self.o_x += 1
            elif msg == 2:
                self.o_y += -1
            elif msg == 3:
                self.o_y += 1
        else :
            if msg == 0:
                self.o_x = -1
            elif msg == 1:
                self.o_x = 1
            elif msg == 2:
                self.o_y = -1
            elif msg == 3:
                self.o_y = 1


        """ Update ball position if movement is possible """

        if self.ball_x0 + self.o_x >= 1 and self.ball_x0 + self.o_x <= self.p.GRID_SIZE-1: 
            self.ball_x3 = self.ball_x2
            self.ball_x2 = self.ball_x1        
            self.ball_x1 = self.ball_x0        
            self.ball_x0 = self.ball_x0 + self.o_x

        if self.ball_y0 + self.o_y >= 1 and self.ball_y0 + self.o_y <= self.p.GRID_SIZE-1: 
            self.ball_y3 = self.ball_y2
            self.ball_y2 = self.ball_y1        
            self.ball_y1 = self.ball_y0        
            self.ball_y0 = self.ball_y0 + self.o_y



        """ Check Ball Position and Generate Reward """
        if self.ball_x0 == self.goal_x and self.ball_y0 == self.goal_y :
            self.reward = 1.0
        elif self.p.RANDOM_HOLE == True and (self.ball_x0 == self.hole_x1 and self.ball_y0 == self.hole_y1 or self.ball_x0 == self.hole_x2 and self.ball_y0 == self.hole_y2 or self.ball_x0 == self.hole_x3 and self.ball_y0 == self.hole_y3):
            self.reward = -0.1
        else :
            self.reward = 0.0

            
        """ Build State Input """
        if self.p.RANDOM_HOLE == True:
            self.state = np.zeros((self.p.GRID_SIZE*self.p.GRID_SIZE*6 + self.p.GYRO_DIM),dtype=np.float32)

            self.state[0*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.ball_x0)*self.p.GRID_SIZE+self.ball_y0] = 1.0
            self.state[1*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.ball_x1)*self.p.GRID_SIZE+self.ball_y1] = 1.0
            self.state[2*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.ball_x2)*self.p.GRID_SIZE+self.ball_y2] = 1.0
            self.state[3*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.ball_x3)*self.p.GRID_SIZE+self.ball_y3] = 1.0
            self.state[4*self.p.GRID_SIZE*self.p.GRID_SIZE+self.goal_x*self.p.GRID_SIZE+self.goal_y] = 1.0

            self.state[5*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.hole_x1)*self.p.GRID_SIZE+self.hole_y1] = 1.0
            self.state[5*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.hole_x2)*self.p.GRID_SIZE+self.hole_y2] = 1.0
            self.state[5*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.hole_x3)*self.p.GRID_SIZE+self.hole_y3] = 1.0

        else :
            self.state = np.zeros((self.p.GRID_SIZE*self.p.GRID_SIZE*5 + self.p.GYRO_DIM),dtype=np.float32)

            self.state[0*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.ball_x0)*self.p.GRID_SIZE+self.ball_y0] = 1.0
            self.state[1*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.ball_x1)*self.p.GRID_SIZE+self.ball_y1] = 1.0
            self.state[2*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.ball_x2)*self.p.GRID_SIZE+self.ball_y2] = 1.0
            self.state[3*self.p.GRID_SIZE*self.p.GRID_SIZE+(self.ball_x3)*self.p.GRID_SIZE+self.ball_y3] = 1.0
            self.state[4*self.p.GRID_SIZE*self.p.GRID_SIZE+self.goal_x*self.p.GRID_SIZE+self.goal_y] = 1.0


        if self.p.GYRO_DIM > 0:
            self.state[-2] = self.o_x
            self.state[-1] = self.o_y
        

        """ For Visualization Purpose """
        """
        image = np.asarray(([255 if item > 0 else 0 for item in self.state[0*self.p.GRID_SIZE*self.p.GRID_SIZE:4*self.p.GRID_SIZE*self.p.GRID_SIZE]]), dtype=np.uint8)
        image1 = np.asarray([ (image[0*self.p.GRID_SIZE*self.p.GRID_SIZE+i], image[1*self.p.GRID_SIZE*self.p.GRID_SIZE+i], image[2*self.p.GRID_SIZE*self.p.GRID_SIZE+i], image[3*self.p.GRID_SIZE*self.p.GRID_SIZE+i]) for i in range(0,self.p.GRID_SIZE*self.p.GRID_SIZE) ])
        self.imagered1_pub.publish(self.bridge.cv2_to_imgmsg(image1.reshape((self.p.GRID_SIZE,self.p.GRID_SIZE,4)),encoding="bgra8"))

        image = np.asarray(([255 if item > 0 else 0 for item in self.state[4*self.p.GRID_SIZE*self.p.GRID_SIZE:5*self.p.GRID_SIZE*self.p.GRID_SIZE]]), dtype=np.uint8).reshape((self.p.GRID_SIZE,self.p.GRID_SIZE))
        self.imageyellow1_pub.publish(self.bridge.cv2_to_imgmsg(image,encoding="mono8"))

        if(self.p.RANDOM_HOLE == True):
            image = np.asarray(([255 if item > 0 else 0 for item in self.state[5*self.p.GRID_SIZE*self.p.GRID_SIZE:6*self.p.GRID_SIZE*self.p.GRID_SIZE]]), dtype=np.uint8).reshape((self.p.GRID_SIZE,self.p.GRID_SIZE))
            self.imageblue1_pub.publish(self.bridge.cv2_to_imgmsg(image,encoding="mono8"))
        """
        return self.state, self.reward, self.isEpisodeFinished(), {'newState' : True}
        
    """    
    """
    def isEpisodeFinished(self):
        if self.reward == 1.0 or self.reward == -0.1:
            return True
        return False