#!/usr/bin/env python

import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from random import randint

class LMazeInterface():

    def __init__(self,FRAME_SKIP,outputDim, gyro, dyna, randomGoal, randomHole, randomBall):
        rospy.init_node("lmaze_interface")

        self.grid = np.array([['W', 'W', 'W', 'W', 'W', 'W', 'W'],
                    ['W', 'S', 'B', 'B', 'B', 'B', 'W'],
                    ['W', 'B', 'B', 'B', 'B', 'B', 'W'],
                    ['W', 'B', 'B', 'B', 'B', 'B', 'W'],
                    ['W', 'B', 'B', 'B', 'B', 'B', 'W'],
                    ['W', 'B', 'B', 'B', 'B', 'X', 'W'],
                    ['W', 'W', 'W', 'W', 'W', 'W', 'W']])

        self.action = np.array(['L', 'R', 'U', 'D'])


        self.gridsize = 7
        self.bridge = CvBridge()
        self.gyro = gyro
        self.dyna = dyna
        self.randomGoal = randomGoal
        self.randomHole = randomHole
        self.randomBall = randomBall
        self.reset()

        #self.goal_x = self.gridsize - 1 - 1
        #self.goal_y = self.gridsize - 1 - 1

        self.imagered1_pub = rospy.Publisher('/interface/ballscene1', Image, queue_size=1)
        self.imageblue1_pub = rospy.Publisher('/interface/holescene1', Image, queue_size=1)
        self.imageyellow1_pub = rospy.Publisher('/interface/goalscene1', Image, queue_size=1)

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

        #value = randint(1, 5)

        self.o_x = 0
        self.o_y = 0
        
        rand = randint(0,3)

        if self.randomGoal == True:
            if rand == 0:
                self.goal_x = 1
                self.goal_y = 1
            elif rand == 1:
                self.goal_x = self.gridsize-2
                self.goal_y = self.gridsize-2
            elif rand == 2:
                self.goal_x = 1
                self.goal_y = self.gridsize-2
            elif rand == 3:
                self.goal_x = self.gridsize-2
                self.goal_y = 1
        else :
                self.goal_x = 4
                self.goal_y = 3

        if self.randomHole == True:
            if rand == 0:
                self.hole_x1 = self.gridsize-2
                self.hole_y1 = self.gridsize-2
                self.hole_x2 = 1
                self.hole_y2 = self.gridsize-2
                self.hole_x3 = self.gridsize-2
                self.hole_y3 = 1
            elif rand == 1:
                self.hole_x1 = 1
                self.hole_y1 = 1
                self.hole_x2 = 1
                self.hole_y2 = self.gridsize-2
                self.hole_x3 = self.gridsize-2
                self.hole_y3 = 1
            elif rand == 2:
                self.hole_x1 = 1
                self.hole_y1 = 1
                self.hole_x2 = self.gridsize-2
                self.hole_y2 = self.gridsize-2
                self.hole_x3 = self.gridsize-2
                self.hole_y3 = 1
            elif rand == 3:
                self.hole_x1 = 1
                self.hole_y1 = 1
                self.hole_x2 = self.gridsize-2
                self.hole_y2 = self.gridsize-2
                self.hole_x3 = 1
                self.hole_y3 = self.gridsize-2
        else :
            self.hole_x1 = -1
            self.hole_y1 = -1
            self.hole_x2 = -1
            self.hole_y2 = -1
            self.hole_x3 = -1
            self.hole_y3 = -1

            

        if self.randomBall == True:
            self.ball_x0 = randint(1, self.gridsize-2)
            self.ball_y0 = randint(1, self.gridsize-2)

            self.ball_x1 = self.ball_x0
            self.ball_y1 = self.ball_y0

            self.ball_x2 = self.ball_x0
            self.ball_y2 = self.ball_y0

            self.ball_x3 = self.ball_x0
            self.ball_y3 = self.ball_y0
        else:
            self.ball_x0 = 3
            self.ball_y0 = 3

            self.ball_x1 = self.ball_x0
            self.ball_y1 = self.ball_y0

            self.ball_x2 = self.ball_x0
            self.ball_y2 = self.ball_y0

            self.ball_x3 = self.ball_x0
            self.ball_y3 = self.ball_y0
            
        self.reward = -0.0


    """    
    """                        
    def step(self,msg):   
        if self.dyna == True:
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


        self.ball_x3 = self.ball_x2
        self.ball_y3 = self.ball_y2

        self.ball_x2 = self.ball_x1        
        self.ball_y2 = self.ball_y1        
        
        self.ball_x1 = self.ball_x0        
        self.ball_y1 = self.ball_y0        


        self.ball_x0 = self.ball_x0 + self.o_x
        self.ball_y0 = self.ball_y0 + self.o_y


        if (self.ball_x0 < 1):
            self.ball_x0 = 1
        if (self.ball_x0 > self.gridsize - 2):
            self.ball_x0 = self.gridsize - 2
        if (self.ball_y0 < 1):
            self.ball_y0 = 1
        if (self.ball_y0 > self.gridsize - 2):
            self.ball_y0 = self.gridsize - 2

        if self.ball_x0 == self.goal_x and self.ball_y0 == self.goal_y :
            self.reward = 1
        elif self.randomHole == True:
            if self.ball_x0 == self.hole_x1 and self.ball_y0 == self.hole_y1 or self.ball_x0 == self.hole_x2 and self.ball_y0 == self.hole_y2 or self.ball_x0 == self.hole_x3 and self.ball_y0 == self.hole_y3:
                self.reward = -0.1
        else :
            self.reward = 0
            

        if self.randomHole == True :
            self.state = np.zeros((self.gridsize*self.gridsize*6 + self.gyro),dtype=np.float32)
        else :
            self.state = np.zeros((self.gridsize*self.gridsize*5 + self.gyro),dtype=np.float32)

        # frame 1 - 0-49 - current
        self.state[0*49+(self.ball_x0)*self.gridsize+self.ball_y0] = 0.99
        # frame 1 - 0-49 - current
        self.state[1*49+(self.ball_x1)*self.gridsize+self.ball_y1] = 0.99
        # frame 1 - 0-49 - current
        self.state[2*49+(self.ball_x2)*self.gridsize+self.ball_y2] = 0.99
        # frame 1 - 0-49 - current
        self.state[3*49+(self.ball_x3)*self.gridsize+self.ball_y3] = 0.99

        self.state[4*49+(self.goal_x)*self.gridsize+self.goal_y] = 0.99

        if(self.randomHole == True):
            self.state[5*49+(self.hole_x1)*self.gridsize+self.hole_y1] = 0.99
            self.state[5*49+(self.hole_x2)*self.gridsize+self.hole_y2] = 0.99
            self.state[5*49+(self.hole_x3)*self.gridsize+self.hole_y3] = 0.99

        if self.gyro > 0:
            self.state[-2] = self.o_x
            self.state[-1] = self.o_y
        
        #rospy.sleep(0.01)

        image = np.asarray(([255 if item > 0 else 0 for item in self.state[0*49:4*49]]), dtype=np.uint8)
        image1 = np.asarray([ (image[0*49+i], image[1*49+i], image[2*49+i], image[3*49+i]) for i in range(0,49) ])
        self.imagered1_pub.publish(self.bridge.cv2_to_imgmsg(image1.reshape((7,7,4)),encoding="bgra8"))


        image = np.asarray(([255 if item > 0 else 0 for item in self.state[4*49:5*49]]), dtype=np.uint8).reshape((self.gridsize,self.gridsize))
        self.imageyellow1_pub.publish(self.bridge.cv2_to_imgmsg(image,encoding="mono8"))

        if(self.randomHole == True):
            image = np.asarray(([255 if item > 0 else 0 for item in self.state[5*49:6*49]]), dtype=np.uint8).reshape((self.gridsize,self.gridsize))
            self.imageblue1_pub.publish(self.bridge.cv2_to_imgmsg(image,encoding="mono8"))


        return self.state, self.reward, self.isEpisodeFinished(), {'newState' : True}
        
    """    
    """
    def isEpisodeFinished(self):
        if self.reward == 1 or self.reward == -0.1:
            return True
        return False