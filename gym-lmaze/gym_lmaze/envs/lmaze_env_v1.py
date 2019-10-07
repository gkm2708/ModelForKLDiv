import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt
plt.ion()
plt.show(block=False)

class LmazeEnv_v1(gym.Env):
  metadata = {'render.modes': ['human']}

  """    
  """
  def __init__(self):
    print("init-init")
    self.action_space = spaces.Discrete(4)
    self.realgrid = 14
    self.expansionRatio = 7
    self.fovea = 5
    self.gridsize = self.fovea * self.expansionRatio
    self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4,self.gridsize,self.gridsize))
    self.negativeNominal = -1.0
    self.positiveNominal = 0.01
    self.positiveFull = 1.0
    self.RANDOM_BALL = False
    self.VISUALIZE = False
    self.f_goal_x = 0
    self.f_goal_y = 0
    self.localDone = False
    self.stepCount = 0
    self.fovealStepCount = 0

    self.visualFile = open("visualize.txt", "r")

    self.grid = np.array(       [['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'S', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'W', 'W', 'W', 'B', 'W', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'B', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'B', 'B', 'W', 'X', 'W', 'B', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'B', 'B', 'B', 'W', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'W', 'W', 'W', 'W', 'W', 'B', 'W', 'B', 'W', 'W', 'W'],
                                ['W','W', 'B', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'W', 'W', 'W'],
                                ['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                                ['W','W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']])

    self.state = np.zeros((self.realgrid * self.realgrid * 4), dtype=np.float32)

    self.reset()
    print("init-end")

  """             Reset-Initialize Handle           """
  """    
  """
  def reset(self):                        # just reset

    self.state = np.zeros((self.realgrid * self.realgrid * 4), dtype=np.float32)
    self.state[0 * self.realgrid * self.realgrid : 1 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "S" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    self.state[1 * self.realgrid * self.realgrid : 2 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "W" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    self.state[2 * self.realgrid * self.realgrid : 3 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "X" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    self.state[3 * self.realgrid * self.realgrid : 4 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "B" or cell == "S" or cell == "X" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    start = np.where(self.grid == 'S')
    self.ball_x0 = start[0][0]
    self.ball_y0 = start[1][0]

    s1 = np.where(self.grid == 'X')
    self.goal_x = s1[0][0]
    self.goal_y = s1[1][0]

    self.originalReward = -0.0
    self.fovealReward = -0.0

    self.stepCount = 0
    #self.fovealStepCount = 0

    self.img = None

    #self.VISUALIZE = self.visualFile.readline()
    #print(self.visualFile.readline())
    return self.getGlobalView()

  """    
  """
  def setFovealGoal(self,msg0,msg1):
      # translate to global goal position for the loacl value
      # set it as the foveal goal
      self.f_goal_x = self.ball_x0 + msg0 - 2
      self.f_goal_y = self.ball_y0 + msg1 - 2
      self.fovealStepCount = 0
      return self.getLocalView()

  """
  """
  def step(self,msg):

    """ Build Action Value to be used for ball position update """
    self.stepCount += 1
    self.fovealStepCount += 1

    self.fovealReward = -0.0
    self.originalReward = -0.0

    self.localDone = False

    o_x, o_y = 0, 0
    if msg == 0:
        o_x, o_y = -1, 0
    elif msg == 1:
        o_x, o_y = 1, 0
    elif msg == 2:
        o_x, o_y = 0, -1
    elif msg == 3:
        o_x, o_y = 0, 1

    if self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'W':
        # position unchanged small negative reward
        self.originalReward = self.negativeNominal
        self.fovealReward = self.negativeNominal

    elif self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'B':
        # position changed small positive reward
        self.state[0 * self.realgrid * self.realgrid + self.ball_x0 * self.realgrid + self.ball_y0] = 0.0

        self.ball_x0 = self.ball_x0 + o_x
        self.ball_y0 = self.ball_y0 + o_y

        self.state[0*self.realgrid*self.realgrid+(self.ball_x0)*self.realgrid+self.ball_y0] = 1.0

        self.originalReward = self.positiveNominal
        self.fovealReward = self.positiveNominal

        # if the goal goes out of view of agent after this movement
        if (self.ball_x0 < self.f_goal_x - 1 or self.ball_x0 > self.f_goal_x +2 \
                or self.ball_y0 < self.f_goal_y - 1 or self.ball_y0 > self.f_goal_y +2):
            #print("Local Goal out of fovea")
            #print(self.ball_x0, self.f_goal_x - 1, self.ball_x0, self.f_goal_x +2, self.ball_y0, self.f_goal_y - 1, self.ball_y0, self.f_goal_y +2)
            self.localDone = True
            self.fovealReward = self.negativeNominal
        elif self.ball_y0 == self.f_goal_y and self.ball_x0 == self.f_goal_x:
            #print("Local Goal acheived")
            #print(self.ball_x0, self.f_goal_x, self.ball_y0, self.f_goal_y)
            self.localDone = True
            self.fovealReward = self.positiveFull

    elif self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'X':
        # position changed full reward
        #print("Global Goal acheived")

        self.state[0*self.realgrid*self.realgrid+(self.ball_x0)*self.realgrid+self.ball_y0] = 0.0

        self.ball_x0 = self.ball_x0 + o_x
        self.ball_y0 = self.ball_y0 + o_y

        self.state[0*self.realgrid*self.realgrid+(self.ball_x0)*self.realgrid+self.ball_y0] = 1.0
        self.originalReward = self.positiveFull

        if self.ball_y0 == self.f_goal_y and self.ball_x0 == self.f_goal_x:
            #print("Local Goal acheived")
            #print(self.ball_x0, self.f_goal_x, self.ball_y0, self.f_goal_y)
            self.localDone = True
            self.fovealReward = self.positiveFull
        else:
            self.fovealReward = self.positiveNominal

    if self.VISUALIZE:
        image = np.asarray(([200 if item == 1.0 else 0 for item in self.state[1 * self.realgrid * self.realgrid : 2 * self.realgrid * self.realgrid]]), dtype=np.uint8)
        image[(self.goal_x) * self.realgrid + self.goal_y] = 100
        image[(self.ball_x0) * self.realgrid + self.ball_y0] = 150

        image2 = np.reshape(image,(self.realgrid,self.realgrid))
        if self.img == None:
            plt.clf()
            self.img = plt.imshow(image2)
        else:
            self.img.set_data(image2)

        plt.pause(.01)
        plt.draw()

    return self.getLocalView(), self.originalReward, self.fovealReward, self.isFovealEpisodeFinished(), self.isEpisodeFinished("print"), msg

  """    
  """
  def getGlobalView(self):

      self.state1 = np.concatenate(
          (np.reshape(self.state[0 * self.realgrid * self.realgrid: 1 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[1 * self.realgrid * self.realgrid: 2 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[2 * self.realgrid * self.realgrid: 3 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[3 * self.realgrid * self.realgrid: 4 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid)))), axis=0)

      self.retState = np.asarray(self.state1[:, self.ball_x0 - 2:self.ball_x0 + 3, self.ball_y0 - 2:self.ball_y0 + 3])

      self.retStateExpanded = np.zeros((4, self.retState.shape[1] * self.expansionRatio,
                                   self.retState.shape[2] * self.expansionRatio), dtype=np.float32)
      channel = 0
      while channel < self.retState.shape[0]:
          i = 0
          while i < self.retState.shape[1]:
              ii = 0
              for ii in range(0, self.expansionRatio):
                  j = 0
                  while j < self.retState.shape[2]:
                      jj = 0
                      for jj in range(0, self.expansionRatio):
                          self.retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                          self.retState[channel][i][j]
                          jj += 1
                      j += 1
                  ii += 1
              i += 1
          channel += 1

      return self.retStateExpanded

  """    
  """
  def getLocalView(self):

      localGoal = np.zeros((self.realgrid*self.realgrid))
      localGoal[self.f_goal_x * self.realgrid + self.f_goal_y] = 1.0

      self.state1 = np.concatenate(
          (np.reshape(self.state[0 * self.realgrid * self.realgrid: 1 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[1 * self.realgrid * self.realgrid: 2 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid))),
           np.reshape(localGoal,((1, self.realgrid, self.realgrid))),
           np.reshape(self.state[3 * self.realgrid * self.realgrid: 4 * self.realgrid * self.realgrid],
                      ((1, self.realgrid, self.realgrid)))), axis=0)

      self.retState = np.asarray(self.state1[:, self.ball_x0 - 2:self.ball_x0 + 3, self.ball_y0 - 2:self.ball_y0 + 3])

      self.retStateExpanded = np.zeros((4, self.retState.shape[1] * self.expansionRatio, self.retState.shape[2] * self.expansionRatio),
                                  dtype=np.float32)

      channel = 0
      while channel < self.retState.shape[0]:
          i = 0
          while i < self.retState.shape[1]:
              ii = 0
              for ii in range(0, self.expansionRatio):
                  j = 0
                  while j < self.retState.shape[2]:
                      jj = 0
                      for jj in range(0, self.expansionRatio):
                          self.retStateExpanded[channel][i * self.expansionRatio + ii][j * self.expansionRatio + jj] = \
                          self.retState[channel][i][j]
                          jj += 1
                      j += 1
                  ii += 1
              i += 1
          channel += 1

      return self.retStateExpanded

  """    
  """
  def render(self, mode='human', close=False):
    if(mode == 'human'):
        self.VISUALIZE = True

  """    
  """
  def initState(self):
    return self.state, self.originalReward, self.isEpisodeFinished(), {'newState' : True}

  """    
  """
  def isEpisodeFinished(self, queryType="plain"):
    if self.originalReward == self.positiveFull or self.stepCount == 200:
        """
        if self.originalReward == self.positiveFull and queryType == "print":
            print(" =====================> Global Episode Done due to Global Goal Reached", self.originalReward, self.fovealReward)
        if self.stepCount == 200 and queryType == "print":
            print(" =====================> Global Episode Done due to Max Steps", self.originalReward, self.fovealReward)
        #self.stepCount = 0
        """
        return True
    return False

  """    
  """
  def isFovealEpisodeFinished(self):
    if self.localDone or self.fovealReward == self.positiveFull or self.fovealStepCount == 10 or self.isEpisodeFinished():
        """
        if self.localDone and self.fovealReward == self.positiveNominal:
            print("Local Episode Done due to Goal out of fovea", self.originalReward, self.fovealReward)
        elif self.fovealStepCount == 10:
            print("Local Episode Done due to Max Steps", self.originalReward, self.fovealReward)
        elif self.isEpisodeFinished():
            print("Local Episode Done due to Global Episode", self.originalReward, self.fovealReward)
        elif self.fovealReward == self.positiveFull:
            print("Local Episode Done due to foveal Goal Acheived", self.originalReward, self.fovealReward)
        else:
            print("Local Episode Done due to unknown reason", self.originalReward, self.fovealReward)
        """
        self.localDone = False
        return True
    return False
