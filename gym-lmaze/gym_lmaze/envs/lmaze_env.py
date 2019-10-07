import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import matplotlib.pyplot as plt
plt.ion()
plt.show(block=False)

class LmazeEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    print("init-init")
    self.action_space = spaces.Discrete(4)
    self.realgrid = 12
    self.expansionRatio = 7
    self.gridsize = self.realgrid * self.expansionRatio
    self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4,self.gridsize,self.gridsize))
    self.negativeNominal = -1.0
    self.positiveNominal = 0.01
    self.positiveFull = 1.0
    self.goalCount = 0
    self.RANDOM_BALL = True
    self.VISUALIZE = True

    """self.gri = np.array(  [['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                            ['W', 'S', 'B', 'B', 'B', 'W', 'W', 'B'],
                            ['W', 'B', 'W', 'W', 'W', 'W', 'W', 'B'],
                            ['W', 'B', 'W', 'B', 'W', 'B', 'B', 'B'],
                            ['W', 'B', 'W', 'B', 'W', 'B', 'W', 'W'],
                            ['W', 'B', 'B', 'B', 'W', 'X', 'W', 'W'],
                            ['W', 'B', 'W', 'B', 'W', 'B', 'W', 'W'],
                            ['W', 'B', 'W', 'B', 'W', 'B', 'W', 'W']])"""

    self.grid = np.array(  [['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W'],
                            ['W', 'S', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'W'],
                            ['W', 'B', 'W', 'W', 'B', 'W', 'W', 'W', 'W', 'W', 'B', 'W'],
                            ['W', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'B', 'B', 'W'],
                            ['W', 'B', 'B', 'W', 'B', 'B', 'B', 'W', 'W', 'W', 'B', 'W'],
                            ['W', 'B', 'W', 'B', 'W', 'X', 'W', 'B', 'B', 'B', 'B', 'W'],
                            ['W', 'B', 'B', 'B', 'W', 'B', 'B', 'W', 'W', 'B', 'B', 'W'],
                            ['W', 'B', 'W', 'B', 'W', 'B', 'B', 'B', 'B', 'W', 'B', 'W'],
                            ['W', 'B', 'W', 'B', 'B', 'B', 'W', 'B', 'B', 'B', 'B', 'W'],
                            ['W', 'B', 'B', 'W', 'W', 'W', 'B', 'B', 'W', 'W', 'B', 'W'],
                            ['W', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'W'],
                            ['W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W']])

    self.state = np.zeros((self.realgrid * self.realgrid * 4), dtype=np.float32)

    self.reset()
    print("init-end")

  def render(self, mode='human', close=False):
    if(mode == 'human'):
        self.VISUALIZE = True
    else :
        self.VISUALIZE = False

  """             Reset-Initialize Handle           """
  """    
  """
  def reset(self):                        # just reset
    #print("RESET")

    self.state = np.zeros((self.realgrid * self.realgrid * 4), dtype=np.float32)
    if self.RANDOM_BALL:
        x = 0
        y = 0
        while self.grid[x][y] == 'W' or self.grid[x][y] == 'X':
            x = random.randint(1, self.realgrid - 2)
            y = random.randint(1, self.realgrid - 2)

        self.ball_x0 = x
        self.ball_y0 = y

    else:
        self.state[0 * self.realgrid * self.realgrid : 1 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "S" else 0.0 for cell in row] for row in self.grid]),
            (-1))

        start = np.where(self.grid == 'S')
        self.ball_x0 = start[0][0]
        self.ball_y0 = start[1][0]

    self.state[1 * self.realgrid * self.realgrid : 2 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "W" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    self.state[2 * self.realgrid * self.realgrid : 3 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "X" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    s1 = np.where(self.grid == 'X')
    self.goal_x = s1[0][0]
    self.goal_y = s1[1][0]

    self.state[3 * self.realgrid * self.realgrid : 4 * self.realgrid * self.realgrid] = np.reshape(
            np.asarray([[1.0 if cell == "B" else 0.0 for cell in row] for row in self.grid]),
            (-1))

    self.reward = -0.0
    self.stepCount = 0
    self.img = None

    retState = np.concatenate((np.reshape(self.state[0 * self.realgrid * self.realgrid : 1 * self.realgrid * self.realgrid],
                     ((1,self.realgrid,self.realgrid))),
                          np.reshape(self.state[1 * self.realgrid * self.realgrid : 2 * self.realgrid * self.realgrid],
                     ((1,self.realgrid,self.realgrid))),
                          np.reshape(self.state[2 * self.realgrid * self.realgrid : 3 * self.realgrid * self.realgrid],
                     ((1,self.realgrid,self.realgrid))),
                          np.reshape(self.state[3 * self.realgrid * self.realgrid : 4 * self.realgrid * self.realgrid],
                     ((1,self.realgrid,self.realgrid)))), axis=0)

    retStateExpanded = np.zeros((4, self.gridsize, self.gridsize), dtype=np.float32)

    channel = 0
    while channel < retState.shape[0]:
        i = 0
        while i < retState.shape[1]:
            ii = 0
            for ii in range(0,self.expansionRatio):
                j = 0
                while j < retState.shape[2]:
                    jj = 0
                    for jj in range(0,self.expansionRatio):
                        retStateExpanded[channel][i*self.expansionRatio + ii][j*self.expansionRatio + jj] = retState[channel][i][j]
                        jj += 1
                    j += 1
                ii += 1
            i += 1
        channel += 1

    return retStateExpanded


  """    
  """
  def step(self,msg):

    """ Build Action Value to be used for ball position update """
    self.stepCount += 1

    o_x = 0
    o_y = 0
    if msg == 0:
        #print("Action 0")
        o_x = -1
        o_y = 0
    elif msg == 1:
        #print("Action 1")
        o_x = 1
        o_y = 0
    elif msg == 2:
        #print("Action 2")
        o_x = 0
        o_y = -1
    elif msg == 3:
        #print("Action 3")
        o_x = 0
        o_y = 1

    if self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'W':
        # position unchanged small negative reward
        self.reward = self.negativeNominal

    elif self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'B':
        # position changed small positive reward
        self.state[0 * self.realgrid * self.realgrid + self.ball_x0 * self.realgrid + self.ball_y0] = 0.0

        self.ball_x0 = self.ball_x0 + o_x
        self.ball_y0 = self.ball_y0 + o_y

        self.state[0*self.realgrid*self.realgrid+(self.ball_x0)*self.realgrid+self.ball_y0] = 1.0
        self.reward = self.positiveNominal

    elif self.grid[self.ball_x0 + o_x][self.ball_y0 + o_y] == 'X':
        # position changed full reward
        self.state[0*self.realgrid*self.realgrid+(self.ball_x0)*self.realgrid+self.ball_y0] = 0.0

        self.ball_x0 = self.ball_x0 + o_x
        self.ball_y0 = self.ball_y0 + o_y

        self.state[0*self.realgrid*self.realgrid+(self.ball_x0)*self.realgrid+self.ball_y0] = 1.0
        self.reward = self.positiveFull
        self.goalCount += 1
        print("Goal Hit Count ",self.goalCount)

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

    retState = np.concatenate((np.reshape(self.state[0 * self.realgrid * self.realgrid : 1 * self.realgrid * self.realgrid],
                     ((1,self.realgrid,self.realgrid))),
                          np.reshape(self.state[1 * self.realgrid * self.realgrid : 2 * self.realgrid * self.realgrid],
                     ((1,self.realgrid,self.realgrid))),
                          np.reshape(self.state[2 * self.realgrid * self.realgrid : 3 * self.realgrid * self.realgrid],
                     ((1,self.realgrid,self.realgrid))),
                          np.reshape(self.state[3 * self.realgrid * self.realgrid : 4 * self.realgrid * self.realgrid],
                     ((1,self.realgrid,self.realgrid)))), axis=0)

    retStateExpanded = np.zeros((4, self.gridsize, self.gridsize), dtype=np.float32)

    channel = 0
    while channel < retState.shape[0]:
        i = 0
        while i < retState.shape[1]:
            ii = 0
            for ii in range(0,self.expansionRatio):
                j = 0
                while j < retState.shape[2]:
                    jj = 0
                    for jj in range(0,self.expansionRatio):
                        retStateExpanded[channel][i*self.expansionRatio + ii][j*self.expansionRatio + jj] = retState[channel][i][j]
                        jj += 1
                    j += 1
                ii += 1
            i += 1
        channel += 1

    return retStateExpanded, self.reward, self.isEpisodeFinished(), msg

    """    
    """
  def initState(self):
    return self.state, self.reward, self.isEpisodeFinished(), {'newState' : True}


  def isEpisodeFinished(self):
    if self.reward == self.positiveFull or self.stepCount == 100:
        return True
    return False
