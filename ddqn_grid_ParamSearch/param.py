import numpy as np
import os
#import rospy

class Param:
    def __init__(self):

        self.HOSTNAME = os.uname()[1]
        
        """ GRID WORLD PARAMS"""
        self.DYNAMICS = False # Should not be true if gyro is not set        
        self.RANDOM_GOAL = True
        self.RANDOM_HOLE = False
        self.RANDOM_BALL = False
        self.GRID_SIZE = 7


        """ SENSORY PARAMS """
        self.FRAME_SKIP = 3
        self.GYRO_DIM = 2

        """ ALL INPUT PARAMS """
        self.IMAGE_DIM = 84

        if self.RANDOM_HOLE == True:
            self.STATE_SIZE = [self.GRID_SIZE*self.GRID_SIZE*6 + self.GYRO_DIM]                      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
            self.STATE_SHAPE = np.zeros((self.GRID_SIZE*self.GRID_SIZE*6 + self.GYRO_DIM), dtype=np.float32)
        else:
            self.STATE_SIZE = [self.GRID_SIZE*self.GRID_SIZE*5 + self.GYRO_DIM]                      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
            self.STATE_SHAPE = np.zeros((self.GRID_SIZE*self.GRID_SIZE*5 + self.GYRO_DIM), dtype=np.float32)


        """ ROLLOUT PARAMS """
        self.MAX_STEPS = 20                             # maximum time step in one episode

        """ OUTPUT PARAM """
        self.ACTION_SIZE = 4                             #
        xdotplus = [1, 0, 0, 0]
        xdotminus = [0, 1, 0, 0]
        ydotplus = [0, 0, 1, 0]
        ydotminus = [0, 0, 0, 1]
        self.POSSIBLE_ACTIONS = [xdotplus, ydotplus, xdotminus, ydotminus]

        """ ###############  Q LEARNING PARAMS ##################### """
        self.GAMMA = 0.99                                # reward discount in TD error
        self.ALPHA = 0.01

        """ Epsilon-Greedy PARAMS """
        self.DECAY_STEP_INIT = 0
        self.DECAY_EPISODE_INIT = 0
        self.EXPLORE_START = 1.0                         # exploration probability at start
        self.EXPLORE_STOP = 0.1                         # minimum exploration probability 
        """ varied for param search """
        self.DECAY_RATE = 0.000002                  


        """ LEARNING RATE PARAMS """
        """ varied for param search """
        self.LEARNING_RATE = 0.0001                    # Alpha 10^-7
        """ varied for param search """
        self.DECAY_WINDOW = 20000

        self.DECAY_BASE = 0.95
        self.DECAY_UNTIL = 1000000

        """ EXPERIENCE REPLAY """
        """ varied for param search """
        self.BATCH_SIZE = 64             
        """ varied for param search """
        self.TRAIN_SIZE = 102
        """ varied for param search """
        self.PRETRAIN_LENGTH = 50000                    # Number of experiences stored in the Memory when initialized for the first time
        """ varied for param search """
        self.MEMORY_SIZE     = 250000                   # Number of experiences the Memory can keep
        self.TRAIN_LENGTH    = 10000000

        """ DQN """
        self.WHEN_UPDATE_TARGET = 100
        self.WHEN_TRAINED = 5

        """ D-DQN """
        """ varied for param search """
        self.SWITCH_CYCLE = 10
        self.SWITCH_CYCLE_HALF = 5

        """ HER """
        self.ENDPOINT_AUGMENT = True
        self.EPISODE_AUGMENT = True
        self.HER_SAMPLE = 3

        """  MISC """
        self.STEP_PENALTY = False

        """ REPORTING PARAMS """
        self.WHEN_BACKUP = 5000
        self.WHEN_SAVED = 5000
        self.WHEN_EVALUATED = 1000
        self.HOW_MANY_EVALUATIONS = 100
        self.WHEN_SAVE_STATS = 100

        """ TRAINING CONTROL """
        self.TRAINING = True
        self.PRE_TRAINING = self.TRAINING
        self.TEST = True

        """ TF NETWORK """
        """ varied for param search """
        self.NUM_LAYERS=4
        """ varied for param search """
        self.LAYER_SIZE=[256,192,128,64]
        
        """ TRAINING SESSION """
        self.RUN_ID = "DDQN_GRID_05_July"
        self.BUFFER_FILE = '/vol/speech/gaurav/datadrive/'+self.HOSTNAME+'/rbuffer.hdf'
        
        
        self.static_goal_x = 3
        self.static_goal_y = 3

        self.static_start_x = 3
        self.static_start_y = 3

        #rospy.loginfo(self.HOSTNAME)

        self.printParam()

    def printParam(self):
    
        os.system('mkdir -p /homes/gkumar/models/'+self.RUN_ID+'/'+self.HOSTNAME)
        file = open("/homes/gkumar/models/"+self.RUN_ID+"/"+self.HOSTNAME+"/initParam.txt","w+") 

        file.write("\n\n GRID WORLD PARAMS\n\n")
        file.write("\n DYNAMICS = "+str(self.DYNAMICS))
        file.write("\n RANDOM_GOAL = "+str(self.RANDOM_GOAL))
        file.write("\n RANDOM_HOLE = "+str(self.RANDOM_HOLE))    
        file.write("\n RANDOM_BALL = "+str(self.RANDOM_BALL))
        file.write("\n GRID_SIZE = "+str(self.GRID_SIZE))

        file.write("\n\n ALL INPUT PARAMS\n\n")
        file.write("\n IMAGE_DIM = "+str(self.IMAGE_DIM))
        file.write("\n STATE_SIZE = "+str(self.STATE_SIZE))
        file.write("\n STATE_SHAPE = "+str(self.STATE_SHAPE.shape))

        file.write("\n\n SENSORY PARAMS\n\n")
        file.write("\n FRAME_SKIP = "+str(self.FRAME_SKIP))    
        file.write("\n GYRO_DIM = "+str(self.GYRO_DIM))

        file.write("\n\n ROLLOUT PARAMS\n\n")
        file.write("\n MAX_STEPS = "+str(self.MAX_STEPS))
        
        file.write("\n\n OUTPUT PARAM\n\n")
        file.write("\n ACTION_SIZE = "+str(self.ACTION_SIZE))
        file.write("\n POSSIBLE_ACTIONS = "+str(self.POSSIBLE_ACTIONS))

        file.write("\n\n\n ###############  Q LEARNING PARAMS #####################\n\n\n")
        file.write("\n GAMMA = "+str(self.GAMMA))
        file.write("\n ALPHA = "+str(self.ALPHA))

        file.write("\n\n Epsilon-Greedy PARAMS\n\n")
        file.write("\n DECAY_STEP_INIT = "+str(self.DECAY_STEP_INIT))
        file.write("\n DECAY_EPISODE_INIT = "+str(self.DECAY_EPISODE_INIT))
        file.write("\n EXPLORE_START = "+str(self.EXPLORE_START))
        file.write("\n EXPLORE_STOP = "+str(self.EXPLORE_STOP))
        file.write("\n DECAY_RATE = "+str(self.DECAY_RATE))
    
        file.write("\n\n LEARNING RATE PARAMS\n\n")    
        file.write("\n LEARNING_RATE = "+str(self.LEARNING_RATE))
        file.write("\n DECAY_WINDOW = "+str(self.DECAY_WINDOW))
        file.write("\n DECAY_BASE = "+str(self.DECAY_BASE))
        file.write("\n DECAY_UNTIL = "+str(self.DECAY_UNTIL))
    
        file.write("\n\n EXPERIENCE REPLAY\n\n")
        file.write("\n BATCH_SIZE = "+str(self.BATCH_SIZE))             
        file.write("\n TRAIN_SIZE = "+str(self.TRAIN_SIZE))
        file.write("\n PRETRAIN_LENGTH = "+str(self.PRETRAIN_LENGTH))
        file.write("\n MEMORY_SIZE = "+str(self.MEMORY_SIZE))
        file.write("\n TRAIN_LENGTH = "+str(self.TRAIN_LENGTH))

        file.write("\n\n DQN\n\n")
        file.write("\n WHEN_UPDATE_TARGET = "+str(self.WHEN_UPDATE_TARGET))
        file.write("\n WHEN_TRAINED = "+str(self.WHEN_TRAINED))

        file.write("\n\n D-DQN\n\n")
        file.write("\n SWITCH_CYCLE = "+str(self.SWITCH_CYCLE))
        file.write("\n SWITCH_CYCLE_HALF = "+str(self.SWITCH_CYCLE_HALF))

        file.write("\n\n HER\n\n")
        file.write("\n ENDPOINT_AUGMENT = "+str(self.ENDPOINT_AUGMENT))
        file.write("\n EPISODE_AUGMENT = "+str(self.EPISODE_AUGMENT))
        file.write("\n HER_SAMPLE = "+str(self.HER_SAMPLE))

        file.write("\n\n MISC\n\n")
        file.write("\n STEP_PENALTY = "+str(self.STEP_PENALTY))

        file.write("\n\n REPORTING PARAMS\n\n")
        file.write("\n WHEN_BACKUP = "+str(self.WHEN_BACKUP))
        file.write("\n WHEN_SAVED = "+str(self.WHEN_SAVED))
        file.write("\n WHEN_EVALUATED = "+str(self.WHEN_EVALUATED))
        file.write("\n HOW_MANY_EVALUATIONS = "+str(self.HOW_MANY_EVALUATIONS))
        file.write("\n WHEN_SAVE_STATS = "+str(self.WHEN_SAVE_STATS))
    
        file.write("\n\n TRAINING CONTROL\n\n")
        file.write("\n TRAINING = "+str(self.TRAINING))
        file.write("\n PRE_TRAINING = "+str(self.PRE_TRAINING))
        file.write("\n TEST = "+str(self.TEST))

        file.write("\n\n TF NETWORK\n\n")
        file.write("\n NUM_LAYERS = "+str(self.NUM_LAYERS))
        file.write("\n LAYER_SIZE = "+str(self.LAYER_SIZE))

        file.write("\n\n TRAINING SESSION\n\n")
        file.write("\n RUN_ID = "+str(self.RUN_ID))
        file.write("\n BUFFER_FILE = "+str(self.BUFFER_FILE))

        file.write("\n\n SIMULATION SESSION\n\n")
        file.write("\n STATIC GOAL = "+str(self.static_goal_x)+", "+str(self.static_goal_y))
        file.write("\n STATIC BALL = "+str(self.static_start_x)+", "+str(self.static_start_y))
    

        file.close() 
