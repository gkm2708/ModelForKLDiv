#!/usr/bin/env python
import tensorflow as tf                     # Deep Learning library
#import rospy

from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

class Network:
    
    def __init__(self, p):
        self.p = p
        #rospy.loginfo(" Created ")
    
    def createNet(self, NAME='dqn'):        
        with tf.variable_scope(NAME):

            """ Placeholders """
            self.inputs_ = Input([self.p.STATE_SIZE[0]], name='A_Image_Input')
            self.actions_ = tf.placeholder(tf.float32, [None, self.p.ACTION_SIZE], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """ add the action vector here """
            x = Dense(self.p.LAYER_SIZE[0], activation='relu', name='main_input')(self.inputs_)

            for i in range(0, self.p.NUM_LAYERS-1):
                x = Dense(self.p.LAYER_SIZE[i+1], activation='relu')(x)

            self.output = Dense(self.p.ACTION_SIZE, activation='linear')(x)

            self.model = Model(inputs=self.inputs_, outputs=self.output)


            self.global_step = tf.Variable(0, trainable=False)
            decayed_lr = tf.train.exponential_decay(self.p.LEARNING_RATE, self.global_step, self.p.DECAY_WINDOW, self.p.DECAY_BASE, staircase=False)
            
            adam  = Adam(lr=decayed_lr)
            self.model.compile(loss="mse", optimizer=adam)

            """ Q is our predicted Q value """
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)            

            """ The loss is the difference between our predicted Q_values and the Q_target; Sum(Qtarget - Q)^2 """
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            #self.optimizer = tf.train.AdamOptimizer(decayed_lr).minimize(self.loss, global_step=self.global_step)

            optimise = tf.train.AdamOptimizer(decayed_lr)
            grads = optimise.compute_gradients(self.loss)
            capped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads if grad is not None]
            self.optimizer = optimise.apply_gradients(capped_grads, global_step=self.global_step)            
            
        return self.inputs_, self.model, self.Q, self.loss, self.optimizer