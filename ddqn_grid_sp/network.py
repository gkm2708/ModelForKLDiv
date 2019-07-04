#!/usr/bin/env python
import tensorflow as tf                     # Deep Learning library
import rospy

from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

class Network:
    
    def __init__(self):
        rospy.loginfo(" Created ")
    
    def createNet(self, STATE_SIZE, ACTION_SIZE, LEARNING_RATE, DECAY_WINDOW, DECAY_BASE, NAME='dqn'):        
        with tf.variable_scope(NAME):

            inputSize = STATE_SIZE[0]
            """ Placeholders """
            self.inputs_ = Input([inputSize], name='A_Image_Input')
            self.actions_ = tf.placeholder(tf.float32, [None, ACTION_SIZE], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            #x = Conv2D(32, (8, 8), padding='same', activation='relu', strides=(4, 4))(self.inputs_)
            #x = Conv2D(84, (4, 4), padding='same', activation='relu', strides=(2, 2))(x)
            #x = Conv2D(128, (4, 4), padding='same', activation='relu', strides=(2, 2))(x)
            #x = Flatten()(x)

            """ add the action vector here """
            x = Dense(256, activation='relu')(self.inputs_)
            x = Dense(128, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            x = Dense(32, activation='relu')(x)
            
            #x = Dense(inputSize, activation='relu')(self.inputs_)
            #x = Dense(inputSize, activation='relu')(x)
            #x = Dense(16, activation='relu')(x)


            self.output = Dense(ACTION_SIZE, activation='linear', name='main_output')(x)

            self.model = Model(inputs=self.inputs_, outputs=self.output)


            self.global_step = tf.Variable(0, trainable=False)
            decayed_lr = tf.train.exponential_decay(LEARNING_RATE, self.global_step, DECAY_WINDOW, DECAY_BASE, staircase=False)
            
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