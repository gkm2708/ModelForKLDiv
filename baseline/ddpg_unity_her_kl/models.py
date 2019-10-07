import tensorflow as tf
from baselines.common.models import get_network_builder
import numpy as np

class Model(object):
    def __init__(self, name, network='mlp', **network_kwargs):
        self.name = name
        self.network_builder = get_network_builder(network)(**network_kwargs)
        self.encoder_builder = get_network_builder("encoder_mlp")(**network_kwargs)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions
        self.layer_norm = True

    def __call__(self, obs, encOnly=False, reuse=False):
        # create encoder part here and feed the output into the next part
        # this way the actor would have a subpart that is trained with
        # it but can be used to calculate the z for critic as well

        """with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.encoder_builder(obs)
            enc_mu = tf.layers.dense(x, 119, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            enc_sigma = tf.layers.dense(x, 119, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            split0, split1 = tf.split(obs, [137, 2], 1)
            x = self.network_builder(tf.concat(axis=1, values=[enc_mu, split0]))
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x, enc_mu, enc_sigma"""
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x = self.encoder_builder(obs)
            enc_mu = tf.layers.dense(x, 119, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            enc_sigma = tf.layers.dense(x, 119, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        if encOnly:
            return enc_mu, enc_sigma

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            split0, split1 = tf.split(obs, [137, 2], 1)
            x = self.network_builder(tf.concat(axis=1, values=[enc_mu, split0]))
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x




class Critic(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, encoder, action, reuse=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            split0, split1 = tf.split(obs, [137, 2], 1)
            x = tf.concat([split0, encoder, action], axis=-1) # this assumes observation and action can be concatenated
            x = self.network_builder(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars