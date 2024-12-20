# -*- coding: utf-8 -*-
"""
Created on Tuesday Sep. 26 2023
@author: Nuocheng Yang, MingzheChen
@github: https://github.com/YangNuoCheng, https://github.com/mzchen0 
"""

import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras import backend as K
from keras.optimizers import *
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, Concatenate, BatchNormalization, Dropout, Multiply

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

MIN_BETA = 0.4
MAX_BETA = 1.0
HUBER_LOSS_DELTA = 1.0

def huber_loss(y_true, y_predict):
    err = y_true - y_predict

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)

    return K.mean(loss)

class Brain(object):
    def __init__(self, state_size, action_size, brain_name, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.weight_backup = brain_name
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']
        self.test = arguments['test']
        self.num_nodes = arguments['number_nodes']
        self.optimizer_model = arguments['optimizer']
        self.model = self.build_model()
        self.model_ = self.build_model()
# THEIRS
#     def build_model(self):
#
#         x = Input(shape=(self.state_size,))
#         # design a neural netwrok model Q(s,a)
#
#
#
#
#
#         ###################################
#         z = Dense(self.action_size, activation="linear")(x)
#
#         model = Model(inputs=x, outputs=z)
#
#         if self.optimizer_model == 'Adam':
#             optimizer = Adam(lr=self.learning_rate, clipnorm=1.)
#         elif self.optimizer_model == 'RMSProp':
#             optimizer = RMSprop(lr=self.learning_rate, clipnorm=1.)
#         else:
#             print('Invalid optimizer!')
#
#         model.compile(loss=huber_loss, optimizer=optimizer)
#
#         if self.test:
#             if not os.path.isfile(self.weight_backup):
#                 print('Error:no file')
#             else:
#                 model.load_weights(self.weight_backup)
#
#         return model

# MINE

    def build_model(self):
        # Input Layer
        x = Input(shape=(self.state_size,))

        attention = Dense(self.state_size, activation="softmax")(x)
        weighted_x = Multiply()([x, attention])
        # First Dense Layer with Batch Normalization and Dropout
        h1 = Dense(128, activation="relu")(weighted_x)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.3)(h1)

        # Second Dense Layer
        h2 = Dense(128, activation="relu")(h1)
        h2 = BatchNormalization()(h2)
        h2 = Dropout(0.3)(h2)

        # Third Dense Layer
        h3 = Dense(64, activation="relu")(h2)

        # Fourth Dense Layer
        h4 = Dense(32, activation="relu")(h3)

        # Output Layer
        z = Dense(self.action_size, activation="linear")(h4)

        # Build the Model
        model = Model(inputs=x, outputs=z)

        # Optimizer Setup
        if self.optimizer_model == 'Adam':
            optimizer = Adam(lr=self.learning_rate, clipnorm=1.)
        elif self.optimizer_model == 'RMSProp':
            optimizer = RMSprop(lr=self.learning_rate, clipnorm=1.)
        else:
            print('Invalid optimizer!')

        # Compile the Model
        model.compile(loss=huber_loss, optimizer=optimizer)

        # Load Pre-Trained Weights (if in test mode)
        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error: no file')
            else:
                model.load_weights(self.weight_backup)

        return model

    # def build_model(self):
    #     # Input layer
    #     x = tf.keras.Input(shape=(self.state_size,), name="input")
    #
    #     # Fully connected layers
    #     h1 = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     h1 = tf.layers.batch_normalization(h1, training=True)
    #     h1 = tf.layers.dropout(h1, rate=0.2, training=True)
    #
    #     h2 = tf.layers.dense(h1, 64, activation=tf.nn.relu)
    #
    #     # Output layer
    #     z = tf.layers.dense(h2, self.action_size, activation=None)
    #
    #     # Define model
    #     model = tf.keras.Model(inputs=x, outputs=z)
    #
    #     # Optimizer setup
    #     if self.optimizer_model == 'Adam':
    #         optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    #     elif self.optimizer_model == 'RMSProp':
    #         optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    #     else:
    #         print('Invalid optimizer!')
    #
    #     # Compile the model
    #     model.compile(loss=huber_loss, optimizer=optimizer)
    #
    #     # Load weights if in test mode
    #     if self.test:
    #         if not os.path.isfile(self.weight_backup):
    #             print('Error: No weight file found!')
    #         else:
    #             model.load_weights(self.weight_backup)
    #
    #     return model

    def predict(self, state, target=False):
        if target:
            return self.model_.predict(state)
        else:
            return self.model.predict(state)

    def predict_one_sample(self, state, target=False):
        return self.predict(state.reshape(1,self.state_size), target=target).flatten()

    def save_model(self):
        self.model.save(self.weight_backup)
        
class Agent(object):
    
    epsilon = MAX_EPSILON
    beta = MIN_BETA

    def __init__(self, state_size, action_size, bee_index, brain_name, arguments):
        self.state_size = state_size
        self.action_size = action_size
        self.bee_index = bee_index
        self.learning_rate = arguments['learning_rate']
        self.gamma = arguments['gamma']
        self.brain = Brain(self.state_size, self.action_size, brain_name, arguments)
        self.memory = UER(arguments['memory_capacity'])
        self.target_type = arguments['target_type']
        self.update_target_frequency = arguments['target_frequency']
        self.max_exploration_step = arguments['maximum_exploration']
        self.batch_size = arguments['batch_size']
        self.step = 0
        self.test = arguments['test']
        if self.test:
            self.epsilon = MIN_EPSILON

    def greedy_actor(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.brain.predict_one_sample(state))

    def find_targets_uer(self, batch):
        batch_len = len(batch)

        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1][self.bee_index]
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                if self.target_type == 'DDQN':
                    t[a] = r + self.gamma * pTarget_[i][np.argmax(p_[i])]
                elif self.target_type == 'DQN':
                    t[a] = r + self.gamma * np.amax(pTarget_[i])
                else:
                    print('Invalid type for target network!')

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y]

    def observe(self, sample):
        self.memory.remember(sample)

    def decay_epsilon(self):
        # slowly decrease Epsilon based on our experience
        self.step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
            self.beta = MAX_BETA
        else:
            if self.step < self.max_exploration_step:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.step)/self.max_exploration_step
                self.beta = MAX_BETA + (MIN_BETA - MAX_BETA) * (self.max_exploration_step - self.step)/self.max_exploration_step
            else:
                self.epsilon = MIN_EPSILON
                
class UER(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def remember(self, sample):
        self.memory.append(sample)

    def sample(self, n, seeds = 1):
        random.seed(seeds)
        n = min(n, len(self.memory))
        sample_batch = random.sample(self.memory, n)

        return sample_batch