
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import sys


# In[2]:


class QLearning:
    
    def __init__(self, n_actions, n_states, gamma=0.9, epsilon=0.9, lr=0.001):
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        
        # Define Graph
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        # Input
        self.state_input = tf.placeholder(tf.float32, shape=[None, self.n_states],
                                         name='input')
        self.q_target = tf.placeholder(tf.float32, shape=[None, self.n_actions],
                                      name='q_target')
        
        # build network
        with tf.variable_scope('q_table'):
            self.q_eval = self.build_network('net_eval')
            
        # params of network
        self.qnet_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                                  scope='q_table/net_eval')
        
        # loss = (target - q_eval)^2 / N
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        
        # optimizer
        self.train = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, var_list=self.qnet_eval_params)
        
        self.sess.run(tf.global_variables_initializer())
        
    def build_network(self, scope):
        with tf.variable_scope(scope):
            x_h1 = tf.layers.dense(self.state_input, units=5, activation=tf.nn.tanh)
            x_h2 = tf.layers.dense(x_h1, units=5, activation=tf.nn.tanh)
        return tf.layers.dense(x_h2, units=self.n_actions)
    
    def choose_action(self, current_state):
        if np.random.uniform() < self.epsilon:
            q_eval = self.sess.run(self.q_eval, feed_dict={
                                        self.state_input: current_state[np.newaxis, :]})
            self.action = np.argmax(q_eval)
        else:
            self.action = np.random.randint(0, self.n_actions)
        
        return self.action
    
    def learn(self, current_state, reward, next_state):
        q_eval = self.sess.run(self.q_eval, feed_dict={self.state_input: current_state[np.newaxis, :]})
        q_eval_next = self.sess.run(self.q_eval, feed_dict={self.state_input: next_state[np.newaxis, :]})
        q_target = q_eval.copy()
        q_target[:, self.action] = reward + self.gamma * q_eval_next.max()
        _, self.cost = self.sess.run([self.train, self.loss], feed_dict={self.state_input:current_state[np.newaxis, :],
                                                                        self.q_target: q_target})

    def model_save(self, model_name):
        saver = tf.train.Saver()
        saver.save(self.sess, "saved_models/{}.ckpt".format(model_name))
    
    def model_restore(self, model_name):
        saver = tf.train.Saver()
        saver.restore(self.sess, "saved_models/{}.ckpt".format(model_name))

