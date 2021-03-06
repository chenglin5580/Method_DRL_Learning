"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import os
import sys

class PPO(object):

    def __init__(self,
                 method,
                 LR_A=0.0001,
                 LR_C=0.0002,
                 A_UPDATE_STEPS=10,
                 C_UPDATE_STEPS=10,
                 ob_dim=3,
                 a_dim=1,
                 tensorboard = True,
                 train=True  # 训练的时候有探索
                 ):
        self.method = method
        self.LR_A = LR_A
        self.LR_C = LR_C
        self.A_UPDATE_STEPS = A_UPDATE_STEPS
        self.C_UPDATE_STEPS = C_UPDATE_STEPS
        self.ob_dim = ob_dim
        self.a_dim = a_dim
        self.train = train
        self.tensorboard = tensorboard

        self.model_path0 = os.path.join(sys.path[0], 'PPO_Net')
        if not os.path.exists(self.model_path0):
            os.mkdir(self.model_path0)
        self.model_path = os.path.join(self.model_path0, 'data.chkp')

        # placeholder
        with tf.variable_scope('placeholder'):
            self.sess = tf.Session()
            self.s_in = tf.placeholder(tf.float32, [None, self.ob_dim], 'state')
            self.v_in = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.a_in = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
            self.adv_in = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # critic
        with tf.variable_scope('critic'):
            self.v = self._build_vnet(self.s_in, 'critic/net', trainable=True)

        with tf.variable_scope('ctrain'):
            self.adv = self.v_in - self.v
            self.closs = tf.reduce_mean(tf.square(self.adv))
            self.ctrain_op = tf.train.AdamOptimizer(self.LR_C).minimize(self.closs)

        # actor
        pi, pi_params, mu, sigma = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params, _, _ = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('choose_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
            self.mu = mu
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]


        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.a_in) / oldpi.prob(self.a_in)
                surr = ratio * self.adv_in
            if self.method['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-self.method['epsilon'], 1.+self.method['epsilon'])*self.adv_in))


        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.LR_A).minimize(self.aloss)

        if self.tensorboard:
            tf.summary.FileWriter("PPO_log/", self.sess.graph)


        self.actor_saver = tf.train.Saver()
        if self.train:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.actor_saver.restore(self.sess, self.model_path)


    def update(self, s, a, r):

        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.adv, {self.s_in: s, self.v_in: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if self.method['name'] == 'kl_pen':
            for _ in range(self.A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.s_in: s, self.a_in: a, self.adv_in: adv, self.tflam: self.method['lam']})
                if kl > 4*self.method['kl_target']:  # this in in google's paper
                    break
            if kl < self.method['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                self.method['lam'] /= 2
            elif kl > self.method['kl_target'] * 1.5:
                self.method['lam'] *= 2
            self.method['lam'] = np.clip(self.method['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.s_in: s, self.a_in: a, self.adv_in: adv})
             for _ in range(self.A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.s_in: s, self.v_in: r}) for _ in range(self.C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.s_in, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params, mu, sigma

    def _build_vnet(self, s, scope, trainable):
        # 建立critic网络
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 100, tf.nn.relu, trainable=trainable)
            v = tf.layers.dense(l1, 1)
        return v

    def choose_action(self, s):
        if self.train:
            s = s[np.newaxis, :]
            a = self.sess.run(self.sample_op, {self.s_in: s})[0]
            return np.clip(a, -2, 2)
        else:
            s = s[np.newaxis, :]
            a = self.sess.run(self.mu, {self.s_in: s})[0]
            return a


    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.s_in: s})[0, 0]

    def net_save(self):
        self.actor_saver.save(self.sess, self.model_path)

