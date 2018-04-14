"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]
Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.
The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.3
gym 0.9.2
"""

import tensorflow as tf
import numpy as np

import gym, threading, queue
import os
import sys
import matplotlib.pyplot as plt


class Para(object):
    def __init__(self,
                 EP_MAX=1000,
                 EP_LEN=200,
                 N_WORKER=8,  # parallel workers
                 GAMMA=0.9, # reward discount factor
                 A_LR=0.0001,  # learning rate for actor
                 C_LR=0.0002,  # learning rate for critic
                 MIN_BATCH_SIZE=64,  # minimum batch size for updating PPO
                 UPDATE_STEP=10,  # loop update operation n-steps
                 EPSILON=0.2,  # for clipping surrogate objective
                 GAME='Pendulum-v0',
                 S_DIM=3,
                 A_DIM=1,  # state and action dimension
                 tensorboard=True,
                 train=True,
                ):
        self.EP_MAX = EP_MAX
        self.EP_LEN = EP_LEN
        self.N_WORKER = N_WORKER  # parallel workers
        self.GAMMA = GAMMA  # reward discount factor
        self.A_LR = A_LR  # learning rate for actor
        self.C_LR = C_LR  # learning rate for critic
        self.MIN_BATCH_SIZE = MIN_BATCH_SIZE  # minimum batch size for updating PPO
        self.UPDATE_STEP = UPDATE_STEP  # loop update operation n-steps
        self.EPSILON = EPSILON  # for clipping surrogate objective
        self.GAME = GAME
        self.S_DIM = S_DIM
        self.A_DIM = A_DIM  # state and action dimension
        self.train = train
        self.tensorboard = tensorboard

        self.GLOBAL_UPDATE_COUNTER = 0
        self.GLOBAL_EP = 0
        self.GLOBAL_RUNNING_R = []
        self.COORD = tf.train.Coordinator()
        self.QUEUE = queue.Queue()  # workers putting data in this queue
        self.UPDATE_EVENT, self.ROLLING_EVENT = threading.Event(), threading.Event()

        self.model_path0 = os.path.join(sys.path[0], 'DPPO_Net')
        if not os.path.exists(self.model_path0):
            os.mkdir(self.model_path0)
        self.model_path = os.path.join(self.model_path0, 'data.chkp')


class DPPO(object):
    def __init__(self, para):
        self.para = para
        self.GLOBAL_PPO = ACnet(self.para)
        self.para.UPDATE_EVENT.clear()  # not update now
        self.para.ROLLING_EVENT.set()  # start to roll out
        self.workers = [Worker(wid=i, para=self.para, GLOBAL_PPO=self.GLOBAL_PPO) for i in range(self.para.N_WORKER)]

        # COORD = tf.train.Coordinator()
        # self.QUEUE = queue.Queue()  # workers putting data in this queue

    def run(self):
        threads = []
        for worker in self.workers:  # worker threads
            t = threading.Thread(target=worker.work, args=())
            t.start()  # training
            threads.append(t)
        # add a PPO updating thread
        threads.append(threading.Thread(target=self.GLOBAL_PPO.update, ))
        threads[-1].start()
        self.para.COORD.join(threads)

        self.GLOBAL_PPO.net_save()

        plt.plot(np.arange(len(self.para.GLOBAL_RUNNING_R)), self.para.GLOBAL_RUNNING_R)
        plt.xlabel('Episode')
        plt.ylabel('Moving reward')
        plt.ion()
        plt.show()




class ACnet(object):
    def __init__(self, para):
        self.para = para
        self.sess = tf.Session()
        with tf.variable_scope('placeholder'):
            self.tfs = tf.placeholder(tf.float32, [None, self.para.S_DIM], 'state')
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.tfa = tf.placeholder(tf.float32, [None, self.para.A_DIM], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # critic definition, loss, and train
        with tf.variable_scope('critic'):
            self.v = self._build_vnet(self.tfs, 'net', trainable=True)
        with tf.variable_scope('c_loss'):
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
        with tf.variable_scope('c_train'):
            self.ctrain_op = tf.train.AdamOptimizer(self.para.C_LR).minimize(self.closs)

        # actor definition, loss, and train
        with tf.variable_scope('actor'): # actor definition
            pi, pi_params, self.mu, sigma = self._build_anet('actor', 'pi', trainable=True)
            oldpi, oldpi_params, oldmu, oldsigma = self._build_anet('actor', 'oldpi', trainable=False)
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
        with tf.variable_scope('a_loss'):    # actor loss
            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
            surr = ratio * self.tfadv  # surrogate loss
            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                surr,
                tf.clip_by_value(ratio, 1. - self.para.EPSILON, 1. + self.para.EPSILON) * self.tfadv))
        with tf.variable_scope('a_train'):  # actor train
            self.atrain_op = tf.train.AdamOptimizer(self.para.A_LR).minimize(self.aloss)

        # net display and save
        with tf.variable_scope('save'):
            self.actor_saver = tf.train.Saver()
            if self.para.train:
                self.sess.run(tf.global_variables_initializer())
            else:
                self.actor_saver.restore(self.sess, self.para.model_path)

            if self.para.tensorboard:
                tf.summary.FileWriter("DPPO_log/", self.sess.graph)


    def update(self):
        # global GLOBAL_UPDATE_COUNTER
        while not self.para.COORD.should_stop():
            if self.para.GLOBAL_EP < self.para.EP_MAX:
                self.para.UPDATE_EVENT.wait()  # wait until get batch of data
                self.sess.run(self.update_oldpi_op)  # copy pi to old pi
                data = [self.para.QUEUE.get() for _ in range(self.para.QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :self.para.S_DIM], data[:, self.para.S_DIM: self.para.S_DIM + self.para.A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.para.UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(self.para.UPDATE_STEP)]
                self.para.UPDATE_EVENT.clear()  # updating finished
                self.para.GLOBAL_UPDATE_COUNTER = 0  # reset counter
                self.para.ROLLING_EVENT.set()  # set roll-out available

    def _build_anet(self, scope, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.para.A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.para.A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope+'/'+name)
        return norm_dist, params, mu, sigma

    def _build_vnet(self, s, scope, trainable):
        # 建立critic网络
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 100, tf.nn.relu, trainable=trainable)
            v = tf.layers.dense(l1, 1)
        return v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        if self.para.train:
            a = self.sess.run(self.sample_op, {self.tfs: s})[0]
            return np.clip(a, -2, 2)
        else:
            a = self.sess.run(self.mu, {self.tfs: s})[0]
            return np.clip(a, -2, 2)

    def net_save(self):
        self.actor_saver.save(self.sess, self.para.model_path)


    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

#
class Worker(object):
    def __init__(self, wid, para, GLOBAL_PPO):
        self.wid = wid
        self.para = para
        self.env = gym.make(self.para.GAME).unwrapped
        self.ppo = GLOBAL_PPO
        # self.QUEUE = queue.Queue()  # workers putting data in this queue

    def work(self):
        # global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not self.para.COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(self.para.EP_LEN):
                if not self.para.ROLLING_EVENT.is_set():  # while global PPO is updating
                    self.para.ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
                s = s_
                ep_r += r

                self.para.GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == self.para.EP_LEN - 1 or self.para.GLOBAL_UPDATE_COUNTER >= self.para.MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.para.GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.para.QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue
                    if self.para.GLOBAL_UPDATE_COUNTER >= self.para.MIN_BATCH_SIZE:
                        self.para.ROLLING_EVENT.clear()  # stop collecting data
                        self.para.UPDATE_EVENT.set()  # globalPPO update

                    if self.para.GLOBAL_EP >= self.para.EP_MAX:  # stop training
                        self.para.COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(self.para.GLOBAL_RUNNING_R) == 0:
                self.para.GLOBAL_RUNNING_R.append(ep_r)
            else:
                self.para.GLOBAL_RUNNING_R.append(self.para.GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            self.para.GLOBAL_EP += 1
            print('{0:.1f}%'.format(self.para.GLOBAL_EP / self.para.EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )

