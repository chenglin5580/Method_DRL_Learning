
import tensorflow as tf
import numpy as np
import sys
###############################  DDPG  ####################################

class ddpg(object):
    def __init__(self, a_dim, s_dim, a_bound, reload_flag=False):

        # DDPG网络参数
        self.method = 'MovFan'
        self.LR_A = 0.001    # learning rate for actor
        self.LR_C = 0.002    # learning rate for critic
        self.GAMMA = 0.9     # reward discount
        self.TAU = 0.01      # soft replacement
        self.MEMORY_CAPACITY = 10000
        self.BATCH_SIZE = 32
        self.pointer = 0
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.iteration=0
        self.modelpath = sys.path[0] + '/data.chkp'


        # DDPG构建
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 + 1), dtype=np.float32)
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.done = tf.placeholder(tf.float32, [None, 1], 'done')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
            # tf.summary.histogram('Actor/eval', self.a)
            # tf.summary.histogram('Actor/target', a_)
            self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
            self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)
            # tf.summary.histogram('Critic/eval', q)
            # tf.summary.histogram('Critic/target', q_)
            self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
            self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')


        # target net replacement
        self.soft_replace = [
            [tf.assign(ta, (1 - self.TAU) * ta + self.TAU * ea), tf.assign(tc, (1 - self.TAU) * tc + self.TAU * ec)]
            for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        # q_target = self.R + self.GAMMA * q_ * (1 - self.done)
        q_target = self.R + self.GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        tf.summary.scalar('td_error', td_error)
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        # tf.summary.scalar('a_loss', a_loss)
        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=self.ae_params)

        self.actor_saver = tf.train.Saver()
        if reload_flag:
            self.actor_saver.restore(self.sess, self.modelpath)
        else:
            self.sess.run(tf.global_variables_initializer())

        # self.merged = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter('./logs/'+self.method+'run'+str(self.LR_A), self.sess.graph)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]


    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
        bs_ = bt[:, self.s_dim + self.a_dim + 1: 2 * self.s_dim + self.a_dim + 1]
        done_ = bt[:, -1:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.done: done_})
        # result_merge = self.sess.run(self.merged, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.done: done_})
        # self.writer.add_summary(result_merge, self.iteration)
        self.iteration += 1

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, r, s_, done))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            net = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[0], name='scaled_a') + self.a_bound[1]


    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            q = tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
            return q

    def net_save(self):
        self.actor_saver.save(self.sess, self.modelpath)

