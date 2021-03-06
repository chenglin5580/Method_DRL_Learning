"""
Simple code for Distributed ES proposed by OpenAI.
Based on this paper: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
Details can be found in : https://arxiv.org/abs/1703.03864
Visit more on my tutorial site: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import gym
import multiprocessing as mp
import time
import copy

class WW(object):
    def __init__(self, CONFIG, Num_WW=20, limit=10, MNum_seeds=10 ):

        self.CONFIG = CONFIG
        self.Num_WW = Num_WW
        self.limit = limit
        self.MNum_seeds = MNum_seeds
        self.maxCycle = 1000
        self.ub = np.array([100, 100])
        self.lb = np.array([-100, -100])


        self.seed_num = self.get_seedNum()
        self.WW_Ini()


    def WW_Ini(self):

        self.Waterweeds = []
        self.Waterweeds_Fit = np.zeros([self.Num_WW])

        for i in range(self.Num_WW):
            # self.Waterweeds[i, :] = np.random.rand(1, self.D) * (self.ub - self.lb) + self.lb
            net_shapes, waterweed = self.build_net()
            self.Waterweeds.append(waterweed)
            self.Waterweeds_Fit[i] = self.get_reward(net_shapes, waterweed, env, CONFIG['ep_max_step'], CONFIG['continuous_a'])
        self.D = len(waterweed)
        self.net_shapes = net_shapes
        self.Waterweeds_sort = np.argsort(self.Waterweeds_Fit)
        self.Waterweeds_life = np.zeros([self.Num_WW])
        index_best = self.Waterweeds_sort[0]
        self.WW_best = self.Waterweeds[index_best].copy()
        self.WW_fit_best = self.Waterweeds_Fit[index_best].copy()

    def build_net(self):

        def linear(n_in, n_out):  # network linear layer
            w = np.random.randn(n_in * n_out).astype(np.float32) * .1
            b = np.random.randn(n_out).astype(np.float32) * .1
            return (n_in, n_out), np.concatenate((w, b))

        s0, p0 = linear(self.CONFIG['n_feature'], 30)
        s1, p1 = linear(30, 20)
        s2, p2 = linear(20, self.CONFIG['n_action'])
        return [s0, s1, s2], np.concatenate((p0, p1, p2))


    def get_seedNum(self):
        fFitness = np.zeros([self.Num_WW], dtype=np.int32)
        for kk in range(self.Num_WW):
            fFitness[kk] = self.MNum_seeds - (kk - 1) * (self.MNum_seeds - 1) / (self.Num_WW - 1)
            fFitness[kk] = np.fix(fFitness[kk] + 0.49999999)
        return fFitness


    def get_reward(self, shapes, params, env, ep_max_step, continuous_a):
        # perturb parameters using seed
        p = self.params_reshape(shapes, params)
        # run episode
        s = env.reset()
        ep_r = 0.
        for step in range(ep_max_step):
            a = self.get_action(p, s, continuous_a)
            s, r, done, _ = env.step(a)
            # mountain car's reward can be tricky
            if env.spec._env_name == 'MountainCar' and s[0] > -0.1: r = 0.
            ep_r += r
            if done: break
        return -ep_r

    def get_action(self, params, x, continuous_a):
        x = x[np.newaxis, :]
        x = np.tanh(x.dot(params[0]) + params[1])
        x = np.tanh(x.dot(params[2]) + params[3])
        x = x.dot(params[4]) + params[5]
        if not continuous_a[0]:
            return np.argmax(x, axis=1)[0]  # for discrete action
        else:
            return (self.CONFIG['a_bound'][1]-self.CONFIG['a_bound'][0])/2 * np.tanh(x)[0]+np.mean(self.CONFIG['a_bound'])
            # return continuous_a[1] * np.tanh(x)[0]  # for continuous action

    def params_reshape(self, shapes, params):  # reshape to be a matrix
        p, start = [], 0
        for i, shape in enumerate(shapes):  # flat params to matrix
            n_w, n_b = shape[0] * shape[1], shape[1]
            p = p + [params[start: start + n_w].reshape(shape),
                     params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
            start += n_w + n_b
        return p


    def run(self):

        for iter in range(self.maxCycle):

            # self.limit = 10 + iter / self.maxCycle * 30

            for i_step in range(self.Num_WW):

                self.waterweed_update(i_step)

            self.Waterweeds_sort = np.argsort(self.Waterweeds_Fit)
            index_best = self.Waterweeds_sort[0]
            if self.WW_fit_best > self.Waterweeds_Fit[index_best]:
                self.WW_best = self.Waterweeds[index_best].copy()
                self.WW_fit_best = self.Waterweeds_Fit[index_best].copy()

            # print(self.Waterweeds_Fit)
            # print('iter', iter, 'Globalfit', self.WW_fit_best, 'x_best', self.WW_best)
            print('iter', iter, 'Globalfit', self.WW_fit_best)

        p = self.params_reshape(self.net_shapes, self.WW_best)
        np.save("WW_Net/model.npy", p)


    def waterweed_update(self, i_step):

        i_ww = self.Waterweeds_sort[i_step]
        mother_WW = self.Waterweeds[i_ww]

        seed_fit_strong = None

        for i_seed in range(self.seed_num[i_step]):

            while True:
                neighbour = int(np.fix(np.random.rand() * (self.Num_WW)))
                if not (neighbour == i_ww):
                    break
            father_WW = self.Waterweeds[neighbour]

            seed = mother_WW.copy()

            for Param2Change in range(self.D):
                if np.random.rand()<0.1:
                    seed[Param2Change] += (father_WW[Param2Change] - mother_WW[Param2Change]) * ((
                                np.random.rand() - 0.5) * 2)

            if seed_fit_strong == None:
                seed_strong = seed
                seed_fit_strong = self.get_reward(self.net_shapes, seed, env, CONFIG['ep_max_step'], CONFIG['continuous_a'])
            else:
                seed_fit = self.get_reward(self.net_shapes, seed, env, CONFIG['ep_max_step'], CONFIG['continuous_a'])
                if seed_fit < seed_fit_strong:
                    seed_strong = seed
                    seed_fit_strong = seed_fit

        # 与母水草展开竞争
        if self.Waterweeds_life[i_ww] < self.limit:
            if self.Waterweeds_Fit[i_ww] > seed_fit_strong:
                self.Waterweeds[i_ww] = seed_strong
                self.Waterweeds_Fit[i_ww] = seed_fit_strong
                self.Waterweeds_life[i_ww] = 0
            else:
                self.Waterweeds_life[i_ww] += 1

        else:
            self.Waterweeds[i_ww] = seed_strong
            self.Waterweeds_Fit[i_ww] = seed_fit_strong
            self.Waterweeds_life[i_ww] = 0

    def display(self, CONFIG):
        # testnet/
        print("\nTESTING....")
        p = np.load("WW_Net/model.npy")
        while True:
            s = env.reset()
            for _ in range(CONFIG['ep_max_step']):
                env.render()
                a = self.get_action(p, s, CONFIG['continuous_a'])
                s, _, done, _ = env.step(a)
                if done: break



if __name__ == "__main__":

    CONFIG = [
        dict(game="CartPole-v0",
             n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
        dict(game="MountainCar-v0",
             n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
        dict(game="Pendulum-v0",
             n_feature=3, n_action=1, continuous_a=[True, 2.], a_bound=[-2., 2.], ep_max_step=200, eval_threshold=-180)
    ][1]  # choose your game

    env = gym.make(CONFIG['game']).unwrapped

    RLmethod = WW(CONFIG=CONFIG,
                  Num_WW=20,
                  limit=10,
                  MNum_seeds=10)

    ## train
    train_flag = True
    # train_flag = False
    if train_flag:
        RLmethod.run()
    else:
        RLmethod.display(CONFIG=CONFIG)



