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
    def __init__(self):
        self.Num_WW = 20
        self.limit = 10
        self.MNum_seeds = 10
        self.maxCycle = 1000
        self.D = 2
        self.ub = np.array([100, 100])
        self.lb = np.array([-100, -100])
        self.limit = 10

        self.seed_num = self.get_seedNum()
        self.WW_Ini()


    def WW_Ini(self):

        self.Waterweeds = np.zeros([self.Num_WW, self.D])
        self.Waterweeds_Fit = np.zeros([self.Num_WW])

        for i in range(self.Num_WW):
            self.Waterweeds[i, :] = np.random.rand(1, self.D) * (self.ub - self.lb) + self.lb
            self.Waterweeds_Fit[i] = self.get_reward(self.Waterweeds[i, :])
        self.Waterweeds_sort = np.argsort(self.Waterweeds_Fit)
        self.Waterweeds_life = np.zeros([self.Num_WW])
        index_best = self.Waterweeds_sort[0]
        self.WW_best = self.Waterweeds[index_best, :].copy()
        self.WW_fit_best = self.Waterweeds_Fit[index_best].copy()


    def get_seedNum(self):
        fFitness = np.zeros([self.Num_WW], dtype=np.int32)
        for kk in range(self.Num_WW):
            fFitness[kk] = self.MNum_seeds - (kk - 1) * (self.MNum_seeds - 1) / (self.Num_WW - 1)
            fFitness[kk] = np.fix(fFitness[kk] + 0.49999999)
        return fFitness

    def get_reward(self, waterweed):
        G1 = 0.5 + ((np.sin((waterweed[0] ** 2 + waterweed[1] ** 2) ** 0.5)) ** 2 - 0.5) \
             / (1 + 0.001 * (waterweed[0] ** 2 + waterweed[1] ** 2)) ** 2
        return G1


    def run(self):

        for iter in range(self.maxCycle):

            self.limit = 10 + iter / self.maxCycle * 30

            for i_step in range(self.Num_WW):

                self.waterweed_update(i_step)

            self.Waterweeds_sort = np.argsort(self.Waterweeds_Fit)
            index_best = self.Waterweeds_sort[0]
            if self.WW_fit_best > self.Waterweeds_Fit[index_best]:
                self.WW_best = self.Waterweeds[index_best, :].copy()
                self.WW_fit_best = self.Waterweeds_Fit[index_best].copy()

            # print(self.Waterweeds_Fit)
            print('iter', iter, 'Globalfit', self.WW_fit_best, 'x_best', self.WW_best)

    def waterweed_update(self, i_step):

        i_ww = self.Waterweeds_sort[i_step]
        mother_WW = self.Waterweeds[i_ww, :]

        seed_fit_strong = None

        for i_seed in range(self.seed_num[i_step]):

            Param2Change = int(np.fix(np.random.rand() * self.D))

            while True:
                neighbour = int(np.fix(np.random.rand() * (self.Num_WW)))
                if not (neighbour == i_ww):
                    break

            father_WW = self.Waterweeds[neighbour, :]
            seed = mother_WW.copy()
            seed[Param2Change] += (mother_WW[Param2Change] - father_WW[Param2Change]) * (
                    np.random.rand() - 0.5) * 2

            if seed[Param2Change] > self.ub[Param2Change]:  seed[Param2Change] = self.ub[Param2Change]
            if seed[Param2Change] < self.lb[Param2Change]:  seed[Param2Change] = self.lb[Param2Change]

            if seed_fit_strong == None:
                seed_strong = seed
                seed_fit_strong = self.get_reward(seed)
            else:
                seed_fit = self.get_reward(seed)
                if seed_fit < seed_fit_strong:
                    seed_strong = seed
                    seed_fit_strong = seed_fit

        # 与母水草展开竞争
        if self.Waterweeds_life[i_ww] < self.limit:
            if self.Waterweeds_Fit[i_ww] > seed_fit_strong:
                self.Waterweeds[i_ww, :] = seed_strong
                self.Waterweeds_Fit[i_ww] = seed_fit_strong
                self.Waterweeds_life[i_ww] = 0
            else:
                self.Waterweeds_life[i_ww] += 1

        else:
            self.Waterweeds[i_ww, :] = seed_strong
            self.Waterweeds_Fit[i_ww] = seed_fit_strong
            self.Waterweeds_life[i_ww] = 0



if __name__ == "__main__":

    CONFIG = [
        dict(game="CartPole-v0",
             n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
        dict(game="MountainCar-v0",
             n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
        dict(game="Pendulum-v0",
             n_feature=3, n_action=1, continuous_a=[True, 2.], a_bound=[-2., 2.], ep_max_step=200, eval_threshold=-180)
    ][2]  # choose your game

    env = gym.make(CONFIG['game']).unwrapped

    RLmethod = WW()

    ## train
    train_flag = True
    # train_flag = False
    if train_flag:
        RLmethod.run()
    else:
        RLmethod.display(CONFIG=CONFIG)



