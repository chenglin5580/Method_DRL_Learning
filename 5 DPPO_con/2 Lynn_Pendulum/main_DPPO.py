

from DPPO import Para
from DPPO import DPPO
import gym
import matplotlib.pyplot as plt
import numpy as np

## import env
env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
a_bound = [env.action_space.low, env.action_space.high]

print('state_dim', state_dim, 'action_dim', action_dim, 'a_bound', a_bound)


## train
train_flag = True
train_flag = False
para = Para(EP_MAX=1000,
            EP_LEN=200,
            N_WORKER=8,  # parallel workers
            GAMMA=0.9, # reward discount factor
            A_LR=0.0001,  # learning rate for actor
            C_LR=0.0002,  # learning rate for critic
            MIN_BATCH_SIZE=64,  # minimum batch size for updating PPO
            UPDATE_STEP=10,  # loop update operation n-steps
            EPSILON=0.2,  # for clipping surrogate objective
            units_a=200,
            units_c=100,
            S_DIM=3,
            A_DIM=1,  # state and action dimension)
            tensorboard=True,
            train=train_flag  # 训练的时候有探索
            )
RLmethod = DPPO(para)
if para.train:
    RLmethod.run()
else:
    # plot reward change and test
    env = gym.make('Pendulum-v0')
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            action = RLmethod.GLOBAL_PPO.choose_action(s)
            s_, r, done, _ = env.step(action*2)
            s = s_



#




