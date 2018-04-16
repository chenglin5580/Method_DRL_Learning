"""
DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  2018.01.19
"""

########################### Package  Input  #################################
import numpy as np
import gym
from DDPG_Morvan import ddpg

############################ Object and Method  ####################################
ENV_NAME = 'MountainCarContinuous-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
print("环境状态空间维度为", s_dim)
print('-----------------------------\t')
a_dim = env.action_space.shape[0]
print("环境动作空间维度为", a_dim)
print('-----------------------------\t')
a_bound = np.zeros([2])
a_bound[0] = env.action_space.high
a_bound[1] = 0
print("环境动作空间的上界为", a_bound)
print('-----------------------------\t')

reload_flag = True
ddpg = ddpg(a_dim, s_dim, a_bound, reload_flag)


###############################  training  ####################################
RENDER = False

max_Episodes = 200
max_Step = 200
var = 3
for i in range(max_Episodes):
    state_new = env.reset()
    ep_reward = 0
    for j in range(max_Step):

        env.render()

        # Add exploration noise
        action = ddpg.choose_action(state_new)
        state_next, reward, done, info = env.step(action)

        state_new = state_next
        ep_reward += reward
        if done or j == max_Step-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:
            break