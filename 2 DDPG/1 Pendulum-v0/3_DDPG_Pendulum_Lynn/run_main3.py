
"""
DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  2018.01.19
"""

########################### Package  Input  #################################
import numpy as np
import gym
from DDPG_Morvan import ddpg

############################ Object and Method  ####################################
ENV_NAME = 'Pendulum-v0'
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

reload_flag = False
ddpg = ddpg(a_dim, s_dim, a_bound)


###############################  training  ####################################
RENDER = False

max_Episodes = 100
max_Step = 200

var = 3  # control exploration
for i in range(max_Episodes):
    state_new = env.reset()
    ep_reward = 0
    for j in range(max_Step):
        if RENDER:
            env.render()

        # Add exploration noise
        action = ddpg.choose_action(state_new)
        action = np.clip(np.random.normal(action, var), -2, 2)    # add randomness to action selection for exploration
        state_next, reward, done, info = env.step(action)

        ddpg.store_transition(state_new, action, reward / 10, state_next, np.array([done * 1.0]))

        if ddpg.pointer > ddpg.MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()
            RENDER = True

        state_new = state_next
        ep_reward += reward
        if j == max_Step-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:
            break
ddpg.net_save()