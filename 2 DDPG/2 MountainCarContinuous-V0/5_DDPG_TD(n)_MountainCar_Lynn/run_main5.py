
"""
DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  2018.01.19
"""

########################### Package  Input  #################################
import numpy as np
import gym
from DDPG_Morvan import ddpg
import matplotlib.pyplot as plt


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

reload_flag = False
ddpg = ddpg(a_dim, s_dim, a_bound)


###############################  training  ####################################
RENDER = False

max_Episodes = 300
Learning_Start = False
var = 5  # control exploration
step_me = np.zeros([max_Episodes])
reward_me = np.zeros([max_Episodes])
for i in range(max_Episodes):
    state_now = env.reset()
    ep_reward = 0
    j = 0
    state_now_sequence = np.empty((0, 2))
    action_sequence = np.empty((0, 1))
    reward_sequence = np.empty((0, 1))
    state_next_sequence = np.empty((0, 2))
    done_sequence = np.empty((0, 1))
    while True:
        if RENDER:
            env.render()

        # Add exploration noise
        action = ddpg.choose_action(state_now)
        action = np.clip(np.random.normal(action, var), -1, 1)    # add randomness to action selection for exploration
        state_next, reward, done, info = env.step(action)

        state_now_sequence = np.vstack((state_now_sequence, state_now))
        action_sequence = np.vstack((action_sequence, action))
        reward_sequence = np.vstack((reward_sequence, reward))
        state_next_sequence = np.vstack((state_next_sequence, state_next))
        done_sequence = np.vstack((done_sequence, done))

        if Learning_Start:
            ddpg.learn()
            var *= .99994  # decay the action randomness
            RENDER = True
        else:
            if ddpg.pointer > ddpg.MEMORY_CAPACITY:
                Learning_Start = True

        state_now = state_next
        ep_reward += reward
        j += 1
        if done:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'step', j,  'Explore: %.2f' % var, )
            # if ep_reward > -300:
            break
    TD_n = 20
    for kk in range(len(state_now_sequence[:, 0])):
        if kk + TD_n - 1 < len(state_now_sequence[:, 0]) - 1:
            state_now = state_now_sequence[kk, :]
            action = action_sequence[kk, 0]
            reward = np.sum(reward_sequence[kk: kk + TD_n, 0])
            state_next = state_next_sequence[kk + TD_n - 1, :]
            done = False
        else:
            state_now = state_now_sequence[kk, :]
            action = action_sequence[kk, 0]
            reward = np.sum(reward_sequence[kk:, 0])
            state_next = state_next_sequence[-1, :]
            done = True
        ddpg.store_transition(state_now, action, reward, state_next, np.array([done * 1.0]))

    step_me[i] = j
    reward_me[i] = ep_reward
    if var < 0.1:
        break

ddpg.net_save()

plt.figure(1)
plt.plot(step_me)
plt.savefig("step_me.png")

plt.figure(2)
plt.plot(reward_me)
plt.savefig("reward_me.png")
plt.show()