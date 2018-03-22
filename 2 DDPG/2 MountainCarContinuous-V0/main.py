"""

DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  17.08.2017

"""

########################### Package  Input  #################################
from DDPG import DDPG
import gym
import numpy as np
import matplotlib.pyplot as plt

############################ Hyper Parameters #################################

max_Episodes = 300
# max_Ep_Steps = 20000
############################ Object and Method  ####################################

## import env
ENV_NAME = 'MountainCarContinuous-v0'
env = gym.make(ENV_NAME).unwrapped
env.seed(1)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
a_bound = [env.action_space.low, env.action_space.high]

print('state_dim', state_dim, 'action_dim', action_dim, 'a_bound', a_bound)



## train
RLmethod = DDPG(
            action_dim,  # 动作的维度
            state_dim,  # 状态的维度
            a_bound,  # 动作的上下限
            var_end=0.1,  # 最后的探索值 0.1倍幅值
            var_liner_times=1000*50,  # 探索值经历多少次学习变成e_end
            var_init=5,  # 表示1倍的幅值作为初始值
            LR_A=0.001,  # Actor的学习率
            LR_C=0.002,  # Critic的学习率
            GAMMA=0.9,  # 衰减系数
            TAU=0.01,  # 软替代率，例如0.01表示学习eval网络0.01的值，和原网络0.99的值
            MEMORY_SIZE=10000,  # 记忆池容量
            BATCH_SIZE=32,  # 批次数量
            units_a=30,  # Actor神经网络单元数
            units_c=30,  # Crtic神经网络单元数
            train=True,  # 训练的时候有探索
            # train=False  # display
            )

###############################  training  ####################################

if RLmethod.train:
    for i in range(max_Episodes):
        observation = env.reset()
        ep_reward = 0

        ob_sequence = np.empty((0, 2))
        action_sequence = np.empty((0, 1))
        reward_sequence = np.empty((0, 1))
        state_next_sequence = np.empty((0, 2))
        done_sequence = np.empty((0, 1))

        # for j in range(max_Ep_Steps):
        while True:

            if RLmethod.pointer > RLmethod.MEMORY_CAPACITY:
                env.render()

            # Add exploration noise
            action = RLmethod.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            ob_sequence = np.vstack((ob_sequence, observation))
            action_sequence = np.vstack((action_sequence, action))
            reward_sequence = np.vstack((reward_sequence, reward))
            state_next_sequence = np.vstack((state_next_sequence, observation_))
            done_sequence = np.vstack((done_sequence, done))

            RLmethod.learn()

            observation = observation_
            ep_reward += reward

            if done:
                # print('action', action_sequence[::50])
                print('Episode:', i, ' Reward: %.5f' % ep_reward, 'Explore: %.5f' % RLmethod.var)
                break

        TD_n = 20
        for kk in range(len(ob_sequence[:, 0])):
            if kk + TD_n - 1 < len(ob_sequence[:, 0]) - 1:
                observation = ob_sequence[kk, :]
                action = action_sequence[kk, 0]
                reward = np.sum(reward_sequence[kk: kk + TD_n, 0])
                observation_ = state_next_sequence[kk + TD_n - 1, :]
                done = False
            else:
                observation = ob_sequence[kk, :]
                action = action_sequence[kk, 0]
                reward = np.sum(reward_sequence[kk:, 0])
                observation_ = state_next_sequence[-1, :]
                done = True
            RLmethod.store_transition(observation, action, reward, observation_, np.array([done * 1.0]))

        # if (var < 0.05) and (ep_reward > - 0.4):
        if (RLmethod.var <= RLmethod.var_end):
            RLmethod.net_save()
            break
else:
    # reset
    observation = env.reset()

    ob_profile = np.empty((0, 4))
    time_profile = np.empty(0)
    action_profile = np.empty(0)

    while True:

        action =RLmethod.choose_action(observation)
        print(action)

        observation_, reward, done, info = env.step(action)

        # memorize the profile
        ob_profile = np.vstack((ob_profile, observation))
        time_profile = np.hstack((time_profile, env.t))
        action_profile = np.hstack((action_profile, action))

        observation = observation_

        if done:
            break

    print('转移轨道时间%d天' % env.t)
    print(env.state)
    plt.figure(1)
    plt.subplot(111, polar=True)
    theta = np.arange(0, 2 * np.pi, 0.02)
    plt.plot(theta, 1 * np.ones_like(theta), 'm')
    plt.plot(theta, 1.524 * np.ones_like(theta), 'b')
    plt.plot(ob_profile[:, 1], ob_profile[:, 0], 'r')

    plt.figure(2)
    plt.plot(time_profile, action_profile)
    # print(action_profile)

    plt.show()








