
"""
DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  17.08.2017
"""

# TODO DRL 有机会尝试

########################### Package  Input  #################################

from A2C import A2C as Method
from QUAD import QUAD as Objective_AI
import numpy as np
import matplotlib.pyplot as plt

############################ Hyper Parameters #################################

max_Episodes = 5000
max_Ep_Steps = 2000
rendering = False
############################ Object and Method  ####################################

env = Objective_AI(random=False, c1=10, c2=0)

ob_dim = env.ob_dim
print("环境状态空间维度为", ob_dim)
print('-----------------------------\t')
a_dim = env.action_dim
print("环境动作空间维度为", a_dim)
print('-----------------------------\t')


## method settting
# tensorboard --logdir="2 QUAD/1 DRL/3 A2C/logs"
method = '0initial/units 200 200/penalty 10+4'

train_flag = True
train_flag = False
RLmethod = Method(
            method,
            env.action_dim,  # 动作的维度
            env.ob_dim,  # 状态的维度
            LR_A=0.00001,  # Actor的学习率
            LR_C=0.001,  # Critic的学习率
            GAMMA=1,  # 衰减系数
            ENTROPY_BETA=0.01,  # 表征探索大小的量，越大结果越不确定
            MEMORY_SIZE=2000,  # 记忆池容量
            BATCH_SIZE=200,  # 批次数量
            units_a=200,  # Actor神经网络单元数
            units_c=200,  # Crtic神经网络单元数
            tensorboard=True,  # 是否存储tensorboard
            train=train_flag  # 训练的时候有探索
            )

###############################  training  ####################################


if RLmethod.train:
    for i in range(max_Episodes):
        observation = env.reset()
        ep_reward = 0

        observation_Now_Seq = np.empty((0, 5))
        action_Seq = np.empty((0, 2))
        reward_Seq = np.empty((0, 1))
        observation_Next_Seq = np.empty((0, 5))
        done_Seq = np.empty((0, 1))

        for j in range(max_Ep_Steps):
            action = RLmethod.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # RLmethod.store_transition(observation, action, reward, observation_, np.array([done * 1.0]))

            observation_Now_Seq = np.vstack((observation_Now_Seq, observation))
            action_Seq = np.vstack((action_Seq, action))
            reward_Seq = np.vstack((reward_Seq, reward))
            observation_Next_Seq = np.vstack((observation_Next_Seq, observation_))
            done_Seq = np.vstack((done_Seq, done))

            RLmethod.learn()
            observation = observation_
            ep_reward += reward


            if done:
                print('step', i, "x_f: %.2f" % env.state[0], "z_f: %.2f" % env.state[1], 'reward: %.2f' % ep_reward, 'total_time %.2f' % env.t)
                break

        TD_n = 10
        for kk in range(len(observation_Now_Seq[:, 0])):
            if kk + TD_n - 1 < len(observation_Now_Seq[:, 0]) - 1:
                observation = observation_Now_Seq[kk, :]
                action = action_Seq[kk, :]
                reward = np.sum(reward_Seq[kk: kk + TD_n, 0])
                observation_ = observation_Next_Seq[kk + TD_n - 1, :]
                done = False
            else:
                observation = observation_Now_Seq[kk, :]
                action = action_Seq[kk, :]
                reward = np.sum(reward_Seq[kk:, 0])
                observation_ = observation_Next_Seq[-1, :]
                done = True
            RLmethod.store_transition(observation, action, reward, observation_, np.array([done * 1.0]))

    RLmethod.net_save()

else:
    # test the critic
    plt.ion()
    plt.grid(color='g', linewidth='0.3', linestyle='--')

    # state_now = env.reset_stable()
    observation = env.reset()
    ep_reward = 0

    for j in range(max_Ep_Steps):

        # Add exploration noise
        action = RLmethod.choose_action(observation)
        observation_, reward, done, info = env.step(action)

        plt.scatter(env.state[0], env.state[1], color='b')
        plt.pause(0.001)

        observation = observation_
        ep_reward += reward

        if done:
            print('time: %.2f' % env.t, 'Reward: %.5f' % ep_reward, "x_f: %.2f" % env.state[0], "z_f: %.2f" % env.state[1],
                  "vx_f: %.2f" % env.state[2],  "vz_f: %.2f" %env.state[3])

            plt.scatter(env.state[0], env.state[1], color='b')
            plt.pause(100000000)
            break











