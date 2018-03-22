

import A3C as A3C
import gym

## import env
ENV_NAME = 'MountainCarContinuous-v0'
env = gym.make(ENV_NAME).unwrapped
env.seed(1)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
a_bound = [env.action_space.low, env.action_space.high]

print('state_dim', state_dim, 'action_dim', action_dim, 'a_bound', a_bound)


## train
train_flag = True
# train_flag = False
para = A3C.Para(env,  # 环境参数包括state_dim,action_dim,abound,step,reset
                state_dim=state_dim,  # 状态的维度
                action_dim=action_dim,  # 动作的维度
                a_bound=a_bound,  # 动作的上下界
                units_a=30,  # 双层网络，第一层的大小
                units_c=30,  # 双层网络，critic第一层的大小
                MAX_GLOBAL_EP=2000,  # 全局需要跑多少轮数
                UPDATE_GLOBAL_ITER=10,  # 多少代进行一次学习，调小一些学的比较快
                gamma=0.9,  # 奖励衰减率
                ENTROPY_BETA=0.01,  # 表征探索大小的量，越大结果越不确定
                LR_A=0.001,  # Actor的学习率
                LR_C=0.001,  # Crtic的学习率
                sigma_mul=1,
                MAX_EP_STEP=10000,  # 控制一个回合的最长长度
                train=train_flag  # 表示训练
                )
RL = A3C.A3C(para)
if para.train:
    RL.run()
else:
    # 1 stable
    # 2 random
    # 3 multi
    RL.display(1)
#




