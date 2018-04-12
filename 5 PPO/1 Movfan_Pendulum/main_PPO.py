

import numpy as np
import matplotlib.pyplot as plt
import gym
from PPO import PPO

#
EP_MAX = 1000
EP_LEN = 200
BATCH = 32
GAMMA = 0.9
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


env = gym.make('Pendulum-v0').unwrapped

train_flag = True
train_flag = False
RLmethod = PPO(
    method=METHOD,
    LR_A=0.0001,
    LR_C=0.0002,
    A_UPDATE_STEPS=10,
    C_UPDATE_STEPS=10,
    ob_dim=3,
    a_dim=1,
    train=train_flag  # 训练的时候有探索
    )

if RLmethod.train:
    all_ep_r = []
    for ep in range(EP_MAX):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):    # in one episode
            # env.render()
            a = RLmethod.choose_action(s)
            s_, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r+8)/8)    # normalize reward, find to be useful
            s = s_
            ep_r += r

            # update ppo
            if (t+1) % BATCH == 0 or t == EP_LEN-1:
                v_s_ = RLmethod.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                RLmethod.update(bs, ba, br)
        if ep == 0: all_ep_r.append(ep_r)
        else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )

    RLmethod.net_save()
    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()

else:
    all_ep_r = []
    for ep in range(EP_MAX):
        s = env.reset()
        ep_r = 0
        for t in range(EP_LEN):  # in one episode
            env.render()
            a = RLmethod.choose_action(s)
            s_, r, done, _ = env.step(a)
            s = s_
            ep_r += r
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )
