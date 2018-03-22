

import matplotlib.pyplot as plt
import numpy as np

def display(A3C, display_flag):

    if display_flag == 1:

        # reset
        observation = A3C.para.env.reset()
        reward_t = 0

        for ep_t in range(A3C.para.MAX_EP_STEP):  # MAX_EP_STEP每个片段的最大个数
            A3C.para.env.render()
            action = A3C.GLOBAL_AC.choose_best(observation)
            print(action)


            observation_, reward, done, info = A3C.para.env.step(action)
            print(reward)

            reward_t += reward

            observation = observation_

            if done:
                break

        print(reward_t)





    elif  display_flag == 2:
        state_now = A3C.para.env.reset_random()
        state_start = state_now

        state_track = []
        action_track = []
        time_track = []
        reward_track = []
        reward_me = 0
        while True:

            omega = A3C.GLOBAL_AC.choose_action(state_now)
            print(omega)
            state_next, reward, done, info = A3C.para.env.step(omega)

            state_track.append(state_next.copy())
            action_track.append(info['action'])
            time_track.append(info['time'])
            reward_track.append(info['reward'])

            state_now = state_next
            reward_me += info['reward']

            if done:
                break

        print('start', state_start)
        print('totla_reward', reward_me)
        print('x_end', A3C.para.env.x)
        plt.figure(1)
        plt.plot(time_track, [x[0] for x in state_track])
        plt.grid()
        plt.title('x')

        #
        plt.figure(2)
        plt.plot(time_track, action_track)
        plt.title('action')
        plt.grid()

        plt.figure(3)
        plt.plot(time_track, reward_track)
        plt.grid()
        plt.title('reward')

        plt.show()
    elif display_flag == 3:

        plt.axis([-0.1, 1.2, -0.1, 1.2])
        plt.ion()
        plt.grid(color='g', linewidth='0.3', linestyle='--')

        ep_num = 10

        state_track = np.zeros([ep_num, 200])
        action_track = np.zeros([ep_num, 200])
        time_track = np.zeros([ep_num, 200])
        reward_track = np.zeros([ep_num, 200])
        step_all = np.zeros([ep_num])

        for ep in range(ep_num):
            print('step', ep)
            state_now = A3C.para.env.reset()
            reward_me = 0
            for step in range(1000):

                action = A3C.GLOBAL_AC.choose_action(state_now)
                state_next, reward, done, info = A3C.para.env.step(action)
                state_track[ep, step]= info['x']
                action_track[ep, step] = info['action']
                time_track[ep, step] = info['time']
                reward_track[ep, step] = info['reward']

                state_now = state_next
                reward_me += info['reward']

                if done:
                    print('x_error', info['x']-1)
                    step_all[ep] = step
                    break

        plt.scatter(1, 1, color='r')
        color_list = ['b', 'c', 'g', 'k', 'm', 'w', 'y','b', 'c', 'g', 'k', 'm', 'w', 'y','b', 'c', 'g', 'k', 'm',
                      'w', 'y','b', 'c', 'g', 'k', 'm', 'w', 'y','b', 'c', 'g', 'k', 'm', 'w', 'y','b', 'c', 'g',
                      'k', 'm', 'w', 'y']
        for aa in range(200):
            for bb in range(ep_num):
                if aa<=step_all[bb]:
                    plt.scatter(time_track[bb, aa], state_track[bb, aa], color=color_list[bb], marker='.')
            plt.pause(0.001)
        plt.pause(100000000)





