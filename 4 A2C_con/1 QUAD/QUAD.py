# 太阳帆运动模型搭建
import numpy as np
import matplotlib.pyplot as plt


class QUAD:

    def __init__(self, random=False, c1=10, c2=0):
        self.t = None
        self.state = None
        self.random = random
        # 归一化参数长度除以AU,时间除以TU
        self.constant = {'g': -9.81, 'c1': 20., 'c2': 2., 'alpha': 0., 'm': 1.,
                         'gamma1': 1, 'gamma2': 1.0,
                         'x_f': 5., 'z_f': 5., 'vx_f': 0., 'vz_f': 0., 'theta_f': 0.}
        self.delta_t = 0.001  # 仿真步长，未归一化，单位天
        observation = self.reset()
        self.ob_dim = len(observation)
        self.action_dim = 2
        self.a_bound = np.array([[0.05, 1], [-1, 1]])
        self.rangeE_old = 1000
        self.c1 = c1
        self.c2 = c2
        self.c3 = 0
        # self.ob_profile = np.empty((0, 5))
        # self.time_profile = np.empty(0)
        # self.theta_SC_profile = np.empty(0)
        # self.control_profile = np.empty((0, 2))


    def render(self):
        pass

    def reset(self):
        self.t = 0
        self.rangeE_old = 1000
        if self.random == True:
            pass
        else:
            pass
        self.state = np.array([0., 0., 0., 0., 0.])  # [r phi u v]
        return self.state.copy()

    def step(self, action):
        # action 解剖
        # print(action)
        u1 = action[0] * (self.a_bound[0, 1]-self.a_bound[0, 0])/2 + np.mean(self.a_bound[0])
        u2 = action[1] * (self.a_bound[1, 1]-self.a_bound[1, 0])/2 + np.mean(self.a_bound[1])

        # 微分方程
        reward = 0
        for step in range(50):
            # print(self.state)
            _x, _z, _vx, _vz, _theta = self.state
            # state control
            # dynamic equation
            x_dot = _vx
            z_dot = _vz
            vx_dot = self.constant['c1'] * u1 * np.sin(_theta)
            vz_dot = self.constant['c1'] * u1 * np.cos(_theta) + self.constant['g']
            theta_dot = self.constant['c2'] * u2

            # 判断下一个状态的距离
            self.state += self.delta_t * np.array([x_dot, z_dot, vx_dot, vz_dot, theta_dot])
            self.t += self.delta_t
            x, z, vx, vz, theta = self.state
            rangeE_now = np.sqrt((x - self.constant['x_f']) ** 2 + (z - self.constant['z_f']) ** 2)
            vE_now = np.sqrt((vx - self.constant['vx_f']) ** 2 + (vz - self.constant['vz_f']) ** 2)
            thetaE_now = np.abs(theta - self.constant['theta_f'])

            # memory
            # self.ob_profile = np.vstack((self.ob_profile, self.state))
            # self.time_profile = np.hstack((self.time_profile, self.t))
            # self.theta_SC_profile = np.hstack((self.theta_SC_profile, theta_SC))
            # self.control_profile = np.vstack((self.control_profile, np.array([u1, u2])))

            # reward 计算
            reward += - self.delta_t
            if 5 > rangeE_now > self.rangeE_old or rangeE_now < 0.01 or self.t > 5:
                done = True
                rangeE_now = 0 if rangeE_now < 0.01 else rangeE_now
                vE_now = 0 if vE_now < 0.01 else vE_now
                Penelty_range = self.c1 *rangeE_now**0.3 if rangeE_now > 1 else self.c1 * rangeE_now
                Penelty_v = self.c2 * vE_now ** 0.3 if vE_now > 1 else self.c2 * vE_now
                reward += 100 - Penelty_range - \
                          Penelty_v - \
                              self.c3 * thetaE_now

                break
            else:
                done = False
                self.rangeE_old = rangeE_now


        return self.state.copy(), reward, done, {}


if __name__ == '__main__':

    env = QUAD()
    action = np.array([1, 1])
    while True:
        observation_, reward, done, info = env.step(action)

        if done:
            break
    print('reward', reward, 'time', env.t)

    # design the weight between DRL and SC
    for i in [10, 5, 4, 3, 2, 1, 0.5, 0.1]:
        weight_u = np.arctan(1 / i ** 2)/np.pi*2
        print(weight_u)

    plt.figure(2)
    plt.plot(env.ob_profile[:, 0], env.ob_profile[:, 1])
    plt.plot(env.constant['x_f'], env.constant['z_f'], 'ro')
    plt.title('x-z')

    plt.figure(3)
    plt.plot(env.ob_profile[:, 2], env.ob_profile[:, 3])
    plt.plot(env.constant['vx_f'], env.constant['vz_f'], 'ro')
    plt.title('v')


    plt.figure(4)
    plt.plot(env.time_profile, env.ob_profile[:, 4])
    plt.plot(env.time_profile, env.theta_SC_profile)
    # plt.plot(env.constant['x_f'], env.constant['z_f'], 'ro')
    plt.title('theta')

    plt.figure(5)
    plt.plot(env.time_profile, env.control_profile[:, 0])
    plt.plot(env.time_profile, env.control_profile[:, 1])
    plt.title('control')

    plt.show()

