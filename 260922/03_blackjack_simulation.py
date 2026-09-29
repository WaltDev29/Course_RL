# 블랙잭 게임 시뮬레이션
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env=gym.make('Blackjack-v1', render_mode='rgb_array')
print(env.observation_space, env.action_space)

s,info=env.reset()
print('상태=',s)
plt.imshow(env.render())
plt.show()

while True:
    a=np.random.choice([0,1],p=(0.5,0.5))
    s1,r,terminated,truncated,info=env.step(a)

    print('행동=',a,'상태=',s1,'보상=',r,'종료?=',terminated)
    plt.imshow(env.render())
    plt.show()

    if terminated or truncated:
        break

env.close()
