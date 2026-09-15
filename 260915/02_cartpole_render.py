# CartPole 과업에서 에피소드 생성과 렌더링

import gymnasium as gym
import matplotlib.pyplot as plt
import time

def generate_episode(env):
    epi=[]
    s,info=env.reset()

    plt.ion()
    image=env.render()
    plt.imshow(image)
    plt.pause(0.1)

    while True:
        a=0 # 항상 0(왼쪽)을 선택하는 정책
        s1,r,terminated,truncated,info=env.step(a)
        epi.append([s,a,r]) # [상태,행동,보상] 추가
        s=s1

        image=env.render()
        plt.imshow(image)
        plt.show()
        plt.pause(0.1)

        if terminated or truncated:
            epi.append([s1,None,None])
            break
    return epi

env=gym.make('CartPole-v1',render_mode='rgb_array')

episode=generate_episode(env)
print('항상 left를 선택하는 정책으로 생성한 에피소드:')
for v in episode:
    print(v)

env.close()
