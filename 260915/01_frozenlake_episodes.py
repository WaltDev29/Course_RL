# FrozenLake 과업에서 에피소드 생성

import gymnasium as gym
import numpy as np

def generate_episode(env,policy):
    epi=[]
    s,info=env.reset()
    while True:
        a=np.random.choice([0,1,2,3],1,p=policy[s])[0]
        s1,r,terminated,truncated,info=env.step(a)
        epi.append([s,a,r]) # [상태,행동,보상] 추가
        s=s1
        if terminated or truncated:
            epi.append([s1,None,None])
            break
    return epi

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

pi1=np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n # 랜덤 정책

episode=generate_episode(env,pi1)
print('랜덤 정책으로 생성한 에피소드:\n',np.array(episode))
env.close()
