# CartPole 과업의 상태 변수 범위 알아내기
import gymnasium as gym
import numpy as np

env=gym.make('CartPole-v1',render_mode='rgb_array')

minv,maxv=[1e10,1e10,1e10,1e10],[-1e10,-1e10,-1e10,-1e10]
for i in range(1000000): # 100만 개 에피소드에 대해
    s,info=env.reset()
    minv,maxv=np.minimum(minv,s),np.maximum(maxv,s)
    while True:
        a=np.random.randint(0,env.action_space.n)
        s,r,terminated,truncated,info=env.step(a)
        minv,maxv=np.minimum(minv,s),np.maximum(maxv,s)
        if terminated or truncated:
            break

for i in range(4):
    print('상태변수',i,'의 범위=(',minv[i],',',maxv[i],')')
