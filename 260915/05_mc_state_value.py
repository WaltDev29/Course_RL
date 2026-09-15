# 몬테카를로 방법으로 정책 평가(상태 가치 함수 계산)

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

def policy_evaluation_by_state_value(env,policy,gamma=0.99):
    V=np.zeros(env.observation_space.n)
    returns=[[] for i in range(env.observation_space.n)]
    for _ in range(10000):
        epi=generate_episode(env,policy)
        R=0
        for t in range(len(epi)-2,-1,-1):
            s,r=epi[t][0],epi[t][2] # 상태와 보상 추출
            R=gamma*R+r
            returns[s].append(R)
            V[s]=np.mean(returns[s])
    return V

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

pi1=[[0.25,0.25,0.25,0.25] for i in range(env.observation_space.n)]
pi2=[[0,0.5,0.5,0] for i in range(env.observation_space.n)]

np.set_printoptions(precision=5)
print('pi1의 상태 가치 함수:\n',policy_evaluation_by_state_value(env,pi1).reshape([4,4]))
print('pi2의 상태 가치 함수:\n',policy_evaluation_by_state_value(env,pi2).reshape([4,4]))
env.close()
