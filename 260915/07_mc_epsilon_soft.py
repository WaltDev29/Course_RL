# 몬테카를로 방법으로 정책 학습(𝜺-소프트 적용)

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

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

def optimal_policy(env,gamma=0.99,eps=0.1):
    policy=[[0.25,0.25,0.25,0.25] for i in range(env.observation_space.n)] # 랜덤 정책으로 초기화
    Q=np.zeros((env.observation_space.n,env.action_space.n))
    returns=[[[] for j in range(env.action_space.n)] for i in range(env.observation_space.n)]
    for i in range(10000):
        epi=generate_episode(env,policy)
        R=0
        for t in range(len(epi)-2,-1,-1):
            s,a,r=epi[t][0],epi[t][1],epi[t][2] # 상태,행동,보상 추출
            R=gamma*R+r
            returns[s][a].append(R)
            Q[s][a]=np.mean(returns[s][a])
            amax=greedy_select(Q[s])
            for a1 in range(env.action_space.n):
                if a1==amax:
                    policy[s][a1]=1-eps+eps/env.action_space.n
                else:
                    policy[s][a1]=eps/env.action_space.n
    return policy,Q

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

pi,Q=optimal_policy(env)
np.set_printoptions(precision=3)
print('최적 정책:\n',np.array(pi),'\nQ 함수:\n',Q)
env.close()
