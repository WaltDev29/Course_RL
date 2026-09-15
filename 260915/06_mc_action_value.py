# 몬테카를로 방법으로 정책 평가(행동 가치 함수 계산)

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

def policy_evaluation_by_action_value(env,policy,gamma=0.99):
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
    return Q

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

pi1=[[0.25,0.25,0.25,0.25] for i in range(env.observation_space.n)] # 랜덤 정책
pi2=[[0,0.5,0.5,0] for i in range(env.observation_space.n)] # 1(D)과 2(R)를 반반씩 선택하는 정책

np.set_printoptions(precision=3)
print('pi1의 행동 가치 함수:\n',policy_evaluation_by_action_value(env,pi1))
print('pi2의 행동 가치 함수:\n',policy_evaluation_by_action_value(env,pi2))
env.close()
