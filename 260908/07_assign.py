# 벨만 방정식으로 랜덤정책 평가

import gymnasium as gym
import numpy as np

gamma=0.9 # 할인율

def policy_evaluation(env,policy):
    V=np.zeros(env.observation_space.n) # 가치 함수를 저장할 표
    while True:
        oldV=V.copy()
        for state in range(env.observation_space.n):
            v=0
            for action,action_prob in enumerate(policy[state]): # 벨만 방정식
                prob,next_state,reward,terminated=env.unwrapped.P[state][action][0]
                v+=action_prob*(reward+gamma*V[next_state])
            V[state]=v
        if max(np.abs(V-oldV))<1e-8:
            break
    return V

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

pi1_14=np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n # 랜덤 정책
pi1_14[14] = [0,0,1,0]
V=policy_evaluation(env,pi1_14)
print('정책 pi1_14의 가치 함수:\n',np.round(V.reshape([4,4]),4))
