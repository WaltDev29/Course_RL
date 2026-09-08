# FrozenLake 과업을 위한 정책 반복 알고리즘

import gymnasium as gym
import numpy as np

gamma=0.9 # 할인율

def policy_iteration(env):
    V=np.zeros(env.observation_space.n)
    pi=[0 for i in range(env.observation_space.n)] # 정책을 행동 0으로 초기화
    while True:
        while True: # E (정책 평가)
            oldV=V.copy()
            for state in range(env.observation_space.n):
                action=pi[state]
                prob,next_state,reward,terminated=env.unwrapped.P[state][action][0]
                v=reward+gamma*V[next_state]
                V[state]=v
            if max(np.abs(V-oldV))<1e-8:
                break
        converged=True # I (정책 개선)
        for state in range(env.observation_space.n):
            old_action=pi[state]
            q=np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                prob,next_state,reward,terminated=env.unwrapped.P[state][action][0]
                q[action]=reward+gamma*V[next_state]
            pi[state]=np.argmax(q)
            if pi[state]!=old_action:
                converged=False
        if converged:
            break
    return V,pi

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')

V,pi=policy_iteration(env)
print('최적 정책:\n',np.array(pi).reshape([4,4]))
print('최적 가치 함수:\n',np.round(V.reshape([4,4]),4))
