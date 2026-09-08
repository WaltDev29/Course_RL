# FrozenLake 과업을 위한 가치 반복 알고리즘

import gymnasium as gym
import numpy as np

gamma=0.9 # 할인율

def value_iteration(env):
    V=np.zeros(env.observation_space.n)
    while True: # 최적 가치 함수 추정
        oldV=V.copy()
        for state in range(env.observation_space.n):
            q=np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                prob,next_state,reward,terminated=env.unwrapped.P[state][action][0]
                q[action]=reward+gamma*V[next_state]
            V[state]=np.max(q)
        if max(np.abs(V-oldV))<1e-8:
            break
    pi=env.observation_space.n*[None] # 최적 가치 함수로부터 최적 정책 구하기
    for state in range(env.observation_space.n):
        q=np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            prob,next_state,reward,terminated=env.unwrapped.P[state][action][0]
            q[action]=reward+gamma*V[next_state]
        pi[state]=np.argmax(q)
    return pi,V

env=gym.make('FrozenLake-v1',is_slippery=False,render_mode='ansi')
pi,V=value_iteration(env)
print('최적 정책:\n',np.array(pi).reshape([4,4]))
print('최적 가치 함수:\n',np.round(V.reshape([4,4]),4))
