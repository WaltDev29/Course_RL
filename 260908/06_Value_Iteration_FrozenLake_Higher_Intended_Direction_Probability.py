# FrozenLake 과업을 위한 가치 반복 알고리즘(의도한 방향의 확률을 높인 스토캐스틱 환경)

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
                for prob,next_state,reward,terminated in env.unwrapped.P[state][action]:
                    q[action]=q[action]+prob*(reward+gamma*V[next_state])
            V[state]=np.max(q)
        if max(np.abs(V-oldV))<1e-8:
            break

    pi=env.observation_space.n*[None] # 최적 가치 함수로부터 최적 정책 구하기
    for state in range(env.observation_space.n):
        q=np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for prob,next_state,reward,terminated in env.unwrapped.P[state][action]:
                q[action]=q[action]+prob*(reward+gamma*V[next_state])
        pi[state]=np.argmax(q)
    return pi,V

env=gym.make('FrozenLake-v1',is_slippery=True,render_mode='ansi')

prob=[0.1,0.8,0.1] # 전이 확률 (1/3,1/3,1/3)을 (0.1,0.8,0.1)로 수정
for state in range(env.observation_space.n):
    for action in range(env.action_space.n):
        if len(env.unwrapped.P[state][action])==3:
            for i,transition in enumerate(env.unwrapped.P[state][action]):
                env.unwrapped.P[state][action][i]=(prob[i],transition[1],transition[2],transition[3])

pi,V=value_iteration(env)
print('최적 정책:\n',np.array(pi).reshape([4,4]))
print('최적 가치 함수:\n',np.round(V.reshape([4,4]),4))