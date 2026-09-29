# 랜덤 정책과 Q-러닝으로 학습한 𝑞 함수의 수익 비교 분석
import gymnasium as gym
import numpy as np
import pickle

n_game=10000 # 게임 횟수

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())
def init_q(): return np.zeros(env.action_space.n)

env=gym.make('Blackjack-v1', render_mode='rgb_array')

n_win,n_lose,n_draw=0,0,0 # 랜덤 정책의 수익 분석
for i in range(n_game):
    s,info=env.reset()
    while True:
        a=np.random.choice([0,1],p=(0.5,0.5))
        s1,r,terminated,truncated,info=env.step(a)
        s=s1
        if terminated or truncated:
            if r==1: n_win+=1
            elif r==-1: n_lose+=1
            elif r==0: n_draw+=1
            break
print('랜덤 정책: 승=',n_win,'패=',n_lose,'비김=',n_draw)

Q=pickle.load(open('f5-4.pkl','rb')) # Q-러닝으로 학습한 q 함수의 수익 분석
n_win,n_lose,n_draw=0,0,0
for i in range(n_game):
    s,info=env.reset()
    while True:
        a=greedy_select(Q[s])
        s1,r,terminated,truncated,info=env.step(a)
        s=s1
        if terminated or truncated:
            if r==1: n_win+=1
            elif r==-1: n_lose+=1
            elif r==0: n_draw+=1
            break
print('Q-러닝으로 학습한 에이전트: 승=',n_win,'패=',n_lose,'비김=',n_draw)

env.close()
