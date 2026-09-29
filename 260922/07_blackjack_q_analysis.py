# Q 함수 해석
import gymnasium as gym
import numpy as np
import pickle

env=gym.make('Blackjack-v1', render_mode='rgb_array')

def init_q(): return np.zeros(env.action_space.n)
Q=pickle.load(open('f5-4.pkl','rb')) # Q-러닝으로 학습한 정책의 수익 분석

print('딕셔너리 크기는',len(Q))

print(3*' '+'상태'+5*' '+'스틱'+4*' '+'히트'+8*' '+'상태'+5*' '+'스틱'+4*' '+'히트')
dealer_card=8
for player in range(1,22):
    not_usable_ace=(player,dealer_card,0)
    usable_ace=(player,dealer_card,1)
    q0_stick=Q[not_usable_ace][0]
    q0_hit=Q[not_usable_ace][1]
    q1_stick=Q[usable_ace][0]
    q1_hit=Q[usable_ace][1]
    print(not_usable_ace,'%7.4f %7.4f'%(q0_stick,q0_hit),' ',end='')
    print(usable_ace,'%7.4f %7.4f'%(q1_stick,q1_hit))
