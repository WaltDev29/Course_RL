# Q 함수 해석
import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model

env=gym.make('Blackjack-v1', render_mode='rgb_array')
s_dim=len(env.observation_space) # 상태 공간 차원

q=load_model('f6-4.keras') # 신경망 불러오기

print(3*' '+'상태'+5*' '+'스틱'+4*' '+'히트'+8*' '+'상태'+5*' '+'스틱'+4*' '+'히트')
dealer_card=8
for player in range(1,22):
    not_usable_ace=(player,dealer_card,0)
    usable_ace=(player,dealer_card,1)
    Q=q.predict(np.array(not_usable_ace).reshape([1,s_dim]),verbose=0)[0]
    q0_stick=Q[0]
    q0_hit=Q[1]
    Q=q.predict(np.array(usable_ace).reshape([1,s_dim]),verbose=0)[0]
    q1_stick=Q[0]
    q1_hit=Q[1]
    print(not_usable_ace,'%7.4f %7.4f'%(q0_stick,q0_hit),' ',end='')
    print(usable_ace,'%7.4f %7.4f'%(q1_stick,q1_hit))
