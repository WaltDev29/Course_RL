# Blackjack 학습
import numpy as np
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tqdm

gamma=0.99 # 할인율
eps=1.0
eps_decay=0.9999
eps_min=0.05
n_episode=10000

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

def build_network(): # 신경망 만들기(MLP)
    mlp=Sequential()
    mlp.add(Dense(32,input_dim=s_dim,activation='relu'))
    mlp.add(Dense(32,activation='relu'))
    mlp.add(Dense(a_dim,activation='linear'))
    mlp.compile(loss='MSE',optimizer=Adam(learning_rate=0.01))
    return mlp

env=gym.make('Blackjack-v1', render_mode='rgb_array')
s_dim=len(env.observation_space) # 상태 공간 차원
a_dim=env.action_space.n # 행동 공간 차원

model=build_network() # 신경망 생성

for i in tqdm.tqdm(range(n_episode)): # 신경망 학습
    s,info=env.reset()
    while True:
        y_=model.predict(np.array(s).reshape([1,s_dim]),verbose=0)[0]
        if np.random.random()<eps:
            a=np.random.randint(0,env.action_space.n)
        else:
            a=greedy_select(y_)
        s1,r,terminated,truncated,info=env.step(a)
        x=s; y=y_
        y1_=model.predict(np.array(s1).reshape([1,s_dim]),verbose=0)[0]

        if terminated:
            y[a]=r
        else:
            y[a]=r+gamma*np.max(y1_)

        model.fit(np.array(x).reshape([1,s_dim]),y.reshape([1,a_dim]),batch_size=1,epochs=1,verbose=0)

        s=s1
        eps=max(eps_min,eps*eps_decay) # 엡실론을 조금씩 줄임

        if terminated or truncated:
            break

model.save('./f6-4.keras') # 신경망 저장
env.close()
