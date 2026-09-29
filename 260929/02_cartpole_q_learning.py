# 신경망 기반 Q-러닝(CartPole 과업의 학습)
import numpy as np
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
gamma=0.99 # 할인율
eps=1.0
eps_decay=0.9999
eps_min=0.05
n_episode=2000

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

env=gym.make('CartPole-v1')
s_dim=env.observation_space.shape[0] # 상태 공간 차원
a_dim=env.action_space.n # 행동 공간 차원

def build_network(): # 신경망 만들기(MLP)
    mlp=Sequential()
    mlp.add(Dense(32,input_dim=s_dim,activation='relu'))
    mlp.add(Dense(32,activation='relu'))
    mlp.add(Dense(2,activation='linear'))
    mlp.compile(loss='MSE',optimizer=Adam(learning_rate=0.01))
    return mlp



model=build_network() # 신경망 생성

epi_length=[] # 에피소드의 길이
for i in tqdm(range(n_episode)): # 신경망 학습
    s,info=env.reset()
    length=0
    while True:
        y_=model.predict(s.reshape([1,s_dim]),verbose=0)[0]
        if np.random.random()<eps:
            a=np.random.randint(0,a_dim)
        else:
            a=greedy_select(y_)
        s1,r,terminated,truncated,info=env.step(a)
        x=s; y=y_
        y1_=model.predict(s1.reshape([1,s_dim]),verbose=0)[0]

        if terminated:
            y[a]=r
        else:
            y[a]=r+gamma*np.max(y1_)

        model.fit(x.reshape([1,s_dim]),y.reshape([1,a_dim]),batch_size=1,epochs=1,verbose=0)

        s=s1
        length+=1
        eps=max(eps_min,eps*eps_decay) # 엡실론 스케줄링

        if terminated or truncated:
            epi_length.append(length)
            break

    if np.min(epi_length[-5:])>=env.spec.max_episode_steps: # 연속 5번 최대 길이 넘으면 수렴
        break
    if (i+1)%10==0:
        print(i+1,'번째 에피소드 길이:',np.mean(epi_length[-10:]))

model.save('./f6-2.keras') # 신경망 저장
env.close()

plt.plot(range(1,len(epi_length)+1),epi_length) # 수렴 곡선 그리기
smooth=np.convolve(epi_length,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('Convergence of Q-network for CartPole-v1')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
