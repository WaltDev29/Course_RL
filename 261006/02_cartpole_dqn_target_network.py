# 행위 신경망을 가진 DQN을 이용한 CartPole 학습
import gymnasium as gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt

gamma=0.99 # 할인율
eps=1.0
eps_decay=0.999
eps_min=0.05
replay_memory_siz=2000
batch_siz=64
copy_interval=50 # 행위 신경망을 목표 신경망에 복사하는 주기
n_episode=500

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

def build_network(): # 신경망 만들기
    mlp=Sequential()
    mlp.add(Dense(32,input_dim=s_dim,activation='relu'))
    mlp.add(Dense(32,activation='relu'))
    mlp.add(Dense(2,activation='linear'))
    mlp.compile(loss='MSE',optimizer=Adam(learning_rate=0.001))
    return mlp

def model_learning(): # DQN 학습
    mini_batch=random.sample(R,batch_siz)
    s=np.asarray([sample[0] for sample in mini_batch])
    a=[sample[1] for sample in mini_batch]
    r=[sample[2] for sample in mini_batch]
    s1=np.asarray([sample[3] for sample in mini_batch])
    terminated=[sample[4] for sample in mini_batch]

    X=s
    Y=model_behavior.predict(s,verbose=0)
    y1_=model_target.predict(s1,verbose=0)

    for i in range(batch_siz):
        if terminated[i]:
            Y[i,a[i]]=r[i]
        else:
            Y[i,a[i]]=r[i]+gamma*np.max(y1_[i])

    model_behavior.fit(X,Y,batch_size=batch_siz,epochs=1,verbose=0)

env=gym.make('CartPole-v1')
s_dim=env.observation_space.shape[0] # 상태 공간 차원
a_dim=env.action_space.n # 행동 공간 차원

model_target=build_network() # 목표 신경망 생성
model_behavior=build_network() # 행위 신경망 생성(데이터 생성용)
R=deque(maxlen=replay_memory_siz) # 리플레이 메모리 초기화
epi_length=[] # 에피소드 길이

n_update=0
for i in range(n_episode): # 신경망 학습
    s,info=env.reset()
    length=0
    while True:
        y_=model_behavior.predict(s.reshape([1,s_dim]),verbose=0)[0]
        if(np.random.random()<eps):
            a=np.random.randint(0,a_dim) # 랜덤 정책
        else:
            a=greedy_select(y_)
        s1,r,terminated,truncated,info=env.step(a)
        R.append((s,a,r,s1,terminated))

        if len(R)>batch_siz*3:
            model_learning()

        n_update+=1
        if n_update%copy_interval==0:
            model_target.set_weights(model_behavior.get_weights())

        s=s1
        length+=1
        eps=max(eps_min,eps*eps_decay) # 엡실론을 조금씩 줄임

        if terminated or truncated:
            epi_length.append(length)
            break

    print(i+1,'번째 에피소드의 길이:',epi_length[i])
    if np.min(epi_length[-5:])>=env.spec.max_episode_steps: # 연속 5번 최대 길이 넘으면 수렴
        break

model_target.save('./f7-2.keras') # 신경망 저장
env.close()

plt.plot(range(1,len(epi_length)+1),epi_length)
plt.title('DQN scores for CartPole-v1 with behavior network')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
