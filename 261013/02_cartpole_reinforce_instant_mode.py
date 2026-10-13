# REINFORCE를 이용한 CartPole 학습([알고리 즘 8 -1] 순간 모드)
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

gamma=0.99 # 할인율
n_episode=1000 # 최대 에피소드 개수

def neural_network(): # 다층 퍼셉트론 신경망 생성
    mlp=Sequential()
    mlp.add(Dense(32,input_dim=s_dim,activation='relu'))
    mlp.add(Dense(2,activation='softmax'))
    return mlp

def learn():
    return_mem=[]
    ret=0
    for t in reversed(range(len(action_mem))): # 각 순간의 이득 계산
        ret=gamma*ret+reward_mem[t]
        return_mem.insert(0,ret)

    for t in range(len(action_mem)):
        with tf.GradientTape() as tape:            
            y_=model(state_mem[t].reshape([1,s_dim]))[0]
            logprob=tf.math.log(y_[action_mem[t]])
            loss-=logprob*return_mem[t] # 음수를 취해 최대화 문제를 최소화 문제로 변환
        grad=tape.gradient(loss,model.trainable_weights)
        optimizer.apply_gradients(zip(grad,model.trainable_weights))

env=gym.make('CartPole-v1')
s_dim=env.observation_space.shape[0]
a_dim=env.action_space.n

model=neural_network() # 정책 신경망 생성
optimizer=tf.optimizers.Adam(learning_rate=0.001)

state_mem=[] # 에피소드 데이터 저장소
action_mem=[]
reward_mem=[]

epi_length=[] # 에피소드 길이
for e in range(n_episode): # 신경망 학습
    s,info=env.reset()
    while True: # 에피소드 생성
        y_=model(s.reshape([1,s_dim]))[0]
        a=np.random.choice(a_dim,p=np.array(y_))
        s1,r,terminated,truncated,info=env.step(a)
        state_mem.append(s)
        action_mem.append(a)
        reward_mem.append(r)
        s=s1
        if terminated or truncated:
            epi_length.append(len(action_mem))
            break

    learn()

    state_mem,action_mem,reward_mem=[],[],[] # 데이터 비우기

    print(e,'번째 에피소드의 길이:',epi_length[-1])
    if np.min(epi_length[-5:])>=env.spec.max_episode_steps: # 연속 5번 최대 길이 넘으면 조기종료
        break

model.save('f8-2.keras') # 신경망 저장
env.close()

plt.plot(range(1,len(epi_length)+1),epi_length)
smooth=np.convolve(epi_length,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('REINFORCE scores for CartPole-v1')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
