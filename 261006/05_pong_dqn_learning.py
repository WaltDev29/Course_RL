# DQN을 이용한 퐁 아타리 게임 학습
import gymnasium as gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import ale_py
from collections import deque
import cv2 as cv
import time

gamma=0.99 # 할인율
eps=1.0
eps_decay=0.99999
eps_min=0.01
replay_memory_siz=20000
batch_siz=32
copy_interval=1000 # 행위 신경망을 목표 신경망에 복사하는 주기
n_match=500
consecutive=4 # 연속 4장 영상으로 상태 표현
rows=84 # 신경망 입력 영상 크기
cols=84
actions=[0,2,3] # 행동: 0(위치 유지), 2(위로 이동), 3(아래로 이동)

greedy_select=lambda x:np.random.choice(np.argwhere(x==np.max(x)).flatten())

def deep_network(): # 신경망 만들기
    cnn=Sequential()
    cnn.add(Conv2D(32,(5,5),activation='relu',strides=2,input_shape=(rows,cols,consecutive)))
    cnn.add(Conv2D(64,(3,3),activation='relu',strides=2))
    cnn.add(Conv2D(64,(3,3),activation='relu',strides=1))
    cnn.add(Flatten())
    cnn.add(Dense(512,activation='relu'))
    cnn.add(Dense(len(actions),activation='linear'))
    cnn.compile(loss='MSE',optimizer=Adam(learning_rate=0.00005))
    return cnn

def preprocess(img): # 화면 영상 전처리(260×180×3 컬러 → 84×84×1 명암)
    crop=cv.cvtColor(img[30:-12,5:-4,:],cv.COLOR_BGR2GRAY) # 경계 제거
    return cv.resize(crop,dsize=(rows,cols)).reshape(rows,cols,1)

def model_learning(): # DQN 학습
    mini_batch=random.sample(range(len(R)),batch_siz)
    s=np.asarray([R[mini_batch[i]][0] for i in range(batch_siz)])
    a=np.asarray([R[mini_batch[i]][1] for i in range(batch_siz)])
    r=np.asarray([R[mini_batch[i]][2] for i in range(batch_siz)])
    s1=np.asarray([R[mini_batch[i]][3] for i in range(batch_siz)])
    terminated=np.asarray([R[mini_batch[i]][4] for i in range(batch_siz)])

    X=s
    Y=np.array(model_behavior(s))
    y1_=np.array(model_target(s1))

    for i in range(batch_siz):
        if terminated[i]:
            Y[i,a[i]]=r[i]
        else:
            Y[i,a[i]]=r[i]+gamma*np.max(y1_[i])

    model_behavior.fit(X,Y,batch_size=batch_siz,epochs=1,verbose=0)

env=gym.make('ALE/Pong-v5',render_mode='rgb_array')

model_target=deep_network() # 목표 신경망 생성
model_behavior=deep_network() # 행위 신경망 생성(데이터 생성용)

R=deque(maxlen=replay_memory_siz) # 리플레이 메모리 초기화
n_update=0 # 누적 프레임 수
scores=[] # 경기(match) 점수 기록
start_time_total=time.time()

for i in range(n_match): # 경기 반복
    start=time.time()
    obs,info=env.reset()
    imem=deque(maxlen=consecutive) # 4장의 연속 영상으로 상태 표현
    for j in range(consecutive): imem.append(preprocess(obs))
    s=np.transpose(np.squeeze(np.asarray(imem)),(1,2,0))

    score=0
    while True:
        if(np.random.random()<eps):
            a=np.random.choice(len(actions)) # 랜덤 선택
        else:
            y_=model_behavior(s.reshape(1,rows,cols,consecutive))[0]
            a=greedy_select(y_) # 탐욕 선택
        obs1,r,terminated,truncated,info=env.step(actions[a])
        imem.append(preprocess(obs1))
        s1=np.transpose(np.squeeze(np.asarray(imem)),(1,2,0))
        R.append((s,a,r,s1,terminated))
        eps=max(eps_min,eps*eps_decay) # 엡실론을 조금씩 줄임

        if len(R)>batch_siz*3 and n_update%4==0:
            model_learning()

        n_update+=1
        if n_update%copy_interval==0:
            model_target.set_weights(model_behavior.get_weights())

        s=s1

        if r!=0: # 게임 종료
            score+=r
        if terminated or truncated: # 경기 종료
            scores.append(score)
            break

    print(i+1,"번째 경기 점수=",scores[-1],'(',int(time.time()-start),'초) eps=',np.round(eps,8))
print('학습에 걸린 총 시간=',int(time.time()-start_time_total),'초')
model_target.save("f7-5.keras") # 신경망 저장
env.close()

plt.plot(range(1,len(scores)+1),scores)
smooth=np.convolve(scores,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('DQN scores for Pong-v5')
plt.ylabel('Score')
plt.xlabel('Match')
plt.grid()
plt.show()
