# REINFORCE를 이용한 InvertedPendulum 학습(배열 연산 버전)
import gymnasium as gym
import numpy as np
import scipy
from tensorflow.keras.layers import Dense, Lambda
import matplotlib.pyplot as plt
import tqdm

def gaussian(x,m,s): # m=mean, s=standard deviation
    return (1.0/(s*tf.math.sqrt(2.0*np.pi)))*tf.math.exp(-((x-m)**2)/(2*s**2))

def log_1(x): return tf.math.log(1+tf.math.exp(x))

def neural_network(): # 신경망 만들기
    input_x = tf.keras.Input(shape=(s_dim,))
    hidden_x = Dense(32,activation='tanh')(input_x)
    hidden_x = Dense(32,activation='tanh')(hidden_x)
    mean = Dense(a_dim,activation='linear')(hidden_x)
    std1 = Dense(a_dim,activation='linear')(hidden_x)
    std = Lambda(log_1,output_shape=(a_dim,))(std1)
    mlp = tf.keras.Model(inputs=input_x,outputs=[mean,std])
    return mlp

def learn(): # 에피소드 e를 가지고 신경망 학습
    def discount_cumulation(x,discount):
        return scipy.signal.lfilter([1],[1,float(-discount)],x[::-1],axis=0)[::-1]

    def learn(): # 에피소드 e를 가지고 신경망 학습
        return_mem = []
        return_mem = discount_cumulation(reward_mem,gamma) # 이득
        return_mem = tf.convert_to_tensor(return_mem,dtype=tf.float32)

    with tf.GradientTape() as tape:
        mean,std = model(tf.convert_to_tensor(state_mem))
        logprob = tf.math.log(gaussian(action_mem,mean,std))
        loss = -tf.reduce_sum(tf.squeeze(logprob)*return_mem)
        gradient = tape.gradient(loss,model.trainable_weights)
        optimizer.apply_gradients(zip(gradient,model.trainable_weights))
gamma = 0.99 # 할인율
n_episode = 1000

env = gym.make('InvertedPendulum-v4',render_mode='rgb_array')
s_dim = env.observation_space.shape[0] # 상태 공간
a_dim = env.action_space.shape[0] # 행동 공간
a_low,a_high = env.action_space.low,env.action_space.high

model = neural_network() # 신경망 생성
optimizer = tf.optimizers.Adam(learning_rate=0.001)

state_mem = [] # 에피소드 데이터 저장소
action_mem = []
reward_mem = []

epi_length = [] # 에피소드 길이
for e in tqdm.tqdm(range(n_episode)): # 신경망 학습
    s,info = env.reset()
    while True: # 에피소드 생성
        mean,std = model(s.reshape([1,s_dim]))
        a = np.clip(np.random.normal(mean[0],std[0],1),a_low[0],a_high[0])
        s1,r,terminated,truncated,info = env.step(a)
        state_mem.append(s)
        action_mem.append(a)
        reward_mem.append(r)
        s = s1

        if terminated or truncated:
            epi_length.append(len(action_mem))
            break

    learn()

    state_mem,action_mem,reward_mem = [],[],[] # 데이터 비우기

    if (e+1)%10==0: print(e+1,'번째 에피소드 평균 길이:',np.mean(epi_length[-10:]))
    if np.min(epi_length[-5:])>=env.spec.max_episode_steps: # 연속 5번 최대 길이 넘으면 수렴
        break

model.save("f8-6.keras") # 신경망 저장
env.close()

plt.figure(figsize=(16,5))
plt.plot(range(1,len(epi_length)+1),epi_length)
smooth = np.convolve(epi_length,10*[0.1],mode='valid')
plt.plot(range(1,len(smooth)+1),smooth)
plt.title('REINFORCE scores for InvertedPendulum-v4')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
