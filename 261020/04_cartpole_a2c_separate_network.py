# A2C를 이용한 CartPole 학습
import gymnasium as gym
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def neural_network_actor(): # actor 신경망
    input_x = tf.keras.Input(shape=(s_dim,))
    hidden_x = Dense(32, activation='tanh')(input_x)
    actor_y = Dense(a_dim, activation='softmax')(hidden_x)
    mlp = tf.keras.Model(inputs=input_x, outputs=actor_y)
    return mlp

def neural_network_critic(): # critic 신경망
    input_x = tf.keras.Input(shape=(s_dim,))
    hidden_x = Dense(32, activation='tanh')(input_x)
    critic_y = Dense(1, activation='linear')(hidden_x)
    mlp = tf.keras.Model(inputs=input_x, outputs=critic_y)
    return mlp

def discount_cumulation(x, discount):
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]

def advantage_MC(rewards, values, gamma): # 몬테카를로 이익 계산
    return discount_cumulation(rewards, gamma) - np.array(values)

actionprob_mem = [] # 데이터 저장소
value_mem = []
reward_mem = []

def episode_generate_and_learn():
    s, info = env.reset()
    with tf.GradientTape(persistent=True) as tape:
        while True:
            action_prob = actor_model(s.reshape([1, s_dim]))
            value = critic_model(s.reshape([1, s_dim]))
            a = np.random.choice(a_dim, p=np.array(action_prob[0]))
            s1, r, terminated, truncated, info = env.step(a)

            actionprob_mem.append(action_prob[0, a])
            value_mem.append(value[0, 0])
            reward_mem.append(r)
            s = s1

            if terminated or truncated:
                break

        returns = discount_cumulation(reward_mem, gamma)
        advantage = advantage_MC(reward_mem, value_mem, gamma)
        advantage = tf.convert_to_tensor(advantage, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        loss_actor = tf.reduce_mean(-advantage * tf.math.log(actionprob_mem))
        loss_critic = tf.reduce_mean(tf.math.square(returns - value_mem))

    # GradientTape 컨텍스트 외부에서 그레이디언트 계산 및 업데이트
    grad_actor = tape.gradient(loss_actor, actor_model.trainable_weights) # actor 학습
    actor_optimizer.apply_gradients(zip(grad_actor, actor_model.trainable_weights))

    grad_critic = tape.gradient(loss_critic, critic_model.trainable_weights) # critic 학습
    critic_optimizer.apply_gradients(zip(grad_critic, critic_model.trainable_weights))

gamma = 0.99 # 할인율
n_episode = 1000

env = gym.make('CartPole-v1')
s_dim = env.observation_space.shape[0] # 상태 공간
a_dim = int(env.action_space.n) # 행동 공간

actor_model = neural_network_actor() # actor 신경망 생성
critic_model = neural_network_critic() # critic 신경망 생성
actor_optimizer = tf.optimizers.Adam(learning_rate=0.01)
critic_optimizer = tf.optimizers.Adam(learning_rate=0.01)

epi_length = [] # 에피소드 길이
for i in range(n_episode):
    episode_generate_and_learn()
    epi_length.append(len(actionprob_mem))
    actionprob_mem, reward_mem, value_mem = [], [], [] # 데이터 비우기

    if (i + 1) % 20 == 0:
        print(i + 1, '번째 에피소드 평균 길이(최근 20개):', np.mean(epi_length[-20:]))
    if np.min(epi_length[-5:]) >= env.spec.max_episode_steps: # 연속 5번 최대 길이 넘으면 조기종료
        break

actor_model.save('f9-4_actor.keras')
critic_model.save('f9-4_critic.keras')
env.close()

plt.figure(figsize=(16, 5))
plt.plot(range(1, len(epi_length) + 1), epi_length)
smooth = np.convolve(epi_length, 10 * [0.1], mode='valid')
plt.plot(range(1, len(smooth) + 1), smooth)
plt.title('A2C scores for CartPole-v1 using separate network')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.savefig('04_cartpole_a2c_separate_network.png') # 비대화형 환경 대비 이미지 저장
try:
    plt.show()
except Exception:
    pass
