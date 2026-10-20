# MountainCarContinuous 시험
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
import cv2 as cv

class ActorNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.actor_mean = tf.keras.Sequential([
            Input(shape=(s_dim,)),
            Dense(64, activation=activation),
            Dense(64, activation=activation),
            Dense(a_dim)
        ])

    def call(self, state):
        mean = self.actor_mean(state)
        return mean

env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
s_dim = env.observation_space.shape[0] # 상태 공간
a_dim = env.action_space.shape[0] # 행동 공간

activation = tf.keras.activations.tanh # 활성 함수
actor = ActorNetwork()
actor.load_weights('f9-7.weights.h5')

score = 0
s, info = env.reset()
while True:
    mean = actor(s.reshape(1, -1))
    action = mean[0] # 평균을 행동으로 취함(탐욕 선택)
    s, r, terminated, truncated, info = env.step(action)
    score += r

    cv.imshow('MountainCarContinuous animation', cv.cvtColor(env.render(), cv.COLOR_BGR2RGB))
    key = cv.waitKey(10)

    if terminated or truncated:
        print("에피소드의 점수:", score)
        break

env.close()
if cv.waitKey() == ord('q'):
    cv.destroyAllWindows()
