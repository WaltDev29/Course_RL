# InvertedPendulum 추론 (비디오 저장 버전)
import os
# EGL 대신 안전한 CPU 오프스크린 렌더러 osmesa 사용 (필요시 "egl"로 변경 가능)
os.environ["MUJOCO_GL"] = "osmesa"

import gymnasium as gym
from tensorflow.keras.models import load_model
import cv2 as cv
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def log_1(x): 
    return tf.math.log(1 + tf.math.exp(x))[0]

model = load_model('f8-6.keras')

env = gym.make('InvertedPendulum-v4', render_mode='rgb_array')
s_dim = env.observation_space.shape[0]  # 상태 공간

length = 0
s, info = env.reset()

# 첫 프레임으로 비디오 해상도 확인 및 VideoWriter 초기화
first_frame = env.render()
h, w, _ = first_frame.shape
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('inverted_pendulum.mp4', fourcc, 30.0, (w, h))

print("추론 및 비디오 녹화 시작...")

while True:
    mean, std = model(s.reshape([1, s_dim]))
    a = mean[0]  # 평균을 행동으로 취함(탐욕 선택)
    s, r, terminated, truncated, info = env.step(a)
    length += r

    # 프레임 저장 (RGB -> BGR)
    frame = cv.cvtColor(env.render(), cv.COLOR_RGB2BGR)
    out.write(frame)

    if terminated or truncated:
        print("에피소드 종료! 총 길이(점수):", length)
        break

env.close()
out.release()
print("비디오 저장 완료: inverted_pendulum.mp4")
