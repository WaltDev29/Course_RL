# InvertedPendulum 추론 (안전 디버깅 버전)
import os
os.environ["MUJOCO_GL"] = "osmesa"
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2 as cv
@tf.keras.utils.register_keras_serializable()
def log_1(x): 
    return tf.math.log(1 + tf.math.exp(x))[0]
print("1. 모델 로드 시작...", flush=True)
model = load_model('f8-6.keras')
print("2. 환경 생성...", flush=True)
env = gym.make('InvertedPendulum-v4', render_mode='rgb_array')
s_dim = env.observation_space.shape[0]
print("3. 환경 리셋 및 첫 렌더링...", flush=True)
s, info = env.reset()
first_frame = env.render()
print(f"-> 렌더링 성공! 해상도: {first_frame.shape}", flush=True)
print("4. 비디오 라이터 초기화...", flush=True)
h, w, _ = first_frame.shape
fourcc = cv.VideoWriter_fourcc(*'avc1')  # 또는 *'mp4v'
out = cv.VideoWriter('inverted_pendulum.mp4', fourcc, 30.0, (w, h))
print("5. 추론 및 녹화 루프 시작...", flush=True)
length = 0
while True:
    mean, std = model(s.reshape([1, s_dim]))
    a = mean[0]
    s, r, terminated, truncated, info = env.step(a)
    length += r
    frame = cv.cvtColor(env.render(), cv.COLOR_RGB2BGR)
    out.write(frame)
    if terminated or truncated:
        print("-> 에피소드 종료! 총 길이:", length, flush=True)
        break
env.close()
out.release()
print("완료: inverted_pendulum.mp4 저장 성공!", flush=True)