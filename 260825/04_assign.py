# Lunar Lander 문제에서 랜덤 정책 테스트

import gymnasium as gym
import numpy as np

env = gym.make('LunarLander-v3', render_mode='human')
print(env.observation_space)  # 상태 공간
print(env.action_space)  # 행동 공간

state, info = env.reset()
print(env.render())

while True:
    action = env.action_space.sample()  # 랜덤 정책
    state1, reward, terminated, truncated, info = env.step(action)
    state = state1
    print(env.render())
    if terminated or truncated:
        break

env.close()
