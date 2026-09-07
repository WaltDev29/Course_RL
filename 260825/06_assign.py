# 8*8 FrozenLake 문제에서 랜덤 정책 테스트

import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', map_name="8x8", render_mode='ansi', is_slippery=False)
print(env.observation_space)  # 상태 공간
print(env.action_space)  # 행동 공간

state, info = env.reset()
print(env.render())

while True:
    action = np.random.choice([0, 1, 2, 3], 1, p=[0.25, 0.25, 0.25, 0.25])[0]  # 랜덤 정책
    state1, reward, terminated, truncated, info = env.step(action)
    state = state1
    print(env.render())
    if terminated or truncated:
        break

env.close()
