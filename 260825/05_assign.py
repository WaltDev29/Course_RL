# HighwayEnv 문제에서 랜덤 정책 테스트

import gymnasium as gym
import highway_env

env = gym.make("highway-v0", render_mode="human")
print(env.observation_space)  # 상태 공간
print(env.action_space)  # 행동 공간

state, info = env.reset()
print(env.render())

while True:
    action = env.action_space.sample()
    state1, reward, terminated, truncated, info = env.step(action)
    state = state1
    print(env.render())
    if terminated or truncated:
        break

env.close()
