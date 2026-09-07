# 4*4, 8*8 FrozenLake 문제에서 랜덤 정책의 기대 이득 계산 시간 차이 확인

import gymnasium as gym
import numpy as np
import time

n = 100000  # 에피소드 개수

start_time = time.perf_counter()

env = gym.make('FrozenLake-v1', render_mode='ansi', is_slippery=False)
env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
episodes = []

for _ in range(n):
    epi = []
    state, info = env.reset()
    epi.append([None, state])

    while True:
        action = np.random.choice([0, 1, 2, 3], 1, p=[0.25, 0.25, 0.25, 0.25])[0]  # 랜덤 정책
        state1, reward, terminated, truncated, info = env.step(action)
        state = state1
        epi.append([action, reward, state])

        if terminated or truncated:
            break

    episodes.append(epi)

env.close()

expected_return = sum(e[-1][1] for e in episodes) / n  # 기대 이득 계산

elapsed_time_4 = time.perf_counter() - start_time # 실행 시간 계산

print('4*4 랜덤 정책의 기대 이득 =', expected_return)
print(f'4*4 기대 이득 계산까지 걸린 시간 = {elapsed_time_4:.3f}초')



start_time = time.perf_counter()
env = gym.make('FrozenLake-v1', map_name="8x8", render_mode='ansi', is_slippery=False)
env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
episodes = []

for _ in range(n):
    epi = []
    state, info = env.reset()
    epi.append([None, state])

    while True:
        action = np.random.choice([0, 1, 2, 3], 1, p=[0.25, 0.25, 0.25, 0.25])[0]  # 랜덤 정책
        state1, reward, terminated, truncated, info = env.step(action)
        state = state1
        epi.append([action, reward, state])

        if terminated or truncated:
            break

    episodes.append(epi)

env.close()

expected_return = sum(e[-1][1] for e in episodes) / n  # 기대 이득 계산

elapsed_time_8 = time.perf_counter() - start_time # 실행 시간 계산

time_ratio = elapsed_time_8 / elapsed_time_4

print('8*8 랜덤 정책의 기대 이득 =', expected_return)
print(f'8*8 기대 이득 계산까지 걸린 시간 = {elapsed_time_8:.3f}초')
print(f'8*8 / 4*4 시간 배수 = {time_ratio:.1f}배')

