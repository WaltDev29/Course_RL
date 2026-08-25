# 최적 정책 기대 이득 계산

import gymnasium as gym
import numpy as np

n = 100000  # 에피소드 개수

pi2 = np.array([
    [0, 0.5, 0.5, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0],
    [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0],
    [0, 0, 1, 0], [0, 0.5, 0.5, 0], [0, 1, 0, 0], [0, 0, 0, 0],
    [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]
])  # 최적 정책

env = gym.make('FrozenLake-v1', render_mode='ansi', is_slippery=False)
env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
episodes = []

for _ in range(n):
    epi = []
    state, info = env.reset()
    epi.append([None, state])

    while True:
        action = np.random.choice([0, 1, 2, 3], 1, p=pi2[state])[0]  # 최적 정책으로 행동 결정
        state1, reward, terminated, truncated, info = env.step(action)
        state = state1
        epi.append([action, reward, state])

        if terminated or truncated:
            break

    if epi[-1][1] != 1:
        print(epi)
        input('a)')

    episodes.append(epi)

env.close()

expected_return = sum(e[-1][1] for e in episodes) / n  # 기대 이득 계산
print('최적 정책의 기대 이득 =', expected_return)
