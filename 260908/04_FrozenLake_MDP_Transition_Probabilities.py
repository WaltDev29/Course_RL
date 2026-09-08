# FrozenLake 과업의 MDP 전이 확률 분포를 확인

import gymnasium as gym
import pprint # 딕셔너리를 깔끔하게 출력

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='ansi')
print('결정론 MDP의 전이 확률 분포:')
pprint.pprint(env.unwrapped.P)

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='ansi')
print('스토캐스틱 MDP의 전이 확률 분포:')
pprint.pprint(env.unwrapped.P)
