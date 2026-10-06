# 퐁 아타리 게임 애니메이션
import gymnasium as gym
import numpy as np
import ale_py
import cv2 as cv

def play_Pong(env):
    s,info=env.reset()
    while True:
        a=np.random.choice([0,2,3],1,p=[1/3,1/3,1/3])[0]  # 0(정지), 2(상), 3(하)
        s,r,terminated,truncated,info=env.step(a)

        cv.imshow('Pong animation',cv.cvtColor(env.render(),cv.COLOR_BGR2RGB))
        cv.waitKey(100)

        if terminated or truncated:
            break

env=gym.make('ALE/Pong-v5',render_mode='rgb_array')
print(env.observation_space,env.action_space)
play_Pong(env)
env.close()

if cv.waitKey()==ord('q'):
    cv.destroyAllWindows()
