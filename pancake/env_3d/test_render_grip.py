import sys, os
sys.path.insert(0, r"d:\course\RL\pancake\env_3d")
from envs.pancake_env_3d import PancakeFlipEnv3D
from PIL import Image
import mujoco

env = PancakeFlipEnv3D(render_mode="rgb_array")
env.reset()

renderer = mujoco.Renderer(env.model, 480, 640)
# 손과 손잡이를 정면 측면에서 선명하게 볼 수 있는 카메라 앵글 설정
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
# 전체 뷰 카메라 설정
cam.lookat = [0.45, 0.15, 0.65]
cam.distance = 1.6
cam.elevation = -18.0
cam.azimuth = 135.0

renderer.update_scene(env.data, camera=cam)
img_arr = renderer.render()

img = Image.fromarray(img_arr)
out_path = r"d:\course\RL\pancake\env_3d\hand_grip_check.png"
img.save(out_path)
print("Saved render check to:", out_path)
