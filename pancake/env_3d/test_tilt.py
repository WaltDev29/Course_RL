import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "envs"))
import pancake_env_3d, mujoco, numpy as np

env = pancake_env_3d.PancakeFlipEnv3D()
obs, _ = env.reset()

print("1. Initial state:")
cake_p = env.data.site_xpos[env._site_cake_center]
pan_p = env.data.site_xpos[env._site_pan]
print(f"Pan pos: {pan_p}, Cake pos: {cake_p}")

# Now, tilt the pan 90 degrees or invert it upside down!
# Let's rotate the wrist pitch by -1.0 rad (~ -57 deg) or invert pan
env.data.qpos[env._j_qpos[6]] = -1.2  # wrist pitch down hard
env.data.ctrl[6] = -1.2
mujoco.mj_forward(env.model, env.data)

print("\n2. Simulating with pan tilted down:")
for step in range(50):
    mujoco.mj_step(env.model, env.data)
    cp = env.data.site_xpos[env._site_cake_center]
    pp = env.data.site_xpos[env._site_pan]
    contacts = []
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if "cake" in (g1 or "") or "cake" in (g2 or ""):
            contacts.append((g1, g2, c.dist))
    if step % 5 == 0:
        print(f"step {step:02d}: cake_z={cp[2]:.4f}, pan_z={pp[2]:.4f}, contacts={contacts}")
