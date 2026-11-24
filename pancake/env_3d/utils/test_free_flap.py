import mujoco
import os

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs", "assets")
top_obj = os.path.join(ASSETS, "cake_sector_top.obj").replace("\\", "/")

xml = f"""
<mujoco>
  <compiler angle="radian"/>
  <asset>
    <mesh name="mesh_cake_sec_top" file="{top_obj}"/>
  </asset>
  <worldbody>
    <body name="cake" pos="0 0 1">
      <geom type="cylinder" size="0.025 0.005" mass="0.01"/>
      <body name="cake_sec_00" pos="0 0 0">
        <joint name="j_pitch" type="hinge" pos="0.025 0 0" axis="0 1 0" stiffness="0.0003" damping="0.00005" range="-1.2 1.2"/>
        <geom type="mesh" mesh="mesh_cake_sec_top" mass="0.003" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)
print("jnt_range in rad:", m.jnt_range)
d = mujoco.MjData(m)
for step in range(500):
    mujoco.mj_step(m, d)
print("With radian compiler, sag angle (deg):", d.qpos[0]*180/3.14159)
