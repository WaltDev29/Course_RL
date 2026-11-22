import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "envs"))
import mujoco, math, numpy as np

# Read base XML
XML_PATH = os.path.join(os.path.dirname(__file__), "envs", "assets", "pancake_env_3d.xml")
with open(XML_PATH, "r", encoding="utf-8") as f:
    xml_str = f.read()

# Build 2-tier flared ramp rim (zero overhang, strictly positive outward draft angle)
# Tier 1 (lower ramp): z = 0.018, r = 0.110, cap_r = 0.009 (r_inner = 0.101, bottom = 0.009 < floor top 0.012)
# Tier 2 (upper rim):  z = 0.030, r = 0.126, cap_r = 0.011 (r_inner = 0.115, bottom = 0.019 < tier 1 top 0.027)
# This forms a smooth 35-degree flared bowl with ZERO undercuts!

N_rim = 20
rim_geoms = []

# Tier 1 (low)
r1, z1, cr1 = 0.110, 0.018, 0.009
for k in range(N_rim):
    a1 = 2 * math.pi * k / N_rim
    a2 = 2 * math.pi * (k + 1) / N_rim
    p1 = f"{r1 * math.cos(a1):.4f} {r1 * math.sin(a1):.4f} {z1:.4f}"
    p2 = f"{r1 * math.cos(a2):.4f} {r1 * math.sin(a2):.4f} {z1:.4f}"
    rim_geoms.append(f'<geom name="rim_l{k+1:02d}" type="capsule" fromto="{p1}  {p2}" size="{cr1:.4f}" rgba="0 0 0 0" contype="1" conaffinity="2" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')

# Tier 2 (high)
r2, z2, cr2 = 0.126, 0.030, 0.011
for k in range(N_rim):
    a1 = 2 * math.pi * k / N_rim
    a2 = 2 * math.pi * (k + 1) / N_rim
    p1 = f"{r2 * math.cos(a1):.4f} {r2 * math.sin(a1):.4f} {z2:.4f}"
    p2 = f"{r2 * math.cos(a2):.4f} {r2 * math.sin(a2):.4f} {z2:.4f}"
    rim_geoms.append(f'<geom name="rim_h{k+1:02d}" type="capsule" fromto="{p1}  {p2}" size="{cr2:.4f}" rgba="0 0 0 0" contype="1" conaffinity="2" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')

rim_xml = "\n                          ".join(rim_geoms)

old_start = xml_str.find('<body name="pan_head"')
old_end = xml_str.find('</body>', old_start) + len('</body>')

new_pan_head = f"""<body name="pan_head" pos="0.34 0 0">
                        <geom name="pan_mesh" type="mesh" mesh="mesh_pan" material="mat_pan" contype="0" conaffinity="0"/>
                        <!-- 바닥 원판: 반경 10.1cm, 높이 z=0.006 (상단 표면 z=0.012) -->
                        <geom name="pan_floor" type="cylinder" size="0.105 0.006" pos="0 0 0.006" rgba="0 0 0 0" contype="1" conaffinity="2" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>
                        <site name="pan_center" pos="0 0 0.012" size="0.008" rgba="1 0 0 1"/>
                        {rim_xml}
                      </body>"""

xml_test = xml_str[:old_start] + new_pan_head + xml_str[old_end:]

# Also update cake contact
cake_old = """      <!-- 상단 Red (Z > 0, 5mm) -->
      <geom name="cake_top" type="cylinder" size="0.065 0.0025" pos="0 0 0.0025" material="mat_cake_top" mass="0.025"/>
      <!-- 하단 Blue (Z < 0, 5mm) -->
      <geom name="cake_bottom" type="cylinder" size="0.065 0.0025" pos="0 0 -0.0025" material="mat_cake_bottom" mass="0.025"/>"""

cake_new = """      <!-- 상단 Red (Z > 0, 5mm) -->
      <geom name="cake_top" type="cylinder" size="0.065 0.0025" pos="0 0 0.0025" material="mat_cake_top" mass="0.025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>
      <!-- 하단 Blue (Z < 0, 5mm) -->
      <geom name="cake_bottom" type="cylinder" size="0.065 0.0025" pos="0 0 -0.0025" material="mat_cake_bottom" mass="0.025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>"""

xml_test = xml_test.replace(cake_old, cake_new)

assets_dir = os.path.join(os.path.dirname(__file__), "envs", "assets").replace("\\", "/")
xml_test = xml_test.replace('<compiler angle="degree"', f'<compiler meshdir="{assets_dir}" angle="degree"')

model = mujoco.MjModel.from_xml_string(xml_test)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

pan_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pan_center")]
cake_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cake_joint")
qpos_adr = model.jnt_qposadr[cake_jnt]
qvel_adr = model.jnt_dofadr[cake_jnt]

# Put pancake on the pan
data.qpos[qpos_adr : qpos_adr + 3] = pan_pos + [0, 0, 0.006]
data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]

# Push pancake horizontally at 0.5 m/s toward rim in 8 directions
print("Testing horizontal sliding and launching in 8 directions:")
for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, -math.pi/4, -math.pi/2, -3*math.pi/4]:
    data.qpos[qpos_adr : qpos_adr + 3] = pan_pos + [0, 0, 0.006]
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]
    data.qvel[qvel_adr : qvel_adr + 3] = [0.6 * math.cos(angle), 0.6 * math.sin(angle), 0]
    
    max_z = 0.0
    for step in range(60):
        mujoco.mj_step(model, data)
        cp = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cake_center")]
        rel_z = cp[2] - pan_pos[2]
        max_z = max(max_z, rel_z)
    
    cp = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cake_center")]
    final_rel_z = cp[2] - pan_pos[2]
    print(f"Angle {math.degrees(angle):4.0f} deg: max_z_lifted={max_z:.4f}m, final_z={final_rel_z:.4f}m")

# Now TEST A REAL UPWARD FLICK / TOSS!
data.qpos[qpos_adr : qpos_adr + 3] = pan_pos + [0, 0, 0.006]
data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]
data.qvel[qvel_adr : qvel_adr + 3] = 0

# Command arm to jerk up and pitch wrist forward
wrist_pitch_idx = 6 # joint_wr_pitch
elbow_pitch_idx = 3 # joint_el_pitch

# Give upward impulse
for i in range(10):
    data.ctrl[wrist_pitch_idx] = -0.6 # flick forward
    data.ctrl[elbow_pitch_idx] = 0.8  # pull up
    mujoco.mj_step(model, data)

# Let fly
for i in range(40):
    mujoco.mj_step(model, data)
    cp = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cake_center")]
    pp = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pan_center")]
    if i % 8 == 0:
        print(f"Toss step {i:02d}: cake_z={cp[2]:.4f}, pan_z={pp[2]:.4f}, separation={(cp[2]-pp[2]):.4f}m")

