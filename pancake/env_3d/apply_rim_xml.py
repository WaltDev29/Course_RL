import math, os

xml_path = r"d:\course\RL\pancake\env_3d\envs\assets\pancake_env_3d.xml"
with open(xml_path, "r", encoding="utf-8") as f:
    xml_str = f.read()

N_rim = 20
rim_geoms = []

# Tier 1 (low ramp): z = 0.018, r = 0.110, cap_r = 0.009 (r_inner = 0.101, bottom = 0.009 < floor top 0.012)
r1, z1, cr1 = 0.110, 0.018, 0.009
for k in range(N_rim):
    a1 = 2 * math.pi * k / N_rim
    a2 = 2 * math.pi * (k + 1) / N_rim
    p1 = f"{r1 * math.cos(a1):.4f} {r1 * math.sin(a1):.4f} {z1:.4f}"
    p2 = f"{r1 * math.cos(a2):.4f} {r1 * math.sin(a2):.4f} {z1:.4f}"
    rim_geoms.append(f'<geom name="rim_l{k+1:02d}" type="capsule" fromto="{p1}  {p2}" size="{cr1:.4f}" rgba="0 0 0 0" contype="1" conaffinity="2" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')

# Tier 2 (high rim): z = 0.030, r = 0.126, cap_r = 0.011 (r_inner = 0.115, bottom = 0.019 < tier 1 top 0.027)
r2, z2, cr2 = 0.126, 0.030, 0.011
for k in range(N_rim):
    a1 = 2 * math.pi * k / N_rim
    a2 = 2 * math.pi * (k + 1) / N_rim
    p1 = f"{r2 * math.cos(a1):.4f} {r2 * math.sin(a1):.4f} {z2:.4f}"
    p2 = f"{r2 * math.cos(a2):.4f} {r2 * math.sin(a2):.4f} {z2:.4f}"
    rim_geoms.append(f'<geom name="rim_h{k+1:02d}" type="capsule" fromto="{p1}  {p2}" size="{cr2:.4f}" rgba="0 0 0 0" contype="1" conaffinity="2" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')

rim_xml = "\n                        ".join(rim_geoms)

old_start = xml_str.find('<body name="pan_head"')
old_end = xml_str.find("</body>", old_start) + len("</body>")

new_pan_head = f"""<body name="pan_head" pos="0.34 0 0">
                        <!-- 사실적인 주방 프라이팬 3D 메쉬 (팬 바디 + 곡면 림 + 손잡이 일체형) -->
                        <geom name="pan_mesh" type="mesh" mesh="mesh_pan" material="mat_pan" contype="0" conaffinity="0"/>
                        <!-- 팬케이크 접촉용 바닥 충돌체: 테프론 저마찰 원판 (반경 0.105m, 상단 z=0.012) -->
                        <geom name="pan_floor" type="cylinder" size="0.105 0.006" pos="0 0 0.006" rgba="0 0 0 0" contype="1" conaffinity="2" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>
                        <site name="pan_center" pos="0 0 0.012" size="0.008" rgba="1 0 0 1"/>

                        <!-- 2단 외향 경사 림(Flared Ramp Rim): 하단 램프(l) + 상단 림(h) 구조로 역경사 끼임 원천 차단 -->
                        {rim_xml}
                      </body>"""

xml_str = xml_str[:old_start] + new_pan_head + xml_str[old_end:]

# 케이크 충돌 속성 업데이트
old_cake = """      <!-- 상단 Red (Z > 0, 5mm) -->
      <geom name="cake_top" type="cylinder" size="0.065 0.0025" pos="0 0 0.0025" material="mat_cake_top" mass="0.025"/>
      <!-- 하단 Blue (Z < 0, 5mm) -->
      <geom name="cake_bottom" type="cylinder" size="0.065 0.0025" pos="0 0 -0.0025" material="mat_cake_bottom" mass="0.025"/>"""

new_cake = """      <!-- 상단 Red (Z > 0, 5mm) -->
      <geom name="cake_top" type="cylinder" size="0.065 0.0025" pos="0 0 0.0025" material="mat_cake_top" mass="0.025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>
      <!-- 하단 Blue (Z < 0, 5mm) -->
      <geom name="cake_bottom" type="cylinder" size="0.065 0.0025" pos="0 0 -0.0025" material="mat_cake_bottom" mass="0.025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>"""

if old_cake in xml_str:
    xml_str = xml_str.replace(old_cake, new_cake)
else:
    print("Warning: old_cake pattern not found!")

with open(xml_path, "w", encoding="utf-8") as f:
    f.write(xml_str)
print("Successfully updated pancake_env_3d.xml")
