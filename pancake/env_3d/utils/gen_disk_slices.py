"""방안 1: 원형 윤곽선(Circular Disc)을 완벽히 유지하는 7분할 탄성 슬라이스 팬케이크 생성기.

- 직경 13cm (반경 0.065m), 두께 1.0cm, 총 질량 50g.
- X축을 따라 7개의 슬라이스로 분할하되, 각 슬라이스의 Y축 양끝을 원의 외곽선에 맞춰 둥근 캡으로 마감.
- 위/측면에서 볼 때 매끄러운 하나의 원반 형태를 유지.
- 인접 슬라이스 간 스프링-댐퍼 힌지(stiffness=0.0008, damping=0.00015)로 프로토타입 특유의 펄럭임/찰랑거림 구현.
"""
import math
import os

XML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs", "assets", "pancake_env_3d.xml")

def generate_disk_slices_xml() -> str:
    R = 0.065  # 반경 6.5cm
    
    # 각 슬라이스 파라미터: (name, width_x, mass_g, stiffness, damping)
    # core: w_x=0.020, center_x=0.0
    # r1/l1: w_x=0.018, center_x=0.019 (상대 pos=0.009)
    # r2/l2: w_x=0.018, center_x=0.037 (상대 pos=0.009)
    # r3/l3: w_x=0.019, center_x=0.0555 (상대 pos=0.0095)
    
    # 1. 중심 슬라이스 (Core)
    w_core = 0.020
    sx_core = w_core / 2.0  # 0.010
    ymax_core = R           # 0.065
    sy_core = ymax_core - sx_core  # 0.055
    
    # 2. 1번 슬라이스 (r1, l1)
    w_1 = 0.018
    sx_1 = w_1 / 2.0        # 0.009
    xc_1 = 0.019
    ymax_1 = math.sqrt(R**2 - xc_1**2)  # ~0.06216
    sy_1 = ymax_1 - sx_1               # ~0.05316
    
    # 3. 2번 슬라이스 (r2, l2)
    w_2 = 0.018
    sx_2 = w_2 / 2.0        # 0.009
    xc_2 = 0.037
    ymax_2 = math.sqrt(R**2 - xc_2**2)  # ~0.05344
    sy_2 = ymax_2 - sx_2               # ~0.04444
    
    # 4. 3번 슬라이스 (r3, l3)
    w_3 = 0.019
    sx_3 = w_3 / 2.0        # 0.0095
    xc_3 = 0.0555
    ymax_3 = math.sqrt(R**2 - xc_3**2)  # ~0.03383
    sy_3 = ymax_3 - sx_3               # ~0.02433

    lines = []
    lines.append('    <!-- [3D 소프트바디 팬케이크] 방안 1: 원형 윤곽선(직경 13cm, 두께 1cm, 50g) 유지 7분할 탄성 슬라이스 -->')
    lines.append('    <body name="cake" pos="0.84 0.22 0.73">')
    lines.append('      <freejoint name="cake_joint"/>')
    lines.append('      <site name="cake_center" pos="0 0 0" size="0.008" rgba="1 1 0 0.7"/>')
    lines.append('      <site name="cake_top_site" pos="0 0 0.005" size="0.006" rgba="1 0 0 0.9"/>')
    lines.append('')
    
    # 코어 지오메트리 (Top / Bottom)
    lines.append(f'      <!-- 코어 슬라이스 (폭 {w_core*100:.1f}cm, 길이 {ymax_core*200:.1f}cm, 8g) -->')
    lines.append(f'      <geom name="cake_core_top" type="box" pos="0 0 0.0025" size="{sx_core:.4f} {sy_core:.4f} 0.0025" material="mat_cake_top" mass="0.003" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'      <geom name="cake_core_bot" type="box" pos="0 0 -0.0025" size="{sx_core:.4f} {sy_core:.4f} 0.0025" material="mat_cake_bottom" mass="0.003" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'      <geom name="cake_core_c1_top" type="cylinder" pos="0 {sy_core:.4f} 0.0025" size="{sx_core:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'      <geom name="cake_core_c1_bot" type="cylinder" pos="0 {sy_core:.4f} -0.0025" size="{sx_core:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'      <geom name="cake_core_c2_top" type="cylinder" pos="0 -{sy_core:.4f} 0.0025" size="{sx_core:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'      <geom name="cake_core_c2_bot" type="cylinder" pos="0 -{sy_core:.4f} -0.0025" size="{sx_core:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append('')

    # 우측 슬라이스 체인 (cake_r1 -> cake_r2 -> cake_r3)
    lines.append('      <!-- [우측 체인] r1 (폭 1.8cm, 7g) -->')
    lines.append(f'      <body name="cake_r1" pos="{sx_core:.4f} 0 0">')
    lines.append('        <joint name="j_r1" type="hinge" axis="0 1 0" stiffness="0.00030" damping="0.00006" range="-1.6 1.6"/>')
    lines.append(f'        <geom name="cake_r1_top" type="box" pos="{sx_1:.4f} 0 0.0025" size="{sx_1:.4f} {sy_1:.4f} 0.0025" material="mat_cake_top" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'        <geom name="cake_r1_bot" type="box" pos="{sx_1:.4f} 0 -0.0025" size="{sx_1:.4f} {sy_1:.4f} 0.0025" material="mat_cake_bottom" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'        <geom name="cake_r1_c1_top" type="cylinder" pos="{sx_1:.4f} {sy_1:.4f} 0.0025" size="{sx_1:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'        <geom name="cake_r1_c1_bot" type="cylinder" pos="{sx_1:.4f} {sy_1:.4f} -0.0025" size="{sx_1:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'        <geom name="cake_r1_c2_top" type="cylinder" pos="{sx_1:.4f} -{sy_1:.4f} 0.0025" size="{sx_1:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'        <geom name="cake_r1_c2_bot" type="cylinder" pos="{sx_1:.4f} -{sy_1:.4f} -0.0025" size="{sx_1:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append('')
    lines.append('        <!-- [우측 체인] r2 (폭 1.8cm, 7g) -->')
    lines.append(f'        <body name="cake_r2" pos="{w_1:.4f} 0 0">')
    lines.append('          <joint name="j_r2" type="hinge" axis="0 1 0" stiffness="0.00020" damping="0.00004" range="-1.6 1.6"/>')
    lines.append(f'          <geom name="cake_r2_top" type="box" pos="{sx_2:.4f} 0 0.0025" size="{sx_2:.4f} {sy_2:.4f} 0.0025" material="mat_cake_top" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'          <geom name="cake_r2_bot" type="box" pos="{sx_2:.4f} 0 -0.0025" size="{sx_2:.4f} {sy_2:.4f} 0.0025" material="mat_cake_bottom" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'          <geom name="cake_r2_c1_top" type="cylinder" pos="{sx_2:.4f} {sy_2:.4f} 0.0025" size="{sx_2:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'          <geom name="cake_r2_c1_bot" type="cylinder" pos="{sx_2:.4f} {sy_2:.4f} -0.0025" size="{sx_2:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'          <geom name="cake_r2_c2_top" type="cylinder" pos="{sx_2:.4f} -{sy_2:.4f} 0.0025" size="{sx_2:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'          <geom name="cake_r2_c2_bot" type="cylinder" pos="{sx_2:.4f} -{sy_2:.4f} -0.0025" size="{sx_2:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append('')
    lines.append('          <!-- [우측 체인] r3 (끝단 캡, 폭 1.9cm, 7g) -->')
    lines.append(f'          <body name="cake_r3" pos="{w_2:.4f} 0 0">')
    lines.append('            <joint name="j_r3" type="hinge" axis="0 1 0" stiffness="0.00015" damping="0.00003" range="-1.6 1.6"/>')
    lines.append(f'            <geom name="cake_r3_top" type="box" pos="{sx_3:.4f} 0 0.0025" size="{sx_3:.4f} {sy_3:.4f} 0.0025" material="mat_cake_top" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <geom name="cake_r3_bot" type="box" pos="{sx_3:.4f} 0 -0.0025" size="{sx_3:.4f} {sy_3:.4f} 0.0025" material="mat_cake_bottom" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <geom name="cake_r3_c1_top" type="cylinder" pos="{sx_3:.4f} {sy_3:.4f} 0.0025" size="{sx_3:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <geom name="cake_r3_c1_bot" type="cylinder" pos="{sx_3:.4f} {sy_3:.4f} -0.0025" size="{sx_3:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <geom name="cake_r3_c2_top" type="cylinder" pos="{sx_3:.4f} -{sy_3:.4f} 0.0025" size="{sx_3:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <geom name="cake_r3_c2_bot" type="cylinder" pos="{sx_3:.4f} -{sy_3:.4f} -0.0025" size="{sx_3:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <site name="site_cake_right" pos="{w_3:.4f} 0 0" size="0.005" rgba="0 1 0 1"/>')
    lines.append('          </body>')
    lines.append('        </body>')
    lines.append('      </body>')
    lines.append('')

    # 좌측 슬라이스 체인 (cake_l1 -> cake_l2 -> cake_l3)
    lines.append('      <!-- [좌측 체인] l1 (폭 1.8cm, 7g) -->')
    lines.append(f'      <body name="cake_l1" pos="-{sx_core:.4f} 0 0">')
    lines.append('        <joint name="j_l1" type="hinge" axis="0 -1 0" stiffness="0.00030" damping="0.00006" range="-1.6 1.6"/>')
    lines.append(f'        <geom name="cake_l1_top" type="box" pos="-{sx_1:.4f} 0 0.0025" size="{sx_1:.4f} {sy_1:.4f} 0.0025" material="mat_cake_top" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'        <geom name="cake_l1_bot" type="box" pos="-{sx_1:.4f} 0 -0.0025" size="{sx_1:.4f} {sy_1:.4f} 0.0025" material="mat_cake_bottom" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'        <geom name="cake_l1_c1_top" type="cylinder" pos="-{sx_1:.4f} {sy_1:.4f} 0.0025" size="{sx_1:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'        <geom name="cake_l1_c1_bot" type="cylinder" pos="-{sx_1:.4f} {sy_1:.4f} -0.0025" size="{sx_1:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'        <geom name="cake_l1_c2_top" type="cylinder" pos="-{sx_1:.4f} -{sy_1:.4f} 0.0025" size="{sx_1:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'        <geom name="cake_l1_c2_bot" type="cylinder" pos="-{sx_1:.4f} -{sy_1:.4f} -0.0025" size="{sx_1:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append('')
    lines.append('        <!-- [좌측 체인] l2 (폭 1.8cm, 7g) -->')
    lines.append(f'        <body name="cake_l2" pos="-{w_1:.4f} 0 0">')
    lines.append('          <joint name="j_l2" type="hinge" axis="0 -1 0" stiffness="0.00020" damping="0.00004" range="-1.6 1.6"/>')
    lines.append(f'          <geom name="cake_l2_top" type="box" pos="-{sx_2:.4f} 0 0.0025" size="{sx_2:.4f} {sy_2:.4f} 0.0025" material="mat_cake_top" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'          <geom name="cake_l2_bot" type="box" pos="-{sx_2:.4f} 0 -0.0025" size="{sx_2:.4f} {sy_2:.4f} 0.0025" material="mat_cake_bottom" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'          <geom name="cake_l2_c1_top" type="cylinder" pos="-{sx_2:.4f} {sy_2:.4f} 0.0025" size="{sx_2:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'          <geom name="cake_l2_c1_bot" type="cylinder" pos="-{sx_2:.4f} {sy_2:.4f} -0.0025" size="{sx_2:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'          <geom name="cake_l2_c2_top" type="cylinder" pos="-{sx_2:.4f} -{sy_2:.4f} 0.0025" size="{sx_2:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'          <geom name="cake_l2_c2_bot" type="cylinder" pos="-{sx_2:.4f} -{sy_2:.4f} -0.0025" size="{sx_2:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append('')
    lines.append('          <!-- [좌측 체인] l3 (끝단 캡, 폭 1.9cm, 7g) -->')
    lines.append(f'          <body name="cake_l3" pos="-{w_2:.4f} 0 0">')
    lines.append('            <joint name="j_l3" type="hinge" axis="0 -1 0" stiffness="0.00015" damping="0.00003" range="-1.6 1.6"/>')
    lines.append(f'            <geom name="cake_l3_top" type="box" pos="-{sx_3:.4f} 0 0.0025" size="{sx_3:.4f} {sy_3:.4f} 0.0025" material="mat_cake_top" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <geom name="cake_l3_bot" type="box" pos="-{sx_3:.4f} 0 -0.0025" size="{sx_3:.4f} {sy_3:.4f} 0.0025" material="mat_cake_bottom" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <geom name="cake_l3_c1_top" type="cylinder" pos="-{sx_3:.4f} {sy_3:.4f} 0.0025" size="{sx_3:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <geom name="cake_l3_c1_bot" type="cylinder" pos="-{sx_3:.4f} {sy_3:.4f} -0.0025" size="{sx_3:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <geom name="cake_l3_c2_top" type="cylinder" pos="-{sx_3:.4f} -{sy_3:.4f} 0.0025" size="{sx_3:.4f} 0.0025" material="mat_cake_top" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <geom name="cake_l3_c2_bot" type="cylinder" pos="-{sx_3:.4f} -{sy_3:.4f} -0.0025" size="{sx_3:.4f} 0.0025" material="mat_cake_bottom" mass="0.0005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append(f'            <site name="site_cake_left" pos="-{w_3:.4f} 0 0" size="0.005" rgba="0 0 1 1"/>')
    lines.append('          </body>')
    lines.append('        </body>')
    lines.append('      </body>')
    lines.append('    </body>')
    return '\n'.join(lines)


def generate_contact_excludes() -> str:
    lines = [
        '  <contact>',
        '    <!-- 프로토타입 탄성 체인 접촉 간섭 배제 (인접 1마디 건너 유연 굽힘 보장) -->',
        '    <exclude body1="cake" body2="cake_r2"/>',
        '    <exclude body1="cake_r1" body2="cake_r3"/>',
        '    <exclude body1="cake" body2="cake_l2"/>',
        '    <exclude body1="cake_l1" body2="cake_l3"/>',
        '  </contact>'
    ]
    return '\n'.join(lines)


def apply_disk_slices():
    with open(XML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # cake body 교체
    start_tag = '    <!-- [3D'
    if start_tag not in content:
        start_tag = '<body name="cake"'
    end_tag = '    </body>\n  </worldbody>'

    s_idx = content.find(start_tag)
    e_idx = content.find(end_tag) + len('    </body>')

    if s_idx == -1 or e_idx == -1:
        print("[ERROR] cake body를 찾을 수 없습니다.")
        return

    new_cake = generate_disk_slices_xml()
    content = content[:s_idx] + new_cake + content[e_idx:]

    # contact 태그 확인 및 추가
    if '<contact>' in content:
        # 기존 contact 내에 exclude 삽입
        c_idx = content.find('<contact>') + len('<contact>\n')
        excludes = generate_contact_excludes().replace('  <contact>\n', '').replace('\n  </contact>', '')
        content = content[:c_idx] + excludes + '\n' + content[c_idx:]
    else:
        # </worldbody> 바로 뒤에 <contact> 추가
        wb_end = content.find('  </worldbody>') + len('  </worldbody>\n')
        content = content[:wb_end] + generate_contact_excludes() + '\n\n' + content[wb_end:]

    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[SUCCESS] 방안 1(원형 원반 유지 7분할 탄성 슬라이스)이 {XML_PATH} 에 완벽히 반영되었습니다!")


if __name__ == "__main__":
    apply_disk_slices()
