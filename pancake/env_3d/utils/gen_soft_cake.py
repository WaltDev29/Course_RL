"""3D 방사형 탄성 소프트바디 팬케이크 XML 생성기.

중심 코어(반경 2.5cm) + 8방향 2단 탄성 플랩 체인(반경 2.5cm~6.5cm, 총 직경 13cm).
총 질량 50g (코어 18g + 외곽 32g).
각 세그먼트 상단 Red / 하단 Blue 적용으로 뒤집힘 시각화.
"""
import math
import os

XML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs", "assets", "pancake_env_3d.xml")

def generate_cake_xml() -> str:
    directions = [
        ("e", 0, "동"),
        ("ne", 45, "북동"),
        ("n", 90, "북"),
        ("nw", 135, "북서"),
        ("w", 180, "서"),
        ("sw", 225, "남서"),
        ("s", 270, "남"),
        ("se", 315, "남동"),
    ]

    lines = []
    lines.append('    <!-- [3D 소프트바디 팬케이크] 프로토타입 탄성 체인 계승: 중심 코어(r=2.5cm) + 8방향 2단 탄성 플랩 (직경 13cm, 50g) -->')
    lines.append('    <body name="cake" pos="0.84 0.22 0.73">')
    lines.append('      <freejoint name="cake_joint"/>')
    lines.append('      <!-- 중심 코어: 상단 Red / 하단 Blue (질량 18g) -->')
    lines.append('      <geom name="cake_top" type="cylinder" size="0.025 0.0025" pos="0 0 0.0025" material="mat_cake_top" mass="0.009" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append('      <geom name="cake_bottom" type="cylinder" size="0.025 0.0025" pos="0 0 -0.0025" material="mat_cake_bottom" mass="0.009" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
    lines.append('      <!-- 법선/중심 추적용 사이트 -->')
    lines.append('      <site name="cake_center" pos="0 0 0" size="0.008" rgba="1 1 0 0.7"/>')
    lines.append('      <site name="cake_top_site" pos="0 0 0.005" size="0.006" rgba="1 0 0 0.9"/>')
    lines.append('')

    for code, deg, name in directions:
        rad = math.radians(deg)
        ux, uy = math.cos(rad), math.sin(rad)
        j1_x, j1_y = 0.025 * ux, 0.025 * uy
        lines.append(f'      <!-- {name} ({deg}도) 방사형 탄성 플랩 2단 체인 (1단 2g + 2단 2g) -->')
        lines.append(f'      <body name="cake_flap_{code}1" pos="{j1_x:.4f} {j1_y:.4f} 0" euler="0 0 {deg}">')
        lines.append(f'        <joint name="j_flap_{code}1" type="hinge" axis="0 1 0" stiffness="0.0012" damping="0.0002" range="-1.0 1.0"/>')
        lines.append(f'        <geom name="cake_{code}1_top" type="box" pos="0.010 0 0.0025" size="0.010 0.0085 0.0025" material="mat_cake_top" mass="0.001" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
        lines.append(f'        <geom name="cake_{code}1_bot" type="box" pos="0.010 0 -0.0025" size="0.010 0.0085 0.0025" material="mat_cake_bottom" mass="0.001" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
        lines.append(f'        <body name="cake_flap_{code}2" pos="0.020 0 0">')
        lines.append(f'          <joint name="j_flap_{code}2" type="hinge" axis="0 1 0" stiffness="0.0009" damping="0.00015" range="-1.2 1.2"/>')
        lines.append(f'          <geom name="cake_{code}2_top" type="box" pos="0.010 0 0.0025" size="0.010 0.0110 0.0025" material="mat_cake_top" mass="0.001" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
        lines.append(f'          <geom name="cake_{code}2_bot" type="box" pos="0.010 0 -0.0025" size="0.010 0.0110 0.0025" material="mat_cake_bottom" mass="0.001" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
        lines.append(f'        </body>')
        lines.append(f'      </body>')

    lines.append('    </body>')
    return '\n'.join(lines)


def apply_to_xml():
    with open(XML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 기존 단일 강체 cake 블록 찾기
    start_tag = '    <!-- [3D 팬케이크]'
    if start_tag not in content:
        start_tag = '<body name="cake"'
    
    end_tag = '    </body>\n  </worldbody>'

    start_idx = content.find(start_tag)
    end_idx = content.find(end_tag) + len('    </body>')

    if start_idx == -1 or end_idx == -1:
        print("[ERROR] cake 블록을 찾지 못했습니다.")
        return

    new_cake = generate_cake_xml()
    new_content = content[:start_idx] + new_cake + content[end_idx:]

    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[SUCCESS] {XML_PATH} 에 3D 소프트바디 팬케이크 적용 완료!")


if __name__ == "__main__":
    apply_to_xml()
