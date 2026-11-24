"""완벽한 3D 원형 원반을 3x3 바둑판 격자(9개 셀)로 분할하는 2D 그리드 소프트바디 생성기.
- 중심 셀 (c): [-w, w] x [-w, w] (정사각형)
- 십자 셀 4개 (f, b, l, r): 바깥쪽 경계가 원호
- 대각선 코너 셀 4개 (fl, fr, bl, br): 바깥쪽 경계가 원호
- 9개 셀이 합쳐지면 틈새 0, 겹침 0의 완벽한 직경 13cm 원형 원반!
"""
import math
import os
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs", "assets")
XML_PATH = os.path.join(ASSETS_DIR, "pancake_env_3d.xml")

def create_prism_obj(filename: str, polygon_2d, x_anchor: float, y_anchor: float,
                     z_min: float, z_max: float):
    """2D 볼록 다각형을 Z축으로 z_min ~ z_max 두께를 갖는 3D 프리즘 OBJ 메쉬로 생성."""
    nv = len(polygon_2d)
    # 정점: 바닥면 (z_min) nv개 + 윗면 (z_max) nv개
    vertices = []
    # 0 ~ nv-1: bot
    for x, y in polygon_2d:
        vertices.append((x - x_anchor, y - y_anchor, z_min))
    # nv ~ 2*nv-1: top
    for x, y in polygon_2d:
        vertices.append((x - x_anchor, y - y_anchor, z_max))
        
    faces = []
    # 1. 바닥면 (Z = z_min, 시계 반대 방향 노멀 아래)
    # 삼각 분할 (fan from vertex 0)
    for i in range(1, nv - 1):
        faces.append((1, i + 2, i + 1))
        
    # 2. 윗면 (Z = z_max, 시계 방향 노멀 위)
    for i in range(1, nv - 1):
        faces.append((nv + 1, nv + i + 1, nv + i + 2))
        
    # 3. 옆면들 (i -> i+1)
    for i in range(nv):
        i_next = (i + 1) % nv
        b1 = i + 1
        b2 = i_next + 1
        t1 = nv + i + 1
        t2 = nv + i_next + 1
        faces.append((b1, b2, t2))
        faces.append((b1, t2, t1))
        
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 3D Grid Cell Prism {os.path.basename(filename)}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def build_3x3_grid_cake():
    R = 0.065
    w = 0.024  # 중심 셀 반-너비 (2.4cm, 중심 정사각형은 4.8cm x 4.8cm)
    
    # 9개 셀의 2D 다각형 정의 (모두 반시계 방향 CCW 순서)
    cells = {}
    
    # 1. 중심 셀 (c)
    cells["c"] = [
        (-w, -w), (w, -w), (w, w), (-w, w)
    ]
    
    # 원호 샘플링 헬퍼
    def sample_arc(th_start, th_end, n=12):
        ths = np.linspace(th_start, th_end, n)
        return [(R * math.cos(th), R * math.sin(th)) for th in ths]
        
    th_w_x = math.asin(w / R)      # Y = w 일 때의 각도 (약 21.7도)
    th_w_y = math.acos(w / R)      # X = w 일 때의 각도 (약 68.3도)
    
    # 2. 앞쪽 셀 (f: +X): Y in [-w, w], X from w to arc
    # arc: theta from -th_w_x to +th_w_x
    arc_f = sample_arc(-th_w_x, th_w_x, n=8)
    poly_f = [(w, -w)] + arc_f + [(w, w)]
    # 중복 정점 제거
    cells["f"] = [(w, -w)] + arc_f + [(w, w)]
    
    # 3. 뒤쪽 셀 (b: -X): Y in [-w, w], X from -w to -arc
    arc_b = sample_arc(math.pi - th_w_x, math.pi + th_w_x, n=8)
    cells["b"] = [(-w, w)] + arc_b + [(-w, -w)]
    
    # 4. 좌측 셀 (l: +Y): X in [-w, w], Y from w to arc
    # arc: theta from th_w_y to math.pi - th_w_y
    arc_l = sample_arc(th_w_y, math.pi - th_w_y, n=8)
    cells["l"] = [(w, w)] + arc_l + [(-w, w)]
    
    # 5. 우측 셀 (r: -Y): X in [-w, w], Y from -w to -arc
    arc_r = sample_arc(-math.pi + th_w_y, -th_w_y, n=8)
    cells["r"] = [(-w, -w)] + arc_r + [(w, -w)]
    
    # 6. 앞-좌 모서리 (fl: +X, +Y)
    # arc from th_w_x to th_w_y
    arc_fl = sample_arc(th_w_x, th_w_y, n=8)
    cells["fl"] = [(w, w)] + arc_fl
    
    # 7. 뒤-좌 모서리 (bl: -X, +Y)
    # arc from math.pi - th_w_y to math.pi - th_w_x
    arc_bl = sample_arc(math.pi - th_w_y, math.pi - th_w_x, n=8)
    cells["bl"] = [(-w, w)] + arc_bl
    
    # 8. 뒤-우 모서리 (br: -X, -Y)
    # arc from math.pi + th_w_x to math.pi + th_w_y
    arc_br = sample_arc(math.pi + th_w_x, math.pi + th_w_y, n=8)
    cells["br"] = [(-w, -w)] + arc_br
    
    # 9. 앞-우 모서리 (fr: +X, -Y)
    # arc from -th_w_y to -th_w_x
    arc_fr = sample_arc(-th_w_y, -th_w_x, n=8)
    cells["fr"] = [(w, -w)] + arc_fr
    
    asset_entries = []
    
    # 앵커 위치 계산 및 OBJ 파일 생성
    anchors = {}
    for name, poly in cells.items():
        # 무게중심을 앵커로 설정
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        xc = float(np.mean(xs)) if name != "c" else 0.0
        yc = float(np.mean(ys)) if name != "c" else 0.0
        anchors[name] = (xc, yc)
        
        # 상단 Red (z: 0 ~ 0.005)
        fn_top = os.path.join(ASSETS_DIR, f"cake_3x3_{name}_top.obj")
        create_prism_obj(fn_top, poly, xc, yc, 0.000, 0.005)
        
        # 하단 Blue (z: -0.005 ~ 0)
        fn_bot = os.path.join(ASSETS_DIR, f"cake_3x3_{name}_bot.obj")
        create_prism_obj(fn_bot, poly, xc, yc, -0.005, 0.000)
        
        asset_entries.append(f'    <mesh name="mesh_cake_3x3_{name}_top" file="cake_3x3_{name}_top.obj"/>')
        asset_entries.append(f'    <mesh name="mesh_cake_3x3_{name}_bot" file="cake_3x3_{name}_bot.obj"/>')

    # XML Body 구성
    # 중심 셀 (c): 질량 10g
    # 주변 8개 셀: 각각 5g (총 50g)
    body_lines = [
        '    <!-- [3D 소프트바디 팬케이크] 3x3 바둑판 격자 (9개 셀, 직경 13cm, 두께 1cm, 총 50g) -->',
        '    <body name="cake" pos="0.84 0.22 0.73">',
        '      <freejoint name="cake_joint"/>',
        '      <!-- 1) 중심 사각 셀 (c, 4.8cm x 4.8cm, 10g) -->',
        '      <geom name="cake_geom_c_top" type="mesh" mesh="mesh_cake_3x3_c_top" material="mat_cake_top" mass="0.005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>',
        '      <geom name="cake_geom_c_bot" type="mesh" mesh="mesh_cake_3x3_c_bot" material="mat_cake_bottom" mass="0.005" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>',
        '      <site name="cake_center" pos="0 0 0" size="0.003" rgba="1 1 0 0"/>',
        '      <site name="cake_top_site" pos="0 0 0.005" size="0.003" rgba="1 0 0 0"/>',
        ''
    ]
    
    # 8개 주변 셀
    cell_names = ["f", "b", "l", "r", "fl", "fr", "bl", "br"]
    for name in cell_names:
        xc, yc = anchors[name]
        # 중심 셀과의 접합 경계면을 회전축 위치(joint_pos)로 지정하여 모멘트 암 확보
        if name == "f":
            jx, jy = w - xc, 0.0
        elif name == "b":
            jx, jy = -w - xc, 0.0
        elif name == "l":
            jx, jy = 0.0, w - yc
        elif name == "r":
            jx, jy = 0.0, -w - yc
        elif name == "fl":
            jx, jy = w - xc, w - yc
        elif name == "fr":
            jx, jy = w - xc, -w - yc
        elif name == "bl":
            jx, jy = -w - xc, w - yc
        elif name == "br":
            jx, jy = -w - xc, -w - yc
        else:
            jx, jy = 0.0, 0.0

        body_lines.append(f'      <!-- 바둑판 셀 ({name}): pos=({xc:.4f}, {yc:.4f}) -->')
        body_lines.append(f'      <body name="cake_{name}" pos="{xc:.4f} {yc:.4f} 0">')
        # 가로 굽힘 힌지 (Pitch) + 세로 굽힘 힌지 (Roll)
        body_lines.append(f'        <joint name="j_{name}_pitch" type="hinge" pos="{jx:.4f} {jy:.4f} 0" axis="0 1 0" stiffness="0.0003" damping="0.00004" armature="0.000001" range="-50 50"/>')
        body_lines.append(f'        <joint name="j_{name}_roll" type="hinge" pos="{jx:.4f} {jy:.4f} 0" axis="1 0 0" stiffness="0.0003" damping="0.00004" armature="0.000001" range="-50 50"/>')
        body_lines.append(f'        <geom name="cake_{name}_top" type="mesh" mesh="mesh_cake_3x3_{name}_top" material="mat_cake_top" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
        body_lines.append(f'        <geom name="cake_{name}_bot" type="mesh" mesh="mesh_cake_3x3_{name}_bot" material="mat_cake_bottom" mass="0.0025" contype="1" conaffinity="3" condim="3" friction="0.2 0.005 0.001" solref="0.004 1"/>')
        body_lines.append('      </body>')
        
    body_lines.append('    </body>')
    
    # XML에 반영
    ref_xml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "env_3d", "envs", "assets", "pancake_env_3d.xml")
    with open(ref_xml_path, "r", encoding="utf-8") as f:
        ref_text = f.read()

    pan_mesh_str = '<mesh name="mesh_pan" file="frying_pan.obj"/>'
    new_assets = pan_mesh_str + '\n' + '\n'.join(asset_entries)
    xml_out = ref_text.replace(pan_mesh_str, new_assets)

    cake_start = xml_out.find('    <!-- 3D 원형 팬케이크')
    if cake_start == -1:
        cake_start = xml_out.find('    <body name="cake"')
    cake_end = xml_out.find('  </worldbody>')
    
    xml_out = xml_out[:cake_start] + '\n'.join(body_lines) + '\n' + xml_out[cake_end:]

    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(xml_out)

    print(f"[SUCCESS] 3x3 바둑판 격자 소프트바디 팬케이크가 {XML_PATH} 에 완벽히 반영되었습니다!")


if __name__ == "__main__":
    build_3x3_grid_cake()
