import numpy as np
import os, math

def generate_frying_pan_obj(filepath):
    # 2D cross-section profile (r, z) in local pan coordinates
    # We define outer bottom, fillet, outer wall, rim lip, inner wall, inner fillet, inner floor
    profile = []
    
    # Inner floor center to edge (z = 0.006, r from 0 to 0.095)
    r_floor = 0.095
    h_floor = 0.006
    h_rim = 0.045
    r_rim = 0.128
    wall_thick = 0.0035
    bottom_thick = 0.005
    
    # Inner surface points:
    inner_pts = []
    for r in np.linspace(0.0, r_floor, 8):
        inner_pts.append((r, h_floor))
    # Fillet from floor to flared wall
    for theta in np.linspace(0.0, math.pi / 4, 6):
        rf = r_floor + 0.010 * math.sin(theta)
        zf = h_floor + 0.010 * (1.0 - math.cos(theta))
        inner_pts.append((rf, zf))
    # Flared inner wall up to rim
    for t in np.linspace(0.1, 1.0, 10):
        r_w = inner_pts[-1][0] * (1 - t) + r_rim * t
        z_w = inner_pts[-1][1] * (1 - t) + h_rim * t
        inner_pts.append((r_w, z_w))
        
    # Rim top round
    rim_pts = []
    for theta in np.linspace(0, math.pi, 6):
        rr = r_rim + (wall_thick / 2.0) * math.sin(theta)
        zr = h_rim + (wall_thick / 2.0) * math.cos(theta)
        rim_pts.append((rr, zr))
        
    # Outer surface down:
    outer_pts = []
    r_outer_rim = r_rim + wall_thick
    r_outer_bottom = r_floor + wall_thick + 0.006
    for t in np.linspace(0.0, 1.0, 10):
        ro = r_outer_rim * (1 - t) + r_outer_bottom * t
        zo = h_rim * (1 - t) + (h_floor - bottom_thick + 0.008) * t
        outer_pts.append((ro, zo))
    # Outer bottom fillet
    for theta in np.linspace(math.pi / 4, 0.0, 6):
        ro = r_floor + 0.012 * math.sin(theta)
        zo = 0.0 + 0.008 * (1.0 - math.cos(theta))
        outer_pts.append((ro, zo))
    # Outer flat bottom to center (z = 0.0)
    for r in np.linspace(outer_pts[-1][0], 0.0, 8):
        outer_pts.append((r, 0.0))
        
    full_profile = inner_pts + rim_pts + outer_pts
    
    # Rotate around Z axis with N segments
    N = 48
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False)
    
    verts = []
    normals = []
    
    # Build surface of revolution
    n_prof = len(full_profile)
    for a in angles:
        ca, sa = math.cos(a), math.sin(a)
        for r, z in full_profile:
            verts.append((r * ca, r * sa, z))
            # Normal approximation
            nr = ca
            ns = sa
            nz = 0.5 if z > h_floor else (-1.0 if z <= 0.001 else 0.0)
            length = math.sqrt(nr*nr + ns*ns + nz*nz)
            normals.append((nr/length, ns/length, nz/length))
            
    # Handle: along -X axis (from pan wall to wrist)
    # Starts at x = -r_rim, extends to x = -r_rim - 0.19 (through the center of hand at x = -0.29)
    # Hand center in pan_head frame: x = 0.05 - 0.34 = -0.29, y = 0.0, z = 0.0
    h_start_x = - (r_rim + 0.002)  # -0.130m
    h_end_x = -0.325                # -0.325m (손 구체를 완전히 관통)
    h_y_w = 0.012
    h_z_w = 0.009
    
    handle_verts_start = len(verts)
    n_h_x = 16
    n_h_cross = 12
    h_x_vals = np.linspace(h_start_x, h_end_x, n_h_x)
    
    for i, x in enumerate(h_x_vals):
        # Slightly ergonomic shape: thicker in middle of grip
        t = i / (n_h_x - 1)
        thick = 1.0 + 0.25 * math.sin(t * math.pi)
        yw = h_y_w * thick
        zw = h_z_w * thick
        
        # 림(-0.13m, z=0.035)에서 손 그립부(-0.24m 이하, z=0.000)로 부드럽게 하향 전환
        if x > -0.24:
            u = (x - (-0.24)) / (h_start_x - (-0.24))  # 0 (-0.24에서) -> 1 (림 접합부에서)
            z_curr = 0.035 * 0.5 * (1.0 - math.cos(math.pi * u))
        else:
            # 손의 정중앙(Z=0)을 관통하는 수평 그립 구간
            z_curr = 0.0
            
        for phi in np.linspace(0, 2*math.pi, n_h_cross, endpoint=False):
            y = yw * math.cos(phi)
            z = z_curr + zw * math.sin(phi)
            verts.append((x, y, z))
            ny, nz = math.cos(phi), math.sin(phi)
            normals.append((0.0, ny, nz))
            
    # Generate faces
    faces = []
    # Body faces
    for i in range(N):
        next_i = (i + 1) % N
        for j in range(n_prof - 1):
            idx1 = i * n_prof + j + 1
            idx2 = i * n_prof + j + 2
            idx3 = next_i * n_prof + j + 2
            idx4 = next_i * n_prof + j + 1
            faces.append((idx1, idx2, idx3, idx4))
            
    # Handle faces
    for i in range(n_h_x - 1):
        for j in range(n_h_cross):
            next_j = (j + 1) % n_h_cross
            idx1 = handle_verts_start + i * n_h_cross + j + 1
            idx2 = handle_verts_start + (i + 1) * n_h_cross + j + 1
            idx3 = handle_verts_start + (i + 1) * n_h_cross + next_j + 1
            idx4 = handle_verts_start + i * n_h_cross + next_j + 1
            faces.append((idx1, idx2, idx3, idx4))
            
    # End cap for handle (centered at z = 0.0)
    cap_center = len(verts) + 1
    verts.append((h_end_x - 0.006, 0.0, 0.0))
    normals.append((-1.0, 0.0, 0.0))
    for j in range(n_h_cross):
        next_j = (j + 1) % n_h_cross
        idx1 = handle_verts_start + (n_h_x - 1) * n_h_cross + j + 1
        idx2 = cap_center
        idx3 = handle_verts_start + (n_h_x - 1) * n_h_cross + next_j + 1
        faces.append((idx1, idx2, idx3))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Photorealistic Chef Frying Pan Mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for vn in normals:
            f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
        for face in faces:
            if len(face) == 4:
                f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]} {face[3]}//{face[3]}\n")
            else:
                f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs", "assets", "frying_pan.obj")
    generate_frying_pan_obj(out)
    print("Generated frying pan mesh at:", out)
