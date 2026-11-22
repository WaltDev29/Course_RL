import csv, os
import numpy as np

csv_path = r"d:\course\RL\pancake\env_3d\runs\run3d_20261121_162758\episodes.csv"
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

total = len(rows)
print(f"Total episodes: {total}")

# 1000 에피소드 단위로 성공률 및 플립률, 보상 추이 분석
chunk_size = 1000
for i in range(0, total, chunk_size):
    chunk = rows[i:i+chunk_size]
    succ = np.mean([float(r["success"]) for r in chunk]) * 100
    flip = np.mean([float(r["flipped"]) for r in chunk]) * 100
    rew = np.mean([float(r["reward"]) for r in chunk])
    spills = {}
    for r in chunk:
        sp = r["spill"]
        spills[sp] = spills.get(sp, 0) + 1
    print(f"Ep {i:5d} ~ {min(i+chunk_size, total):5d} | Succ: {succ:5.1f}% | Flip: {flip:5.1f}% | Rew: {rew:6.1f} | Spills: {spills}")

# 6000 에피소드 이후의 실패 원인 집중 분석
late = rows[6000:]
spills_late = {}
for r in late:
    sp = r["spill"]
    spills_late[sp] = spills_late.get(sp, 0) + 1
print("\nLate Spills (6000+):", spills_late)
