import os, sys, shutil, subprocess, time
import json

ENV3D_DIR = r"d:\course\RL\pancake\env_3d"
sys.path.insert(0, ENV3D_DIR)
sys.path.insert(0, os.path.join(ENV3D_DIR, "ui"))
import plotting

models_dir = os.path.join(ENV3D_DIR, "models")
runs_dir = os.path.join(ENV3D_DIR, "runs")

# 최근 모델 폴더 찾기
candidates = [os.path.join(models_dir, d) for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
candidates.sort(key=os.path.getmtime, reverse=True)
if not candidates:
    print("No model found to test resume.")
    sys.exit(0)

target_model = candidates[0]
print(f"Target model for resume test: {target_model}")

# 임시 테스트 세션 실행
test_session = f"test_resume_{int(time.time())}"
test_session_dir = os.path.join(runs_dir, test_session)

cmd = [
    sys.executable,
    os.path.join(ENV3D_DIR, "training", "train_worker_3d.py"),
    "--session", test_session,
    "--episodes", "2",
    "--resume", target_model,
    "--n-envs", "1",
    "--speed", "max",
]

print("Launching train_worker_3d with resume...")
proc = subprocess.Popen(cmd, cwd=ENV3D_DIR, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
ret = proc.wait(timeout=60)
print(f"Worker exited with code: {ret}")

# 검증
ep_csv = os.path.join(test_session_dir, "episodes.csv")
meta_json = os.path.join(target_model, "meta.json")
with open(meta_json, "r", encoding="utf-8") as f:
    prev_meta = json.load(f)
prev_eps = int(prev_meta.get("total_episodes", prev_meta.get("episodes", 0)))

data = plotting.load_csv(ep_csv, plotting.EP_FIELDS)
eps_arr = data["episode"]
print(f"Previous episodes in meta: {prev_eps}")
print(f"Total episodes in new session CSV: {len(eps_arr)}")
print(f"First 3 episodes: {eps_arr[:3]}")
print(f"Last 3 episodes: {eps_arr[-3:]}")

assert len(eps_arr) == prev_eps + 2, f"Expected {prev_eps + 2} episodes, got {len(eps_arr)}"
assert int(eps_arr[-1]) == prev_eps + 2, f"Expected last episode {prev_eps + 2}, got {eps_arr[-1]}"
print("\n>>> ALL RESUME CHECKS PASSED SUCCESSFULLY! <<<")
