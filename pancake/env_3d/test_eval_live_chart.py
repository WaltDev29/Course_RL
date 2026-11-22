import os, sys, subprocess, time

ENV3D_DIR = r"d:\course\RL\pancake\env_3d"
model_p = os.path.join(ENV3D_DIR, "models", "20261121_144738_6205_succ88", "model.pt")
session_name = f"test_live_chart_{int(time.time())}"
session_dir = os.path.join(ENV3D_DIR, "runs", session_name)
chart_path = os.path.join(session_dir, "eval_chart.png")

cmd = [
    sys.executable,
    os.path.join(ENV3D_DIR, "training", "test_eval_3d.py"),
    "--model", model_p,
    "--session", session_dir,
    "--episodes", "3",
    "--speed", "max"
]

print("Starting eval process...")
proc = subprocess.Popen(cmd, cwd=ENV3D_DIR, env=dict(os.environ, PYTHONIOENCODING="utf-8"))

chart_updates = 0
last_mtime = 0.0

start_t = time.time()
while proc.poll() is None:
    if os.path.exists(chart_path):
        mtime = os.path.getmtime(chart_path)
        if mtime > last_mtime:
            last_mtime = mtime
            chart_updates += 1
            print(f"[{chart_updates}번째 실시간 차트 갱신 감지!] mtime: {mtime}")
    time.sleep(0.3)
    if time.time() - start_t > 30:
        break

proc.wait()
# 최종 확인
if os.path.exists(chart_path):
    mtime = os.path.getmtime(chart_path)
    if mtime > last_mtime:
        chart_updates += 1
        print(f"[{chart_updates}번째 최종 차트 갱신 감지!]")

print(f"Total live chart updates during test: {chart_updates}")
assert chart_updates >= 3, f"Expected at least 3 live updates, got {chart_updates}"
print("\n>>> ALL REAL-TIME EVAL CHART TESTS PASSED! <<<")
