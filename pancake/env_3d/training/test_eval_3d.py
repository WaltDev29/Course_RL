"""3D 팬케이크 모델 평가 전용 러너 (독립 실행 및 대시보드 연동)."""
import argparse
import csv
import json
import logging
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ENV3D_DIR = os.path.dirname(HERE)
sys.path.insert(0, ENV3D_DIR)
sys.path.insert(0, os.path.join(ENV3D_DIR, "envs"))
sys.path.insert(0, os.path.join(ENV3D_DIR, "utils"))

import matplotlib
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils.fileio import read_json, write_json_atomic
from envs.pancake_env_3d import PancakeFlipEnv3D

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have a glyph for.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph 8722.*")


def make_eval_env():
    return Monitor(PancakeFlipEnv3D(render_mode="rgb_array"))


def _render_eval_chart(results: list, total_target_eps: int, model_name: str, out_png: str) -> None:
    """테스트 진행 중 및 완료 시 에피소드별 보상/성공/스텝 수 차트를 실시간 갱신."""
    if not results:
        return
    n = len(results)
    n_flipped = sum(1 for r in results if r["flipped"])
    n_success = sum(1 for r in results if r["success"])
    avg_reward = sum(r["reward"] for r in results) / n

    from matplotlib.patches import Patch
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 9.5))

    eps_list = [r["episode"] for r in results]
    rews = [r["reward"] for r in results]
    colors = ["#2ecc71" if r["success"] else ("#f39c12" if r["flipped"] else "#95a5a6") for r in results]

    ax1.bar(eps_list, rews, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
    ax1.set_xlim(0.5, max(total_target_eps, n) + 0.5)
    ax1.margins(y=0.20)
    ax1.set_ylabel("에피소드 총 보상")
    status_str = f"진행 중 ({n}/{total_target_eps}회)" if n < total_target_eps else "평가 완료"
    ax1.set_title(f"3D 모델 테스트 [{status_str}]: {model_name}\n"
                  f"(성공률: {n_success/n*100:.1f}% | 뒤집기 달성률: {n_flipped/n*100:.1f}% | 평균 보상: {avg_reward:.1f})",
                  pad=8)
    ax1.grid(alpha=0.3, axis="y")
    legend_elements = [
        Patch(facecolor="#2ecc71", label=f"1초 안착 성공 ({n_success}회)"),
        Patch(facecolor="#f39c12", label=f"뒤집기만 달성 ({n_flipped - n_success}회)"),
        Patch(facecolor="#95a5a6", label=f"실패/낙하 ({n - n_flipped}회)")
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)

    lens = [r["length"] for r in results]
    ax2.plot(eps_list, lens, marker="o", color="#2980b9", linewidth=2, label="에피소드 지속 스텝 수 (최대 400)")
    ax2.axhline(y=400, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlim(0.5, max(total_target_eps, n) + 0.5)
    ax2.margins(y=0.15)
    ax2.set_xlabel("테스트 에피소드 번호")
    ax2.set_ylabel("스텝 수 (1스텝=0.02초)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.subplots_adjust(top=0.91, bottom=0.07, left=0.10, right=0.95, hspace=0.32)
    tmp_path = out_png.replace(".png", "_tmp.png")
    fig.savefig(tmp_path, format="png", dpi=120)
    plt.close(fig)
    try:
        os.replace(tmp_path, out_png)
    except OSError:
        pass


def run_evaluation(model_path: str, episodes: int = 20, session_dir: str = "",
                   sim_speed: float = 1.0) -> dict:
    if not session_dir:
        session_dir = os.path.join(ENV3D_DIR, "runs", f"eval_{int(time.time())}")
    os.makedirs(session_dir, exist_ok=True)
    write_json_atomic(os.path.join(session_dir, "control.json"), {"cmd": "run", "speed": sim_speed})

    if os.path.isfile(model_path):
        m_file = model_path
        parent_dir = os.path.dirname(model_path)
    else:
        parent_dir = model_path
        m_file = ""
        for cand in ("model.pt", "model.zip"):
            cp = os.path.join(model_path, cand)
            if os.path.exists(cp):
                m_file = cp
                break
        if not m_file and os.path.isdir(model_path):
            import glob
            pts = glob.glob(os.path.join(model_path, "model*.pt"))
            if pts:
                m_file = sorted(pts)[-1]
        if not m_file:
            m_file = os.path.join(model_path, "model.pt")

    if not os.path.exists(m_file):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {m_file}")

    venv = DummyVecEnv([make_eval_env])
    vec_path = os.path.join(parent_dir, "vecnormalize.pkl")
    if os.path.exists(vec_path):
        venv = VecNormalize.load(vec_path, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"[test_eval_3d] 정규화 통계 로드 완료: {vec_path}")

    model = SAC.load(m_file, env=venv, device="auto")
    print(f"[test_eval_3d] 3D 모델 로드 성공: {m_file}")

    csv_path = os.path.join(session_dir, "eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["episode", "reward", "steps", "success", "flipped", "dist_xy", "spill"])

    results = []
    fps_target = 50.0  # 기본 환경 주파수 50Hz
    frame_path = os.path.join(session_dir, "live_frame.jpg")

    for ep in range(1, episodes + 1):
        obs = venv.reset()
        done, total_r, step_cnt = False, 0.0, 0
        succ, flip, spill_info, final_dist = False, False, "none", 0.0

        while not done:
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = venv.step(action)
            total_r += float(r[0])
            step_cnt += 1
            info = infos[0]

            if info.get("flipped", False):
                flip = True
            if info.get("success", False):
                succ = True
            if "spill" in info:
                spill_info = info["spill"]
            if "dist_xy" in info:
                final_dist = info["dist_xy"]

            done = dones[0]

            # 3D 쿼터뷰 실시간 화면 갱신 (Windows 파일 락 방지)
            if step_cnt % 3 == 0:
                frame = venv.env_method("render")[0]
                if frame is not None:
                    tmp = frame_path + ".tmp"
                    try:
                        Image.fromarray(frame).save(tmp, format="JPEG", quality=80)
                        os.replace(tmp, frame_path)
                    except OSError:
                        pass

            # 제어 신호 확인 (정지 명령 및 실시간 속도 조절)
            ctl = read_json(os.path.join(session_dir, "control.json"))
            if ctl.get("cmd") == "stop":
                print("[test_eval_3d] 사용자에 의해 테스트 중단")
                done = True
                break
            cur_speed = ctl.get("speed", sim_speed)

            # 속도 페이싱 (실제 로봇 속도 관찰 지원)
            if cur_speed != "max" and cur_speed > 0:
                try:
                    target_dt = (1.0 / fps_target) / float(cur_speed)
                    sleep_rem = target_dt - (time.time() - t0)
                    if sleep_rem > 0:
                        time.sleep(sleep_rem)
                except Exception:
                    pass

        results.append(dict(episode=ep, reward=round(total_r, 2), length=step_cnt,
                            flipped=flip, success=succ, dist_xy=round(final_dist, 4), spill=spill_info))

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, round(total_r, 2), step_cnt, int(succ), int(flip), round(final_dist, 4), spill_info])

        n_done = len(results)
        s_rate = sum(1 for x in results if x["success"]) / n_done
        f_rate = sum(1 for x in results if x["flipped"]) / n_done
        r_mean = sum(x["reward"] for x in results) / n_done

        write_json_atomic(os.path.join(session_dir, "status.json"), {
            "state": "evaluating",
            "episodes": ep,
            "session_episodes": ep,
            "total_episodes": episodes,
            "reward_mean": round(r_mean, 2),
            "success_rate": round(s_rate, 3),
            "flip_rate": round(f_rate, 3),
            "best_name": os.path.basename(m_file),
        })
        print(f"[test_eval_3d] Ep {ep}/{episodes} | R: {total_r:.1f} | Flip: {flip} | Succ: {succ} | Spill: {spill_info}")

        # 매 에피소드 종료 즉시 실시간 검증 차트 생성 및 갱신 (대시보드 실시간 표출)
        chart_path = os.path.join(session_dir, "eval_chart.png")
        _render_eval_chart(results, episodes, os.path.basename(m_file), chart_path)

        if ctl.get("cmd") == "stop":
            break

    venv.close()

    # 최종 검증 차트 갱신 및 모델 폴더(parent_dir) 영구 보존
    n = len(results)
    if n > 0:
        n_flipped = sum(1 for r in results if r["flipped"])
        n_success = sum(1 for r in results if r["success"])
        avg_reward = sum(r["reward"] for r in results) / n

        chart_path = os.path.join(session_dir, "eval_chart.png")
        _render_eval_chart(results, episodes, os.path.basename(m_file), chart_path)

        # 모델 폴더(parent_dir)에도 검증 결과 및 그래프 영구 저장
        try:
            import shutil
            if os.path.isdir(parent_dir):
                shutil.copyfile(chart_path, os.path.join(parent_dir, "eval_chart.png"))
                shutil.copyfile(csv_path, os.path.join(parent_dir, "eval_results.csv"))
                print(f"[test_eval_3d] 검증 차트 및 결과가 모델 폴더에 영구 저장되었습니다: {parent_dir}")
        except Exception as e:
            print(f"[test_eval_3d] 모델 폴더 저장 실패: {e}")

        write_json_atomic(os.path.join(session_dir, "status.json"), {
            "state": "eval_done",
            "episodes": n,
            "session_episodes": n,
            "total_episodes": n,
            "success_rate": round(n_success / n, 3),
            "flip_rate": round(n_flipped / n, 3),
            "reward_mean": round(avg_reward, 2),
            "chart_path": chart_path,
        })
        return {"success_rate": n_success / n, "flip_rate": n_flipped / n, "reward_mean": avg_reward}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="평가할 3D 모델 경로")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--session", default="")
    ap.add_argument("--speed", default="1.0")
    args = ap.parse_args()

    sp = args.speed
    if sp != "max":
        try:
            sp = float(sp)
        except ValueError:
            sp = 1.0
    run_evaluation(args.model, args.episodes, args.session, sp)


if __name__ == "__main__":
    main()
