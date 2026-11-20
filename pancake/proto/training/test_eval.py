"""저장된 모델 평가/테스트 전용 러너 (독립 실행 및 대시보드 연동).

- 지정된 모델을 불러와 일정 에피소드 수만큼 평가 (deterministic=True)
- 실시간 뷰(live_frame.jpg) 프레임 갱신으로 대시보드에서 주행 관찰 가능
- 테스트 완료 시 에피소드별 보상, 플립/성공 여부 막대 차트(eval_results.png) 저장
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO_DIR = os.path.dirname(HERE)
sys.path.insert(0, PROTO_DIR)
sys.path.insert(0, os.path.join(PROTO_DIR, "envs"))
sys.path.insert(0, os.path.join(PROTO_DIR, "utils"))

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
from envs.pancake_env import PancakeFlipEnv

matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have a glyph for.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph 8722.*")


def make_eval_env():
    return Monitor(PancakeFlipEnv(render_mode="rgb_array"))


def run_evaluation(model_path: str, episodes: int = 20, session_dir: str = "",
                   sim_speed: float = 1.0) -> dict:
    """모델 평가 실행 함수 (모델 zip 파일 또는 폴더 경로 지원)."""
    if not session_dir:
        session_dir = os.path.join(PROTO_DIR, "runs", f"eval_{int(time.time())}")
    os.makedirs(session_dir, exist_ok=True)
    write_json_atomic(os.path.join(session_dir, "control.json"), {"cmd": "run", "speed": sim_speed})

    if os.path.isfile(model_path):
        m_file = model_path
        parent_dir = os.path.dirname(model_path)
    else:
        m_file = os.path.join(model_path, "model.pt")
        if not os.path.exists(m_file):
            m_file = os.path.join(model_path, "model.zip")
        parent_dir = model_path

    if not os.path.exists(m_file):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {m_file}")

    print(f"[test_eval] 모델 파일 로드: {m_file}")
    venv = VecNormalize(DummyVecEnv([make_eval_env]), norm_obs=True, norm_reward=False, training=False)
    vecnorm_path = os.path.join(parent_dir, "vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv.venv)
        venv.training = False
        venv.norm_reward = False
        print(f"[test_eval] 정규화 통계 로드 완료: {vecnorm_path}")

    model = SAC.load(m_file, env=venv, device="auto")

    results = []
    fps_target = 50.0  # 기본 환경 주파수 50Hz
    live_frame_path = os.path.join(session_dir, "live_frame.jpg")

    for ep in range(episodes):
        obs = venv.reset()
        ep_reward = 0.0
        ep_len = 0
        flipped = False
        success = False
        spill = ""

        while True:
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = venv.step(action)
            ep_reward += float(reward[0])
            ep_len += 1

            info = infos[0]
            if info.get("flipped"):
                flipped = True
            if info.get("success"):
                success = True
            if "spill" in info:
                spill = info["spill"]

            # 실시간 프레임 저장 (약 15fps)
            if ep_len % 3 == 0:
                raw_env = venv.venv.envs[0].unwrapped
                frame = raw_env.render()
                if frame is not None:
                    tmp = live_frame_path + ".tmp"
                    Image.fromarray(frame).save(tmp, "JPEG", quality=80)
                    try:
                        os.replace(tmp, live_frame_path)
                    except OSError:
                        pass

            # 제어 폴링 (속도 조절 및 중지)
            ctl = read_json(os.path.join(session_dir, "control.json"))
            if ctl.get("cmd") == "stop":
                print("[test_eval] 사용자에 의해 테스트 중단")
                break
            cur_speed = ctl.get("speed", sim_speed)

            # 속도 페이싱 (실제 시뮬레이터 속도 조절)
            if cur_speed != "max" and cur_speed > 0:
                target_dt = (1.0 / fps_target) / float(cur_speed)
                sleep_rem = target_dt - (time.time() - t0)
                if sleep_rem > 0:
                    time.sleep(sleep_rem)

            # 상태 업데이트
            write_json_atomic(os.path.join(session_dir, "status.json"), dict(
                state="evaluating", session_episodes=ep + 1, total_episodes=episodes,
                target_episodes=episodes, steps=ep_len, fps=round(1.0 / max(time.time() - t0, 1e-6), 1),
                speed=cur_speed, rolling_success=sum(r["success"] for r in results) / max(len(results), 1),
                rolling_reward=sum(r["reward"] for r in results) / max(len(results), 1),
                best_name=os.path.basename(m_file), updated=time.time()))

            if done[0]:
                break

        results.append(dict(episode=ep + 1, reward=round(ep_reward, 1), length=ep_len,
                            flipped=flipped, success=success, spill=spill))
        print(f"[test_eval] Ep {ep + 1}/{episodes} | Reward: {ep_reward:.1f} | "
              f"Flipped: {flipped} | Success: {success} | Spill: {spill or 'None'}")

    # 결과 통계
    n = len(results)
    if n == 0:
        return {}

    n_flipped = sum(1 for r in results if r["flipped"])
    n_success = sum(1 for r in results if r["success"])
    avg_reward = sum(r["reward"] for r in results) / n
    summary = dict(total_episodes=n, flip_rate=round(n_flipped / n * 100, 1),
                   success_rate=round(n_success / n * 100, 1), avg_reward=round(avg_reward, 1))

    # 결과 CSV 저장
    csv_path = os.path.join(session_dir, "eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["episode", "reward", "length", "flipped", "success", "spill"])
        w.writeheader()
        w.writerows(results)

    # 결과 막대 차트 생성 및 저장
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    eps = [r["episode"] for r in results]
    rews = [r["reward"] for r in results]
    colors = ["tab:green" if r["success"] else ("tab:orange" if r["flipped"] else "tab:gray") for r in results]

    ax1.bar(eps, rews, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("에피소드 총 보상")
    ax1.set_title(f"모델 테스트 결과: {os.path.basename(m_file)}\n"
                  f"(성공률: {summary['success_rate']}% | 뒤집기 달성률: {summary['flip_rate']}% | 평균 보상: {summary['avg_reward']})")
    ax1.grid(alpha=0.3, axis="y")
    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="tab:green", label=f"2초 유지 성공 ({n_success}회)"),
        Patch(facecolor="tab:orange", label=f"뒤집기만 달성 ({n_flipped - n_success}회)"),
        Patch(facecolor="tab:gray", label=f"실패/낙하 ({n - n_flipped}회)")
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)

    lens = [r["length"] for r in results]
    ax2.plot(eps, lens, marker="o", color="tab:blue", label="에피소드 지속 스텝 수 (최대 400)")
    ax2.set_xlabel("테스트 에피소드 번호")
    ax2.set_ylabel("스텝 수 (1스텝=0.02초)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    chart_path = os.path.join(session_dir, "eval_chart.png")
    fig.savefig(chart_path, dpi=110)
    plt.close(fig)

    # 모델 디렉토리에도 복사
    try:
        import shutil
        if os.path.isdir(parent_dir):
            shutil.copyfile(chart_path, os.path.join(parent_dir, "eval_chart.png"))
            shutil.copyfile(csv_path, os.path.join(parent_dir, "eval_results.csv"))
    except Exception:
        pass

    m_name = os.path.basename(m_file)
    write_json_atomic(os.path.join(session_dir, "status.json"), dict(
        state="eval_finished", session_episodes=n, total_episodes=n, target_episodes=n,
        steps=0, fps=0, speed=sim_speed, rolling_success=summary["success_rate"] / 100.0,
        rolling_reward=summary["avg_reward"], best_name=m_name,
        summary=summary, updated=time.time()))

    print(f"\n================ 테스트 완료 ================")
    print(f"테스트 모델: {m_name}")
    print(f"총 에피소드: {n}회")
    print(f"뒤집기 달성률: {summary['flip_rate']}%")
    print(f"2초 안착 성공률: {summary['success_rate']}%")
    print(f"평균 보상: {summary['avg_reward']}")
    print(f"결과 차트 저장: {chart_path}")
    print(f"============================================\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="평가할 모델 폴더 (models/<name>)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--session", default="")
    ap.add_argument("--speed", default="max", help="시뮬레이터 속도: 1(실시간 50Hz), 2, 5, max")
    args = ap.parse_args()

    sp = args.speed
    if sp != "max":
        try:
            sp = float(sp)
        except ValueError:
            sp = "max"

    run_evaluation(args.model, args.episodes, args.session, sp)


if __name__ == "__main__":
    main()
