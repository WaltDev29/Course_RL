"""3D 팬케이크 뒤집기 SAC 멀티프로세싱 학습 백엔드 워커."""
import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
import traceback
import warnings
from collections import deque
from datetime import datetime

import matplotlib
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have a glyph for.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph 8722.*")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV3D_DIR = os.path.dirname(HERE)
sys.path.insert(0, ENV3D_DIR)
sys.path.insert(0, os.path.join(ENV3D_DIR, "envs"))
sys.path.insert(0, os.path.join(ENV3D_DIR, "utils"))
sys.path.insert(0, os.path.join(ENV3D_DIR, "ui"))

from matplotlib.figure import Figure
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from ui import plotting
from utils.fileio import read_json, write_json_atomic
from envs.pancake_env_3d import PancakeFlipEnv3D

MODELS_DIR = os.path.join(ENV3D_DIR, "models")
RUNS_DIR = os.path.join(ENV3D_DIR, "runs")
BEST_MIN_GAP_EPISODES = 20
MAX_SPEED = 32


def make_env_fn(rank: int, render: bool = True):
    def _init():
        mode = "rgb_array" if render else None
        return Monitor(PancakeFlipEnv3D(render_mode=mode))
    return _init


class DashboardCallback3D(BaseCallback):
    def __init__(self, session_dir: str, target_episodes: int, start_episodes: int = 0,
                 initial_best_score: float = -float("inf"), window: int = 50,
                 initial_speed: float = 1.0, verbose: int = 0):
        super().__init__(verbose)
        self.session_dir = session_dir
        self.target_episodes = target_episodes
        self.window = window
        self.sim_speed = initial_speed
        self.control_path = os.path.join(session_dir, "control.json")
        self.status_path = os.path.join(session_dir, "status.json")
        self.ep_csv = os.path.join(session_dir, "episodes.csv")
        self.loss_csv = os.path.join(session_dir, "losses.csv")
        self.frame_path = os.path.join(session_dir, "live_frame.jpg")

        self.episodes_done = start_episodes      # 전체 누적 에피소드 수
        self.session_episodes = 0                # 이번 세션에서 새로 수행한 에피소드 수
        self.best_score = initial_best_score     # 이전 세션 최고 점수 승계
        self.best_save_ep = start_episodes - BEST_MIN_GAP_EPISODES
        self.best_dir = None
        self.last_status_t = 0.0
        self.last_frame_t = 0.0
        self.last_chart_t = 0.0
        self.start_wall_t = time.time()
        self.prev_steps = 0
        self.prev_time = self.start_wall_t

        self.ret_buf = deque(maxlen=window)
        self.succ_buf = deque(maxlen=window)
        self.flip_buf = deque(maxlen=window)
        self.render_enabled = True
        self.fig = Figure(figsize=(8.5, 11.5))
        self.should_stop = False

        self._init_csvs()

    def _init_csvs(self):
        if not os.path.exists(self.ep_csv):
            with open(self.ep_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(plotting.EP_FIELDS)
        if not os.path.exists(self.loss_csv):
            with open(self.loss_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(plotting.LOSS_FIELDS)

    def _log_loss(self):
        """SAC 신경망 학습 손실(critic, actor loss, ent_coef)을 losses.csv에 기록."""
        try:
            if hasattr(self.model, "logger") and self.model.logger is not None:
                v = self.model.logger.name_to_value
                if "train/critic_loss" in v:
                    with open(self.loss_csv, "a", newline="", encoding="utf-8") as fp:
                        csv.writer(fp).writerow([
                            self.num_timesteps,
                            round(float(v.get("train/actor_loss", 0.0)), 4),
                            round(float(v["train/critic_loss"]), 4),
                            round(float(v.get("train/ent_coef", 0.0)), 4)
                        ])
        except Exception:
            pass

    def _poll_control(self):
        data = read_json(self.control_path)
        if not data:
            return
        cmd = data.get("cmd", "run")
        sp = data.get("speed", self.sim_speed)
        self.sim_speed = MAX_SPEED if sp == "max" else float(sp)
        self.render_enabled = bool(data.get("render", True))

        if cmd == "stop":
            print("[DashboardCallback3D] 사용자 정지 명령 수신 -> 종료")
            self.should_stop = True
        elif cmd == "pause":
            print("[DashboardCallback3D] 일시정지 진입")
            self._write_status("paused")
            while True:
                time.sleep(0.1)
                d2 = read_json(self.control_path)
                if not d2:
                    continue
                c2 = d2.get("cmd")
                sp2 = d2.get("speed")
                if sp2 is not None:
                    self.sim_speed = MAX_SPEED if sp2 == "max" else float(sp2)
                if c2 == "run":
                    print("[DashboardCallback3D] 재개")
                    break
                if c2 == "stop":
                    self.should_stop = True
                    break

    def _write_status(self, state: str = "running"):
        now = time.time()
        fps = (self.num_timesteps - self.prev_steps) / max(1e-3, now - self.prev_time)
        self.prev_steps = self.num_timesteps
        self.prev_time = now

        r_mean = float(np.mean(self.ret_buf)) if self.ret_buf else 0.0
        s_rate = float(np.mean(self.succ_buf)) if self.succ_buf else 0.0
        f_rate = float(np.mean(self.flip_buf)) if self.flip_buf else 0.0

        write_json_atomic(self.status_path, {
            "state": state,
            "steps": int(self.num_timesteps),
            "episodes": int(self.episodes_done),
            "total_episodes": int(self.episodes_done),
            "session_episodes": int(self.session_episodes),
            "fps": round(fps, 1),
            "reward_mean": round(r_mean, 2),
            "success_rate": round(s_rate, 3),
            "flip_rate": round(f_rate, 3),
            "elapsed_sec": round(now - self.start_wall_t, 1),
            "best_score": round(self.best_score, 2) if self.best_score != -float("inf") else None,
            "speed": self.sim_speed,
            "render": self.render_enabled,
        })

    def _save_checkpoint(self, is_best: bool = False, save_to_models: bool = False, label: str = ""):
        now_dt = datetime.now()
        stamp = f"{now_dt:%Y%m%d_%H%M%S}_{self.episodes_done}"
        if is_best or save_to_models:
            tag = label if label else ("best" if is_best else "final")
            target_dir = os.path.join(MODELS_DIR, f"{stamp}_{tag}")
        else:
            target_dir = os.path.join(self.session_dir, "last")
        os.makedirs(target_dir, exist_ok=True)

        ep_num = self.episodes_done
        # 모델 저장 (기본 model.pt 및 에피소드 번호가 명시된 model_epXXXX.pt 병기)
        self.model.save(os.path.join(target_dir, "model.pt"))
        self.model.save(os.path.join(target_dir, f"model_ep{ep_num}.pt"))
        self.model.save(os.path.join(target_dir, "model.zip"))
        self.model.save(os.path.join(target_dir, f"model_ep{ep_num}.zip"))
        try:
            import torch
            torch.save(self.model.policy.state_dict(), os.path.join(target_dir, "policy.pt"))
            torch.save(self.model.policy.state_dict(), os.path.join(target_dir, f"policy_ep{ep_num}.pt"))
        except Exception as e:
            print(f"[DashboardCallback3D] policy.pt 저장 실패: {e}")

        # 정규화 통계 및 리플레이 버퍼 저장
        venv = self.training_env
        if isinstance(venv, VecNormalize):
            venv.save(os.path.join(target_dir, "vecnormalize.pkl"))
        try:
            self.model.save_replay_buffer(os.path.join(target_dir, "replay_buffer.pkl"))
        except Exception as e:
            print(f"[DashboardCallback3D] 리플레이 버퍼 저장 생략/실패: {e}")

        # 메타 정보 및 지표 복사
        r_mean = float(np.mean(self.ret_buf)) if self.ret_buf else 0.0
        s_rate = float(np.mean(self.succ_buf)) if self.succ_buf else 0.0
        f_rate = float(np.mean(self.flip_buf)) if self.flip_buf else 0.0
        meta = {
            "saved_at": now_dt.isoformat(),
            "episodes": int(self.episodes_done),
            "total_episodes": int(self.episodes_done),
            "session_episodes": int(self.session_episodes),
            "steps": int(self.num_timesteps),
            "reward_mean": round(r_mean, 2),
            "success_rate": round(s_rate, 3),
            "flip_rate": round(f_rate, 3),
            "is_best": bool(is_best),
            "model_file": f"model_ep{ep_num}.pt",
        }
        write_json_atomic(os.path.join(target_dir, "meta.json"), meta)
        for name in ("episodes.csv", "losses.csv"):
            src = os.path.join(self.session_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(target_dir, name))

        # 학습 곡선 시각화 이미지(curves.png) 자동 렌더링 및 저장
        try:
            ep_data = plotting.load_csv(os.path.join(target_dir, "episodes.csv"), plotting.EP_FIELDS)
            ls_data = plotting.load_csv(os.path.join(target_dir, "losses.csv"), plotting.LOSS_FIELDS)
            fig = Figure(figsize=(8.5, 11.5))
            plotting.draw_curves(fig, ep_data, ls_data, self.window)
            fig.savefig(os.path.join(target_dir, "curves.png"), dpi=110)
        except Exception as e:
            print(f"[DashboardCallback3D] curves.png 저장 실패: {e}")

        if is_best:
            # 같은 학습 세션에서 이전에 저장했던 하위 베스트 모델은 자동 삭제 (세션당 오직 1개의 최종 최고 모델만 유지)
            if self.best_dir and os.path.isdir(self.best_dir) and self.best_dir != target_dir:
                shutil.rmtree(self.best_dir, ignore_errors=True)
            self.best_dir = target_dir
            print(f"[DashboardCallback3D] ★ 베스트 모델 갱신: {target_dir} (model_ep{ep_num}.pt)")

    def _sync_final_metrics_to_best(self):
        """학습 세션 종료 시, 베스트 모델 폴더의 지표(CSV 및 curves.png)를 최종 완주 상태로 동기화."""
        if self.best_dir and os.path.isdir(self.best_dir):
            try:
                for name in ("episodes.csv", "losses.csv"):
                    src = os.path.join(self.session_dir, name)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(self.best_dir, name))
                ep_data = plotting.load_csv(os.path.join(self.best_dir, "episodes.csv"), plotting.EP_FIELDS)
                ls_data = plotting.load_csv(os.path.join(self.best_dir, "losses.csv"), plotting.LOSS_FIELDS)
                fig = Figure(figsize=(8.5, 11.5))
                plotting.draw_curves(fig, ep_data, ls_data, self.window)
                fig.savefig(os.path.join(self.best_dir, "curves.png"), dpi=110)
                print(f"[DashboardCallback3D] 베스트 모델 폴더에 최종 완주 지표 동기화 완료: {self.best_dir}")
            except Exception as e:
                print(f"[DashboardCallback3D] 베스트 폴더 최종 지표 동기화 실패: {e}")

    def _on_step(self) -> bool:
        if self.should_stop:
            return False

        now = time.time()
        # 1. 제어 신호 확인
        if now - self.last_status_t > 0.1:
            self._poll_control()
            if self.should_stop:
                return False
            self._write_status("running")
            self.last_status_t = now

        # 2. 실시간 프레임 갱신 (렌더링 켜져 있을 때만)
        if self.render_enabled and (self.n_calls % 6 == 0):
            try:
                frame = self.training_env.env_method("render", indices=[0])[0]
                if frame is not None:
                    from PIL import Image
                    tmp = self.frame_path + ".tmp"
                    Image.fromarray(frame).save(tmp, format="JPEG", quality=75)
                    os.replace(tmp, self.frame_path)
            except Exception:
                pass

        # 2-1. 신경망 Loss 기록 (매 200 스텝 주기)
        if self.num_timesteps % 200 == 0:
            self._log_loss()

        # 3. 에피소드 종료 지표 수집
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episodes_done += 1
                self.session_episodes += 1
                ep_r = float(info["episode"]["r"])
                ep_l = int(info["episode"]["l"])
                succ = 1.0 if info.get("success", False) else 0.0
                flip = 1.0 if info.get("flipped", False) else 0.0
                spill = info.get("spill", "none")

                self.ret_buf.append(ep_r)
                self.succ_buf.append(succ)
                self.flip_buf.append(flip)

                with open(self.ep_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([self.episodes_done, self.num_timesteps, round(ep_r, 2), ep_l, succ, flip, spill])

                # 베스트 모델 갱신 검사
                s_rate = float(np.mean(self.succ_buf))
                r_mean = float(np.mean(self.ret_buf))
                score = s_rate * 1000.0 + r_mean
                if (score > self.best_score + 1.0 and 
                    self.episodes_done - self.best_save_ep >= BEST_MIN_GAP_EPISODES and 
                    len(self.succ_buf) >= 10):
                    self.best_score = score
                    self.best_save_ep = self.episodes_done
                    self._save_checkpoint(is_best=True, label=f"succ{int(s_rate*100):02d}")

                if self.session_episodes >= self.target_episodes:
                    print(f"[DashboardCallback3D] 목표 에피소드({self.target_episodes}) 도달 -> 종료")
                    return False

        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="세션 디렉터리 이름")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--speed", default="max")
    ap.add_argument("--n-envs", type=int, default=4, help="병렬 환경 수")
    ap.add_argument("--resume", default="", help="이어학습할 체크포인트 폴더")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    session_dir = os.path.join(RUNS_DIR, args.session)
    os.makedirs(session_dir, exist_ok=True)
    write_json_atomic(os.path.join(session_dir, "control.json"), {"cmd": "run", "speed": args.speed, "render": True})

    n_envs = max(1, args.n_envs)
    print(f"[train_worker_3d] 3D 환경 병렬 실행: n_envs={n_envs}, 장치={args.device}")

    env_fns = [make_env_fn(i, render=True) for i in range(n_envs)]
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = VecNormalize(vec_cls(env_fns), norm_obs=True, norm_reward=False, clip_obs=10.0)

    start_eps = 0
    initial_best_score = -float("inf")

    # 이어학습 시 이전 지표 파일(episodes.csv, losses.csv) 및 메타 정보 승계
    if args.resume and os.path.isdir(args.resume):
        meta_path = os.path.join(args.resume, "meta.json")
        if os.path.exists(meta_path):
            try:
                meta = read_json(meta_path)
                start_eps = int(meta.get("total_episodes", meta.get("episodes", 0)))
                s_rate = float(meta.get("success_rate", 0.0))
                r_mean = float(meta.get("reward_mean", 0.0))
                initial_best_score = s_rate * 1000.0 + r_mean
                print(f"[train_worker_3d] 이전 학습 메타 승계: 누적 {start_eps} 에피소드, 베스트 기준점수 {initial_best_score:.1f}")
            except Exception as e:
                print(f"[train_worker_3d] meta.json 읽기 실패: {e}")

        for fn in ("episodes.csv", "losses.csv"):
            src = os.path.join(args.resume, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(session_dir, fn))
                print(f"[train_worker_3d] 이전 지표 승계 완료: {fn}")

    cb = DashboardCallback3D(session_dir, target_episodes=args.episodes,
                             start_episodes=start_eps, initial_best_score=initial_best_score,
                             window=args.window,
                             initial_speed=MAX_SPEED if args.speed == "max" else float(args.speed))

    m_file = ""
    if args.resume:
        for cand in ("model.pt", "model.zip"):
            cp = os.path.join(args.resume, cand)
            if os.path.exists(cp):
                m_file = cp
                break

    if m_file:
        print(f"[train_worker_3d] 이전 모델 로드 ({os.path.basename(m_file)}): {args.resume}")
        model = SAC.load(m_file, env=venv, device=args.device)
        vec_path = os.path.join(args.resume, "vecnormalize.pkl")
        if os.path.exists(vec_path):
            norm_stats = VecNormalize.load(vec_path, venv.venv)
            venv.obs_rms = norm_stats.obs_rms
            venv.training = True

        rb_path = os.path.join(args.resume, "replay_buffer.pkl")
        if os.path.exists(rb_path):
            try:
                model.load_replay_buffer(rb_path)
                print(f"[train_worker_3d] 리플레이 버퍼 로드 완료: {rb_path}")
            except Exception as e:
                print(f"[train_worker_3d] 리플레이 버퍼 로드 실패 (워밍업 재설정): {e}")
                model.learning_starts = model.num_timesteps + 1000
        else:
            model.learning_starts = model.num_timesteps + 1000
            print(f"[train_worker_3d] 리플레이 버퍼 없음 -> 버퍼 채우기 워밍업({model.learning_starts} 스텝까지) 설정")
    else:
        model = SAC(
            "MlpPolicy", venv, device=args.device,
            learning_rate=3e-4, buffer_size=300_000, batch_size=256,
            learning_starts=3_000, gamma=0.99, tau=0.005,
            train_freq=TrainFreq(4, TrainFrequencyUnit.STEP), gradient_steps=4,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            verbose=0
        )

    try:
        model.learn(total_timesteps=args.episodes * 400, callback=cb, reset_num_timesteps=False)
    except Exception as e:
        print(f"[train_worker_3d] 학습 예외 발생: {e}")
        traceback.print_exc()
    finally:
        # 1. 베스트 모델 폴더에 전체 학습 기간의 최종 그래프와 지표 동기화
        cb._sync_final_metrics_to_best()
        # 2. 세션 last 폴더에 최종 상태 저장 (이어학습용)
        cb._save_checkpoint(is_best=False)
        # 3. models 폴더에도 완주 체크포인트(final) 영구 보존
        cb._save_checkpoint(is_best=False, save_to_models=True, label="final")
        cb._write_status("stopped")
        venv.close()
        print("[train_worker_3d] 세션 종료 완료")


if __name__ == "__main__":
    main()
