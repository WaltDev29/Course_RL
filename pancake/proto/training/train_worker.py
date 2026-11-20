"""대시보드가 서브프로세스로 띄우는 학습 워커.

- 제어: <session>/control.json {"cmd": "run"|"pause"|"stop"}
- 상태: <session>/status.json, 지표: episodes.csv / losses.csv
- 베스트 저장: models/<YYYYMMDD_HHMMSS_에피소드>/ 에 이어학습에 필요한 전부(모델+리플레이버퍼+정규화+메타+지표+그래프)
"""
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

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO_DIR = os.path.dirname(HERE)
sys.path.insert(0, PROTO_DIR)
sys.path.insert(0, os.path.join(PROTO_DIR, "envs"))
sys.path.insert(0, os.path.join(PROTO_DIR, "utils"))
sys.path.insert(0, os.path.join(PROTO_DIR, "ui"))

from matplotlib.figure import Figure  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # noqa: E402

from ui import plotting  # noqa: E402
from utils.fileio import read_json, write_json_atomic  # noqa: E402
from envs.pancake_env import PancakeFlipEnv  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

MODELS_DIR = os.path.join(PROTO_DIR, "models")
RUNS_DIR = os.path.join(PROTO_DIR, "runs")
BEST_MIN_GAP_EPISODES = 20  # 리플레이 버퍼 포함 저장이 무거우므로 베스트 저장 최소 간격
MAX_SPEED = 32  # 학습 가속 배율 상한 (환경 N스텝당 업데이트 1회)


def make_env_fn(rank: int, render: bool = True):
    """병렬 환경 생성자 (SB3 VecEnv 규격에 맞춰 render_mode를 통일하되, 호출 시에만 0번 환경에서 렌더링)."""
    def _init():
        mode = "rgb_array" if render else None
        return Monitor(PancakeFlipEnv(render_mode=mode))
    return _init


def save_full_state(model: SAC, venv: VecNormalize, out_dir: str, session_dir: str, meta: dict,
                    window: int, save_rb: bool = True) -> None:
    """모델만이 아니라 이어학습에 필요한 상태를 저장한다 (.pt 및 .zip). save_rb=False 시 수백MB 버퍼 I/O 프리징 방지."""
    os.makedirs(out_dir, exist_ok=True)
    # 1) SB3 모델 저장 (.pt 및 호환용 .zip)
    model.save(os.path.join(out_dir, "model.pt"))
    model.save(os.path.join(out_dir, "model.zip"))
    # 2) 순수 PyTorch 신경망 가중치 (state_dict) 저장
    try:
        import torch
        torch.save(model.policy.state_dict(), os.path.join(out_dir, "policy.pt"))
    except Exception as e:
        print(f"[worker] policy.pt 저장 실패: {e}")

    # 베스트 저장 시에는 가벼운 가중치만 저장하고, 세션 종료(last/) 시에만 리플레이 버퍼를 전체 저장
    if save_rb:
        model.save_replay_buffer(os.path.join(out_dir, "replay_buffer.pkl"))
    venv.save(os.path.join(out_dir, "vecnormalize.pkl"))
    for fn in ("episodes.csv", "losses.csv"):
        src = os.path.join(session_dir, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, fn))
    fig = Figure(figsize=(9, 9))
    plotting.draw_curves(fig, plotting.load_csv(os.path.join(out_dir, "episodes.csv"), plotting.EP_FIELDS),
                         plotting.load_csv(os.path.join(out_dir, "losses.csv"), plotting.LOSS_FIELDS), window)
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=110)
    write_json_atomic(os.path.join(out_dir, "meta.json"), meta)


class ControlCallback(BaseCallback):
    def __init__(self, session_dir: str, target_episodes: int, start_episodes: int, window: int,
                 speed="max"):
        super().__init__()
        self.session_dir = session_dir
        self.target = target_episodes
        self.sim_speed = speed
        self.total_eps = start_episodes
        self.session_eps = 0
        self.window = window
        self.enable_render = True
        self.recent = deque(maxlen=window)  # (success, reward)
        self.best_score = (-1.0, -1e18)
        self.best_name = ""
        self.best_dir = ""
        self.last_best_ep = -10 ** 9
        self.state = "running"
        self.stop_flag = False
        self._last_status = 0.0
        self._t0 = time.time()
        self._steps0 = 0
        self._step_t0 = time.time()

    # ---------- 제어/상태 ----------
    def _read_control(self) -> dict:
        return read_json(os.path.join(self.session_dir, "control.json"))  # 실패 시 {} -> 현재 동작 유지

    def _read_cmd(self) -> str:
        return self._read_control().get("cmd", "run")

    def _apply_speed(self, speed) -> None:
        """실제 시뮬레이터 속도 조절 (신경망 업데이트는 1스텝당 1회로 고정하여 샘플 효율 100% 보존).
        - 'max': 딜레이 없이 CPU 최대 속도로 학습 (기본값)
        - 1.0: 실시간 1.0배속 (50Hz = 스텝당 20ms 대기하여 실제 로봇 속도로 관찰)
        - 2.0, 4.0: 2배속, 4배속 시뮬레이션
        """
        if speed == "max" or speed == 0 or speed == "MAX":
            self.sim_speed = "max"
        else:
            try:
                self.sim_speed = max(0.2, float(speed))
            except (TypeError, ValueError):
                self.sim_speed = "max"

    def _roll(self):
        if not self.recent:
            return 0.0, 0.0
        s = sum(x[0] for x in self.recent) / len(self.recent)
        r = sum(x[1] for x in self.recent) / len(self.recent)
        return s, r

    def _write_status(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_status < 1.0:
            return
        self._last_status = now
        s, r = self._roll()
        fps = (self.num_timesteps - self._steps0) / max(now - self._t0, 1e-6)
        write_json_atomic(os.path.join(self.session_dir, "status.json"), dict(
            state=self.state, session_episodes=self.session_eps, total_episodes=self.total_eps,
            target_episodes=self.target, steps=int(self.num_timesteps), fps=round(fps, 1),
            speed=self.sim_speed, rolling_success=s, rolling_reward=r, best_name=self.best_name,
            updated=now))

    def _poll_control(self) -> None:
        ctl = self._read_control()
        self._apply_speed(ctl.get("speed", self.sim_speed))
        self.enable_render = ctl.get("render", True)
        cmd = ctl.get("cmd", "run")
        if cmd == "stop":
            self.stop_flag = True
            return
        if cmd == "pause":
            self.state = "paused"
            self._write_status(force=True)
            while True:
                time.sleep(0.3)
                ctl = self._read_control()
                cmd = ctl.get("cmd", "run")
                if cmd == "stop":
                    self.stop_flag = True
                    break
                if cmd == "run":
                    self._apply_speed(ctl.get("speed", self.sim_speed))
                    self.enable_render = ctl.get("render", True)
                    break
            # 일시정지 시간은 FPS 계산에서 제외
            self._t0, self._steps0 = time.time(), self.num_timesteps
        self.state = "running"

    # ---------- 콜백 ----------
    def _on_training_start(self) -> None:
        self._steps0 = self.num_timesteps
        self._step_t0 = time.time()
        self._write_status(force=True)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "episode" in info:
                self._on_episode(info)
        if self.num_timesteps % 500 == 0:
            self._log_loss()
        if self.num_timesteps % 20 == 0:
            self._poll_control()
        # 대시보드 실시간 뷰용 프레임 저장 (대시보드 뷰 활성화 시에만 약 8~10 FPS로 저장하여 I/O 부하 최소화)
        if self.enable_render and self.num_timesteps % 6 == 0:
            self._save_live_frame()

        # 실제 시뮬레이터 속도 페이싱 (실시간 1배속 또는 지정 배속으로 관찰)
        if self.sim_speed != "max" and self.sim_speed > 0:
            target_dt = (1.0 / 50.0) / float(self.sim_speed)
            elapsed = time.time() - self._step_t0
            sleep_rem = target_dt - elapsed
            if sleep_rem > 0:
                time.sleep(sleep_rem)
        self._step_t0 = time.time()

        self._write_status()
        if self.session_eps >= self.target:
            self.stop_flag = True
        return not self.stop_flag

    def _save_live_frame(self) -> None:
        """대시보드 GUI에 보여줄 최신 시각화 프레임을 파일로 저장 (DummyVecEnv/SubprocVecEnv 호환)."""
        try:
            venv = self.model.env
            if hasattr(venv, "venv"):
                venv = venv.venv
            img = None
            if hasattr(venv, "env_method"):
                imgs = venv.env_method("render", indices=0)
                img = imgs[0] if imgs else None
            elif hasattr(venv, "envs"):
                img = venv.envs[0].unwrapped.render()

            if img is not None:
                from PIL import Image
                tmp = os.path.join(self.session_dir, "live_frame.jpg.tmp")
                path = os.path.join(self.session_dir, "live_frame.jpg")
                Image.fromarray(img).save(tmp, "JPEG", quality=75)
                os.replace(tmp, path)
        except Exception as e:
            # 첫 1회 디버깅용 로그
            if not hasattr(self, "_logged_render_err"):
                self._logged_render_err = True
                print(f"[worker] 렌더 프레임 저장 실패: {type(e).__name__}: {e}")

    def _log_loss(self) -> None:
        v = self.model.logger.name_to_value
        if "train/critic_loss" not in v:
            return
        with open(os.path.join(self.session_dir, "losses.csv"), "a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow([self.num_timesteps, v.get("train/actor_loss", 0.0),
                                     v["train/critic_loss"], v.get("train/ent_coef", 0.0)])

    def _on_episode(self, info: dict) -> None:
        self.total_eps += 1
        self.session_eps += 1
        success = 1 if info.get("success") else 0
        reward = float(info["episode"]["r"])
        self.recent.append((success, reward))
        with open(os.path.join(self.session_dir, "episodes.csv"), "a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow([self.total_eps, self.num_timesteps, reward, int(info["episode"]["l"]),
                                     success, int(bool(info.get("flipped"))), info.get("spill", "")])
        if len(self.recent) < self.window:
            return
        score = self._roll()
        if score > self.best_score and self.total_eps - self.last_best_ep >= BEST_MIN_GAP_EPISODES:
            self.best_score = score
            self.last_best_ep = self.total_eps
            self._save_best()

    def meta(self, tag: str) -> dict:
        s, r = self._roll()
        return dict(tag=tag, total_episodes=self.total_eps, num_timesteps=int(self.num_timesteps),
                    rolling_success=s, rolling_reward=r, window=self.window,
                    saved_at=datetime.now().isoformat(timespec="seconds"),
                    session=os.path.basename(self.session_dir))

    def _save_best(self) -> None:
        name = f"{datetime.now():%Y%m%d_%H%M%S}_{self.total_eps}"
        new_dir = os.path.join(MODELS_DIR, name)
        try:
            # 베스트 경신 시에는 수백MB 버퍼 I/O를 생략하여 0.05초 초고속 저장 (세션 종료 시 last/ 에 풀 저장)
            save_full_state(self.model, self.model.get_vec_normalize_env(), new_dir, self.session_dir,
                            self.meta("best"), self.window, save_rb=False)
        except Exception as e:  # 저장 실패가 학습 전체를 죽이지 않도록
            print(f"[worker] 베스트 저장 실패 {new_dir}: {type(e).__name__}: {e}")
            return
        # 같은 세션에서 이전에 만든 베스트는 갱신되었으므로 정리 (이전 세션 모델은 건드리지 않음)
        if self.best_dir and os.path.isdir(self.best_dir):
            shutil.rmtree(self.best_dir, ignore_errors=True)
        self.best_dir, self.best_name = new_dir, name
        print(f"[worker] 베스트 갱신: {name} score={self.best_score}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--episodes", type=int, default=2000, help="이번 세션에서 추가로 학습할 에피소드 수")
    ap.add_argument("--resume", default="", help="이어학습할 models/<이름> 경로")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--learning-starts", type=int, default=5000)
    ap.add_argument("--speed", default="max", help="시뮬레이터 속도: 1(실시간 50Hz), 2, 5, max(최대)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-envs", type=int, default=4, help="병렬 환경 수 (기본 4개, CPU 멀티코어 가속)")
    args = ap.parse_args()

    sp = args.speed
    if sp != "max":
        try:
            sp = float(sp)
        except ValueError:
            sp = "max"

    session_dir = os.path.join(RUNS_DIR, args.session)
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    write_json_atomic(os.path.join(session_dir, "control.json"), {"cmd": "run", "speed": sp, "render": True})

    start_eps, cb, model, venv = 0, None, None, None
    try:
        # CPU 멀티코어 병렬 환경 생성 (랭크 0만 렌더 활성화, 나머지는 헤드리스로 최적화)
        if args.n_envs > 1:
            env_fns = [make_env_fn(i, render=True) for i in range(args.n_envs)]
            raw_venv = SubprocVecEnv(env_fns)
        else:
            raw_venv = DummyVecEnv([make_env_fn(0, render=True)])
        venv = VecNormalize(raw_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if args.resume:
            r = args.resume
            with open(os.path.join(r, "meta.json"), encoding="utf-8") as fp:
                start_eps = int(json.load(fp)["total_episodes"])
            for fn in ("episodes.csv", "losses.csv"):  # 그래프가 이어지도록 이전 지표 승계
                if os.path.exists(os.path.join(r, fn)):
                    shutil.copyfile(os.path.join(r, fn), os.path.join(session_dir, fn))
            venv = VecNormalize.load(os.path.join(r, "vecnormalize.pkl"), venv.venv)
            venv.training = True
            m_path = os.path.join(r, "model.pt")
            if not os.path.exists(m_path):
                m_path = os.path.join(r, "model.zip")
            model = SAC.load(m_path, env=venv, device=args.device)
            rb = os.path.join(r, "replay_buffer.pkl")
            if os.path.exists(rb):
                model.load_replay_buffer(rb)
            else:  # 빈 버퍼로 학습하면 샘플링 오류가 나므로 워밍업 재설정
                model.learning_starts = model.num_timesteps + args.learning_starts
            print(f"[worker] 이어학습: {r} (누적 {start_eps} 에피소드, {model.num_timesteps} 스텝)")
        else:
            for fn, fields in (("episodes.csv", plotting.EP_FIELDS), ("losses.csv", plotting.LOSS_FIELDS)):
                with open(os.path.join(session_dir, fn), "w", newline="", encoding="utf-8") as fp:
                    csv.writer(fp).writerow(fields)
            model = SAC("MlpPolicy", venv, seed=args.seed, device=args.device, learning_rate=3e-4,
                         buffer_size=500_000, batch_size=256, learning_starts=args.learning_starts,
                         gamma=0.99, tau=0.005, train_freq=4, gradient_steps=4,
                         policy_kwargs=dict(net_arch=[256, 256]), verbose=0)
        cb = ControlCallback(session_dir, args.episodes, start_eps, args.window, sp)
        model.learn(10 ** 9, callback=cb, reset_num_timesteps=False, progress_bar=False)
        cb.state = "finished"
    except Exception as e:
        print(f"[worker] 학습 중 오류: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        if cb is not None:
            cb.state = "error"
    finally:
        # 종료 시 이어학습용 전체 상태를 세션의 last/ 에 저장
        if cb is not None and model is not None:
            try:
                save_full_state(model, venv, os.path.join(session_dir, "last"), session_dir,
                                cb.meta("last"), args.window)
                print(f"[worker] 최종 상태 저장: {os.path.join(session_dir, 'last')}")
            except Exception as e:
                print(f"[worker] 최종 저장 실패: {type(e).__name__}: {e}")
            cb._write_status(force=True)


if __name__ == "__main__":
    main()
