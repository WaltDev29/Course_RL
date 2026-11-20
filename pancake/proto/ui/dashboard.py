"""학습 관제 대시보드 (tkinter + matplotlib).

학습은 train_worker.py 서브프로세스로 실행하고, 파일(control/status/csv)로 제어·모니터링한다.
  - 중지  = 일시정지(재개 가능)
  - 종료  = 상태 저장 후 완전 종료
"""
import json
import logging
import os
import subprocess
import sys
import tkinter as tk
import warnings
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import matplotlib
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have a glyph for.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph 8722.*")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO_DIR = os.path.dirname(HERE)
sys.path.insert(0, PROTO_DIR)
sys.path.insert(0, os.path.join(PROTO_DIR, "utils"))
sys.path.insert(0, HERE)

from ui import plotting
from utils.fileio import read_json, write_json_atomic

MODELS_DIR = os.path.join(PROTO_DIR, "models")
RUNS_DIR = os.path.join(PROTO_DIR, "runs")
NEW_LABEL = "(새 학습)"
REFRESH_MS = 100  # 화면 모니터링 주기 (100ms = 10 FPS)

SPEED_LEVELS = [1.0, 2.0, 3.0, 5.0, "max"]
SPEED_NAMES = ["1.0x (실제속도)", "2.0x", "3.0x", "5.0x", "MAX (최대가속)"]


class Dashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("팬케이크 뒤집기 RL 통합 관제 대시보드")
        self.geometry("1440x940")
        self.proc = None
        self.session_dir = ""
        self.is_evaluating = False
        self._sizes = (-1, -1)
        self.window = 50
        self._cmd = "run"
        self._tick_count = 0
        self._current_photo = None  # GC 방지용 레퍼런스
        self._last_frame_mtime = 0.0

        self._build_ui()
        self._set_buttons("idle")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(REFRESH_MS, self._tick)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        # 1구역: [훈련] 새 학습 및 이어학습 (폴더 단위 불러오기)
        train_box = ttk.LabelFrame(top, text=" [훈련 제어] 새 학습 및 이어학습 (폴더 단위 불러오기) ", padding=6)
        train_box.pack(fill="x", pady=(0, 4))

        t_row1 = ttk.Frame(train_box)
        t_row1.pack(fill="x", pady=2)
        ttk.Label(t_row1, text="이어학습 폴더:", width=14).pack(side="left")
        self.train_model_dir_var = tk.StringVar(value=NEW_LABEL)
        self.train_entry = ttk.Entry(t_row1, textvariable=self.train_model_dir_var, state="readonly")
        self.train_entry.pack(side="left", fill="x", expand=True, padx=4)
        self.btn_browse_train = ttk.Button(t_row1, text="📂 폴더 선택...", command=self._browse_train_dir)
        self.btn_browse_train.pack(side="left", padx=2)
        self.btn_reset_train = ttk.Button(t_row1, text="🔄 새 학습으로 초기화", command=self._reset_train_dir)
        self.btn_reset_train.pack(side="left", padx=2)

        t_row2 = ttk.Frame(train_box)
        t_row2.pack(fill="x", pady=2)
        ttk.Label(t_row2, text="학습 에피소드:", width=14).pack(side="left")
        self.ep_var = tk.StringVar(value="2000")
        self.ep_spin = ttk.Spinbox(t_row2, from_=10, to=1_000_000, increment=100,
                                   textvariable=self.ep_var, width=8)
        self.ep_spin.pack(side="left", padx=(0, 10))

        ttk.Label(t_row2, text="병렬 환경 수:", width=11).pack(side="left")
        self.n_envs_var = tk.StringVar(value="4")
        self.n_envs_spin = ttk.Spinbox(t_row2, from_=1, to=16, increment=1,
                                       textvariable=self.n_envs_var, width=5)
        self.n_envs_spin.pack(side="left", padx=(0, 14))

        self.btn_start = ttk.Button(t_row2, text="▶ 학습 시작", command=self._start)
        self.btn_pause = ttk.Button(t_row2, text="⏸ 중지(일시정지)", command=self._toggle_pause)
        self.btn_stop = ttk.Button(t_row2, text="⏹ 종료(저장)", command=self._stop)
        for b in (self.btn_start, self.btn_pause, self.btn_stop):
            b.pack(side="left", padx=4)

        # 2구역: [테스트 / 평가] 모델 파일(.pt / .zip) 불러오기 및 실행
        eval_box = ttk.LabelFrame(top, text=" [테스트 / 평가] 저장된 모델 주행 테스트 (모델 .pt / .zip 파일 불러오기) ", padding=6)
        eval_box.pack(fill="x", pady=(0, 4))

        e_row1 = ttk.Frame(eval_box)
        e_row1.pack(fill="x", pady=2)
        ttk.Label(e_row1, text="테스트 모델 파일:", width=14).pack(side="left")
        self.eval_model_file_var = tk.StringVar(value="")
        self.eval_entry = ttk.Entry(e_row1, textvariable=self.eval_model_file_var, state="readonly")
        self.eval_entry.pack(side="left", fill="x", expand=True, padx=4)
        self.btn_browse_eval = ttk.Button(e_row1, text="📄 모델 파일(.pt/.zip) 선택...", command=self._browse_eval_file)
        self.btn_browse_eval.pack(side="left", padx=2)
        self.btn_reset_eval = ttk.Button(e_row1, text="❌ 비우기", command=self._reset_eval_file)
        self.btn_reset_eval.pack(side="left", padx=2)

        e_row2 = ttk.Frame(eval_box)
        e_row2.pack(fill="x", pady=2)
        ttk.Label(e_row2, text="테스트 회수:", width=14).pack(side="left")
        self.test_ep_var = tk.StringVar(value="20")
        self.test_ep_spin = ttk.Spinbox(e_row2, from_=1, to=500, increment=5,
                                        textvariable=self.test_ep_var, width=10)
        self.test_ep_spin.pack(side="left", padx=(0, 16))

        self.btn_eval = ttk.Button(e_row2, text="🧪 모델 테스트 실행", command=self._start_eval)
        self.btn_eval.pack(side="left", padx=4)

        # 3구역: 속도 슬라이더 및 상태 표시
        ctrl_bar = ttk.Frame(top, padding=2)
        ctrl_bar.pack(fill="x", pady=(2, 2))
        ttk.Label(ctrl_bar, text="시뮬레이터 속도:").pack(side="left")
        self.speed_idx_var = tk.IntVar(value=4)  # 기본값: MAX (인덱스 4)
        self.speed_label = ttk.Label(ctrl_bar, text=SPEED_NAMES[4], width=16)
        self.speed_scale = ttk.Scale(ctrl_bar, from_=0, to=4, orient="horizontal", length=300,
                                     command=self._on_speed_drag)
        self.speed_scale.set(4)
        self.speed_scale.pack(side="left", padx=6)
        self.speed_label.pack(side="left")
        ttk.Label(ctrl_bar, text="(기본 MAX=최대 가속, 1.0x=실제 50Hz 로봇 속도로 관찰)",
                  foreground="gray").pack(side="left", padx=8)
        self.speed_scale.bind("<ButtonRelease-1>", lambda e: self._apply_speed())

        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(self, textvariable=self.status_var, font=("Malgun Gothic", 10), padding=(10, 2),
                  justify="left").pack(fill="x")

        # 2단 분할 (좌측: MuJoCo 실시간 뷰, 우측: 지표 그래프)
        content = ttk.PanedWindow(self, orient="horizontal")
        content.pack(fill="both", expand=True, padx=6, pady=4)

        # 좌측: MuJoCo 실시간 화면
        left_frame = ttk.LabelFrame(content, text=" 실시간 시뮬레이션 (MuJoCo 횡 뷰) ", padding=6)
        content.add(left_frame, weight=1)

        opt_frame = ttk.Frame(left_frame)
        opt_frame.pack(fill="x", pady=(0, 4))
        self.live_view_var = tk.BooleanVar(value=True)
        self.chk_live = ttk.Checkbutton(opt_frame, text="실시간 화면 켜기 (끄면 3~4배 터보 가속)",
                                        variable=self.live_view_var, command=self._on_toggle_render)
        self.chk_live.pack(side="left")

        self.view_label = ttk.Label(left_frame, text="\n\n학습이나 테스트가 시작되면\nMuJoCo 시뮬레이션 화면이\n여기에 실시간 표시됩니다.",
                                    anchor="center", justify="center", font=("Malgun Gothic", 11),
                                    background="#1e1e1e", foreground="#cccccc")
        self.view_label.pack(fill="both", expand=True)

        # 우측: 지표 그래프 4종 (성공률 / 보상 / 에피소드 스텝 수 / Loss)
        right_frame = ttk.LabelFrame(content, text=" 학습 지표 그래프 (성공률 / 보상 / 스텝 수 / Loss) ", padding=4)
        content.add(right_frame, weight=1)

        self.fig = Figure(figsize=(7, 8.2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _set_buttons(self, mode: str) -> None:
        on, off = "normal", "disabled"
        self.btn_start.config(state=on if mode == "idle" else off)
        self.btn_pause.config(state=on if mode in ("running", "paused") else off)
        self.btn_stop.config(state=on if mode in ("running", "paused", "evaluating") else off)
        self.btn_eval.config(state=on if mode == "idle" else off)
        self.btn_browse_train.config(state=on if mode == "idle" else off)
        self.btn_reset_train.config(state=on if mode == "idle" else off)
        self.btn_browse_eval.config(state=on if mode == "idle" else off)
        self.btn_reset_eval.config(state=on if mode == "idle" else off)
        self.btn_pause.config(text="▶ 재개" if mode == "paused" else "⏸ 중지(일시정지)")

    # ---------- 파일 / 폴더 브라우저 ----------
    def _browse_train_dir(self) -> None:
        """사용자가 직접 이어학습할 모델 폴더를 선택."""
        initial = MODELS_DIR if os.path.exists(MODELS_DIR) else HERE
        path = filedialog.askdirectory(title="이어학습할 모델 폴더 선택", initialdir=initial)
        if path:
            has_model = os.path.exists(os.path.join(path, "model.pt")) or os.path.exists(os.path.join(path, "model.zip"))
            if not has_model:
                messagebox.showwarning("주의", f"선택한 폴더에 'model.pt' 또는 'model.zip' 파일이 없습니다:\n{path}\n\n정상적인 체크포인트 폴더인지 확인해 주세요.")
            self.train_model_dir_var.set(path)

    def _reset_train_dir(self) -> None:
        self.train_model_dir_var.set(NEW_LABEL)

    def _browse_eval_file(self) -> None:
        """사용자가 직접 테스트할 모델 파일(.pt 또는 .zip)을 선택."""
        initial = MODELS_DIR if os.path.exists(MODELS_DIR) else HERE
        path = filedialog.askopenfilename(
            title="테스트할 모델 파일(.pt / .zip) 선택",
            initialdir=initial,
            filetypes=[
                ("PyTorch Model (*.pt, *.zip)", "*.pt;*.zip"),
                ("PyTorch Model (*.pt)", "*.pt"),
                ("Zip Model (*.zip)", "*.zip"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.eval_model_file_var.set(path)

    def _reset_eval_file(self) -> None:
        self.eval_model_file_var.set("")

    # ---------- 제어 ----------
    def _current_speed(self):
        try:
            idx = int(round(float(self.speed_scale.get())))
            idx = max(0, min(len(SPEED_LEVELS) - 1, idx))
            return SPEED_LEVELS[idx]
        except (tk.TclError, ValueError):
            return "max"

    def _on_speed_drag(self, _val=None) -> None:
        try:
            idx = int(round(float(self.speed_scale.get())))
            idx = max(0, min(len(SPEED_LEVELS) - 1, idx))
            self.speed_label.config(text=SPEED_NAMES[idx])
        except (tk.TclError, ValueError):
            pass

    def _on_toggle_render(self) -> None:
        """실시간 화면 렌더링을 끄면 워커가 렌더 I/O를 완전 중단하여 3~4배 터보 가속."""
        if not self.live_view_var.get():
            self.view_label.configure(image="", text="⚡ [터보 가속 모드 가동 중]\n\n화면 렌더링이 완전히 비활성화되어\nCPU/GPU 자원을 오직 신경망 학습에만 집중합니다.\n(다시 보시려면 체크박스를 켜세요)")
        if self.session_dir and self.proc is not None and self.proc.poll() is None:
            self._send(self._cmd)

    def _apply_speed(self) -> None:
        """슬라이더 값을 정수로 스냅하고 실행 중인 워커/테스터에 전달 (명령 상태는 유지)."""
        try:
            idx = int(round(float(self.speed_scale.get())))
            idx = max(0, min(len(SPEED_LEVELS) - 1, idx))
            self.speed_scale.set(idx)
            self.speed_label.config(text=SPEED_NAMES[idx])
        except (tk.TclError, ValueError):
            pass
        if self.session_dir and self.proc is not None and self.proc.poll() is None:
            self._send(self._cmd)

    def _send(self, cmd: str) -> None:
        self._cmd = cmd
        ok = write_json_atomic(os.path.join(self.session_dir, "control.json"),
                               {"cmd": cmd, "speed": self._current_speed(),
                                "render": bool(self.live_view_var.get())})
        if not ok:
            messagebox.showerror("제어 실패", f"명령 전송 실패({cmd}). 다시 시도해 주세요.")

    def _start(self) -> None:
        try:
            episodes = int(self.ep_var.get())
            if episodes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("입력 오류", "에피소드 수는 1 이상의 정수여야 합니다.")
            return

        sid = f"{datetime.now():%Y%m%d_%H%M%S}"
        self.session_dir = os.path.join(RUNS_DIR, sid)
        os.makedirs(self.session_dir, exist_ok=True)
        self.is_evaluating = False

        n_envs = self.n_envs_var.get().strip() or "4"
        cmd = [sys.executable, os.path.join(PROTO_DIR, "training", "train_worker.py"), "--session", sid,
               "--episodes", str(episodes), "--window", str(self.window),
               "--speed", str(self._current_speed()), "--n-envs", n_envs]

        resume_dir = self.train_model_dir_var.get().strip()
        if resume_dir and resume_dir != NEW_LABEL:
            if not os.path.isdir(resume_dir):
                messagebox.showerror("오류", f"선택한 이어학습 폴더가 존재하지 않습니다:\n{resume_dir}")
                return
            cmd += ["--resume", resume_dir]

        try:
            log = open(os.path.join(self.session_dir, "log.txt"), "w", encoding="utf-8")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=PROTO_DIR, env=env)
        except OSError as e:
            messagebox.showerror("시작 실패", f"워커 실행 실패: {e}")
            return
        self._sizes = (-1, -1)
        self._set_buttons("running")
        self.status_var.set(f"시작 중... (세션 {sid})")

    def _start_eval(self) -> None:
        model_file = self.eval_model_file_var.get().strip()
        if not model_file:
            messagebox.showwarning("모델 파일 선택", "테스트할 모델 파일(.pt 또는 .zip)을 먼저 선택해 주세요.")
            return
        if not os.path.isfile(model_file):
            messagebox.showerror("오류", f"선택한 모델 파일이 존재하지 않습니다:\n{model_file}")
            return
        try:
            episodes = int(self.test_ep_var.get())
            if episodes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("입력 오류", "테스트 회수는 1 이상의 정수여야 합니다.")
            return

        sid = f"eval_{datetime.now():%Y%m%d_%H%M%S}"
        self.session_dir = os.path.join(RUNS_DIR, sid)
        os.makedirs(self.session_dir, exist_ok=True)
        self.is_evaluating = True

        cur_sp = self._current_speed()
        cmd = [sys.executable, os.path.join(PROTO_DIR, "training", "test_eval.py"),
               "--model", model_file,
               "--session", self.session_dir,
               "--episodes", str(episodes),
               "--speed", str(cur_sp)]
        try:
            log = open(os.path.join(self.session_dir, "eval_log.txt"), "w", encoding="utf-8")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=PROTO_DIR, env=env)
        except OSError as e:
            messagebox.showerror("시작 실패", f"테스터 실행 실패: {e}")
            self.is_evaluating = False
            return

        self._sizes = (-1, -1)
        self._set_buttons("evaluating")
        self.status_var.set(f"테스트 평가 시작... (모델: {os.path.basename(model_file)}, 총 {episodes}회, 세션: {sid})")

    def _toggle_pause(self) -> None:
        paused = self.btn_pause.cget("text").startswith("▶")
        self._send("run" if paused else "pause")
        self._set_buttons("running" if paused else "paused")

    def _stop(self) -> None:
        if messagebox.askyesno("종료", "학습을 종료하고 현재 상태를 저장할까요?"):
            self._send("stop")
            self.btn_pause.config(state="disabled")
            self.btn_stop.config(state="disabled")
            self.status_var.set("종료 및 저장 중... (잠시 기다려 주세요)")

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("확인", "프로세스가 진행 중입니다. 종료하고 창을 닫을까요?"):
                return
            self._send("stop")
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                messagebox.showwarning("경고", "지연으로 인해 프로세스를 강제 종료합니다.")
                self.proc.kill()
        self.destroy()

    # ---------- 모니터링 ----------
    def _read_status(self) -> dict:
        return read_json(os.path.join(self.session_dir, "status.json"))

    def _tick(self) -> None:
        try:
            self._tick_count += 1
            if self.session_dir:
                if self.live_view_var.get():
                    self._update_live_frame()
                if self._tick_count % 10 == 0:  # 약 1초마다 상태 및 그래프 갱신
                    self._update_status()
                    self._update_plots()
        except Exception as e:
            self.status_var.set(f"모니터링 오류: {type(e).__name__}: {e}")
        finally:
            self.after(REFRESH_MS, self._tick)

    def _update_live_frame(self) -> None:
        frame_path = os.path.join(self.session_dir, "live_frame.jpg")
        if not os.path.exists(frame_path):
            return
        try:
            mtime = os.path.getmtime(frame_path)
            if mtime <= self._last_frame_mtime:
                return
            self._last_frame_mtime = mtime
            img = Image.open(frame_path)
            img = img.resize((600, 450), Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self._current_photo = photo
            self.view_label.configure(image=photo, text="")
        except Exception:
            pass

    def _update_status(self) -> None:
        st = self._read_status()
        alive = self.proc is not None and self.proc.poll() is None
        if st:
            if self.is_evaluating or st.get("state") == "evaluating":
                self.status_var.set(
                    f"상태: [테스트 평가 진행 중]   |   에피소드: {st['session_episodes']}/{st['target_episodes']}   |   "
                    f"스텝: {st['steps']}   |   {st['fps']} FPS (속도: {st.get('speed', 'max')})\n"
                    f"진행 성공률: {st['rolling_success'] * 100:.1f}%   |   평균 보상: {st['rolling_reward']:.1f}   |   "
                    f"대상 모델: {st['best_name']}")
            else:
                self.status_var.set(
                    f"상태: {st['state']}   |   에피소드: 이번 {st['session_episodes']}/{st['target_episodes']}"
                    f" (누적 {st['total_episodes']})   |   스텝: {st['steps']:,}   |   {st['fps']} FPS"
                    f" (실시간 대비 {st['fps'] / 50:.1f}배속, 속도 {st.get('speed', 'max')})\n"
                    f"최근 {self.window}ep 성공률: {st['rolling_success'] * 100:.1f}%   |   "
                    f"평균 보상: {st['rolling_reward']:.1f}   |   베스트 모델: {st['best_name'] or '-'}")
        if self.proc is not None and not alive:
            self._set_buttons("idle")
            if self.is_evaluating:
                summary = st.get("summary") if st else None
                if summary:
                    self.status_var.set(
                        f"[테스트 완료] 모델: {st.get('best_name', '-')} | "
                        f"성공률: {summary['success_rate']}% | 뒤집기 달성률: {summary['flip_rate']}% | 평균 보상: {summary['avg_reward']}\n"
                        f"결과 차트(eval_chart.png) 및 CSV가 저장되었습니다.")
                else:
                    self.status_var.set("테스트 프로세스 종료됨.")
                self.is_evaluating = False
            else:
                if st:
                    self.status_var.set(self.status_var.get() + f"\n[{st['state']}] 워커 종료됨")
            self.proc = None
        elif alive and st and not self.is_evaluating:
            self._set_buttons("paused" if st["state"] == "paused" else "running")

    def _update_plots(self) -> None:
        if self.is_evaluating:
            chart_path = os.path.join(self.session_dir, "eval_chart.png")
            if os.path.exists(chart_path):
                mtime = os.path.getmtime(chart_path)
                if mtime != getattr(self, "_last_eval_chart_mtime", 0.0):
                    self._last_eval_chart_mtime = mtime
                    try:
                        self.fig.clear()
                        ax = self.fig.add_subplot(111)
                        img = Image.open(chart_path)
                        ax.imshow(img)
                        ax.axis("off")
                        self.fig.tight_layout()
                        self.canvas.draw_idle()
                    except Exception:
                        pass
            return

        ep_path = os.path.join(self.session_dir, "episodes.csv")
        ls_path = os.path.join(self.session_dir, "losses.csv")
        sizes = tuple(os.path.getsize(p) if os.path.exists(p) else 0 for p in (ep_path, ls_path))
        if sizes == self._sizes:
            return  # 변화 없으면 다시 그리지 않음
        self._sizes = sizes
        plotting.draw_curves(self.fig, plotting.load_csv(ep_path, plotting.EP_FIELDS),
                             plotting.load_csv(ls_path, plotting.LOSS_FIELDS), self.window)
        self.canvas.draw_idle()


if __name__ == "__main__":
    Dashboard().mainloop()
