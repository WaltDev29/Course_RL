"""3D 팬케이크 뒤집기 관제 대시보드 (tkinter + matplotlib).

프로토타입 대시보드와 완전히 동일한 UI 레이아웃을 채택하여,
- 상단: 1구역(훈련 제어), 2구역(테스트/평가), 3구역(속도 슬라이더), 상태 표시줄
- 하단: 좌측(3인칭 쿼터뷰 실시간 시뮬레이션 화면 + 터보 가속 토글), 우측(실시간 학습 곡선 4종 그래프)
로 구성됩니다.
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
ENV3D_DIR = os.path.dirname(HERE)
sys.path.insert(0, ENV3D_DIR)
sys.path.insert(0, os.path.join(ENV3D_DIR, "utils"))
sys.path.insert(0, HERE)

from ui import plotting
from utils.fileio import read_json, write_json_atomic

MODELS_DIR = os.path.join(ENV3D_DIR, "models")
RUNS_DIR = os.path.join(ENV3D_DIR, "runs")
NEW_LABEL = "(새 학습)"
REFRESH_MS = 100

SPEED_LEVELS = [1.0, 2.0, 3.0, 5.0, "max"]
SPEED_NAMES = ["1.0x (실제속도)", "2.0x", "3.0x", "5.0x", "MAX (최대가속)"]


class Dashboard3D(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("완전 3D 팬케이크 뒤집기 RL 관제 대시보드 (8-DOF)")
        self.geometry("1440x940")
        self.proc = None
        self.session_dir = ""
        self.is_evaluating = False
        self._sizes = (-1, -1)
        self.window = 50
        self._cmd = "run"
        self._tick_count = 0
        self._current_photo = None
        self._last_frame_mtime = 0.0

        self._build_ui()
        self._set_buttons("idle")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(REFRESH_MS, self._tick)

    # ---------- UI (프로토타입과 동일한 3구역 상단 + 좌우 2단 분할 레이아웃) ----------
    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        # 1구역: [훈련 제어] 새 학습 및 이어학습 (폴더 단위 불러오기)
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

        # 2구역: [테스트 / 평가] 모델 파일(.zip) 불러오기 및 실행
        eval_box = ttk.LabelFrame(top, text=" [테스트 / 평가] 저장된 모델 주행 테스트 (모델 .zip 파일 불러오기) ", padding=6)
        eval_box.pack(fill="x", pady=(0, 4))

        e_row1 = ttk.Frame(eval_box)
        e_row1.pack(fill="x", pady=2)
        ttk.Label(e_row1, text="테스트 모델 파일:", width=14).pack(side="left")
        self.eval_model_file_var = tk.StringVar(value="")
        self.eval_entry = ttk.Entry(e_row1, textvariable=self.eval_model_file_var, state="readonly")
        self.eval_entry.pack(side="left", fill="x", expand=True, padx=4)
        self.btn_browse_eval = ttk.Button(e_row1, text="📄 모델 파일(.zip) 선택...", command=self._browse_eval_file)
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

        self.btn_eval = ttk.Button(e_row2, text="🧪 3D 모델 테스트 실행", command=self._start_eval)
        self.btn_eval.pack(side="left", padx=4)

        # 3구역: 속도 슬라이더 및 상태 표시
        ctrl_bar = ttk.Frame(top, padding=2)
        ctrl_bar.pack(fill="x", pady=(2, 2))
        ttk.Label(ctrl_bar, text="시뮬레이터 속도:").pack(side="left")
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

        # 2단 분할 (좌측: MuJoCo 3D 쿼터뷰 실시간 뷰, 우측: 지표 그래프)
        content = ttk.PanedWindow(self, orient="horizontal")
        content.pack(fill="both", expand=True, padx=6, pady=4)

        # 좌측: MuJoCo 실시간 화면
        left_frame = ttk.LabelFrame(content, text=" 실시간 시뮬레이션 (MuJoCo 3D 쿼터 뷰) ", padding=6)
        content.add(left_frame, weight=1)

        opt_frame = ttk.Frame(left_frame)
        opt_frame.pack(fill="x", pady=(0, 4))
        self.live_view_var = tk.BooleanVar(value=True)
        self.chk_live = ttk.Checkbutton(opt_frame, text="실시간 화면 켜기 (끄면 3~4배 터보 가속)",
                                        variable=self.live_view_var, command=self._on_toggle_render)
        self.chk_live.pack(side="left")

        self.view_label = ttk.Label(left_frame, text="\n\n3D 학습이나 테스트가 시작되면\nMuJoCo 3D 쿼터뷰 화면이\n여기에 실시간 표시됩니다.",
                                    anchor="center", justify="center", font=("Malgun Gothic", 11),
                                    background="#1e1e1e", foreground="#cccccc")
        self.view_label.pack(fill="both", expand=True)

        # 우측: 지표 그래프 4종 (성공률 / 보상 / 스텝 수 / Loss)
        right_frame = ttk.LabelFrame(content, text=" 3D 학습 지표 그래프 (성공률 / 보상 / 스텝 수 / Loss) ", padding=4)
        content.add(right_frame, weight=1)

        self.fig = Figure(figsize=(7.5, 9.5))
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

    def _browse_train_dir(self) -> None:
        path = filedialog.askdirectory(title="이어학습할 3D 모델 폴더 선택", initialdir=MODELS_DIR)
        if path:
            has_model = any(os.path.exists(os.path.join(path, f)) for f in ("model.pt", "model.zip"))
            if not has_model:
                messagebox.showwarning("모델 파일 없음", f"선택한 폴더 안에 model.pt 또는 model.zip이 없습니다:\n{path}\n올바른 체크포인트 폴더를 선택해주세요.")
                return
            self.train_model_dir_var.set(path)
            meta_path = os.path.join(path, "meta.json")
            extra_msg = ""
            if os.path.exists(meta_path):
                meta = read_json(meta_path)
                eps = meta.get("total_episodes", meta.get("episodes", 0))
                s_rate = meta.get("success_rate", 0) * 100
                extra_msg = f" (누적 {eps:,} ep, 성공률 {s_rate:.1f}%)"
            self.status_var.set(f"이어학습 대상 선택됨: {os.path.basename(path)}{extra_msg}")

    def _reset_train_dir(self) -> None:
        self.train_model_dir_var.set(NEW_LABEL)
        self.status_var.set("새 학습 모드로 설정되었습니다.")

    def _browse_eval_file(self) -> None:
        path = filedialog.askopenfilename(
            title="테스트할 3D 모델 파일(.pt / .zip) 선택",
            initialdir=MODELS_DIR,
            filetypes=[
                ("Model Files (*.pt;*.zip)", "*.pt;*.zip"),
                ("PyTorch Model (*.pt)", "*.pt"),
                ("Zip Model (*.zip)", "*.zip"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.eval_model_file_var.set(path)

    def _reset_eval_file(self) -> None:
        self.eval_model_file_var.set("")

    def _current_speed(self):
        try:
            idx = int(round(float(self.speed_scale.get())))
            idx = max(0, min(len(SPEED_LEVELS) - 1, idx))
            return SPEED_LEVELS[idx]
        except Exception:
            return "max"

    def _on_speed_drag(self, _val=None) -> None:
        try:
            idx = int(round(float(self.speed_scale.get())))
            idx = max(0, min(len(SPEED_LEVELS) - 1, idx))
            self.speed_label.config(text=SPEED_NAMES[idx])
        except Exception:
            pass

    def _on_toggle_render(self) -> None:
        if not self.live_view_var.get():
            self.view_label.configure(image="", text="⚡ [3D 터보 가속 모드 가동 중]\n\n화면 렌더링이 비활성화되어\nGPU 자원을 오직 3D 강화학습에만 집중합니다.\n(다시 보시려면 체크박스를 켜세요)")
        if self.session_dir and self.proc is not None and self.proc.poll() is None:
            self._send(self._cmd)

    def _apply_speed(self) -> None:
        try:
            idx = int(round(float(self.speed_scale.get())))
            idx = max(0, min(len(SPEED_LEVELS) - 1, idx))
            self.speed_scale.set(idx)
            self.speed_label.config(text=SPEED_NAMES[idx])
        except Exception:
            pass
        if self.session_dir and self.proc is not None and self.proc.poll() is None:
            self._send(self._cmd)

    def _send(self, cmd: str) -> None:
        self._cmd = cmd
        write_json_atomic(os.path.join(self.session_dir, "control.json"),
                          {"cmd": cmd, "speed": self._current_speed(), "render": bool(self.live_view_var.get())})

    def _start(self) -> None:
        try:
            episodes = int(self.ep_var.get())
            if episodes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("입력 오류", "에피소드 수는 1 이상의 정수여야 합니다.")
            return

        sid = f"run3d_{datetime.now():%Y%m%d_%H%M%S}"
        self.session_dir = os.path.join(RUNS_DIR, sid)
        os.makedirs(self.session_dir, exist_ok=True)
        self.is_evaluating = False

        n_envs = self.n_envs_var.get().strip() or "4"
        cmd = [sys.executable, os.path.join(ENV3D_DIR, "training", "train_worker_3d.py"), "--session", sid,
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
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ENV3D_DIR, env=env)
        except OSError as e:
            messagebox.showerror("시작 실패", f"워커 실행 실패: {e}")
            return

        self._sizes = (-1, -1)
        self._set_buttons("running")
        self.status_var.set(f"3D 학습 시작 중... (세션 {sid})")

    def _start_eval(self) -> None:
        model_file = self.eval_model_file_var.get().strip()
        if not model_file or not os.path.isfile(model_file):
            messagebox.showwarning("모델 선택", "평가할 3D 모델 파일(.zip)을 선택해 주세요.")
            return

        try:
            episodes = int(self.test_ep_var.get())
            if episodes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("입력 오류", "테스트 회수는 1 이상의 정수여야 합니다.")
            return

        sid = f"eval3d_{datetime.now():%Y%m%d_%H%M%S}"
        self.session_dir = os.path.join(RUNS_DIR, sid)
        os.makedirs(self.session_dir, exist_ok=True)
        self.is_evaluating = True

        cmd = [sys.executable, os.path.join(ENV3D_DIR, "training", "test_eval_3d.py"),
               "--model", model_file, "--session", self.session_dir,
               "--episodes", str(episodes), "--speed", str(self._current_speed())]

        try:
            log = open(os.path.join(self.session_dir, "eval_log.txt"), "w", encoding="utf-8")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ENV3D_DIR, env=env)
        except OSError as e:
            messagebox.showerror("시작 실패", f"평가 실행 실패: {e}")
            self.is_evaluating = False
            return

        self._sizes = (-1, -1)
        self._last_eval_chart_mtime = 0.0
        self.fig.clear()
        self.canvas.draw_idle()
        self._set_buttons("evaluating")
        self.status_var.set(f"3D 평가 시작... (총 {episodes}회, 세션 {sid})")

    def _toggle_pause(self) -> None:
        paused = self.btn_pause.cget("text").startswith("▶")
        self._send("run" if paused else "pause")
        self._set_buttons("running" if paused else "paused")

    def _stop(self) -> None:
        if messagebox.askyesno("종료", "3D 학습을 종료하고 현재 상태를 저장할까요?"):
            self._send("stop")
            self.btn_pause.config(state="disabled")

    def _tick(self) -> None:
        self._tick_count += 1
        alive = self.proc is not None and self.proc.poll() is None
        if self.proc is not None and not alive:
            self._set_buttons("idle")
            self.status_var.set("학습/평가 완료 (대기 중)")
            self.proc = None

        if self.session_dir:
            # 1. 상태 갱신
            st = read_json(os.path.join(self.session_dir, "status.json"))
            if st:
                state = st.get('state', '')
                if state in ('evaluating', 'eval_done'):
                    cur_ep = st.get('session_episodes', st.get('episodes', 0))
                    tot_ep = st.get('total_episodes', self.test_ep_var.get())
                    txt = f"[{'평가 중' if state == 'evaluating' else '평가 완료'}] 진행: {cur_ep}/{tot_ep}회 | 성공률: {st.get('success_rate', 0)*100:.1f}% | 플립률: {st.get('flip_rate', 0)*100:.1f}% | 평균 보상: {st.get('reward_mean', 0.0):.1f}"
                else:
                    tot_ep = st.get('total_episodes', st.get('episodes', 0))
                    sess_ep = st.get('session_episodes', st.get('episodes', 0))
                    txt = f"상태: {state} | 누적 {tot_ep:,} ep (이번 세션 {sess_ep}/{self.ep_var.get()}) | 스텝: {st.get('steps', 0):,} | 리워드: {st.get('reward_mean', 0.0):.1f} | 성공률: {st.get('success_rate', 0)*100:.1f}% | 플립률: {st.get('flip_rate', 0)*100:.1f}% | FPS: {st.get('fps', 0)}"
                self.status_var.set(txt)

            # 2. 실시간 쿼터뷰 프레임 갱신 (파일 락 방지)
            if self.live_view_var.get():
                frame_path = os.path.join(self.session_dir, "live_frame.jpg")
                if os.path.exists(frame_path):
                    try:
                        mtime = os.path.getmtime(frame_path)
                        if mtime > self._last_frame_mtime:
                            self._last_frame_mtime = mtime
                            with Image.open(frame_path) as orig:
                                img = orig.copy()
                            lw = max(100, self.view_label.winfo_width())
                            lh = max(100, self.view_label.winfo_height())
                            img = img.resize((lw, lh), Image.Resampling.BILINEAR)
                            photo = ImageTk.PhotoImage(img)
                            self.view_label.configure(image=photo, text="")
                            self._current_photo = photo
                    except Exception:
                        pass

            # 3. 차트 갱신 (1초마다)
            if self._tick_count % 10 == 0:
                self._update_plots()

        self.after(REFRESH_MS, self._tick)

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
                        with Image.open(chart_path) as orig:
                            img = orig.copy()
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
            return
        self._sizes = sizes
        plotting.draw_curves(self.fig, plotting.load_csv(ep_path, plotting.EP_FIELDS),
                             plotting.load_csv(ls_path, plotting.LOSS_FIELDS), self.window)
        self.canvas.draw_idle()

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("종료 확인", "3D 학습 워커가 동작 중입니다. 종료하시겠습니까?"):
                return
            self._send("stop")
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        self.destroy()


def main():
    app = Dashboard3D()
    app.mainloop()


if __name__ == "__main__":
    main()
