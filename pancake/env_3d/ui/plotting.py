"""학습 지표 CSV 로딩 및 그래프 렌더링 (GUI/워커 공용).

워커는 pyplot 없이 Figure.savefig만 쓰므로 헤드리스에서도 동작한다.
"""
import csv
import logging
import os
import warnings
from typing import Dict

import matplotlib
import numpy as np
from matplotlib import rcParams
from matplotlib.figure import Figure

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have a glyph for.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph 8722.*")

EP_FIELDS = ["episode", "steps", "reward", "length", "success", "flipped", "spill"]
LOSS_FIELDS = ["steps", "actor_loss", "critic_loss", "ent_coef"]


def load_csv(path: str, fields) -> Dict[str, np.ndarray]:
    """쓰는 도중의 불완전한 마지막 줄은 건너뛰고 읽는다."""
    cols = {f: [] for f in fields}
    if not os.path.exists(path):
        return {f: np.array([]) for f in fields}
    try:
        with open(path, newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                try:
                    vals = {f: float(row[f]) for f in fields if f != "spill"}
                except (TypeError, ValueError, KeyError):
                    continue
                for f, v in vals.items():
                    cols[f].append(v)
    except OSError as e:
        print(f"[plotting] CSV 읽기 실패 {path}: {e}")
    return {f: np.array(v) for f, v in cols.items() if f != "spill"}


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) == 0:
        return x
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def draw_curves(fig: Figure, ep: Dict[str, np.ndarray], ls: Dict[str, np.ndarray], window: int = 50) -> None:
    """성공률 / 보상 / 에피소드 스텝 수(길이) / Loss 4개 그래프를 fig에 그린다."""
    fig.clear()
    axes = fig.subplots(4, 1)
    ax1, ax2, ax3, ax4 = axes
    n = len(ep.get("episode", []))
    if n:
        x = ep["episode"]
        # 1. 성공률 & 뒤집기 달성률
        ax1.plot(x, rolling_mean(ep["success"], window) * 100, color="tab:green", label=f"성공률 (최근 {window})")
        ax1.plot(x, rolling_mean(ep["flipped"], window) * 100, color="tab:orange", alpha=0.7, label="뒤집기 달성률")
        # 2. 보상
        ax2.plot(x, ep["reward"], color="tab:blue", alpha=0.25, label="에피소드 보상")
        ax2.plot(x, rolling_mean(ep["reward"], window), color="tab:blue", label=f"이동평균 ({window})")
        # 3. 에피소드 스텝 수 (길이)
        if "length" in ep and len(ep["length"]):
            ax3.plot(x, ep["length"], color="tab:purple", alpha=0.3, label="스텝 수 (최대 400)")
            ax3.plot(x, rolling_mean(ep["length"], window), color="tab:purple", label=f"이동평균 ({window})")
            ax3.axhline(400, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    
    ax1.set_ylabel("%")
    ax1.set_ylim(-5, 105)
    ax1.set_title("성공률 및 플립율", fontsize=10, pad=6)

    ax2.set_ylabel("reward")
    if n and len(ep.get("reward", [])):
        r_max = float(np.nanmax(ep["reward"]))
        top_y = r_max + max(20.0, abs(r_max) * 0.15) if r_max > 0 else 50.0
        ax2.set_ylim(bottom=-500, top=top_y)
    else:
        ax2.margins(y=0.20)
    ax2.set_title("에피소드 총 보상", fontsize=10, pad=6)

    ax3.set_ylabel("steps (0.02s)")
    ax3.set_ylim(-10, 430)
    ax3.set_title("에피소드 지속 스텝 수 (생존/완료 시간)", fontsize=10, pad=6)

    # 4. Loss
    if len(ls.get("steps", [])):
        ax4.plot(ls["steps"], ls["critic_loss"], color="tab:red", label="critic loss")
        ax4.set_yscale("log")
        ax4.margins(y=0.20)
        ax4b = ax4.twinx()
        ax4b.plot(ls["steps"], ls["actor_loss"], color="tab:cyan", alpha=0.8, label="actor loss")
        ax4b.set_ylabel("actor loss")
        ax4b.margins(y=0.20)
        h = ax4.get_legend_handles_labels()[0] + ax4b.get_legend_handles_labels()[0]
        ax4.legend(h, [l.get_label() for l in h], loc="upper left", fontsize=8)
    ax4.set_ylabel("critic loss (log)")
    ax4.set_xlabel("total steps")
    ax4.set_title("신경망 Loss", fontsize=10, pad=6)

    for ax in (ax1, ax2, ax3):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
    ax4.grid(alpha=0.3)
    fig.subplots_adjust(top=0.96, bottom=0.05, left=0.12, right=0.88, hspace=0.45)
