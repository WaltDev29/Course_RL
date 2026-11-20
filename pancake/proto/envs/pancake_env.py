"""팬케이크 뒤집기 프로토타입 Gymnasium 환경 (강체 팬케이크, 횡 시점 2D).

관측 18D / 행동 3D(팬 중심 x,z,theta의 delta) / 50Hz 제어.
"""
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

try:
    from . import kinematics as kin
except ImportError:  # 스크립트 직접 실행 대비
    import kinematics as kin

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "pancake_env.xml")


@dataclass
class RewardConfig:
    """보상 하이퍼파라미터 (소프트바디 지원 및 치팅 방지)."""
    # 1) 런치 (뒤집기 전 고도 유도)
    z_threshold: float = 0.04      # R_throw 시작 고도(팬 대비, m)
    z_throw_cap: float = 0.35      # R_throw 상한 고도(m)
    w_throw: float = 1.0           # 고도 보상 가중치
    max_throw_reward: float = 12.0 # 에피소드당 던지기 보상 누적 상한 (저글링/통통 튀기기 파밍 차단)

    # 2) 회전 진행도 및 반전 보너스
    w_flip: float = 1.5            # 180도 회전 과정 스텝 보상
    flip_bonus: float = 15.0       # 최초 180도 반전 달성 1회성 보너스 (norm_z < -0.8)
    flip_norm_z: float = -0.8      # 반전 판정 임계값

    # 3) 소프트바디 평평도/구겨짐 방지 보상 (R_flat)
    w_flat: float = 10.0           # 평평도 보상 가중치 (-w_flat * |deform_curl|)

    # 4) 재착지 및 1초 안정 유지 (Hold & Settle)
    alpha: float = 15.0            # 거리 감쇠
    w_catch: float = 1.0           # 하강 안착 유도 가중치
    r_safe: float = 0.10           # 안정 착지 판정 반경(m, 팬 반경 12cm)
    hold_step: float = 0.5         # 유지 스텝 보상
    beta: float = 5.0              # 속도 감쇠
    gamma: float = 5.0             # 수평 감쇠
    hold_steps_required: int = 50  # 1.0초 (50Hz)
    hold_speed_max: float = 0.15   # 안정 판정 상대속도 상한(m/s)
    success_bonus: float = 50.0    # 1초 유지 최종 성공 보너스

    # 5) 치팅 방지 (Anti-Cheating / Exploit Prevention)
    w_spin_penalty: float = 0.02   # 과회전 팽이 스핀 억제 가중치 (|omega| > max_safe_spin)
    max_safe_spin: float = 16.0    # 허용 회전 각속도 상한 (rad/s, 약 2.5 rev/s)
    over_flip_penalty: float = -10.0 # 뒤집힌 후 다시 360도 회전해 원래 면으로 복귀 시 페널티
    idle_penalty: float = -0.02    # 팬 위에 가만히 누워있을 때 미세 시간 페널티
    idle_timeout_steps: int = 150  # 3초(150스텝) 동안 한 번도 띄우지 않으면 조기 종료

    # 6) 페널티
    spill_penalty: float = -20.0
    spill_after_flip_penalty: float = -10.0  # 뒤집기 성공 후 낙하 시 감점을 완화하여 뒤집기 탐색 장려
    lambda_action: float = 0.01


class PancakeFlipEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # 팬 목표 포즈(어깨 기준 x, z, theta)의 초기값과 작업공간 한계
    # 사람 조리 자세: 어깨(Z=0.90m) 앞쪽 0.60m, 아래쪽 -0.25m (지면 기준 Z=0.65m), 팬 수평
    HOME = np.array([0.60, -0.25, 0.0])
    LOW = np.array([0.40, -0.50, -1.2])
    HIGH = np.array([0.78, 0.05, 1.2])
    ACTION_SCALE = np.array([0.04, 0.04, 0.3])  # step당 최대 이동량 (m, m, rad)

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 400,
                 reward_cfg: Optional[RewardConfig] = None, frame_skip: int = 10):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.cfg = reward_cfg or RewardConfig()
        self.frame_skip = frame_skip

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._jq = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in ("shoulder", "elbow", "wrist", "cake_x", "cake_z", "cake_rot")}
        self._jv = {n: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                    for n in self._jq}
        self._site_wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "wrist_site")
        self._site_pan = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pan_center")
        self._site_cake_center = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "site_cake_center")
        self._site_cake_left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "site_cake_left")
        self._site_cake_right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "site_cake_right")

        self._cake_geoms = {i for i in range(m.ngeom)
                            if "cake_geom" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")}
        pan_geom_names = ("pan_floor",
                          "pan_rim_f1", "pan_rim_f2", "pan_rim_f3", "pan_rim_f4", "pan_rim_f5",
                          "pan_rim_b1", "pan_rim_b2", "pan_rim_b3", "pan_rim_b4", "pan_rim_b5")
        self._pan_geoms = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                           for name in pan_geom_names
                           if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0}
        self._shoulder_pos = np.array([0.01, 0.0, 0.90])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # 소프트바디 상태 공간 (20차원: norm_x 및 deform_curl 포함)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float32)

        self._renderer = None
        self._target = self.HOME.copy()
        self._reset_episode_state()

    # ---------- 내부 상태 ----------
    def _reset_episode_state(self) -> None:
        self.steps = 0
        self.flipped = False           # 180도 반전 래치
        self.over_flipped = False      # 360도 과회전(재회전) 여부
        self.accumulated_throw_reward = 0.0  # 에피소드 누적 던지기 보상 (저글링 파밍 방지용)
        self.max_z_reached = 0.0       # 에피소드 중 도달한 팬케이크 최대 고도 (무작동 판정용)
        self.hold_count = 0

    def _set_arm_pose(self, target: np.ndarray) -> None:
        """목표 팬 포즈를 IK로 풀어 관절 목표(ctrl)로 설정."""
        sol = kin.inverse(*target)
        if sol is None:
            return  # 도달 불가 목표는 무시하고 이전 ctrl 유지 (안전)
        self.data.ctrl[:] = sol

    # ---------- 관측 ----------
    def _contact_with_floor(self) -> bool:
        """케이크의 어느 세그먼트라도 프라이팬과 닿아 있는지 확인."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self._cake_geoms and c.geom2 in self._pan_geoms) or \
               (c.geom2 in self._cake_geoms and c.geom1 in self._pan_geoms):
                return True
        return False

    def _site_vel(self, site_id: int) -> np.ndarray:
        res = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, site_id, res, 0)
        return res[3:]  # 선속도(월드 좌표)

    def _get_state_dict(self) -> Dict[str, float]:
        d, q, v = self.data, self._jq, self._jv
        wrist = d.site_xpos[self._site_wrist]
        pan = d.site_xpos[self._site_pan]
        wv, pv = self._site_vel(self._site_wrist), self._site_vel(self._site_pan)
        q_sum = kin.wrap_angle(d.qpos[q["shoulder"]] + d.qpos[q["elbow"]] + d.qpos[q["wrist"]])
        w_sum = d.qvel[v["shoulder"]] + d.qvel[v["elbow"]] + d.qvel[v["wrist"]]
        cake_pos = np.array([d.qpos[q["cake_x"]], d.qpos[q["cake_z"]]])
        cake_vel = np.array([d.qvel[v["cake_x"]], d.qvel[v["cake_z"]]])
        phi = d.qpos[q["cake_rot"]]
        omega_cake = d.qvel[v["cake_rot"]]

        # 소프트바디 법선 벡터 및 굽힘 변형도(deform_curl)
        norm_x = -math.sin(phi)
        norm_z = math.cos(phi)
        pos_c = d.site_xpos[self._site_cake_center]
        pos_l = d.site_xpos[self._site_cake_left]
        pos_r = d.site_xpos[self._site_cake_right]
        # 로컬 법선 벡터 (norm_x, norm_z) 방향 처짐/굽힘도 (deform_curl: (z_left + z_right)/2 - z_center)
        curl = float(((pos_l[0] - pos_c[0] + pos_r[0] - pos_c[0]) * norm_x +
                      (pos_l[2] - pos_c[2] + pos_r[2] - pos_c[2]) * norm_z) / 2.0)

        return dict(
            q_sh=d.qpos[q["shoulder"]], dq_sh=d.qvel[v["shoulder"]],
            q_el=d.qpos[q["elbow"]], dq_el=d.qvel[v["elbow"]],
            q_wr=d.qpos[q["wrist"]], dq_wr=d.qvel[v["wrist"]],
            rel_x_pan=pan[0] - wrist[0], rel_z_pan=pan[2] - wrist[2],
            rel_vx_pan=pv[0] - wv[0], rel_vz_pan=pv[2] - wv[2],
            theta_pan=q_sum, omega_pan=w_sum,
            rel_x_cake=cake_pos[0] - pan[0], rel_z_cake=cake_pos[1] - pan[2],
            rel_vx_cake=cake_vel[0] - pv[0], rel_vz_cake=cake_vel[1] - pv[2],
            norm_x=norm_x,
            norm_z=norm_z,
            deform_curl=curl,
            is_contact=float(self._contact_with_floor()),
            omega_cake=omega_cake,
        )

    # 20차원 상태 공간 (계획안 03_plan.md 5.2절 준수)
    OBS_KEYS = ("q_sh", "dq_sh", "q_el", "dq_el", "q_wr", "dq_wr",
                "rel_x_pan", "rel_z_pan", "rel_vx_pan", "rel_vz_pan", "theta_pan", "omega_pan",
                "rel_x_cake", "rel_z_cake", "rel_vx_cake", "rel_vz_cake",
                "norm_x", "norm_z", "deform_curl", "is_contact")

    def _obs(self, s: Dict[str, float]) -> np.ndarray:
        return np.array([s[k] for k in self.OBS_KEYS], dtype=np.float32)

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._reset_episode_state()
        self._target = self.HOME.copy() + np.array([*self.np_random.uniform(-0.01, 0.01, 2), 0.0])
        sol = kin.inverse(*self._target)
        q = self._jq
        for name, val in zip(("shoulder", "elbow", "wrist"), sol):
            self.data.qpos[q[name]] = val
        self.data.ctrl[:] = sol
        mujoco.mj_forward(self.model, self.data)
        # 팬케이크를 팬 중심 약간 위에 배치 후 안착시킴
        pan = self.data.site_xpos[self._site_pan]
        self.data.qpos[q["cake_x"]] = pan[0] + self.np_random.uniform(-0.01, 0.01)
        self.data.qpos[q["cake_z"]] = pan[2] + 0.012
        for _ in range(100):  # 0.2초 정착
            mujoco.mj_step(self.model, self.data)
        s = self._get_state_dict()
        self._prev_action = np.zeros(3)
        return self._obs(s), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._target = np.clip(self._target + action * self.ACTION_SCALE, self.LOW, self.HIGH)
        self._set_arm_pose(self._target)
        try:
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
        except mujoco.FatalError as e:  # 시뮬레이션 발산 시 에피소드 종료
            print(f"[PancakeFlipEnv] 물리 발산 step={self.steps}: {e}")
            s = self._get_state_dict()
            return self._obs(s), self.cfg.spill_penalty, True, False, {"diverged": True}
        self.steps += 1

        s = self._get_state_dict()
        reward, terminated, info = self._compute_reward(s, action)
        truncated = (not terminated) and self.steps >= self.max_steps
        if not np.isfinite(reward):
            reward, terminated = self.cfg.spill_penalty, True
        return self._obs(s), float(reward), terminated, truncated, info

    # ---------- 보상 ----------
    def _compute_reward(self, s: Dict[str, float], action: np.ndarray) -> Tuple[float, bool, dict]:
        c = self.cfg
        r = 0.0
        info: dict = {}
        terminated = False

        # 최대 고도 갱신 (무작동 판정용)
        self.max_z_reached = max(self.max_z_reached, s["rel_z_cake"])

        # 180도 반전 판정
        newly_flipped = False
        if not self.flipped and not self.over_flipped and s["norm_z"] < c.flip_norm_z:
            self.flipped, newly_flipped = True, True

        # [치팅 방지 1] 360도 재회전(Over-flip) 차단:
        # 이미 180도 뒤집혔던 상태에서 멈추지 않고 계속 회전하여 다시 원래 바닥면(norm_z > 0.0)으로 복귀한 경우
        if self.flipped and not self.over_flipped and s["norm_z"] > 0.0:
            self.over_flipped = True
            self.flipped = False  # 뒤집힌 상태 박탈
            r += c.over_flip_penalty
            info["over_flipped"] = True

        # 1) 던지기(런치) 보상: 뒤집기 전 팬 위로 띄우는 동작 유도
        # [치팅 방지 2] 저글링/통통 튀기기 파밍 차단: 에피소드 총 누적 던지기 보상에 상한선(max_throw_reward) 적용
        if not self.flipped and not self.over_flipped:
            if self.accumulated_throw_reward < c.max_throw_reward:
                throw_z = min(max(0.0, s["rel_z_cake"] - c.z_threshold), c.z_throw_cap - c.z_threshold)
                potential_r = c.w_throw * throw_z
                allowed_r = min(potential_r, c.max_throw_reward - self.accumulated_throw_reward)
                r += allowed_r
                self.accumulated_throw_reward += allowed_r

        # 2) 회전 보상: 케이크가 뒤집히는 방향(norm_z가 1 -> -1)으로 회전 유도 (dense)
        if not self.flipped and not self.over_flipped:
            r += c.w_flip * max(0.0, (1.0 - s["norm_z"]) / 2.0)

        # 180도 반전 달성 1회성 보너스 (래치로 1회만 지급)
        if newly_flipped:
            r += c.flip_bonus

        # [치팅 방지 3] 과도한 팽이 스핀(|omega_cake| > max_safe_spin) 억제 페널티
        omega_abs = abs(s.get("omega_cake", 0.0))
        if omega_abs > c.max_safe_spin:
            r -= c.w_spin_penalty * ((omega_abs - c.max_safe_spin) ** 2)

        # [치팅 방지 4] 가만히 있기(Inaction / Idling) 방지:
        # 뒤집지 않고 팬 위에 가만히 누워있는 상태일 때 미세 페널티 부여
        if not self.flipped and s["is_contact"] > 0.5 and s["rel_z_cake"] < c.z_threshold:
            r += c.idle_penalty

        # 3) 재착지 유도 (반전 후 공중에서 팬으로 하강하는 구간)
        dist2 = s["rel_x_cake"] ** 2 + s["rel_z_cake"] ** 2
        if self.flipped and s["is_contact"] < 0.5:
            r += c.w_catch * math.exp(-c.alpha * dist2)

        # 4) 1초 안정 유지 (현재도 뒤집힌 상태 + 바닥 접촉 + 팬 중심부)
        hold = (self.flipped and s["norm_z"] < c.flip_norm_z
                and s["is_contact"] > 0.5 and abs(s["rel_x_cake"]) < c.r_safe)
        speed = math.hypot(s["rel_vx_cake"], s["rel_vz_cake"])
        if hold:
            r += c.hold_step
            r += math.exp(-c.beta * speed ** 2)
            r += math.exp(-c.gamma * s["theta_pan"] ** 2)
            self.hold_count = self.hold_count + 1 if speed < c.hold_speed_max else 0
        else:
            self.hold_count = 0

        # 5) 소프트바디 평평도/구겨짐 방지 보상 (R_flat = -w_flat * |deform_curl|)
        r -= c.w_flat * abs(s.get("deform_curl", 0.0))

        # 6) 불필요한 과도한 관절 행동 억제 페널티
        r -= c.lambda_action * float(np.sum(action ** 2))

        # [치팅 방지 5] 무작동 조기 종료:
        # idle_timeout_steps 동안 팬케이크를 한 번도 띄우지 않고 팬 위에 방치한 경우 조기 종료
        if not self.flipped and self.steps >= c.idle_timeout_steps and self.max_z_reached < c.z_threshold:
            r += c.spill_penalty * 0.5
            terminated = True
            info["idle_timeout"] = True

        # 7) 이탈(Spill) 및 최종 성공 판정
        spilled = s["rel_z_cake"] < -0.30 or abs(s["rel_x_cake"]) > 0.40
        if spilled:
            r += c.spill_after_flip_penalty if self.flipped else c.spill_penalty
            terminated = True
            info["spill"] = "after_flip" if self.flipped else "plain"
        # [최종 성공] 1초 안정 안착 연속 50스텝 유지 달성!
        elif self.hold_count >= c.hold_steps_required:
            r += c.success_bonus
            terminated = True
            info["success"] = True

        info.update(flipped=self.flipped, hold_count=self.hold_count, curl=s.get("deform_curl", 0.0))
        return r, terminated, info

    # ---------- 렌더 ----------
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="side")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
