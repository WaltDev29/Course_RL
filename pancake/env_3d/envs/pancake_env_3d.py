"""완전 3D 팬케이크 뒤집기 Gymnasium 환경 (8자유도 인체 조리자 팔 및 3D 원형 팬).

- 관측 공간: 41차원 (관절 각/속도, 손목 기준 팬 상대위치/속도, 팬 법선 및 각속도, 케이크 상대위치/속도, 케이크 3D 법선 및 각속도, 접촉)
- 행동 공간: 8차원 (8개 관절 목표 각도 증분 제어)
- 제어 주기: 50Hz (MuJoCo timestep 0.002초 x frame_skip 10)
"""
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "pancake_env_3d.xml")


@dataclass
class RewardConfig3D:
    """3D 보상 및 치팅 방지 하이퍼파라미터."""
    # 1) 런치 (뒤집기 전 고도 유도)
    z_threshold: float = 0.04        # R_throw 시작 고도 (팬 기준, m)
    z_throw_cap: float = 0.35        # R_throw 상한 고도 (m)
    w_throw: float = 2.0             # 고도 보상 가중치 (초기 띄우기 탐색 유도)
    w_throw_align: float = 5.0       # 수평 이탈 방지 가중치 (exp(-w_align * dist_xy^2))
    max_throw_reward: float = 12.0   # 에피소드당 던지기 보상 누적 상한 (저글링 파밍 차단)
    w_lift_hint: float = 0.5         # 초반 위로 튕겨올리는 수직 속도(Vz > 0) 탐험 힌트 가중치
    max_lift_reward: float = 3.0     # 초반 튕기기 힌트 누적 한도 (파밍 방지용 작은 캡)

    # 2) 회전 진행도 및 반전 보너스
    w_flip: float = 1.5              # 180도 회전 과정 스텝 보상
    flip_bonus: float = 15.0         # 최초 180도 반전 달성 1회성 보너스 (norm_z < -0.80)
    flip_norm_z: float = -0.80       # 반전 판정 임계값 (상향 법선 Z성분)
    step_penalty_before_flip: float = -0.05  # 뒤집기 전 시간 경과 벌점 (빠른 토스/뒤집기 촉구)
    pan_tilt_penalty: float = -0.5   # 뒤집기 전 팬 과도 기울임 감점 (옆으로 눕혀 비비기 방지)
    pan_tilt_thresh: float = 0.70    # 팬 수평 법선 Z성분 하한 (cos(45도) ≈ 0.707)

    # 3) 3D 원형 안착 및 50스텝(1.0초) 안정 유지 (Hold & Settle)
    r_safe: float = 0.095            # 안정 안착 판정 반경 (프로토타입 팬 바닥 반경 10.1cm 중 9.5cm)
    hold_step: float = 0.4           # 유지 스텝 기본 보상
    hold_speed_max: float = 0.22     # 안정 판정 3D 상대속도 상한 (m/s)
    hold_steps_required: int = 50    # 1.0초 (50Hz x 50스텝)
    max_hold_reward: float = 35.0    # 에피소드당 유지 보상 누적 한도 (딱 50스텝치만 지급하여 400스텝 파밍 차단)
    success_bonus: float = 150.0     # 50스텝 연속 유지 성공 별도 보너스

    # 4) 치팅 방지 및 안정화 페널티
    w_spin_penalty: float = 0.02     # 과도한 3D 팽이 회전 억제 (|omega| > max_safe_spin)
    max_safe_spin: float = 16.0      # 허용 회전 각속도 상한 (rad/s)
    over_flip_penalty: float = -10.0 # 뒤집힌 후 다시 360도 회전해 원래 면으로 복귀 시 페널티
    idle_penalty: float = -0.06      # 팬 위에 가만히 누워있을 때 미세 감점 (부동자세 방지)
    idle_timeout_steps: int = 150    # 3초(150스텝) 동안 띄우지 못하면 조기 종료
    w_posture: float = 0.0           # 자연스러운 탐색을 위해 기본 자세 구속 해제 (0.0)
    lambda_action: float = 0.002     # 과도한 관절 행동 억제 완화 (시원한 스윙 보장)

    # 5) 이탈(Spill) 페널티
    spill_penalty: float = -20.0
    spill_after_flip_penalty: float = -10.0  # 뒤집기 성공 후 낙하 시 감점 완화


class PancakeFlipEnv3D(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # 8개 관절 명칭 (Shoulder 3 + Elbow 2 + Wrist 3)
    JOINT_NAMES = (
        "joint_sh_yaw", "joint_sh_pitch", "joint_sh_roll",
        "joint_el_pitch", "joint_el_yaw",
        "joint_wr_roll", "joint_wr_pitch", "joint_wr_yaw"
    )

    # 기본 조리 자세 (Home Pose: 팬 수평 정면 전방 배치)
    HOME_QPOS = np.zeros(8, dtype=np.float64)
    # 스텝당 최대 관절 회전량 (역동적인 스냅과 런치 가속을 위해 상향)
    ACTION_SCALE = np.array([0.08, 0.12, 0.08, 0.14, 0.08, 0.12, 0.18, 0.08], dtype=np.float64)

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 400,
                 reward_cfg: Optional[RewardConfig3D] = None, frame_skip: int = 10):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.cfg = reward_cfg or RewardConfig3D()
        self.frame_skip = frame_skip

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        m = self.model

        # 8개 관절 및 액추에이터 주소 매핑
        self._j_qpos = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.JOINT_NAMES]
        self._j_qvel = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.JOINT_NAMES]
        self._j_range = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.JOINT_NAMES])
        # MuJoCo Python 바인딩의 m.jnt_range는 이미 라디안 단위이므로 그대로 사용 (중복 변환 제거)
        self._j_range_rad = self._j_range.copy()

        # 팬케이크 6 DoF 자유체 조인트 주소
        cake_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cake_joint")
        self._cake_qpos_adr = m.jnt_qposadr[cake_jnt_id]
        self._cake_qvel_adr = m.jnt_dofadr[cake_jnt_id]

        # 추적용 사이트 ID
        self._site_wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "wrist_site")
        self._site_pan = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pan_center")
        self._site_cake_center = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cake_center")
        self._site_cake_top = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cake_top_site")

        # 충돌 판정용 지오메트리 세트 (팬 바닥 + 2단 외향 경사 림 40개 충돌체 + 룸 바닥)
        # [소프트바디 지원] cake_ 코어 및 8방향 탄성 플랩의 모든 geom을 자동 수집
        self._cake_geoms = {i for i in range(m.ngeom)
                            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("cake_")}
        pan_geom_names = ["pan_floor"] + [f"rim_l{i:02d}" for i in range(1, 21)] + [f"rim_h{i:02d}" for i in range(1, 21)]
        self._pan_geoms = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in pan_geom_names
                           if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0}
        self._floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # 행동 공간 (8D) 및 관측 공간 (41D)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(41,), dtype=np.float32)

        self._target_q = self.HOME_QPOS.copy()
        self._renderer = None
        self._reset_episode_state()

    # ---------- 내부 상태 초기화 ----------
    def _reset_episode_state(self) -> None:
        self.steps = 0
        self.flipped = False
        self.over_flipped = False
        self.accumulated_throw_reward = 0.0
        self.accumulated_lift_reward = 0.0
        self.accumulated_hold_reward = 0.0
        self.max_z_reached = 0.0
        self.hold_count = 0

    # ---------- 관측 연산 ----------
    def _contact_with_pan(self) -> bool:
        """팬케이크가 프라이팬 바닥이나 림과 접촉 중인지 확인."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self._cake_geoms and c.geom2 in self._pan_geoms) or \
               (c.geom2 in self._cake_geoms and c.geom1 in self._pan_geoms):
                return True
        return False

    def _contact_with_floor(self) -> bool:
        """팬케이크가 바닥(floor)과 물리적으로 충돌했는지 확인."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self._cake_geoms and c.geom2 == self._floor_geom) or \
               (c.geom2 in self._cake_geoms and c.geom1 == self._floor_geom):
                return True
        return False

    def _get_site_spatial_vel(self, site_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """사이트의 월드 3D 각속도와 선속도 반환 (ang_vel, lin_vel)."""
        res = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, site_id, res, 0)
        return res[0:3], res[3:6]

    def _get_state_dict(self) -> Dict[str, np.ndarray]:
        d = self.data
        # 1) 관절 각도 및 각속도 (8D + 8D)
        q_arm = np.array([d.qpos[idx] for idx in self._j_qpos], dtype=np.float32)
        dq_arm = np.array([d.qvel[idx] for idx in self._j_qvel], dtype=np.float32)

        # 2) 위치 및 속도 추출
        wrist_pos = d.site_xpos[self._site_wrist]
        pan_pos = d.site_xpos[self._site_pan]
        cake_pos = d.site_xpos[self._site_cake_center]
        cake_top = d.site_xpos[self._site_cake_top]

        pan_ang_vel, pan_lin_vel = self._get_site_spatial_vel(self._site_pan)
        _, wrist_lin_vel = self._get_site_spatial_vel(self._site_wrist)
        cake_ang_vel, cake_lin_vel = self._get_site_spatial_vel(self._site_cake_center)

        # 3) 손목 기준 프라이팬 바닥 중심의 상대 위치 및 상대 선속도 (3D + 3D)
        rel_pan_pos = (pan_pos - wrist_pos).astype(np.float32)
        rel_pan_vel = (pan_lin_vel - wrist_lin_vel).astype(np.float32)

        # 4) 프라이팬 바닥의 3D 상향 법선 단위 벡터 (3D)
        # site_xmat는 3x3 회전행렬, 2열(0,1,2 중 z축)은 로컬 Z축 = 수직 윗방향
        pan_mat = d.site_xmat[self._site_pan].reshape(3, 3)
        pan_up_vec = pan_mat[:, 2].astype(np.float32)

        # 5) 프라이팬 3D 각속도 (3D)
        pan_ang_vel_f32 = pan_ang_vel.astype(np.float32)

        # 6) 프라이팬 중심 기준 케이크 3D 상대 위치 및 상대 선속도 (프라이팬 로컬 프레임 투영, 3D + 3D)
        # pan_mat.T를 곱해 프라이팬 바닥면 기준 [앞/뒤(Local X), 좌/우(Local Y), 팬 바닥 수직고도(Local Z)]로 변환
        rel_cake_pos_world = cake_pos - pan_pos
        rel_cake_vel_world = cake_lin_vel - pan_lin_vel
        rel_cake_pos = (pan_mat.T @ rel_cake_pos_world).astype(np.float32)
        rel_cake_vel = (pan_mat.T @ rel_cake_vel_world).astype(np.float32)

        # 7) 케이크 윗면(Red)의 3D 상향 법선 단위 벡터 (3D)
        diff = cake_top - cake_pos
        norm = np.linalg.norm(diff)
        cake_up_vec = (diff / (norm + 1e-8)).astype(np.float32)

        # 8) 케이크 3D 각속도 (3D)
        cake_ang_vel_f32 = cake_ang_vel.astype(np.float32)

        # 9) 접촉 여부 (1D)
        is_contact = np.array([float(self._contact_with_pan())], dtype=np.float32)

        return dict(
            q_arm=q_arm, dq_arm=dq_arm,
            rel_pan_pos=rel_pan_pos, rel_pan_vel=rel_pan_vel,
            pan_up_vec=pan_up_vec, pan_ang_vel=pan_ang_vel_f32,
            rel_cake_pos=rel_cake_pos, rel_cake_vel=rel_cake_vel,
            cake_up_vec=cake_up_vec, cake_ang_vel=cake_ang_vel_f32,
            is_contact=is_contact
        )

    def _obs(self, s: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([
            s["q_arm"], s["dq_arm"],
            s["rel_pan_pos"], s["rel_pan_vel"],
            s["pan_up_vec"], s["pan_ang_vel"],
            s["rel_cake_pos"], s["rel_cake_vel"],
            s["cake_up_vec"], s["cake_ang_vel"],
            s["is_contact"]
        ]).astype(np.float32)

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._reset_episode_state()

        # 1. 조리 기본 자세(Home Pose) 설정 + 미세 노이즈
        noise = self.np_random.uniform(-0.02, 0.02, size=8)
        self._target_q = np.clip(self.HOME_QPOS + noise, self._j_range_rad[:, 0], self._j_range_rad[:, 1])
        for idx, val in zip(self._j_qpos, self._target_q):
            self.data.qpos[idx] = val
        self.data.ctrl[:] = self._target_q
        mujoco.mj_forward(self.model, self.data)

        # 2. 팬케이크 초기 위치 무작위화: 프로토타입 팬 바닥(반경 10.1cm) 내에서 절대 벗어나지 않게 배치
        # 팬 바닥 반경 10.1cm, 케이크 반경 6.5cm -> 안전 마진 포함 최대 반경 2.5cm 이내 무작위화
        r_offset = 0.025 * math.sqrt(float(self.np_random.uniform(0.0, 1.0)))
        phi_offset = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        local_dx = r_offset * math.cos(phi_offset)
        local_dy = r_offset * math.sin(phi_offset)

        # 팬 바닥의 3D 자세(회전행렬)에 맞추어 오프셋 투영
        pan_mat = self.data.site_xmat[self._site_pan].reshape(3, 3)
        pan_pos = self.data.site_xpos[self._site_pan]
        # pan_mat[:, 0]은 로컬 X축, pan_mat[:, 1]은 로컬 Y축, pan_mat[:, 2]는 상향 Z축
        spawn_pos = pan_pos + pan_mat[:, 0] * local_dx + pan_mat[:, 1] * local_dy + pan_mat[:, 2] * 0.012

        self.data.qpos[self._cake_qpos_adr : self._cake_qpos_adr + 3] = spawn_pos
        # 초기 쿼터니언: 수평 (상단 Red가 위를 향함)
        self.data.qpos[self._cake_qpos_adr + 3 : self._cake_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._cake_qvel_adr : self._cake_qvel_adr + 6] = 0.0

        # 0.2초(100스텝) 정착 물리 시뮬레이션
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        s = self._get_state_dict()
        return self._obs(s), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # 8개 관절 목표 각도 증분 제어
        delta = action * self.ACTION_SCALE
        self._target_q = np.clip(self._target_q + delta, self._j_range_rad[:, 0], self._j_range_rad[:, 1])
        self.data.ctrl[:] = self._target_q

        try:
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
        except mujoco.FatalError as e:
            print(f"[PancakeFlipEnv3D] 물리 발산 step={self.steps}: {e}")
            s = self._get_state_dict()
            return self._obs(s), self.cfg.spill_penalty, True, False, {"diverged": True}

        self.steps += 1
        s = self._get_state_dict()
        reward, terminated, info = self._compute_reward(s, action)
        truncated = (not terminated) and self.steps >= self.max_steps
        if not np.isfinite(reward):
            reward, terminated = self.cfg.spill_penalty, True

        return self._obs(s), float(reward), terminated, truncated, info

    # ---------- 보상 함수 및 치팅 방지 ----------
    def _compute_reward(self, s: Dict[str, np.ndarray], action: np.ndarray) -> Tuple[float, bool, dict]:
        c = self.cfg
        r = 0.0
        info: dict = {}
        terminated = False

        cake_z = float(s["rel_cake_pos"][2])
        dist_xy = float(math.hypot(s["rel_cake_pos"][0], s["rel_cake_pos"][1]))
        cake_nz = float(s["cake_up_vec"][2])
        is_contact = float(s["is_contact"][0])
        pan_nz = float(s["pan_up_vec"][2])  # 팬 상향 수평도 (1.0에 가까울수록 수평)

        self.max_z_reached = max(self.max_z_reached, cake_z)

        # 팬케이크 공중 체공(비접촉 또는 기준 고도 이상) 여부 판정
        is_airborne = (is_contact < 0.5 or cake_z > c.z_threshold)

        # 180도 반전 판정 (상향 벡터 Z성분이 -0.80 이하이며, 공중에 분리 체공 중이어야 인정)
        newly_flipped = False
        if not self.flipped and not self.over_flipped and cake_nz < c.flip_norm_z and is_airborne:
            self.flipped, newly_flipped = True, True

        # [치팅 방지 1] 360도 재회전(Over-flip) 차단:
        # 뒤집기 달성 후 멈추지 않고 계속 돌아 다시 원래 바닥면(cake_nz > 0.0)으로 복귀 시 페널티
        if self.flipped and not self.over_flipped and cake_nz > 0.0:
            self.over_flipped = True
            self.flipped = False
            r += c.over_flip_penalty
            info["over_flipped"] = True

        # 1) 런치(던지기) 보상: 뒤집기 전 수직 고도 유도 + 수평 정렬 결합
        # [치팅 방지 2] 저글링/통통 튀기기 누적 캡(max_throw_reward) 적용
        if not self.flipped and not self.over_flipped:
            if self.accumulated_throw_reward < c.max_throw_reward:
                throw_z = min(max(0.0, cake_z - c.z_threshold), c.z_throw_cap - c.z_threshold)
                align_factor = math.exp(-c.w_throw_align * (dist_xy ** 2))
                potential_r = c.w_throw * throw_z * align_factor
                allowed_r = min(potential_r, c.max_throw_reward - self.accumulated_throw_reward)
                r += allowed_r
                self.accumulated_throw_reward += allowed_r

        # 1-1) 뒤집기 전 시간 경과 벌점 (빨리 띄워 뒤집도록 촉구)
        if not self.flipped and not self.over_flipped:
            r += c.step_penalty_before_flip

        # 1-2) [치팅 방지] 뒤집기 전 팬을 옆으로 눕혀 비비는 꼼수 차단 (팬이 45도 이상 기울어지면 감점)
        if not self.flipped and not self.over_flipped and pan_nz < c.pan_tilt_thresh:
            r += c.pan_tilt_penalty * (c.pan_tilt_thresh - pan_nz)

        # 1-3) [초반 탐험 촉진] 뒤집기 전 수직 상향 도약(Lift) 힌트 보상 (최대 3.0점 캡)
        # 8개 관절이 허공에서 바들바들 떨지 않고 위로 쳐올리는 모션을 초반에 신속히 발견하도록 유도
        if not self.flipped and not self.over_flipped and cake_z < c.z_threshold:
            cake_vz = float(s["rel_cake_vel"][2])
            if cake_vz > 0.1 and self.accumulated_lift_reward < c.max_lift_reward:
                lift_r = min(c.w_lift_hint * cake_vz, c.max_lift_reward - self.accumulated_lift_reward)
                r += lift_r
                self.accumulated_lift_reward += lift_r

        # 2) 180도 회전 진행도 보상
        # [치팅 방지] 팬에 붙어있는 상태(is_contact > 0.5 & cake_z <= z_threshold)에서 비비는 경우 0점 처리 (공중 체공 필수)
        if not self.flipped and not self.over_flipped and is_airborne:
            r += c.w_flip * max(0.0, (1.0 - cake_nz) / 2.0)

        # 180도 반전 달성 1회성 보너스
        if newly_flipped:
            r += c.flip_bonus

        # [치팅 방지 3] 3D 과도한 팽이 스핀 억제 페널티
        cake_omega_norm = float(np.linalg.norm(s["cake_ang_vel"]))
        if cake_omega_norm > c.max_safe_spin:
            r -= c.w_spin_penalty * ((cake_omega_norm - c.max_safe_spin) ** 2)

        # [치팅 방지 4] 가만히 있기(Inaction) 방지
        if not self.flipped and is_contact > 0.5 and cake_z < c.z_threshold:
            r += c.idle_penalty

        # 3) 3D 수평 안착 및 50스텝(1.0초) 안정 유지 판정
        rel_speed_3d = float(np.linalg.norm(s["rel_cake_vel"]))

        hold = (self.flipped and cake_nz < c.flip_norm_z and is_contact > 0.5 and dist_xy < c.r_safe)
        if hold:
            # 유지 보수는 딱 50스텝치(max_hold_reward)까지만 누적 지급하여 400스텝 파밍 원천 차단!
            if self.accumulated_hold_reward < c.max_hold_reward:
                potential_hold_r = c.hold_step + 0.15 * math.exp(-4.0 * (rel_speed_3d ** 2)) + 0.15 * math.exp(-4.0 * ((1.0 - pan_nz) ** 2))
                allowed_hold_r = min(potential_hold_r, c.max_hold_reward - self.accumulated_hold_reward)
                r += allowed_hold_r
                self.accumulated_hold_reward += allowed_hold_r

            if rel_speed_3d < c.hold_speed_max:
                self.hold_count += 1
            else:
                # 1스텝 속도 튐 발생 시 0으로 즉시 리셋하지 않고 완충 감점 (모터 노이즈 방어)
                self.hold_count = max(0, self.hold_count - 3)
        else:
            self.hold_count = max(0, self.hold_count - 5)

        # 4) 자연스러운 조리 기본 자세(Home Pose) 유지 보상
        q_err = np.linalg.norm(s["q_arm"] - self.HOME_QPOS)
        r -= c.w_posture * float(q_err)

        # 5) 급격한 관절 조작 억제
        r -= c.lambda_action * float(np.sum(action ** 2))

        # [치팅 방지 5] 무작동 조기 종료
        if not self.flipped and self.steps >= c.idle_timeout_steps and self.max_z_reached < c.z_threshold:
            r += c.spill_penalty * 0.5
            terminated = True
            info["idle_timeout"] = True

        # 6) 3D 이탈(Spill) 및 바닥 낙하 판정
        cake_world_z = float(self.data.site_xpos[self._site_cake_center][2])
        hit_floor = self._contact_with_floor() or (cake_world_z <= 0.035)
        spilled = hit_floor or cake_z < -0.25 or dist_xy > 0.35
        if spilled:
            r += c.spill_after_flip_penalty if self.flipped else c.spill_penalty
            terminated = True
            info["spill"] = "floor" if hit_floor else ("after_flip" if self.flipped else "plain")
        # [최종 성공] 50스텝(1.0초) 연속 수평 안착 달성!
        elif self.hold_count >= c.hold_steps_required:
            r += c.success_bonus
            terminated = True
            info["success"] = True

        info.update(flipped=self.flipped, hold_count=self.hold_count, dist_xy=dist_xy)
        return r, terminated, info

    # ---------- 렌더링 ----------
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        # 3인칭 쿼터뷰 카메라로 렌더링
        self._renderer.update_scene(self.data, camera="quarter_view")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
