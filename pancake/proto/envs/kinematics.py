"""3-DoF 성인 남성 체형 평면 로봇팔의 순/역기구학 모듈.

인체 치수:
- L1 (상완, 어깨 ~ 팔꿈치): 0.32m (32cm)
- L2 (전완, 팔꿈치 ~ 손목): 0.28m (28cm)
- L3_PAN (손목 ~ 둥근 손 ~ 손잡이 ~ 팬 중심): 0.34m (34cm)

각도 규약: X-Z 평면에서 +X 기준 반시계(CCW, Z가 위쪽)가 양(+).
- 어깨: 아래 방향(음수) ~ 앞쪽 수평(0) ~ 위쪽(양수)
- 팔꿈치: 0(펴짐) ~ 양수(사람처럼 위/앞으로 굽힘, Elbow-Down 형태 유지)
"""
import math
from typing import Optional, Tuple

L1 = 0.32  # 성인 남성 상완 길이
L2 = 0.28  # 성인 남성 전완 길이
L3_PAN = 0.34  # 손목 -> 손 -> 손잡이 -> 팬 중심 거리


def wrap_angle(a: float) -> float:
    """각도를 [-pi, pi]로 정규화 (관절 목표의 불연속 점프 방지)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def forward(q1: float, q2: float, q3: float) -> Tuple[float, float, float]:
    """관절각 -> 팬 중심 (x, z, theta). 어깨 원점 기준."""
    a1, a2, a3 = q1, q1 + q2, q1 + q2 + q3
    wx = L1 * math.cos(a1) + L2 * math.cos(a2)
    wz = L1 * math.sin(a1) + L2 * math.sin(a2)
    return wx + L3_PAN * math.cos(a3), wz + L3_PAN * math.sin(a3), a3


def inverse(px: float, pz: float, theta: float) -> Optional[Tuple[float, float, float]]:
    """팬 중심 포즈(어깨 기준) -> 관절각(Elbow-Down, 사람 팔 자세). 도달 불가 시 None."""
    # 손목 위치 = 팬 중심 - 팬 방향 * L3_PAN
    wx = px - L3_PAN * math.cos(theta)
    wz = pz - L3_PAN * math.sin(theta)
    d2 = wx * wx + wz * wz
    c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1.0:
        return None
    q2 = math.acos(c2)  # 사람 팔처럼 팔꿈치가 아래로 향하는 해 (q2 >= 0)
    q1 = math.atan2(wz, wx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    q3 = wrap_angle(theta - q1 - q2)
    return q1, q2, q3
