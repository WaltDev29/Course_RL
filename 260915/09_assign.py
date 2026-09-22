# 몬테카를로 방법으로 원주율 π 추정
# 정사각형, 원, 무작위 점을 샘플 개수별로 시각화

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# 샘플 개수
sample_counts = 10 ** np.arange(1, 7)  # 10, 100, ..., 1,000,000

# 최대 샘플 개수만큼 무작위 점 생성
max_samples = max(sample_counts)

x = np.random.uniform(-1, 1, max_samples)
y = np.random.uniform(-1, 1, max_samples)

# 원 내부 여부
inside_circle = (x ** 2 + y ** 2) <= 1

# 누적 원 내부 점 개수
cumulative_inside = np.cumsum(inside_circle)

# π 추정값과 오차 저장
estimates = []
errors = []

for n in sample_counts:
    pi_estimate = 4 * cumulative_inside[n - 1] / n
    estimates.append(pi_estimate)
    errors.append(abs(pi_estimate - np.pi))


# 그래프 생성
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


# -------------------------
# 왼쪽: 정사각형과 원
# -------------------------

# 정사각형 테두리
square_x = [-1, 1, 1, -1, -1]
square_y = [-1, -1, 1, 1, -1]

ax1.plot(square_x, square_y, linewidth=2)

# 원
theta = np.linspace(0, 2 * np.pi, 500)

ax1.plot(
    np.cos(theta),
    np.sin(theta),
    linewidth=2
)

ax1.set_xlim(-1.05, 1.05)
ax1.set_ylim(-1.05, 1.05)

ax1.set_aspect('equal')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Monte Carlo Sampling')

ax1.grid(True, alpha=0.3)


# -------------------------
# 오른쪽: π 추정 오차
# -------------------------

ax2.set_xscale('log')
ax2.set_yscale('log')

ax2.set_xlabel('Number of Samples')
ax2.set_ylabel('Absolute Error')

ax2.set_title('Error of Monte Carlo Estimation')

ax2.grid(True, which='both', alpha=0.3)


# 실제 π 값과 추정값의 차이 그래프
ax2.plot(
    sample_counts,
    errors,
    marker='o',
    label='Absolute Error'
)

ax2.legend()


# -------------------------
# 애니메이션
# -------------------------

# 점을 저장할 객체
scatter_inside = ax1.scatter([], [], s=5, label='Inside Circle')
scatter_outside = ax1.scatter([], [], s=5, label='Outside Circle')

ax1.legend(loc='upper right')


def update(frame):
    """
    프레임마다 샘플 개수를 늘려가며 점을 갱신
    """

    n = sample_counts[frame]

    # 현재 샘플까지의 점
    current_x = x[:n]
    current_y = y[:n]

    current_inside = inside_circle[:n]

    # 원 내부 점
    scatter_inside.set_offsets(
        np.column_stack((
            current_x[current_inside],
            current_y[current_inside]
        ))
    )

    # 원 외부 점
    scatter_outside.set_offsets(
        np.column_stack((
            current_x[~current_inside],
            current_y[~current_inside]
        ))
    )

    # 제목에 현재 정보 표시
    pi_estimate = estimates[frame]
    error = errors[frame]

    ax1.set_title(
        f'Monte Carlo Sampling\n'
        f'Samples: {n:,} | π ≈ {pi_estimate:.8f}'
    )

    return scatter_inside, scatter_outside


# 애니메이션 실행
ani = FuncAnimation(
    fig,
    update,
    frames=len(sample_counts),
    interval=1000,
    repeat=True
)

plt.tight_layout()
plt.show()