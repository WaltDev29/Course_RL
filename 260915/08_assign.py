# 몬테카를로 방법으로 원주율 π 추정
# 샘플 개수를 10^1, 10^2, 10^3, ... 순으로 증가시키며
# 추정값과 실제 π 값의 차이를 그래프로 시각화

import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_pi(sample_counts):
    """
    몬테카를로 방법으로 π를 추정하고,
    각 샘플 개수에 대한 추정값과 오차를 반환
    """

    max_samples = max(sample_counts)

    # 정사각형 [-1, 1] × [-1, 1] 안에 무작위 점 생성
    x = np.random.uniform(-1, 1, max_samples)
    y = np.random.uniform(-1, 1, max_samples)

    # 원점으로부터의 거리가 1 이하이면 원 내부
    inside_circle = (x ** 2 + y ** 2) <= 1

    # 누적 원 내부 점 개수
    cumulative_inside = np.cumsum(inside_circle)

    # 샘플 개수별 π 추정값
    estimates = []

    for n in sample_counts:
        inside_count = cumulative_inside[n - 1]
        pi_estimate = 4 * inside_count / n
        estimates.append(pi_estimate)

    return np.array(estimates)


# 샘플 개수: 10^1, 10^2, ..., 10^6
sample_counts = 10 ** np.arange(1, 7)

# 몬테카를로 π 추정
estimates = monte_carlo_pi(sample_counts)

# 실제 π 값과의 차이 (절댓값 오차)
errors = np.abs(estimates - np.pi)


# 결과 출력
print("샘플 개수별 π 추정 결과")
print("-" * 45)
print(f"{'샘플 개수'} {'추정값':>11} {'오차':>11}")

for n, estimate, error in zip(sample_counts, estimates, errors):
    print(f"{n:>12} {estimate:>15.8f} {error:>15.8f}")


# 그래프 시각화
plt.figure(figsize=(10, 6))

plt.plot(
    sample_counts,
    errors,
    marker='o',
    label='Monte Carlo Error'
)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of Samples')
plt.ylabel('Absolute Error |Estimated π - Actual π|')
plt.title('Monte Carlo Estimation of π')

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.show()