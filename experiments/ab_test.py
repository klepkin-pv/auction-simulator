"""
A/B тестирование аукционных стратегий.

Проверяем гипотезы:
  H1: GSP даёт больше выручки, чем VCG
  H2: ML-оптимизированные ставки увеличивают CTR рекламодателя
  H3: Позиция 1 даёт статистически значимо больше кликов, чем позиция 2

Используем t-test и bootstrap для оценки значимости.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


def ttest_two_groups(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Двухвыборочный t-test с выводом результата."""
    t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)
    significant = p_value < alpha
    lift = (treatment.mean() - control.mean()) / control.mean() * 100

    return {
        "control_mean": round(control.mean(), 6),
        "treatment_mean": round(treatment.mean(), 6),
        "lift_pct": round(lift, 2),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": significant,
    }


def bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.mean,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap доверительный интервал для произвольной статистики."""
    rng = np.random.default_rng(seed)
    stats_boot = [
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    return (
        round(np.quantile(stats_boot, alpha), 6),
        round(np.quantile(stats_boot, 1 - alpha), 6),
    )


def run_position_experiment(impressions: pd.DataFrame) -> dict:
    """
    H3: Позиция 1 vs Позиция 2 — разница в CTR.
    """
    pos1 = impressions[impressions["rank"] == 1]["click"].values
    pos2 = impressions[impressions["rank"] == 2]["click"].values

    result = ttest_two_groups(pos2, pos1)
    ci_pos1 = bootstrap_ci(pos1)
    ci_pos2 = bootstrap_ci(pos2)

    print("=== Position 1 vs Position 2 CTR ===")
    print(f"  Position 1 CTR: {result['treatment_mean']:.4f}  95% CI: {ci_pos1}")
    print(f"  Position 2 CTR: {result['control_mean']:.4f}  95% CI: {ci_pos2}")
    print(f"  Lift: {result['lift_pct']}%  p-value: {result['p_value']}  significant: {result['significant']}")
    return result


def run_mechanism_revenue_experiment(
    gsp_revenues: np.ndarray,
    vcg_revenues: np.ndarray,
) -> dict:
    """
    H1: GSP vs VCG — разница в выручке платформы.
    """
    result = ttest_two_groups(vcg_revenues, gsp_revenues)
    print("=== GSP vs VCG Revenue ===")
    print(f"  GSP mean revenue: {result['treatment_mean']:.4f}")
    print(f"  VCG mean revenue: {result['control_mean']:.4f}")
    print(f"  Lift: {result['lift_pct']}%  p-value: {result['p_value']}  significant: {result['significant']}")
    return result


def run_bid_strategy_experiment(
    baseline_clicks: np.ndarray,
    optimized_clicks: np.ndarray,
) -> dict:
    """
    H2: ML-оптимизированные ставки vs базовые — разница в кликах.
    """
    result = ttest_two_groups(baseline_clicks, optimized_clicks)
    print("=== Baseline vs Optimized Bids ===")
    print(f"  Optimized clicks/round: {result['treatment_mean']:.4f}")
    print(f"  Baseline clicks/round:  {result['control_mean']:.4f}")
    print(f"  Lift: {result['lift_pct']}%  p-value: {result['p_value']}  significant: {result['significant']}")
    return result
