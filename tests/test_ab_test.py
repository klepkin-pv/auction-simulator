from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.ab_test import bootstrap_ci, run_mechanism_revenue_experiment, run_position_experiment, ttest_two_groups


def test_ttest_detects_significant_difference() -> None:
    rng = np.random.default_rng(0)
    control = rng.normal(loc=10.0, scale=1.0, size=100)
    treatment = rng.normal(loc=12.0, scale=1.0, size=100)

    result = ttest_two_groups(control, treatment, alpha=0.05)
    assert result["significant"] is True
    assert result["lift_pct"] > 0


def test_ttest_no_difference_when_equal() -> None:
    rng = np.random.default_rng(0)
    control = rng.normal(loc=10.0, scale=1.0, size=100)
    treatment = rng.normal(loc=10.0, scale=1.0, size=100)

    result = ttest_two_groups(control, treatment, alpha=0.05)
    assert result["significant"] is False


def test_bootstrap_ci() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower, upper = bootstrap_ci(data, n_bootstrap=1000, seed=0)

    assert lower < upper
    assert lower <= np.mean(data) <= upper


def test_run_position_experiment() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "rank": np.repeat([1, 2], 100),
        "click": np.concatenate([
            rng.binomial(1, 0.15, size=100),
            rng.binomial(1, 0.08, size=100),
        ]),
    })

    result = run_position_experiment(df)
    assert "treatment_mean" in result
    assert "control_mean" in result
    assert "p_value" in result
    assert "significant" in result


def test_run_mechanism_revenue_experiment() -> None:
    rng = np.random.default_rng(0)
    gsp = rng.normal(loc=50.0, scale=5.0, size=30)
    vcg = rng.normal(loc=48.0, scale=5.0, size=30)

    result = run_mechanism_revenue_experiment(gsp, vcg)
    assert "treatment_mean" in result
    assert "p_value" in result
