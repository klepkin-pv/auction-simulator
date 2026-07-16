from __future__ import annotations

import numpy as np

from models.bid_optimizer import (
    BidOptimizerConfig,
    GradientBidOptimizer,
    analytical_optimal_bid,
    simulate_budget_pacing,
)


def test_analytical_optimal_bid() -> None:
    bid = analytical_optimal_bid(predicted_ctr=0.05, target_cpc=100.0, quality_score=1.0)
    assert bid == 5.0


def test_analytical_optimal_bid_floor() -> None:
    bid = analytical_optimal_bid(predicted_ctr=0.0, target_cpc=100.0)
    assert bid == 0.01


def test_gradient_optimizer_returns_positive_bid() -> None:
    config = BidOptimizerConfig(budget=1000.0, target_cpc=50.0, n_slots=3)
    optimizer = GradientBidOptimizer(config, competitor_bids=np.array([40.0, 30.0, 20.0]))
    best_bid = optimizer.optimize(predicted_ctr=0.05, n_steps=50, lr=0.5)

    assert isinstance(best_bid, float)
    assert best_bid >= config.reserve_price


def test_simulate_budget_pacing() -> None:
    stats = simulate_budget_pacing(
        bids_per_round=[10.0, 20.0],
        ctrs=[0.1, 0.2],
        budget=100.0,
        n_rounds=100,
    )

    assert stats["total_spent"] <= 100.0
    assert stats["total_clicks"] >= 0
    assert 0 <= stats["budget_utilization"] <= 1
    assert stats["avg_cpc"] >= 0
