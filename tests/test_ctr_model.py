from __future__ import annotations

import numpy as np
import pandas as pd

from models.ctr_model import LogRegCTR, NeuralCTR, build_features


def _make_impressions(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "rank": rng.integers(1, 4, size=n),
        "bid": rng.uniform(10.0, 100.0, size=n),
        "user_quality": rng.uniform(0.0, 1.0, size=n),
        "category": rng.choice(["electronics", "fashion", "food"], size=n),
        "click": rng.binomial(1, 0.1, size=n).astype(float),
    })


def test_build_features_shape() -> None:
    df = _make_impressions(n=50)
    X, y = build_features(df)

    assert X.shape[0] == 50
    assert X.shape[1] == 4
    assert y.shape[0] == 50


def test_logreg_ctr_trains_and_evaluates() -> None:
    df = _make_impressions(n=100)
    X, y = build_features(df)

    model = LogRegCTR()
    model.fit(X, y)
    metrics = model.evaluate(X, y)

    assert "auc" in metrics
    assert "log_loss" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["log_loss"] >= 0.0


def test_neural_ctr_trains_and_evaluates() -> None:
    df = _make_impressions(n=60)
    X, y = build_features(df)

    model = NeuralCTR(epochs=2, lr=1e-2, batch_size=16)
    model.fit(X, y)
    metrics = model.evaluate(X, y)

    assert "auc" in metrics
    assert "log_loss" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["log_loss"] >= 0.0
