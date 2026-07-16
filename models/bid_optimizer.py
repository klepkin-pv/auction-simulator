"""
Оптимизация ставок для рекламодателя.

Задача: максимизировать количество кликов при ограниченном бюджете.
Используем два подхода:
  1. Аналитический — на основе предсказанного CTR и цены аукциона.
  2. Градиентный — оптимизируем ставку через PyTorch autograd.
"""

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class BidOptimizerConfig:
    budget: float
    target_cpc: float        # целевая цена за клик
    n_slots: int = 3
    reserve_price: float = 1.0


def analytical_optimal_bid(
    predicted_ctr: float,
    target_cpc: float,
    quality_score: float = 1.0,
) -> float:
    """
    В GSP-аукционе оптимальная ставка при известном CTR:
    bid* = target_cpc * predicted_ctr * quality_score

    Это упрощённая версия — в реальности нужно учитывать
    распределение ставок конкурентов.
    """
    return max(target_cpc * predicted_ctr * quality_score, 0.01)


class GradientBidOptimizer:
    """
    Дифференцируемый симулятор аукциона для оптимизации ставки.

    Идея: аппроксимируем дискретный аукцион мягкой функцией (softmax),
    чтобы можно было считать градиент по ставке.
    """

    def __init__(self, config: BidOptimizerConfig, competitor_bids: np.ndarray):
        self.config = config
        # Фиксируем ставки конкурентов
        self.competitor_bids = torch.tensor(competitor_bids, dtype=torch.float32)

    def _soft_win_prob(self, bid: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """Вероятность выиграть топ-слот через softmax-аппроксимацию."""
        all_bids = torch.cat([bid, self.competitor_bids])
        probs = torch.softmax(all_bids / temperature, dim=0)
        return probs[0]  # вероятность нашей ставки быть первой

    def optimize(
        self,
        predicted_ctr: float,
        n_steps: int = 200,
        lr: float = 0.5,
    ) -> float:
        """Возвращает оптимальную ставку, максимизирующую ожидаемые клики."""
        bid = torch.tensor([self.config.target_cpc], requires_grad=True)
        optimizer = torch.optim.Adam([bid], lr=lr)

        ctr = torch.tensor(predicted_ctr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            win_prob = self._soft_win_prob(bid)
            expected_clicks = win_prob * ctr
            # Минимизируем отрицательные клики (= максимизируем клики)
            loss = -expected_clicks
            loss.backward()
            optimizer.step()
            # Ограничиваем ставку снизу
            with torch.no_grad():
                bid.clamp_(min=self.config.reserve_price)

        return round(float(bid.item()), 4)


def simulate_budget_pacing(
    bids_per_round: list[float],
    ctrs: list[float],
    budget: float,
    n_rounds: int = 1000,
) -> dict:
    """
    Симулируем расход бюджета по раундам.
    Возвращаем статистику: потраченный бюджет, клики, средний CPC.
    """
    spent = 0.0
    clicks = 0
    rng = np.random.default_rng(0)

    for i in range(n_rounds):
        if spent >= budget:
            break
        bid = bids_per_round[i % len(bids_per_round)]
        ctr = ctrs[i % len(ctrs)]
        click = int(rng.random() < ctr)
        cost = bid * click
        if spent + cost <= budget:
            spent += cost
            clicks += click

    return {
        "total_spent": round(spent, 2),
        "total_clicks": clicks,
        "avg_cpc": round(spent / clicks, 4) if clicks > 0 else 0,
        "budget_utilization": round(spent / budget, 4),
    }
