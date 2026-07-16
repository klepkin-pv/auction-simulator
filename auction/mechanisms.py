"""
Реализация аукционных механизмов: GSP и VCG.

GSP (Generalized Second Price) — именно этот механизм используется
в большинстве рекламных платформ, включая Ozon.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Bid:
    advertiser_id: int
    bid: float
    quality_score: float = 1.0  # множитель качества объявления

    @property
    def effective_bid(self) -> float:
        """Эффективная ставка с учётом качества (rank score)."""
        return self.bid * self.quality_score


@dataclass
class AuctionResult:
    advertiser_id: int
    rank: int
    bid: float
    price_paid: float  # цена за клик (CPC)
    quality_score: float


def run_gsp(bids: List[Bid], n_slots: int = 3) -> List[AuctionResult]:
    """
    Generalized Second Price аукцион.

    Победители ранжируются по effective_bid.
    Каждый платит ставку следующего конкурента / своё quality_score.
    Последний слот платит reserve_price / quality_score.
    """
    reserve_price = 1.0
    sorted_bids = sorted(bids, key=lambda b: b.effective_bid, reverse=True)
    winners = sorted_bids[:n_slots]

    results = []
    for rank, winner in enumerate(winners):
        if rank + 1 < len(sorted_bids):
            next_effective = sorted_bids[rank + 1].effective_bid
        else:
            next_effective = reserve_price

        # CPC = следующая эффективная ставка / quality_score победителя
        price_paid = next_effective / winner.quality_score
        results.append(AuctionResult(
            advertiser_id=winner.advertiser_id,
            rank=rank + 1,
            bid=winner.bid,
            price_paid=round(price_paid, 4),
            quality_score=winner.quality_score,
        ))

    return results


def run_vcg(bids: List[Bid], n_slots: int = 3, ctr_by_slot: List[float] = None) -> List[AuctionResult]:
    """
    VCG (Vickrey–Clarke–Groves) аукцион.

    Теоретически оптимален (truthful), но сложнее в реализации.
    Каждый победитель платит externality — ущерб, нанесённый остальным.

    ctr_by_slot: CTR для каждого слота (позиционные дисконты).
    """
    if ctr_by_slot is None:
        ctr_by_slot = [1 / (i + 1) ** 0.5 for i in range(n_slots + 1)]

    sorted_bids = sorted(bids, key=lambda b: b.effective_bid, reverse=True)
    winners = sorted_bids[:n_slots]

    def total_value(allocation: List[Bid]) -> float:
        return sum(
            b.effective_bid * ctr_by_slot[i]
            for i, b in enumerate(allocation)
        )

    results = []
    for rank, winner in enumerate(winners):
        # Считаем социальную ценность без этого победителя
        others = [b for b in sorted_bids if b.advertiser_id != winner.advertiser_id]
        value_without = total_value(others[:n_slots])
        value_with_others = total_value(winners[:rank] + winners[rank + 1:n_slots] + others[:1])

        externality = value_without - value_with_others
        price_paid = max(externality / ctr_by_slot[rank], 0)

        results.append(AuctionResult(
            advertiser_id=winner.advertiser_id,
            rank=rank + 1,
            bid=winner.bid,
            price_paid=round(price_paid, 4),
            quality_score=winner.quality_score,
        ))

    return results


def compare_mechanisms(bids: List[Bid], n_slots: int = 3):
    """Сравниваем выручку GSP vs VCG для одного аукциона."""
    gsp = run_gsp(bids, n_slots)
    vcg = run_vcg(bids, n_slots)

    gsp_revenue = sum(r.price_paid for r in gsp)
    vcg_revenue = sum(r.price_paid for r in vcg)

    print(f"GSP revenue: {gsp_revenue:.4f}")
    print(f"VCG revenue: {vcg_revenue:.4f}")
    print(f"Difference:  {gsp_revenue - vcg_revenue:.4f} (GSP {'>' if gsp_revenue > vcg_revenue else '<='} VCG)")
    return gsp, vcg
