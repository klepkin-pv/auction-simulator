from __future__ import annotations

from auction.mechanisms import Bid, compare_mechanisms, run_gsp, run_vcg


def test_gsp_basic_three_slots() -> None:
    bids = [
        Bid(advertiser_id=1, bid=10.0, quality_score=1.0),
        Bid(advertiser_id=2, bid=8.0, quality_score=1.0),
        Bid(advertiser_id=3, bid=5.0, quality_score=1.0),
        Bid(advertiser_id=4, bid=2.0, quality_score=1.0),
    ]
    results = run_gsp(bids, n_slots=3)

    assert len(results) == 3
    assert [r.advertiser_id for r in results] == [1, 2, 3]
    assert results[0].price_paid == 8.0  # next effective bid / qs
    assert results[1].price_paid == 5.0
    assert results[2].price_paid == 1.0  # reserve price


def test_gsp_quality_score_changes_price() -> None:
    bids = [
        Bid(advertiser_id=1, bid=10.0, quality_score=2.0),
        Bid(advertiser_id=2, bid=8.0, quality_score=1.0),
    ]
    results = run_gsp(bids, n_slots=2)

    # effective bids: 20 vs 8
    assert results[0].advertiser_id == 1
    assert results[0].price_paid == 4.0  # 8 / 2.0


def test_gsp_empty_bids() -> None:
    assert run_gsp([], n_slots=3) == []


def test_vcg_basic_three_slots() -> None:
    bids = [
        Bid(advertiser_id=1, bid=10.0, quality_score=1.0),
        Bid(advertiser_id=2, bid=8.0, quality_score=1.0),
        Bid(advertiser_id=3, bid=5.0, quality_score=1.0),
        Bid(advertiser_id=4, bid=2.0, quality_score=1.0),
    ]
    ctr_by_slot = [1.0, 0.7, 0.5, 0.3]
    results = run_vcg(bids, n_slots=3, ctr_by_slot=ctr_by_slot)

    assert len(results) == 3
    assert [r.advertiser_id for r in results] == [1, 2, 3]
    assert all(r.price_paid >= 0 for r in results)


def test_compare_mechanisms_prints_revenue(capsys) -> None:
    bids = [
        Bid(advertiser_id=1, bid=10.0),
        Bid(advertiser_id=2, bid=8.0),
        Bid(advertiser_id=3, bid=5.0),
    ]
    gsp, vcg = compare_mechanisms(bids, n_slots=2)
    assert len(gsp) == len(vcg) == 2

    captured = capsys.readouterr()
    assert "GSP revenue" in captured.out
    assert "VCG revenue" in captured.out
