"""
Генерация синтетических данных для симулятора аукциона.
Имитируем рекламодателей, товары и пользовательские запросы.
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

RNG = np.random.default_rng(42)


def generate_advertisers(n: int = 50) -> pd.DataFrame:
    """Создаём рекламодателей с бюджетами и базовыми ставками."""
    return pd.DataFrame({
        "advertiser_id": range(n),
        "budget": RNG.uniform(1000, 50000, n).round(2),
        "base_bid": RNG.uniform(5, 100, n).round(2),
        "category": RNG.choice(["electronics", "fashion", "home", "sports", "beauty"], n),
    })


def generate_queries(n: int = 10_000) -> pd.DataFrame:
    """Генерируем поисковые запросы пользователей."""
    categories = ["electronics", "fashion", "home", "sports", "beauty"]
    return pd.DataFrame({
        "query_id": range(n),
        "category": RNG.choice(categories, n),
        "user_quality": RNG.beta(2, 5, n).round(4),  # склонность к клику
    })


def generate_impressions(
    advertisers: pd.DataFrame,
    queries: pd.DataFrame,
    slots: int = 3,
) -> pd.DataFrame:
    """
    Для каждого запроса проводим аукцион среди рекламодателей той же категории.
    Возвращаем таблицу показов с реальными CTR и кликами.
    """
    records = []
    for _, q in queries.iterrows():
        candidates = advertisers[advertisers["category"] == q["category"]]
        if len(candidates) < slots:
            candidates = advertisers.sample(slots, random_state=int(q["query_id"]))

        # Добавляем шум к ставкам — имитируем стратегическое поведение
        bids = candidates["base_bid"].values * RNG.uniform(0.8, 1.2, len(candidates))
        top_idx = np.argsort(bids)[::-1][:slots]

        for rank, idx in enumerate(top_idx):
            adv = candidates.iloc[idx]
            # CTR зависит от позиции, качества пользователя и рекламодателя
            position_discount = 1 / (rank + 1) ** 0.5
            true_ctr = q["user_quality"] * position_discount * RNG.beta(2, 8)
            click = int(RNG.random() < true_ctr)

            records.append({
                "query_id": int(q["query_id"]),
                "advertiser_id": int(adv["advertiser_id"]),
                "rank": rank + 1,
                "bid": round(float(bids[idx]), 4),
                "true_ctr": round(float(true_ctr), 6),
                "click": click,
                "category": q["category"],
                "user_quality": q["user_quality"],
            })

    return pd.DataFrame(records)


def save_to_db(engine, advertisers, queries, impressions):
    advertisers.to_sql("advertisers", engine, if_exists="replace", index=False)
    queries.to_sql("queries", engine, if_exists="replace", index=False)
    impressions.to_sql("impressions", engine, if_exists="replace", index=False)
    print(f"Saved: {len(advertisers)} advertisers, {len(queries)} queries, {len(impressions)} impressions")


if __name__ == "__main__":
    engine = create_engine("sqlite:///data/auction.db")

    advertisers = generate_advertisers(50)
    queries = generate_queries(10_000)
    impressions = generate_impressions(advertisers, queries)

    save_to_db(engine, advertisers, queries, impressions)
