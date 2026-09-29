"""LambdaMART ranking, offline NDCG, and exact online value under the true click model."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .market import SHOWN, Market, click_book_probs, examine, utility
from .store import FEATURES, build_rows


def label(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["booked"], 2, np.where(df["clicked"], 1, 0))


def train(df: pd.DataFrame, seed: int = 0) -> lgb.LGBMRanker:
    df = df.sort_values(["session", "hotel"])
    groups = df.groupby("session", sort=True).size().to_numpy()
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31,
                       min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                       random_state=seed, verbose=-1)
    m.fit(df[FEATURES], label(df), group=groups)
    return m


def ndcg_at(df: pd.DataFrame, scores: np.ndarray, k: int = 10) -> float:
    d = df.assign(s=scores, y=label(df))
    out = []
    for _, g in d.groupby("session", sort=False):
        y = g["y"].to_numpy()
        if y.max() == 0:
            continue
        gains = (2.0 ** y[np.argsort(-g["s"].to_numpy())][:k] - 1) / np.log2(np.arange(2, min(k, len(y)) + 2))
        ideal = (2.0 ** np.sort(y)[::-1][:k] - 1) / np.log2(np.arange(2, min(k, len(y)) + 2))
        out.append(gains.sum() / ideal.sum())
    return float(np.mean(out))


def candidates(m: Market, sessions: pd.DataFrame) -> pd.DataFrame:
    """Every launched hotel in the searched city, as of the search time."""
    h = m.hotels[["hotel", "city", "launch"]]
    rows = sessions[["session", "city", "t"]].merge(h, on="city")
    rows = rows[rows["launch"] <= rows["t"]]
    return rows[["session", "hotel", "t"]].assign(clicked=False, booked=False)


def true_propensity(m: Market, rows: pd.DataFrame, sessions: pd.DataFrame):
    """Per (session, hotel): P(click | examined) and P(book | click) under the true model."""
    hs = m.hotels.set_index("hotel")
    ps = sessions.set_index("session")["price_sens"]
    day = rows["t"].astype(int).clip(upper=m.prices.shape[1] - 1).to_numpy()
    u = utility(hs.loc[rows["hotel"], "quality"].to_numpy(), m.prices[rows["hotel"].to_numpy(), day],
                hs.loc[rows["hotel"], "distance"].to_numpy(), ps.loc[rows["session"]].to_numpy())
    return click_book_probs(u)


def online_value(rows: pd.DataFrame, scores: np.ndarray, pc: np.ndarray, pb: np.ndarray) -> float:
    """Exact expected bookings per session when each session shows its top-SHOWN hotels by score."""
    d = rows[["session"]].assign(s=scores, v=pc * pb)
    total, n = 0.0, 0
    ex = examine(np.arange(SHOWN))
    for _, g in d.groupby("session", sort=False):
        v = g["v"].to_numpy()[np.argsort(-g["s"].to_numpy())][:SHOWN]
        total += float((ex[:len(v)] * v).sum())
        n += 1
    return total / max(1, n)


def serving_rows(m: Market, sessions: pd.DataFrame) -> pd.DataFrame:
    return build_rows(m, candidates(m, sessions), "pit")
