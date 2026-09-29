"""Synthetic accommodation marketplace that emits an event log and knows the truth.

Hotels have a latent quality. Over 120 days: prices move daily, reviews arrive and slowly reveal
quality, new hotels launch, and search sessions are ranked by a production popularity ranker. Shoppers
examine positions with decaying attention (position bias), click on attractive hotels and sometimes
book. Every event is timestamped, so any feature can be reconstructed as of any moment, which is what
point-in-time correctness means.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

CITIES = 5
PER_CITY = 60
DAYS = 150                                   # 0-120 logged for training and holdout, 120-150 serve online
TRAIN_END = 96
LOG_END = 120
SESSIONS_PER_CITY_DAY = 20
SHOWN = 25                                   # results shown per search


def examine(k):
    return 1.0 / (1.0 + k) ** 0.7              # probability the shopper looks at position k (0-based)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Market:
    hotels: pd.DataFrame
    prices: np.ndarray                         # (n_hotels, DAYS)
    reviews: pd.DataFrame                      # hotel, t, score
    impressions: pd.DataFrame                  # session, hotel, t, pos, clicked, booked
    sessions: pd.DataFrame                     # session, city, t, price_sens
    bookings: pd.DataFrame                     # hotel, t


def utility(quality, price, distance, price_sens):
    return 1.1 * (quality - 7.0) - price_sens * np.log(price / 100.0) - 0.25 * distance


def click_book_probs(u):
    return sigmoid(u - 0.4), sigmoid(u - 3.4)  # P(click | examined), P(book | click)


def expected_bookings(order_quality_price, sess_price_sens) -> float:
    """Exact expected bookings of one ranked list under the true model."""
    q, p, d = order_quality_price
    u = utility(q, p, d, sess_price_sens)
    pc, pb = click_book_probs(u)
    k = np.arange(len(u))
    return float((examine(k) * pc * pb).sum())


def simulate(seed: int = 0) -> Market:
    r = np.random.default_rng(seed)
    n = CITIES * PER_CITY
    city = np.repeat(np.arange(CITIES), PER_CITY)
    quality = np.clip(r.normal(7.4, 0.9, n), 4.5, 9.8)
    stars = np.clip(np.round(quality / 2 + r.normal(0, 0.6, n)), 1, 5)
    distance = np.round(r.gamma(2.0, 1.2, n), 2)
    base = 70 * np.exp(0.35 * (stars - 3) + r.normal(0, 0.25, n))
    launch = np.where(r.random(n) < 0.25, r.uniform(10, 100, n), 0.0)  # a quarter of hotels launch mid-period
    walk = np.cumsum(r.normal(0, 0.05, (n, DAYS)), axis=1)
    weekend = np.array([1.15 if d % 7 in (4, 5) else 1.0 for d in range(DAYS)])
    prices = np.round(base[:, None] * np.exp(walk - walk.mean(1, keepdims=True)) * weekend[None, :], 2)
    hotels = pd.DataFrame({"hotel": np.arange(n), "city": city, "quality": quality, "stars": stars,
                           "distance": distance, "launch": launch})

    rev = []
    for h in range(n):
        k = r.poisson(0.35 * (DAYS - launch[h]))
        ts = np.sort(r.uniform(launch[h], DAYS, k))
        rev += [(h, t, float(np.clip(quality[h] + r.normal(0, 1.6), 1, 10))) for t in ts]
    reviews = pd.DataFrame(rev, columns=["hotel", "t", "score"])

    # chronological sessions ranked by the production ranker: popularity (bookings in the last 7 days)
    recent = [deque() for _ in range(n)]
    imps, sess, books = [], [], []
    sid = 0
    for day in range(DAYS):
        for c in range(CITIES):
            for t in np.sort(day + r.random(SESSIONS_PER_CITY_DAY)):
                hs = np.where((city == c) & (launch <= t))[0]
                for h in hs:
                    while recent[h] and recent[h][0] < t - 7:
                        recent[h].popleft()
                pop = np.array([len(recent[h]) for h in hs]) + r.random(len(hs)) * 0.5
                shown = hs[np.argsort(-pop)][:SHOWN]
                ps = float(np.clip(r.normal(0.9, 0.35), 0.2, 1.8))
                u = utility(quality[shown], prices[shown, day], distance[shown], ps)
                pc, pb = click_book_probs(u)
                ex = r.random(len(shown)) < examine(np.arange(len(shown)))
                clicked = ex & (r.random(len(shown)) < pc)
                booked = clicked & (r.random(len(shown)) < pb)
                if booked.sum() > 1:                                   # a shopper books at most one hotel
                    keep = np.flatnonzero(booked)[0]
                    booked[:] = False
                    booked[keep] = True
                for pos, h in enumerate(shown):
                    imps.append((sid, int(h), float(t), pos, bool(clicked[pos]), bool(booked[pos])))
                    if booked[pos]:
                        recent[h].append(t)
                        books.append((int(h), float(t)))
                sess.append((sid, c, float(t), ps))
                sid += 1
    return Market(hotels, prices,
                  reviews,
                  pd.DataFrame(imps, columns=["session", "hotel", "t", "pos", "clicked", "booked"]),
                  pd.DataFrame(sess, columns=["session", "city", "t", "price_sens"]),
                  pd.DataFrame(books, columns=["hotel", "t"]))
