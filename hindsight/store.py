"""Feature store: an offline store that builds training rows two ways, and an online store.

  naive join            dynamic features from the latest snapshot (what a batch table holds today)
  point-in-time join    every feature reconstructed from events strictly before the row's timestamp,
                        using DuckDB ASOF joins over cumulative event tables

The online store serves the same definitions at request time, with freshness metadata and request
coalescing so a burst of identical lookups hits the backend once.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import duckdb
import numpy as np
import pandas as pd

from .market import LOG_END, Market

PRIOR_SCORE, PRIOR_WEIGHT = 7.5, 3.0
FEATURES = ["log_price", "price_rel", "review_score", "log_reviews", "bookings_7d", "ctr", "clicks_today", "imps_today",
            "bookings_today", "stars", "distance", "age_days"]


def _cum(df: pd.DataFrame, value_cols: dict) -> pd.DataFrame:
    df = df.sort_values(["hotel", "t"]).copy()
    for out, col in value_cols.items():
        df[out] = df.groupby("hotel")[col].cumsum() if col else df.groupby("hotel").cumcount() + 1
    return df


def build_rows(m: Market, rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    """rows: session, hotel, t (+ label columns). mode: 'pit' or 'naive'. Returns rows plus FEATURES."""
    con = duckdb.connect()
    h = m.hotels[["hotel", "stars", "distance", "launch"]]
    day = np.minimum(rows["t"].to_numpy().astype(int), m.prices.shape[1] - 1)
    rows = rows.assign(day=day)
    if mode == "naive":
        cut = float(rows["t"].max()) if len(rows) else LOG_END      # "latest snapshot" at the time of the batch job
        cut = max(cut, LOG_END)
        rev = m.reviews[m.reviews.t <= cut].groupby("hotel").agg(cnt=("score", "size"), ssum=("score", "sum"))
        b7 = m.bookings[(m.bookings.t > cut - 7) & (m.bookings.t <= cut)].groupby("hotel").size().rename("b7")
        imp = m.impressions[m.impressions.t <= cut].groupby("hotel").agg(imps=("clicked", "size"), clicks=("clicked", "sum"))
        snap = h.set_index("hotel").join([rev, b7, imp]).fillna(0).reset_index()
        out = rows.merge(snap, on="hotel", how="left")
        # daily aggregates from the batch table: complete days, so they include the row's own outcome
        im = m.impressions.assign(day=m.impressions.t.astype(int))
        daily = im.groupby(["hotel", "day"]).agg(imps_td=("clicked", "size"), clicks_td=("clicked", "sum"),
                                                 books_td=("booked", "sum")).reset_index()
        out = out.merge(daily, on=["hotel", "day"], how="left").fillna({"imps_td": 0, "clicks_td": 0, "books_td": 0})
        out["price"] = m.prices[out["hotel"].to_numpy(), int(min(cut, m.prices.shape[1] - 1))]
    else:
        rc = _cum(m.reviews, {"cnt": None, "ssum": "score"})[["hotel", "t", "cnt", "ssum"]]
        bc = _cum(m.bookings, {"bcum": None})[["hotel", "t", "bcum"]]
        ic = m.impressions.assign(clicks=m.impressions.clicked.astype(int))
        ic = _cum(ic, {"imps": None, "cclicks": "clicks"})[["hotel", "t", "imps", "cclicks"]]
        ib = m.impressions.assign(b=m.impressions.booked.astype(int), c=m.impressions.clicked.astype(int))
        ib = ib.sort_values(["hotel", "t", "session"])
        ib["ci"] = ib.groupby("hotel").cumcount() + 1
        ib["cc"] = ib.groupby("hotel")["c"].cumsum()
        ib["cb"] = ib.groupby("hotel")["b"].cumsum()
        ib = ib.groupby(["hotel", "t"], as_index=False)[["ci", "cc", "cb"]].max()   # one row per (hotel, instant)
        con.register("ib", ib)
        con.register("q", rows.assign(t7=rows["t"] - 7.0, t0=np.floor(rows["t"])))
        con.register("rc", rc)
        con.register("bc", bc)
        con.register("ic", ic)
        out = con.execute("""
            SELECT q.*, coalesce(rc.cnt, 0) AS cnt, coalesce(rc.ssum, 0) AS ssum,
                   coalesce(b1.bcum, 0) - coalesce(b0.bcum, 0) AS b7,
                   coalesce(ic.imps, 0) AS imps, coalesce(ic.cclicks, 0) AS clicks,
                   coalesce(n1.ci, 0) - coalesce(n0.ci, 0) AS imps_td,
                   coalesce(n1.cc, 0) - coalesce(n0.cc, 0) AS clicks_td,
                   coalesce(n1.cb, 0) - coalesce(n0.cb, 0) AS books_td
            FROM q
            ASOF LEFT JOIN rc ON q.hotel = rc.hotel AND q.t  > rc.t
            ASOF LEFT JOIN bc b1 ON q.hotel = b1.hotel AND q.t  > b1.t
            ASOF LEFT JOIN bc b0 ON q.hotel = b0.hotel AND q.t7 > b0.t
            ASOF LEFT JOIN ic ON q.hotel = ic.hotel AND q.t  > ic.t
            ASOF LEFT JOIN ib n1 ON q.hotel = n1.hotel AND q.t  > n1.t
            ASOF LEFT JOIN ib n0 ON q.hotel = n0.hotel AND q.t0 > n0.t
        """).df()
        out = out.merge(h, on="hotel", how="left")
        out["price"] = m.prices[out["hotel"].to_numpy(), out["day"].to_numpy()]
    out["review_score"] = (PRIOR_SCORE * PRIOR_WEIGHT + out["ssum"]) / (PRIOR_WEIGHT + out["cnt"])
    out["log_reviews"] = np.log1p(out["cnt"])
    out["bookings_7d"] = out["b7"]
    out["ctr"] = (out["clicks"] + 1.0) / (out["imps"] + 10.0)
    out["log_price"] = np.log(out["price"])
    out["price_rel"] = out["log_price"] - out.groupby("session")["log_price"].transform("mean")
    out["age_days"] = np.maximum(0.0, out["t"] - out["launch"])
    out["clicks_today"], out["imps_today"], out["bookings_today"] = out["clicks_td"], out["imps_td"], out["books_td"]
    return out.sort_values(["session", "hotel"]).reset_index(drop=True)


# -------------------------------------------------------------------------------- online
class OnlineStore:
    """Key-value serving with freshness metadata and single-flight request coalescing."""

    def __init__(self, backend, ttl_s: float = 0.0, coalesce: bool = True):
        self.backend, self.ttl, self.coalesce = backend, ttl_s, coalesce
        self.cache: dict = {}
        self.inflight: dict = {}
        self.lock = threading.Lock()
        self.backend_calls = 0

    def get(self, key):
        now = time.monotonic()
        with self.lock:
            hit = self.cache.get(key)
            if hit and now - hit[1] <= self.ttl:
                return hit[0], now - hit[1]
            fut = self.inflight.get(key) if self.coalesce else None
            leader = fut is None
            if leader:
                fut = Future()
                if self.coalesce:
                    self.inflight[key] = fut
                self.backend_calls += 1
        if leader:
            try:
                val = self.backend(key)
                fut.set_result(val)
                with self.lock:
                    self.cache[key] = (val, time.monotonic())
            finally:
                with self.lock:
                    self.inflight.pop(key, None)
        return fut.result(), 0.0
