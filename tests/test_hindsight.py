import threading
import time

import numpy as np
import pandas as pd
import pytest

from hindsight.interleave import team_draft
from hindsight.market import LOG_END, SHOWN, examine, simulate
from hindsight.rank import ndcg_at, online_value
from hindsight.store import OnlineStore, build_rows


@pytest.fixture(scope="module")
def market():
    return simulate(3)


@pytest.fixture(scope="module")
def rows(market):
    imp = market.impressions[market.impressions.t < LOG_END]
    sample = imp[imp.session.isin(imp.session.drop_duplicates().sample(300, random_state=0))]
    return sample[["session", "hotel", "t", "pos", "clicked", "booked"]]


def test_simulation_is_deterministic():
    a, b = simulate(9), simulate(9)
    pd.testing.assert_frame_equal(a.impressions, b.impressions)


def test_point_in_time_features_only_use_strictly_earlier_events(market, rows):
    pit = build_rows(market, rows, "pit")
    rv, bk, im = market.reviews, market.bookings, market.impressions
    for _, r in pit.sample(150, random_state=1).iterrows():
        assert r.cnt == ((rv.hotel == r.hotel) & (rv.t < r.t)).sum()
        assert r.bookings_7d == ((bk.hotel == r.hotel) & (bk.t < r.t) & (bk.t >= r.t - 7)).sum()
        same_day = (im.hotel == r.hotel) & (im.t < r.t) & (im.t >= np.floor(r.t))
        assert r.imps_today == same_day.sum() and r.clicks_today == (same_day & im.clicked).sum()


def test_naive_daily_features_contain_the_rows_own_outcome(market, rows):
    naive = build_rows(market, rows, "naive")
    booked = naive[naive.booked]
    assert len(booked) and (booked.bookings_today >= 1).all()
    pit = build_rows(market, rows, "pit")
    assert (pit.clicks_today.to_numpy() <= naive.clicks_today.to_numpy()).all()


def test_ndcg_is_one_for_the_ideal_order(rows):
    y = np.where(rows.booked, 2, np.where(rows.clicked, 1, 0))
    assert ndcg_at(rows, y.astype(float)) == pytest.approx(1.0)


def test_sorting_by_true_propensity_is_optimal():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"session": np.repeat(np.arange(50), 40)})
    pc, pb = rng.random(len(rows)), rng.random(len(rows))
    best = online_value(rows, pc * pb, pc, pb)
    for _ in range(5):
        assert online_value(rows, rng.random(len(rows)), pc, pb) <= best + 1e-12


def test_team_draft_is_a_fair_duplicate_free_merge():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a, b = rng.permutation(40), rng.permutation(40)
        lst, team = team_draft(a, b, rng)
        assert len(lst) == SHOWN == len(set(lst))
        c = np.cumsum(np.where(np.asarray(team) == 0, 1, -1))
        assert np.abs(c).max() <= 1


def test_examination_decays():
    assert np.all(np.diff(examine(np.arange(10))) < 0)


def test_single_flight_collapses_concurrent_identical_lookups():
    calls = []

    def backend(k):
        calls.append(k)
        time.sleep(0.05)
        return k * 2

    store = OnlineStore(backend, coalesce=True)
    out = []
    ts = [threading.Thread(target=lambda: out.append(store.get(7)[0])) for _ in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out == [14] * 16 and len(calls) == 1
    plain = OnlineStore(backend, coalesce=False)
    calls.clear()
    ts = [threading.Thread(target=lambda: plain.get(7)) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(calls) == 8
