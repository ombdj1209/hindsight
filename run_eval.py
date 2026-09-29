"""Reproduces every number in the README.

    python run_eval.py            # 3 simulated markets
    python run_eval.py --quick    # 1 market, fewer power simulations
"""
from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np

from hindsight.interleave import power
from hindsight.market import LOG_END, TRAIN_END, simulate
from hindsight.rank import ndcg_at, online_value, serving_rows, train, true_propensity
from hindsight.report import write_report
from hindsight.store import FEATURES, OnlineStore, build_rows

OUT = Path(__file__).resolve().parent / "results"


def one_market(seed: int, power_sims: int, rng):
    m = simulate(seed)
    logged = m.impressions[m.impressions.t < LOG_END][["session", "hotel", "t", "pos", "clicked", "booked"]]
    pit, naive = build_rows(m, logged, "pit"), build_rows(m, logged, "naive")
    tr = lambda d: d[d.t < TRAIN_END]
    ho = lambda d: d[d.t >= TRAIN_END]
    m_naive, m_pit = train(tr(naive), seed), train(tr(pit), seed)

    offline = {
        "naive model, naive holdout": ndcg_at(ho(naive), m_naive.predict(ho(naive)[FEATURES])),
        "naive model, point-in-time holdout": ndcg_at(ho(pit), m_naive.predict(ho(pit)[FEATURES])),
        "point-in-time model, point-in-time holdout": ndcg_at(ho(pit), m_pit.predict(ho(pit)[FEATURES])),
        "production popularity, point-in-time holdout": ndcg_at(ho(pit), ho(pit)["bookings_7d"].to_numpy()),
    }

    online_sessions = m.sessions[m.sessions.t >= LOG_END]
    rows = serving_rows(m, online_sessions)
    pc, pb = true_propensity(m, rows, online_sessions)
    noise = np.random.default_rng(seed).random(len(rows)) * 1e-3
    scores = {"production popularity": rows["bookings_7d"].to_numpy() + noise,
              "naive model": m_naive.predict(rows[FEATURES]),
              "point-in-time model": m_pit.predict(rows[FEATURES]),
              "oracle (true propensity)": pc * pb}
    online = {k: 1000 * online_value(rows, s, pc, pb) for k, s in scores.items()}

    imp = {name: dict(sorted(zip(FEATURES, (mod.booster_.feature_importance("gain") /
                                           mod.booster_.feature_importance("gain").sum()).round(4)),
                             key=lambda kv: -kv[1])[:5]) for name, mod in (("naive", m_naive), ("point_in_time", m_pit))}

    pw = None
    if power_sims:
        sess = []
        d = rows.assign(sa=scores["point-in-time model"], sb=scores["production popularity"], pc=pc, pb=pb)
        for _, g in d.groupby("session", sort=False):
            sess.append((np.argsort(-g["sa"].to_numpy()), np.argsort(-g["sb"].to_numpy()),
                         g["pc"].to_numpy(), g["pb"].to_numpy()))
        pw = [power(sess, n, power_sims, rng) for n in (50, 100, 250, 500, 1000, 2000)]
    return {"seed": seed, "rows_logged": len(logged), "online_sessions": len(online_sessions),
            "bookings_per_session_logged": float(logged.booked.sum() / logged.session.nunique()),
            "offline_ndcg10": offline, "online_bookings_per_1000": online, "top_features": imp, "power": pw}


def coalescing_bench(requests=2000, threads=32, keys=40, backend_ms=3.0):
    out = {}
    for mode, kw in (("no coalescing", dict(coalesce=False)), ("single-flight", dict(coalesce=True)),
                     ("single-flight + 50 ms TTL", dict(coalesce=True, ttl_s=0.05))):
        store = OnlineStore(lambda k: (time.sleep(backend_ms / 1000), k)[1], **kw)
        lat, lock = [], threading.Lock()
        rng = np.random.default_rng(0)
        hot = rng.zipf(1.6, requests) % keys                       # a few hotels get most lookups

        def worker(chunk):
            for k in chunk:
                s = time.perf_counter()
                store.get(int(k))
                with lock:
                    lat.append((time.perf_counter() - s) * 1e3)

        ts = [threading.Thread(target=worker, args=(hot[i::threads],)) for i in range(threads)]
        t0 = time.perf_counter()
        [t.start() for t in ts]
        [t.join() for t in ts]
        out[mode] = {"backend_calls": store.backend_calls, "requests": requests,
                     "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99)),
                     "wall_s": time.perf_counter() - t0}
    return out


def main(quick):
    t0 = time.time()
    rng = np.random.default_rng(0)
    seeds = [0] if quick else [0, 1, 2]
    markets = [one_market(s, (30 if quick else 100) if i == 0 else 0, rng) for i, s in enumerate(seeds)]
    for mk in markets:
        print("seed", mk["seed"], json.dumps(mk["offline_ndcg10"]), json.dumps(mk["online_bookings_per_1000"]))
    agg = lambda key: {k: {"mean": float(np.mean([mk[key][k] for mk in markets])),
                           "min": float(np.min([mk[key][k] for mk in markets])),
                           "max": float(np.max([mk[key][k] for mk in markets]))} for k in markets[0][key]}
    res = {"environment": {"python": platform.python_version(), "machine": platform.machine(), "quick": quick},
           "markets": len(markets), "offline_ndcg10": agg("offline_ndcg10"),
           "online_bookings_per_1000": agg("online_bookings_per_1000"), "per_market": markets,
           "power": markets[0]["power"], "coalescing": coalescing_bench()}
    res["environment"]["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    write_report(res, OUT / "report.html")
    print(json.dumps(res["power"], indent=1))
    print(json.dumps(res["coalescing"], indent=1))
    print("top features", json.dumps(markets[0]["top_features"]))
    print(f"done in {time.time() - t0:.0f}s -> results/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
