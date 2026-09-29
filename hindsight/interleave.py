"""Team-draft interleaving versus A/B testing: how many sessions to detect a better ranker?"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, norm

from .market import SHOWN, examine


def team_draft(a: np.ndarray, b: np.ndarray, rng) -> tuple[list, list]:
    """a, b: item indices in rank order. Returns the interleaved list and team labels (0 = A, 1 = B)."""
    out, team, used, ia, ib, na, nb = [], [], set(), 0, 0, 0, 0
    while len(out) < SHOWN and (ia < len(a) or ib < len(b)):
        pick_a = na < nb or (na == nb and rng.random() < 0.5)
        src, idx = (a, ia) if pick_a else (b, ib)
        while idx < len(src) and src[idx] in used:
            idx += 1
        if idx >= len(src):
            pick_a = not pick_a
            src, idx = (a, ia) if pick_a else (b, ib)
            while idx < len(src) and src[idx] in used:
                idx += 1
            if idx >= len(src):
                break
        item = src[idx]
        used.add(item)
        out.append(item)
        team.append(0 if pick_a else 1)
        if pick_a:
            ia, na = idx + 1, na + 1
        else:
            ib, nb = idx + 1, nb + 1
    return out, team


def power(sessions, n: int, sims: int, rng) -> dict:
    """sessions: list of (order_a, order_b, pc, pb). A is expected to be better. Returns detection rates."""
    ex = examine(np.arange(SHOWN))
    hit_il = hit_ab_book = hit_ab_click = 0
    for _ in range(sims):
        idx = rng.integers(0, len(sessions), n)
        wins_a = wins_b = 0
        ab = {0: [0, 0, 0], 1: [0, 0, 0]}          # sessions, bookings, clicks
        for j, i in enumerate(idx):
            oa, ob, pc, pb = sessions[i]
            lst, team = team_draft(oa, ob, rng)
            clicks = (rng.random(len(lst)) < ex[:len(lst)]) & (rng.random(len(lst)) < pc[lst])
            ca, cb = int((clicks & (np.asarray(team) == 0)).sum()), int((clicks & (np.asarray(team) == 1)).sum())
            wins_a += ca > cb
            wins_b += cb > ca
            arm = j % 2
            order = (oa if arm == 0 else ob)[:SHOWN]
            c = (rng.random(len(order)) < ex[:len(order)]) & (rng.random(len(order)) < pc[order])
            bk = c & (rng.random(len(order)) < pb[order])
            ab[arm][0] += 1
            ab[arm][1] += int(bk.any())
            ab[arm][2] += int(c.sum())
        if wins_a + wins_b and wins_a > wins_b and binomtest(wins_a, wins_a + wins_b, 0.5).pvalue < 0.05:
            hit_il += 1
        for k, counter in ((1, "book"), (2, "click")):
            p1, p2 = ab[0][k] / ab[0][0], ab[1][k] / ab[1][0]
            if counter == "book":
                pp = (ab[0][k] + ab[1][k]) / (ab[0][0] + ab[1][0])
                se = np.sqrt(max(pp * (1 - pp), 1e-12) * (1 / ab[0][0] + 1 / ab[1][0]))
            else:
                se = np.sqrt((p1 + p2) / ab[0][0])          # Poisson approximation for click counts
            z = (p1 - p2) / max(se, 1e-12)
            if z > norm.ppf(0.975):
                if counter == "book":
                    hit_ab_book += 1
                else:
                    hit_ab_click += 1
    return {"n": n, "interleaving": hit_il / sims, "ab_bookings": hit_ab_book / sims, "ab_clicks": hit_ab_click / sims}
