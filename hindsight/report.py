"""Self-contained HTML report generated from results.json."""
from __future__ import annotations

import html
from pathlib import Path

CSS = """
:root{--ink:#10243e;--muted:#566479;--line:#dfe5ee;--bg:#fbfcfe}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "Source Sans 3","Segoe UI",system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:48px 24px 80px}h1{font-size:2rem;line-height:1.15;margin:0 0 8px}
h2{font-size:1.2rem;margin:44px 0 8px}p{max-width:70ch;color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:12px 0;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}
.n{text-align:right}.wrap{overflow-x:auto}
"""


def _t(headers, rows):
    h = "".join(f'<th class="{"n" if i else ""}">{html.escape(x)}</th>' for i, x in enumerate(headers))
    b = "".join("<tr>" + "".join(f'<td class="{"n" if i else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="wrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def write_report(r: dict, path: Path) -> None:
    off = _t(["Model and holdout", "NDCG@10 (mean)", "min", "max"],
             [[html.escape(k), f'{v["mean"]:.4f}', f'{v["min"]:.4f}', f'{v["max"]:.4f}'] for k, v in r["offline_ndcg10"].items()])
    on = _t(["Ranker at serving time", "Expected bookings per 1,000 searches", "min", "max"],
            [[html.escape(k), f'{v["mean"]:.1f}', f'{v["min"]:.1f}', f'{v["max"]:.1f}'] for k, v in r["online_bookings_per_1000"].items()])
    pw = _t(["Sessions", "Interleaving (clicks)", "A/B on clicks", "A/B on bookings"],
            [[p["n"], f'{p["interleaving"]:.0%}', f'{p["ab_clicks"]:.0%}', f'{p["ab_bookings"]:.0%}'] for p in (r["power"] or [])])
    co = _t(["Online store", "Backend calls", "p50 ms", "p99 ms"],
            [[k, f'{v["backend_calls"]} of {v["requests"]}', f'{v["p50_ms"]:.2f}', f'{v["p99_ms"]:.2f}'] for k, v in r["coalescing"].items()])
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hindsight results</title><style>{CSS}</style></head><body><main>
<h1>The offline gain that disappears in production</h1>
<p>{r["markets"]} simulated markets. Runtime {r["environment"]["runtime_s"]} s.</p>
<h2>Offline</h2>{off}<h2>At serving time</h2>{on}
<h2>How many sessions to detect the better ranker?</h2><p>Share of simulated experiments that detect the point-in-time model as better than production popularity at p &lt; 0.05.</p>{pw}
<h2>Online store under a hot-key burst</h2>{co}
</main></body></html>"""
    Path(path).write_text(doc)
