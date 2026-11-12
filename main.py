"""Roblox Up-and-Coming top-10 forecaster (stdlib only, deterministic).

Method (validated on the practice backtest: -35.22 vs house baseline -43.51,
second split -41.2 vs -56.9):
per game, fit rank velocity over the last 12h of reads, diffuse the rank with a
Gaussian random walk (sigma 1.0 ranks/sqrt(hour), momentum decay tau 24h), and
average the rank distribution over horizons 0..72h. 28-day age expiry is
applied when a game's created date is known (house-recorder reads). All games
seen in the history stay candidates: a game that left the last read's chart
can re-enter during the window (backtest: restricting to last-read games
costs ~2.7 log points). Place probabilities are the
window-averaged masses, normalised to leave an unnamed share of 0.2 per place
for churn and new entrants.
"""
import json
import math
import os
from datetime import datetime, timedelta

SIGMA = 1.0        # ranks per sqrt(hour) of rank diffusion
TAU = 24.0         # hours: momentum decay
VEL_HOURS = 12.0   # hours of reads used for the velocity fit
UNNAMED = 0.2      # share of each place left unnamed
MAX_RANK = 110
TOP_PER_PLACE = 40
CHART_AGE_DAYS = 28


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(in_dir, "history.ndjson"), encoding="utf-8") as f:
        reads = [json.loads(line) for line in f if line.strip()]
    reads.sort(key=lambda r: r["at"])
    t_end = parse_ts(reads[-1]["at"])

    # per-game rank series + created date (from details when available)
    series = {}
    created = {}
    for r in reads:
        t = parse_ts(r["at"])
        det = r.get("details") or {}
        for g in r["up_and_coming"]:
            uid = g["universe_id"]
            series.setdefault(uid, []).append((t, g["rank"]))
            d = det.get(str(uid)) or det.get(uid)
            if isinstance(d, dict) and d.get("created"):
                created[uid] = parse_ts(d["created"])

    horizons = list(range(0, 73, 6))
    dists = {}
    for uid in sorted(series):
        pts = sorted(series[uid])
        r0 = pts[-1][1]
        w = [p for p in pts if (t_end - p[0]) <= timedelta(hours=VEL_HOURS)]
        if len(w) < 2:
            w = pts[-2:]
        if len(w) >= 2:
            t0 = w[0][0]
            xs = [(p[0] - t0).total_seconds() / 3600.0 for p in w]
            ys = [p[1] for p in w]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs)
            v = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den > 0 else 0.0
        else:
            v = 0.0
        v = max(-3.0, min(3.0, v))
        exit_h = None
        if uid in created:
            exit_h = (created[uid] + timedelta(days=CHART_AGE_DAYS) - t_end).total_seconds() / 3600.0
        dist = [0.0] * (MAX_RANK + 1)
        for h in horizons:
            if exit_h is not None and h > exit_h:
                continue
            mu = r0 + v * h * math.exp(-h / TAU)
            sd = SIGMA * math.sqrt(h + 1.0) + 0.5
            for k in range(1, MAX_RANK + 1):
                dist[k] += norm_cdf((k + 0.5 - mu) / sd) - norm_cdf((k - 0.5 - mu) / sd)
        s = sum(dist)
        if s <= 0:
            continue
        # conditional distribution (given the game is still on the chart),
        # scaled by the fraction of the window it survives: a game that
        # leaves mid-window keeps only that fraction of its mass
        n_alive = sum(1 for h in horizons if exit_h is None or h <= exit_h)
        dists[uid] = [d / s * (n_alive / len(horizons)) for d in dist]

    probs = []
    for k in range(1, 11):
        cand = sorted(((dists[u][k], u) for u in dists if k < len(dists[u])), reverse=True)
        cand = cand[:TOP_PER_PLACE]
        tot = sum(p for p, _ in cand)
        if tot <= 0:
            continue
        scale = (1.0 - UNNAMED) / tot
        for p, u in cand:
            probs.append({"universe_id": u, "place": k, "p": round(p * scale, 6)})
    probs.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    in_top10 = {u: sum(d[1:11]) for u, d in dists.items()}
    top10 = [u for u, _ in sorted(in_top10.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]

    out = {"probabilities": probs, "top10": top10}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "forecast.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(out, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
