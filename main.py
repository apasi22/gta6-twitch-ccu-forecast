"""Comparable-kernel forecaster for Twitch launch-day viewers.

For each case, the program reads the hour-1 print and the pre-launch hourly
history, weights past launches (comparables) by similarity of four numeric
features under a Gaussian kernel, and forecasts hours 2..24 as five
quantiles per hour from the comparables' log-ratios to their own hour 1.

Method (all constants fixed in advance, tuned only on leave-one-out runs
over the comparables): hour-split kernel scales (early hours 2-8 vs late
hours 9-24); per-comparable weight cap of 0.3; 72-hour pre-launch peak
window; median shrinkage toward the persistence baseline; spread scaled by
0.85 and widened when the effective sample size is small; outer quantiles
floored at 0.8x the baseline sigma; 3-hour median smoothing. A comparable
enters the pool only when it has a usable release-day curve (a positive
hour-1 print and at least 12 of hours 2..24 printed). Whenever the pool is
empty or anything unexpected happens, the program falls back to the
persistence baseline, so a valid forecast is always written.
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
MAXQ = 1e9
EARLY_SCALES = (1.0, 1.0, 2.0, 2.0)
LATE_SCALES = (0.7, 2.0, 3.0, 1.0)
WEIGHT_CAP = 0.3


def parse_time(s):
    if not isinstance(s, str):
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def is_num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(v)
    except (OverflowError, ValueError):
        return False


def to_float(v):
    """Finite float or None; huge ints and non-finite values become None
    instead of raising, so one bad point never kills a whole forecast."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except (OverflowError, ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def hour1_print(case):
    if not isinstance(case, dict):
        return None
    known = case.get("known")
    rows = known if isinstance(known, list) else [known] if isinstance(known, dict) else []
    for row in rows:
        if isinstance(row, dict):
            f = to_float(row.get("viewers"))
            if f is not None and f > 0:
                return f
    return None


def split_points(points, day_start):
    """Split hourly points into release-day hours 1..24 and pre-launch points."""
    release = {}
    pre = []
    for p in points or []:
        if not isinstance(p, dict):
            continue
        t = parse_time(p.get("from"))
        f = to_float(p.get("viewers"))
        if t is None or f is None:
            continue
        if t < day_start:
            pre.append((t, f))
        else:
            h = int((t - day_start).total_seconds() // 3600) + 1
            if 1 <= h <= 24 and h not in release:
                release[h] = f
    pre.sort()
    return release, pre


def window_mean(pre, start, end):
    vals = [max(v, 1.0) for (t, v) in pre if start <= t < end]
    if not vals:
        return None
    return sum(vals) / len(vals)


def extract_features(hour1, pre, day_start):
    """Four numeric features: log level, ramp, pre-midnight level, peak."""
    level = math.log(max(hour1, 1.0))
    recent = window_mean(pre, day_start - timedelta(hours=24), day_start)
    prior = window_mean(pre, day_start - timedelta(hours=48), day_start - timedelta(hours=24))
    if recent is None or prior is None:
        recent = window_mean(pre, day_start - timedelta(hours=72), day_start)
        prior = window_mean(pre, day_start - timedelta(days=7), day_start - timedelta(hours=72))
    if recent is None or prior is None or recent <= 0 or prior <= 0:
        ramp = 0.0
    else:
        ramp = math.log(recent) - math.log(prior)
    pm = window_mean(pre, day_start - timedelta(hours=3), day_start)
    if pm is None:
        pm = max(pre[-1][1], 1.0) if pre else max(hour1, 1.0)
    pre_mid = math.log(max(pm, 1.0))
    w72 = [max(v, 1.0) for (t, v) in pre if day_start - timedelta(hours=72) <= t < day_start]
    if w72:
        peak = max(w72)
    elif pre:
        peak = max(max(v, 1.0) for (_, v) in pre)
    else:
        peak = max(hour1, 1.0)
    return (level, ramp, pre_mid, math.log(peak))


def kernel_weights(tfeat, cfeats, scales):
    ws = []
    for cf in cfeats:
        d2 = 0.0
        for a, b, s in zip(tfeat, cf, scales):
            d2 += ((a - b) / s) ** 2
        ws.append(math.exp(-0.5 * d2))
    tot = sum(ws)
    if tot <= 0 or not math.isfinite(tot):
        n = len(ws)
        return [1.0 / n] * n if n else []
    return [w / tot for w in ws]


def cap_weights(ws, cap=WEIGHT_CAP):
    ws = [float(w) for w in ws]
    n = len(ws)
    if n < 4:
        return ws
    for _ in range(50):
        over = [i for i, w in enumerate(ws) if w > cap + 1e-12]
        if not over:
            break
        rest = [i for i in range(n) if ws[i] <= cap + 1e-12]
        room = 1.0 - cap * len(over)
        rsum = sum(ws[i] for i in rest)
        for i in over:
            ws[i] = cap
        if rsum > 0:
            for i in rest:
                ws[i] = room * ws[i] / rsum
        elif rest:
            for i in rest:
                ws[i] = room / len(rest)
    tot = sum(ws)
    if tot > 0:
        ws = [w / tot for w in ws]
    return ws


def weighted_quantile(pairs, tau):
    ordered = sorted(pairs, key=lambda pv: pv[0])
    cum = 0.0
    for v, w in ordered:
        cum += w
        if cum >= tau - 1e-12:
            return v
    return ordered[-1][0]


def forecast_log_quantiles(p, tfeat, comps):
    """Per-hour log-space quantiles; comps are (features, release dict)."""
    logp = math.log(max(p, 1.0))
    out = {}
    feats = [cf for (cf, _) in comps]
    for h in range(2, 25):
        scales = EARLY_SCALES if h <= 8 else LATE_SCALES
        ws = cap_weights(kernel_weights(tfeat, feats, scales))
        pairs = []
        for (_, rel), w in zip(comps, ws):
            y = rel.get(h)
            h1 = rel.get(1)
            if y is None or h1 is None or y <= 0 or h1 <= 0:
                continue
            pairs.append((math.log(y / h1), w))
        sig = 0.15 * math.sqrt(h - 1)
        if not pairs:
            out[h] = [logp + z * sig for z in Z]
            continue
        tot = sum(w for (_, w) in pairs)
        pairs = [(v, w / tot) for (v, w) in pairs]
        ess = 1.0 / sum(w * w for (_, w) in pairs)
        widen = 1.0 + 1.5 * max(0.0, (4.0 - ess) / 4.0)
        raw = [weighted_quantile(pairs, tau) for tau in LEVELS]
        shrink = 0.5 * (0.5 ** (h - 2))
        med = (1.0 - shrink) * raw[2]
        k = 0.85 * widen
        qs = [med + k * (r - raw[2]) for r in raw]
        need = 0.8 * 1.645 * sig
        if qs[4] - med < need:
            qs[4] = med + need
        if med - qs[0] < need:
            qs[0] = med - need
        out[h] = [logp + q for q in qs]
    return out


def smooth_and_finalize(logqs):
    hours = sorted(logqs)
    series = [[logqs[h][i] for h in hours] for i in range(5)]
    rows = []
    for hi, h in enumerate(hours):
        row = []
        for i in range(5):
            s = series[i]
            win = sorted(s[max(0, hi - 1):min(len(s), hi + 2)])
            m = len(win)
            med = win[m // 2] if m % 2 == 1 else 0.5 * (win[m // 2 - 1] + win[m // 2])
            row.append(med)
        for i in range(1, 5):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        q = [max(round(math.exp(v), 2), 0.01) for v in row]
        for i in range(1, 5):
            if q[i] < q[i - 1]:
                q[i] = q[i - 1]
        q = [min(v, MAXQ) for v in q]
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def baseline_rows(p):
    logp = math.log(max(p, 1.0))
    rows = []
    for h in range(2, 25):
        sig = 0.15 * math.sqrt(h - 1)
        q = [min(max(round(math.exp(logp + z * sig), 2), 0.01), MAXQ) for z in Z]
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def forecast_case(case):
    try:
        return _forecast_case(case)
    except Exception:
        try:
            p = hour1_print(case) or 1.0
        except Exception:
            p = 1.0
        return baseline_rows(p)


def _forecast_case(case):
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    day_start = parse_time(case.get("day_start"))
    if day_start is None:
        raise ValueError("day_start missing")
    p = hour1_print(case)
    rel_t, pre_t = split_points(case.get("history_hourly") or [], day_start)
    if p is None:
        v = rel_t.get(1)
        p = float(v) if v and v > 0 else 1.0
    tfeat = extract_features(p, pre_t, day_start)
    comps = []
    for c in case.get("comparables") or []:
        if not isinstance(c, dict):
            continue
        cds = parse_time(c.get("day_start"))
        if cds is None:
            continue
        rel, pre = split_points(c.get("hourly") or [], cds)
        h1 = rel.get(1)
        if h1 is None or h1 <= 0:
            continue
        nrel = sum(1 for h in range(2, 25) if rel.get(h) and rel[h] > 0)
        if nrel < 12:
            continue
        comps.append((extract_features(h1, pre, cds), rel))
    if not comps:
        return baseline_rows(p)
    return smooth_and_finalize(forecast_log_quantiles(p, tfeat, comps))


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except Exception:
        names = []
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    for name in names:
        cid = name[:-3]
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
                case = json.load(f)
        except Exception:
            case = {}
        try:
            fc = forecast_case(case)
        except Exception:
            fc = baseline_rows(1.0)
        data = json.dumps(fc, separators=(",", ":")).encode("utf-8")
        if len(data) > 65536:
            try:
                p = hour1_print(case) or 1.0
            except Exception:
                p = 1.0
            data = json.dumps(baseline_rows(p), separators=(",", ":")).encode("utf-8")
        tmp = os.path.join(out_dir, cid + ".out.tmp")
        final = os.path.join(out_dir, cid + ".out")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
