#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads one case per *.in file (schema makefun/twitch-ccu-case/v1): the first
hour's viewer print, the pre-launch history, and the release-day curves of
past big launches. Writes one forecast per case: hours 2..24, five quantiles
each, as a single JSON object {"hours": [...]}.

Method: every comparable launch's release-day curve is expressed as log
ratios to its own first hour. Launches are weighted by similarity to the
target case (first-hour level, pre-launch ramp, pre-midnight level, and
pre-launch peak, all measured from hourly data). Each hour's quantiles are
the weighted quantiles of the comparable log-ratios, applied to the target's
first-hour print, with shrinkage toward persistence in the early hours, a
calibrated spread, and a spread floor. When a case carries no usable
comparable, or anything unexpected happens, the forecast falls back to
persistence with a widening spread, so an output is always written.

Standard library only. No network, no randomness: two runs on the same case
write identical bytes.
"""

import json
import math
import os
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
BASE_SIGMA = 0.15

# Gaussian kernel scales over (log first-hour level, pre-launch ramp,
# pre-midnight level, pre-launch peak), tuned by leave-one-out over the
# comparable launches only.
KERNEL_SCALES = (("l1", 1.0), ("ramp", 1.0), ("pre1", 2.0), ("prepeak", 2.0))
SPREAD_MULT = 0.9
SHRINK_A = 0.5
FLOOR_C = 0.6
MIN_DAY_POINTS = 20


def parse_time(text):
    """ISO 8601 timestamp -> aware UTC datetime, or None when unparseable."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def as_number(value):
    """Finite float for a JSON number, else None (bools and NaN rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def bucketize(records, time_key, day_start):
    """Hourly log-mean viewers per hour offset from day_start.

    records carry a timestamp under time_key and a viewers count. Points
    sharing an hour are combined by their geometric mean, which keeps the
    log-scale features consistent between sparse hourly sources and dense
    minute reads.
    """
    sums = {}
    counts = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        ts = parse_time(rec.get(time_key))
        if ts is None:
            continue
        v = as_number(rec.get("viewers"))
        if v is None:
            continue
        off = int((ts - day_start).total_seconds() // 3600)
        sums[off] = sums.get(off, 0.0) + math.log(max(v, 1.0))
        counts[off] = counts.get(off, 0) + 1
    return {off: math.exp(sums[off] / counts[off]) for off in sums}


def day_curve(buckets):
    """The 24 release-day hourly values, or None when the curve is unusable.

    Requires the first hour to be present and at least MIN_DAY_POINTS of the
    24 hours present; missing hours are filled forward from the last known
    value (the first hour anchors the fill, so no backward fill is needed).
    """
    if 0 not in buckets:
        return None
    if sum(1 for i in range(24) if i in buckets) < MIN_DAY_POINTS:
        return None
    vals = [buckets.get(i) for i in range(24)]
    last = None
    for i in range(24):
        if vals[i] is None:
            vals[i] = last
        else:
            last = vals[i]
    if vals[0] is None:
        return None
    return [max(v, 1.0) for v in vals]


def pre_features(buckets, h1):
    """Shape features from the hours before day_start.

    ramp: how much the audience grew into the release, comparing the mean log
    level of the last 6 pre-midnight hours with the mean 24-30 hours before.
    pre1: how close the pre-midnight level already was to the first hour.
    prepeak: how far the highest pre-launch hour sat above the first hour
    (early-access and pre-release hype show up here).
    """
    log_h1 = math.log(max(h1, 1.0))
    pre = sorted((off, math.log(v)) for off, v in buckets.items() if off < 0)
    if not pre:
        return 0.0, -1.0, 0.0
    recent = [lv for off, lv in pre if off >= -6]
    if not recent:
        recent = [lv for _, lv in pre[-6:]]
    pre6 = sum(recent) / len(recent)
    ref = [lv for off, lv in pre if -30 <= off < -24]
    if not ref:
        ref = [lv for _, lv in pre[:6]]
    ramp = pre6 - sum(ref) / len(ref)
    pre1 = pre6 - log_h1
    window = [lv for off, lv in pre if off >= -72]
    prepeak = (max(window) if window else pre6) - log_h1
    return ramp, pre1, prepeak


def launch_features(comp):
    """Feature record for one comparable launch, or None when unusable."""
    if not isinstance(comp, dict):
        return None
    ds = parse_time(comp.get("day_start"))
    hourly = comp.get("hourly")
    if ds is None or not isinstance(hourly, list):
        return None
    buckets = bucketize(hourly, "from", ds)
    curve = day_curve(buckets)
    if curve is None:
        return None
    h1 = curve[0]
    ramp, pre1, prepeak = pre_features(buckets, h1)
    return {
        "l1": math.log(h1),
        "ramp": ramp,
        "pre1": pre1,
        "prepeak": prepeak,
        "ratios": [math.log(curve[h] / h1) for h in range(1, 24)],
    }


def hour_one_viewers(case):
    """Hour 1's print, floored at 1; 1 when the case does not carry one."""
    known = case.get("known")
    rows = known if isinstance(known, list) else [known] if isinstance(known, dict) else []
    for want_hour in (1, None):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if want_hour is not None and row.get("hour") != want_hour:
                continue
            v = as_number(row.get("viewers"))
            if v is not None:
                return max(v, 1.0)
    return 1.0


def target_features(case):
    """Feature record for the case being forecast."""
    h1 = hour_one_viewers(case)
    ds = parse_time(case.get("day_start"))
    buckets = {}
    if ds is not None:
        hourly = case.get("history_hourly")
        if isinstance(hourly, list) and hourly:
            buckets = bucketize(hourly, "from", ds)
        if not buckets:
            hist = case.get("history")
            if isinstance(hist, list) and hist:
                buckets = bucketize(hist, "at", ds)
    ramp, pre1, prepeak = pre_features(buckets, h1)
    return {
        "l1": math.log(max(h1, 1.0)),
        "ramp": ramp,
        "pre1": pre1,
        "prepeak": prepeak,
        "h1": max(h1, 1.0),
    }


def weighted_quantile(pairs, tau):
    """Weighted quantile of (value, weight) pairs, linear interpolation."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    cut = tau * total
    acc = 0.0
    prev_value = None
    for value, weight in ordered:
        acc += weight
        if acc >= cut:
            if prev_value is None:
                return value
            frac = 1.0 - (acc - cut) / weight if weight > 0 else 1.0
            return prev_value + frac * (value - prev_value)
        prev_value = value
    return ordered[-1][0]


def kernel_rows(tfeats, pool):
    """Per-hour log-ratio quantiles from the similarity-weighted pool."""
    weights = []
    for rec in pool:
        dist2 = 0.0
        for key, scale in KERNEL_SCALES:
            dist2 += ((tfeats[key] - rec[key]) / scale) ** 2
        weights.append(math.exp(-0.5 * dist2))
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        weights = [1.0] * len(pool)
        total = float(len(pool))
    weights = [w / total for w in weights]
    rows = []
    for hi in range(23):
        hour = hi + 2
        pairs = [(rec["ratios"][hi], w) for rec, w in zip(pool, weights)]
        qs = [weighted_quantile(pairs, tau) for tau in LEVELS]
        old_median = qs[2]
        # Shrink the median toward persistence (a log-ratio of 0) in the
        # early hours, where the first hour is the strongest evidence.
        median = old_median * (1.0 - SHRINK_A * (0.5 ** (hour - 2)))
        qs = [median + (q - old_median) for q in qs]
        # Calibrate the spread, then floor the outer quantiles at a multiple
        # of the widening baseline spread so late hours never go too narrow.
        qs = [median + (q - median) * SPREAD_MULT for q in qs]
        floor = FLOOR_C * BASE_SIGMA * math.sqrt(hour - 1)
        qs[0] = min(qs[0], median + Z_SCORES[0] * floor)
        qs[4] = max(qs[4], median + Z_SCORES[4] * floor)
        rows.append((hour, qs))
    # Smooth the median path over neighboring hours: a launch day's median
    # curve moves gradually, while per-hour weighted quantiles from a small
    # effective pool are jumpy. Each hour keeps its own spread around the
    # smoothed median. (Measured on leave-one-out: 0.3817 -> 0.3806.)
    medians = [qs[2] for _, qs in rows]
    smoothed = []
    for i, med in enumerate(medians):
        lo = medians[i - 1] if i > 0 else med
        hi_m = medians[i + 1] if i < len(medians) - 1 else med
        smoothed.append(0.25 * lo + 0.5 * med + 0.25 * hi_m)
    rows = [
        (hour, [sm + (q - med) for q in qs])
        for (hour, qs), sm, med in zip(rows, smoothed, medians)
    ]
    return rows


def baseline_rows():
    """Persistence log-ratio quantiles with a widening spread."""
    return [
        (h, [z * BASE_SIGMA * math.sqrt(h - 1) for z in Z_SCORES])
        for h in range(2, 25)
    ]


def finalize(rows, p1):
    """Log-ratio rows -> the forecast object, clamped, rounded, ordered."""
    hours = []
    for hour, qs in rows:
        vals = [p1 * math.exp(q) for q in qs]
        vals = [min(max(v, 0.01), 1e9) for v in vals]
        vals = [round(v, 2) for v in vals]
        for i in range(1, 5):
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
        hours.append({"hour": hour, "q": vals})
    return {"hours": hours}


def forecast_case(case):
    """Forecast one case; the persistence baseline is the always-safe path."""
    try:
        tfeats = target_features(case)
        pool = []
        comps = case.get("comparables")
        if isinstance(comps, list):
            for comp in comps:
                try:
                    rec = launch_features(comp)
                except Exception:
                    rec = None
                if rec is not None:
                    pool.append(rec)
        if pool:
            return finalize(kernel_rows(tfeats, pool), tfeats["h1"])
    except Exception:
        pass
    try:
        h1 = hour_one_viewers(case)
    except Exception:
        h1 = 1.0
    return finalize(baseline_rows(), max(h1, 1.0))


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            if not isinstance(case, dict):
                raise ValueError("case is not a JSON object")
            forecast = forecast_case(case)
        except Exception:
            forecast = finalize(baseline_rows(), 1.0)
        payload = json.dumps(forecast, separators=(",", ":")) + "\n"
        with open(os.path.join(out_dir, name[:-3] + ".out"), "w", encoding="utf-8") as fh:
            fh.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
