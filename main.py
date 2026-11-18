"""Forecast GTA VI Twitch viewers from comparable launch curves.

Method: each comparable launch's release-day curve is expressed as log
ratios to its own hour-1 count. A Gaussian kernel over four numeric
pre-launch features (log hour-1 level, pre-launch ramp, pre-midnight
level, pre-launch peak) weights the comparables; per-hour weighted
quantiles of their log ratios, applied to this case's hour-1 print, give
the forecast. The kernel uses one set of feature scales for the early
hours (2-8) and a level/peak-weighted set for the late hours (9-24),
averages an ensemble of three bandwidths, and caps any single
comparable's weight so no one launch dominates. The band is shrunk
toward persistence early on, narrowed, widened when the effective
sample size is small, floored at a fraction of the baseline spread, and
smoothed over three hours. When the comparable pool is empty or
anything about a case is unexpected, the persistence baseline is
emitted instead, so an output is always written.

Stdlib only, offline, deterministic: no randomness, no wall clock, no
network. Reads $MAKEFUN_INPUT_DIR (else /input/), every *.in file in
name order; writes $MAKEFUN_OUTPUT_DIR/<id>.out (else /output/) per
case, one JSON object {"hours": [23 rows, hours 2..24]}.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Kernel feature scales: early hours weight the unlock-wave shape,
# late hours weight the launch's level and pre-launch peak.
SCALES_EARLY = {"l1": 1.0, "ramp": 1.0, "pre1": 2.0, "prepeak": 2.0}
SCALES_LATE = {"l1": 0.7, "ramp": 2.0, "pre1": 3.0, "prepeak": 1.0}
LATE_FROM_HOUR = 9
ENSEMBLE_MULTS = (0.7, 1.0, 1.5)
WEIGHT_CAP = 0.3

SHRINK_A = 0.5
SHRINK_DECAY = 0.5
SPREAD_MULT = 0.85
ESS_ALPHA = 1.5
ESS_REF = 4.0
FLOOR_C = 0.8
PREPEAK_WIN_HOURS = 72

BASE_SIGMA = 0.15
Q_MIN = 1e-9
Q_MAX = 1e9
RATIO_CLAMP = 50.0


def _num(value):
    """Finite float for a JSON number, else None (bools are not numbers)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (OverflowError, ValueError):
            return None
        if math.isfinite(f):
            return f
    return None


def _parse_ts(value):
    """Parse an ISO-8601 timestamp to an aware datetime, else None.

    A timestamp without an offset is read as UTC, matching the data's
    convention, so naive and aware stamps can always be compared.
    """
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _rows_by_offset(rows, day_start):
    """Bucket hourly rows by whole-hour offset from day_start.

    Offset is the floor of (from - day_start) / 3600s, so a row in the
    hour before day_start lands at -1. Later rows win on duplicates.
    Rows with unusable timestamps or viewer counts are skipped.
    """
    by_off = {}
    if not isinstance(rows, list):
        return by_off
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = _parse_ts(row.get("from"))
        if start is None or day_start is None:
            continue
        viewers = _num(row.get("viewers"))
        if viewers is None:
            continue
        off = int((start - day_start).total_seconds() // 3600)
        by_off[off] = viewers
    return by_off


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def _window_vals(pre, lo, hi):
    return [v for (o, v) in pre if lo <= o < hi and v > 0]


def _logmean(pre, lo, hi):
    m = _mean(_window_vals(pre, lo, hi))
    return math.log(m) if m else None


def _features(pre, h1):
    """Four numeric pre-launch features from (offset, viewers) points.

    pre: ascending-offset pre-release points (offsets < 0). h1: the
    launch's hour-1 print. Missing windows fall back gracefully; a
    launch with no usable pre-release data gets neutral zeros.
    """
    p = max(float(h1), 1.0)
    lh1 = math.log(p)
    m24 = _logmean(pre, -24, 0)
    me = _logmean(pre, -72, -24)
    if me is None:
        me = _logmean(pre, -336, -24)
    ramp = (m24 - me) if (m24 is not None and me is not None) else 0.0
    m6 = _logmean(pre, -6, 0)
    if m6 is not None:
        pre1 = m6 - lh1
    else:
        positives = [v for (_o, v) in pre if v > 0]
        pre1 = math.log(positives[-1]) - lh1 if positives else 0.0
    peak_vals = [v for (o, v) in pre if v > 0 and o >= -PREPEAK_WIN_HOURS]
    prepeak = math.log(max(peak_vals)) - lh1 if peak_vals else 0.0
    return {"l1": lh1, "ramp": ramp, "pre1": pre1, "prepeak": prepeak}


def _kernel_weights(tfeat, pool_feats, scales):
    """Gaussian kernel weights over feature differences, normalized."""
    keys = list(scales.keys())
    w = []
    for pf in pool_feats:
        d2 = 0.0
        for k in keys:
            d = (tfeat[k] - pf[k]) / scales[k]
            d2 += d * d
        w.append(math.exp(-0.5 * d2))
    tot = sum(w)
    if tot <= 0.0 or not math.isfinite(tot):
        return [1.0 / len(w)] * len(w)
    return [x / tot for x in w]


def _cap_weights(weights, cap):
    """Cap any single weight at `cap`, redistributing the excess.

    Iterating cap + renormalize converges quickly; eight passes are
    far more than enough for a pool of this size.
    """
    if not weights or max(weights) <= cap:
        return list(weights)
    w = list(weights)
    for _ in range(8):
        w = [min(x, cap) for x in w]
        tot = sum(w)
        if tot <= 0.0:
            return list(weights)
        w = [x / tot for x in w]
        if max(w) <= cap + 1e-9:
            break
    return w


def _wquantile(pairs, tau):
    """Weighted quantile with linear interpolation on cumulative weight."""
    pairs = sorted(pairs)
    tot = sum(w for _v, w in pairs)
    if tot <= 0:
        return pairs[len(pairs) // 2][0]
    thresh = tau * tot
    acc = 0.0
    prev_v = pairs[0][0]
    prev_c = 0.0
    for v, w in pairs:
        acc += w
        if acc >= thresh:
            if w > 0 and acc > prev_c:
                frac = (thresh - prev_c) / (acc - prev_c)
                return prev_v + frac * (v - prev_v)
            return v
        prev_v, prev_c = v, acc
    return pairs[-1][0]


def _baseline_sigma(h):
    return BASE_SIGMA * math.sqrt(h - 1)


def _shape_rows(rows):
    """Clamp quantiles into (0, 1e9] and enforce non-decreasing order."""
    out = []
    for qs in rows:
        shaped = []
        prev = 0.0
        for q in qs:
            q = min(max(float(q), Q_MIN), Q_MAX)
            q = max(q, prev)
            shaped.append(q)
            prev = q
        out.append(shaped)
    return out


def _baseline_rows(p):
    """Persistence baseline: q_tau = p * exp(z_tau * sigma_h)."""
    rows = []
    for hi in range(23):
        h = hi + 2
        sig = _baseline_sigma(h)
        rows.append([p * math.exp(Z[ti] * sig) for ti in range(5)])
    return _shape_rows(rows)


def _build_pool(comparables):
    """Pool entries with a usable release-day curve.

    A comparable is usable only when its hour-1 row and all 23
    following hourly rows exist with positive viewer counts; anything
    else (including a category history with no release-day curve) is
    skipped by this data rule, never by name.
    """
    pool = []
    if not isinstance(comparables, list):
        return pool
    for comp in comparables:
        if not isinstance(comp, dict):
            continue
        day_start = _parse_ts(comp.get("day_start"))
        if day_start is None:
            continue
        by_off = _rows_by_offset(comp.get("hourly"), day_start)
        h1 = by_off.get(0)
        if h1 is None or h1 <= 0:
            continue
        answers = [by_off.get(o) for o in range(1, 24)]
        if any(a is None or a <= 0 for a in answers):
            continue
        lh1 = math.log(float(h1))
        ratios = []
        for a in answers:
            r = math.log(float(a)) - lh1
            ratios.append(max(-RATIO_CLAMP, min(RATIO_CLAMP, r)))
        pre = sorted((o, v) for o, v in by_off.items() if o < 0)
        pool.append({"feats": _features(pre, h1), "ratios": ratios})
    return pool


def _target_pre(case, day_start):
    """Pre-release (offset, viewers) points for the target case.

    Prefers history_hourly; falls back to averaging the minute-level
    history rows within each hour offset when hourly rows are absent.
    """
    by_off = _rows_by_offset(case.get("history_hourly"), day_start)
    if by_off:
        return sorted((o, v) for o, v in by_off.items() if o < 0)
    history = case.get("history")
    if isinstance(history, list) and day_start is not None:
        acc = {}
        for row in history:
            if not isinstance(row, dict):
                continue
            at = _parse_ts(row.get("at"))
            viewers = _num(row.get("viewers"))
            if at is None or viewers is None:
                continue
            off = int((at - day_start).total_seconds() // 3600)
            acc.setdefault(off, []).append(viewers)
        by_off = {o: sum(v) / len(v) for o, v in acc.items()}
        return sorted((o, v) for o, v in by_off.items() if o < 0)
    return []


def _target_h1(case, day_start):
    """The case's hour-1 print, from `known` or the hour-1 history row."""
    known = case.get("known")
    rows = None
    if isinstance(known, dict):
        rows = [known]
    elif isinstance(known, list):
        rows = known
    if rows:
        chosen = None
        for row in rows:
            if isinstance(row, dict) and row.get("hour") == 1:
                chosen = row
                break
        if chosen is None:
            for row in rows:
                if isinstance(row, dict):
                    chosen = row
                    break
        if chosen is not None:
            v = _num(chosen.get("viewers"))
            if v is not None:
                return v
    by_off = _rows_by_offset(case.get("history_hourly"), day_start)
    v = by_off.get(0)
    if v is not None:
        return v
    return None


def _forecast_rows(case):
    """Kernel forecast rows for one case; raises on unusable input."""
    if not isinstance(case, dict):
        raise ValueError("case is not an object")
    day_start = _parse_ts(case.get("day_start"))
    h1 = _target_h1(case, day_start)
    if h1 is None:
        raise ValueError("no hour-1 print")
    p = min(max(float(h1), 1.0), Q_MAX)
    pool = _build_pool(case.get("comparables"))
    if not pool:
        raise ValueError("empty comparable pool")
    pre = _target_pre(case, day_start)
    tfeat = _features(pre, p)
    pool_feats = [c["feats"] for c in pool]

    # Effective sample size from the base (early-scale, uncapped)
    # weights: unusual targets get wider bands.
    base_w = _kernel_weights(tfeat, pool_feats, SCALES_EARLY)
    ess = 1.0 / sum(w * w for w in base_w)
    width_mult = 1.0 + ESS_ALPHA * max(0.0, (ESS_REF - ess) / ESS_REF)

    # Raw log-ratio bands, averaged over the bandwidth ensemble.
    bands = [[0.0] * 5 for _ in range(23)]
    for mult in ENSEMBLE_MULTS:
        for hi in range(23):
            h = hi + 2
            scales = SCALES_EARLY if h < LATE_FROM_HOUR else SCALES_LATE
            sc = {k: v * mult for k, v in scales.items()}
            weights = _cap_weights(
                _kernel_weights(tfeat, pool_feats, sc), WEIGHT_CAP)
            pairs = [(c["ratios"][hi], w) for c, w in zip(pool, weights)]
            for ti in range(5):
                bands[hi][ti] += _wquantile(pairs, LEVELS[ti]) / len(ENSEMBLE_MULTS)

    # Shrink the band toward persistence (log-ratio 0), decaying fast.
    for hi in range(23):
        h = hi + 2
        wsh = SHRINK_A * (SHRINK_DECAY ** (h - 2))
        delta = wsh * (0.0 - bands[hi][2])
        bands[hi] = [r + delta for r in bands[hi]]

    # Narrow the spread, widening again for low effective samples.
    spread = SPREAD_MULT * width_mult
    for hi in range(23):
        med = bands[hi][2]
        bands[hi] = [med + spread * (r - med) for r in bands[hi]]

    # Floor the outer quantiles at a fraction of the baseline spread.
    for hi in range(23):
        h = hi + 2
        med = bands[hi][2]
        sig = _baseline_sigma(h)
        for ti in range(5):
            if ti == 2:
                continue
            need = FLOOR_C * abs(Z[ti]) * sig
            d = bands[hi][ti] - med
            if ti < 2 and d > -need:
                bands[hi][ti] = med - need
            elif ti > 2 and d < need:
                bands[hi][ti] = med + need

    # Smooth the median series with a 3-hour running median, shifting
    # each hour's whole band by the median's movement.
    meds = [bands[hi][2] for hi in range(23)]
    smoothed = []
    for hi in range(23):
        lo = max(0, hi - 1)
        hi2 = min(23, hi + 2)
        win = sorted(meds[lo:hi2])
        smoothed.append(win[len(win) // 2])
    for hi in range(23):
        delta = smoothed[hi] - bands[hi][2]
        bands[hi] = [r + delta for r in bands[hi]]

    rows = []
    for hi in range(23):
        qs = []
        for r in bands[hi]:
            r = max(-RATIO_CLAMP, min(RATIO_CLAMP, r))
            qs.append(p * math.exp(r))
        rows.append(qs)
    return _shape_rows(rows)


def _case_output(case):
    """Valid forecast rows for one parsed case, baseline on any failure."""
    try:
        rows = _forecast_rows(case)
    except Exception:
        rows = None
    if rows is None:
        p = 1.0
        try:
            if isinstance(case, dict):
                day_start = _parse_ts(case.get("day_start"))
                h1 = _target_h1(case, day_start)
                if h1 is not None and math.isfinite(float(h1)):
                    p = min(max(float(h1), 1.0), Q_MAX)
        except Exception:
            p = 1.0
        rows = _baseline_rows(p)
    if len(rows) != 23 or any(len(r) != 5 for r in rows):
        return _baseline_rows(1.0)
    for qs in rows:
        for q in qs:
            if not (isinstance(q, float) and math.isfinite(q)
                    and 0.0 < q <= Q_MAX):
                return _baseline_rows(1.0)
    return rows


def main(argv):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    try:
        names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    except OSError:
        names = []
    os.makedirs(output_dir, exist_ok=True)
    for name in names:
        case_id = name[:-3]
        try:
            with open(os.path.join(input_dir, name), "r",
                      encoding="utf-8") as fh:
                case = json.load(fh)
        except Exception:
            case = None
        rows = _case_output(case)
        payload = {"hours": [{"hour": hi + 2, "q": rows[hi]}
                             for hi in range(23)]}
        text = json.dumps(payload, separators=(",", ":"))
        out_path = os.path.join(output_dir, case_id + ".out")
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            # One unwritable output must not drop the remaining cases.
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
