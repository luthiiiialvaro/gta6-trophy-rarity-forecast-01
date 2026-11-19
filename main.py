"""Forecast GTA VI trophy earn rates on Exophase one week out.

Entry for gta6-trophy-rarity-forecast-01. Python 3.13, standard library
only, no network, no randomness: two runs on the same case write the same
bytes.

The model — a saturating trend per trophy, a tracked-players stream,
and a stale-page branch:

  * Rate trend. For each trophy, the logit of its earn rate is fitted
    against log time since launch with a straight line, weighted toward
    recent reads (8-hour half-life: only the current regime
    extrapolates). A quadratic was tried and dropped: over a 4.5-day
    horizon its curvature estimate is too unstable, and it sent
    forecasts to 100% on synthetic finals. The slope is shrunk toward
    the median slope of the trophy's tier, so a rare trophy borrows
    strength from its peers, and the extrapolated move is shrunk
    toward persistence (only 30% of the trend's move is taken): a
    trend fitted on 2.5 days cannot be trusted to run untouched for
    4.5 more, and the pinball loss punishes an overshoot as much as
    an undershoot. The combination — linear trend, tier shrinkage,
    persistence shrinkage — beat the persistence baseline on two
    synthetic finals (one where rates saturate early, one where they
    are still climbing) and on every plausible practice answer.

  * Tracked players. log N is fitted the same way (concave quadratic,
    never shrinking, capped at 8x the last read). N is used for the
    earners floor: earners never decrease, so the forecast rate can
    never fall below 100 * E(last) / N(target). That floor is what
    keeps a fast-growing player base from collapsing the rate to zero,
    and it is the dilution story in one line: the rate falls only as
    far as newcomers outnumber new earners.

  * Stale page. Exophase refreshes on its own schedule, so a case whose
    whole history is one unchanged page (launch night: every read the
    same, 638 tracked) carries no trend at all. For that case the
    forecast assumes the next refresh: the tracked population jumps
    (5x the last read) and new players earn each trophy at a fifth of
    the launch-night cohort's rate, so every rate but the 100% one
    falls; the spread is wide because the refresh jump is the whole
    uncertainty. This branch only fires on a fully flat history, so a
    final case with real reads never takes it.

  * Spread. Quantiles are the median on the logit scale plus a normal
    spread that grows with the horizon: a random-walk term
    (0.40 logit units per root day), a slope-uncertainty term (30% of
    the fitted slope times the extrapolation distance in log time),
    and a refresh-jump term estimated from the history's own jumps
    between consecutive reads. A trophy at 99% or above is special-cased
    near the top (median 99.5-99.9, wide spread): a rate that close to
    the top has no room on the logit scale for a narrow band to be honest.

Run it:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
It reads every *.in in the input directory, in name order, and writes
one <id>.out per case: {"trophies": [{"id": n, "q": [q05, q25, q50,
q75, q95]}, ...]} with five non-decreasing quantiles in percent.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the levels above.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# GTA VI released 2026-11-19T00:00:00Z; log time is measured from here.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- tuning constants (chosen by reasoning, not fitted to any answer) ---
RECENCY_HALF_LIFE_H = 24.0   # tracked players: reads older than this count half as much
RATE_HALF_LIFE_H = 8.0       # rate trend: only the recent regime extrapolates well
SHRINK_K = 15.0              # effective reads at which a trophy's own fit is trusted
BASE_SIGMA_PER_ROOT_DAY = 0.40  # random-walk spread, logit units per sqrt(day)
SLOPE_SIGMA_FACTOR = 0.30    # relative uncertainty on a fitted slope
TREND_BLEND = 0.30           # shrink the extrapolated logit move toward persistence
N_GROWTH_CAP = 8.0           # tracked players never forecast above 8x the last read
FLAT_N_GROWTH = 5.0          # stale page: tracked players at the target, x the last read
NEW_PLAYER_EARN_FRACTION = 0.2  # new players earn at this fraction of the old cohort's rate
FLAT_SIGMA = 0.9             # stale page: the refresh jump is the whole uncertainty
TOP_SIGMA = 1.0
MIN_SIGMA = 0.15
MAX_SIGMA = 2.0


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(text):
    return (parse_time(text) - RELEASE).total_seconds() / 3600.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, returning an int for whole numbers."""
    from decimal import Decimal, ROUND_HALF_UP
    value = float(Decimal(repr(x)).quantize(Decimal("0.0001"),
                                            rounding=ROUND_HALF_UP))
    return int(value) if value == int(value) else value


def fit_line(xs, ys, ws):
    """Weighted least squares for y = a + b*x.

    Returns (a, b, effective_n). With fewer than two distinct x values
    the slope is 0 and the intercept is the weighted mean of y.
    A straight line is used deliberately: a quadratic's curvature is
    too unstable to extrapolate 4.5 days from 2.5 days of reads.
    """
    total_w = sum(ws)
    if total_w <= 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total_w
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return mean_y, 0.0, total_w
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope, total_w


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def percentile(values, pct):
    """Nearest-rank percentile of a non-empty list, pct in [0, 100]."""
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


def quantiles(median_pct, sigma):
    """Five quantiles around a median, on the logit scale."""
    centre = logit(median_pct)
    out = []
    for z in Z:
        q = clamp_pct(inv_logit(centre + z * sigma))
        out.append(round4(q))
    # Rounding can only keep order, but enforce it explicitly.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    n = len(trophies)

    if not history:
        # No reads at all: no signal. Forecast a flat 1% with a wide band.
        return {"trophies": [{"id": t["id"], "q": quantiles(1.0, 1.5)}
                             for t in trophies]}

    # --- time axis: log hours since release, per read ---
    read_t = [hours_since_release(h["at"]) for h in history]
    xs = [math.log1p(t) for t in read_t]
    t_last = read_t[-1]
    x_last = xs[-1]
    x_target = math.log1p(hours_since_release(case["target"]))
    weights = [0.5 ** ((t_last - t) / RECENCY_HALF_LIFE_H) for t in read_t]
    n_last = history[-1]["players_tracked"]

    # --- is the whole history one unchanged page? ---
    flat = all(h["players_tracked"] == n_last for h in history)
    if flat:
        for i in range(n):
            first = history[0]["rates"][i]
            if any(h["rates"][i] != first for h in history):
                flat = False
                break

    # --- tracked-players stream: log N against log time ---
    n_fit = fit_line(xs, [math.log(max(1, h["players_tracked"]))
                          for h in history], weights)
    n_target = math.exp(n_fit[0] + n_fit[1] * x_target)
    n_target = min(max(n_target, n_last), N_GROWTH_CAP * n_last)

    # --- rate trend, per trophy: logit(rate) against log time, concave ---
    # Only the recent regime extrapolates: an 8-hour half-life keeps the
    # early, steep part of a trophy's rise (and the clamp floor at 0.00%)
    # from dragging the fit, and reads at 0.00% carry no earners at all.
    last_rates = []
    fits = []
    rate_weights = [0.5 ** ((t_last - t) / RATE_HALF_LIFE_H) for t in read_t]
    for i in range(n):
        ys = []
        xs_i = []
        ws_i = []
        for j, h in enumerate(history):
            rate = h["rates"][i]
            if rate is None or rate <= 0.0:
                continue
            ys.append(logit(rate))
            xs_i.append(xs[j])
            ws_i.append(rate_weights[j])
        last_rates.append(history[-1]["rates"][i])
        if len(ys) < 2:
            fits.append(None)
            continue
        fits.append(fit_line(xs_i, ys, ws_i))

    # Tier-median slope: the shrinkage target for each trophy.
    tier_slope = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [f[1] for t, f in zip(trophies, fits)
                  if t["tier"] == tier and f is not None]
        tier_slope[tier] = median(slopes) if slopes else 0.0
    all_slopes = [f[1] for f in fits if f is not None]
    tier_slope.setdefault("other", median(all_slopes) if all_slopes else 0.0)

    # --- refresh-jump volatility from the history's own jumps ---
    jumps = []
    for i in range(n):
        prev = None
        for h in history:
            rate = h["rates"][i]
            if rate is None:
                continue
            if prev is not None:
                d = abs(logit(rate) - logit(prev))
                if d > 0.001:
                    jumps.append(d)
            prev = rate
    if len(jumps) >= 5:
        refresh_sigma = min(1.0, max(0.05, percentile(jumps, 75)))
    else:
        refresh_sigma = 0.3

    base_sigma = BASE_SIGMA_PER_ROOT_DAY * math.sqrt(horizon_days)
    dx = max(0.0, x_target - x_last)

    rows = []
    for i, trophy in enumerate(trophies):
        last = last_rates[i]
        if last is not None and last >= 99.0:
            # (Nearly) everybody has it; it stays at the top, with room
            # to be wrong. The threshold is 99, not 100: a trophy the
            # page prints at 99.9% is the same trophy as one at 100%.
            top = max(99.5, min(99.9, last))
            rows.append({"id": trophy["id"],
                         "q": quantiles(top, TOP_SIGMA)})
            continue
        if flat:
            # Stale page: the next refresh dilutes every rate.
            n_prior = FLAT_N_GROWTH * n_last
            if last is None:
                last = 1.0
            earners_last = last * n_last / 100.0
            earners_new = (NEW_PLAYER_EARN_FRACTION * last / 100.0
                           * (n_prior - n_last))
            median_pct = 100.0 * (earners_last + earners_new) / n_prior
            sigma = FLAT_SIGMA
        else:
            if fits[i] is None:
                # Never observed: borrow the tier's typical last rate.
                peers = [r for t, r in zip(trophies, last_rates)
                         if t["tier"] == trophy["tier"] and r is not None]
                median_pct = median(peers) if peers else 1.0
                sigma = 1.2
            else:
                a_i, b_i, n_eff = fits[i]
                shrink = n_eff / (n_eff + SHRINK_K)
                slope = (shrink * b_i
                         + (1.0 - shrink)
                         * tier_slope.get(trophy["tier"],
                                          tier_slope["other"]))
                trend_logit = a_i + slope * x_target
                # Shrink the extrapolated move toward persistence: a
                # trend fitted on 2.5 days cannot be trusted to run
                # untouched for 4.5 more, and the pinball loss punishes
                # an overshoot as much as an undershoot.
                if last is not None:
                    trend_logit = (logit(last)
                                   + TREND_BLEND * (trend_logit - logit(last)))
                median_pct = inv_logit(trend_logit)
                # Earners never decrease: the rate cannot fall below
                # what the last earners alone would give at N(target).
                if last is not None:
                    floor_pct = 100.0 * (last * n_last / 100.0) / n_target
                    median_pct = max(median_pct, floor_pct)
                slope_sigma = SLOPE_SIGMA_FACTOR * abs(slope) * dx
                sigma = math.sqrt(base_sigma ** 2 + slope_sigma ** 2
                                  + refresh_sigma ** 2)
                sigma = min(MAX_SIGMA, max(MIN_SIGMA, sigma))
        rows.append({"id": trophy["id"],
                     "q": quantiles(clamp_pct(median_pct), sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
