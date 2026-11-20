"""GTA VI trophy earn-rate forecaster (make.fun: gta6-trophy-rarity-forecast-01).

Reads one case file per run from the input directory and writes, for every
trophy in the case, five quantiles (5/25/50/75/95%) of the trophy's Exophase
earn rate at the case's target instant.

Model summary (full write-up in README.md):

  * The house's reads repeat verbatim between Exophase refreshes, so the
    history is first compressed to its distinct snapshots. Only those carry
    information.
  * Players tracked at the target is forecast in log space by a small
    ensemble of growth curves (two power laws, a quadratic in log-age,
    and a saturating exponential), weighted by a backtest inside the
    case's own history.
  * Earners per trophy (rate x players / 100) are fitted with a
    saturating lognormal CDF over hours since release: the spread is
    shared by all trophies, each trophy gets its own location (shrunk
    toward the shared location) and its own ceiling. The forecast rate
    is forecast earners over forecast players, which is what makes
    dilution (players growing faster than earners) pull most rates down
    while fast-progressing story trophies can still rise.
  * Medians are clipped to the story funnel (Chapter 1 >= Chapter 2 >=
    Chapter 3 >= Chapter 4 >= final mission >= 100% completion >=
    platinum), which must hold for the true rates by construction.
  * The spread sigma is estimated by pseudo-backtests inside the case
    (forecast the last snapshot from earlier cut points), scaled by the
    square root of the horizon ratio, with a floor and per-trophy widening
    near the 0%/100% edges and for the platinum.
  * A degenerate history (fewer than 3 distinct snapshots, as in the
    practice cases) cannot support any of that, so it falls back to a
    persistence median shifted down 0.30 logit (the dilution a refresh
    brings) with a wide sigma of 1.0.

The program is deterministic, uses only the Python standard library, does
no network access, and never raises on a well-formed case: any internal
failure degrades to a wide persistence forecast rather than an invalid
output file.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Standard-normal quantiles at the five scored levels, in order.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

PCT_LO = 0.01
PCT_HI = 99.99


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------

def clamp_pct(x):
    return min(PCT_HI, max(PCT_LO, x))


def logit_pct(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def round4(x):
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def solve_ols(rows, ys):
    """Least squares via the normal equations; None when (near-)singular."""
    n = len(rows[0])
    ata = [[0.0] * n for _ in range(n)]
    aty = [0.0] * n
    for row, y in zip(rows, ys):
        for i in range(n):
            aty[i] += row[i] * y
            for j in range(n):
                ata[i][j] += row[i] * row[j]
    for i in range(n):
        ata[i][i] += 1e-9
    aug = [ata[i][:] + [aty[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pv
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                for j in range(col, n + 1):
                    aug[r][j] -= f * aug[col][j]
    return [aug[i][n] for i in range(n)]


# --------------------------------------------------------------------------
# Case preparation: distinct snapshots
# --------------------------------------------------------------------------

class Snap:
    __slots__ = ("when", "age", "players", "rates")

    def __init__(self, when, age, players, rates):
        self.when = when
        self.age = age
        self.players = players
        self.rates = rates


def release_time(case, first_when):
    raw = (case.get("game") or {}).get("release_date") or ""
    try:
        if len(raw) == 10:
            return datetime.fromisoformat(raw + "T00:00:00+00:00")
        return parse_time(raw)
    except Exception:
        return first_when


def distinct_snapshots(case):
    history = case.get("history") or []
    parsed = []
    for read in history:
        try:
            when = parse_time(read.get("at") or read.get("slot"))
            players = float(read.get("players_tracked"))
            rates = list(read.get("rates") or [])
        except Exception:
            continue
        parsed.append((when, players, rates))
    if not parsed:
        return []
    rel = release_time(case, parsed[0][0])
    snaps = []
    prev_key = None
    for when, players, rates in parsed:
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        age = max(0.25, (when - rel).total_seconds() / 3600.0)
        snaps.append(Snap(when, age, players, rates))
    return snaps


def last_known_rate(snaps, idx):
    """(rate, players at that snapshot) for the latest non-null rate."""
    for snap in reversed(snaps):
        if idx < len(snap.rates) and snap.rates[idx] is not None:
            return snap.rates[idx], snap.players
    return None, None


# --------------------------------------------------------------------------
# Players-tracked forecast
# --------------------------------------------------------------------------

def _player_features(kind, age, elapsed):
    if kind == 0:
        return [1.0, math.log(age)]
    if kind == 1:
        return [1.0, math.log1p(elapsed)]
    la = math.log(age)
    return [1.0, la, la * la]


def _fit_players(snaps, kind):
    """Fit one growth candidate; returns a params tuple or None.

    Kinds 0-2 are log-space OLS curves (see _player_features). Kind 3 is a
    saturating exponential P = c0 + c1 * exp(-age / tau) with c1 < 0, fit
    in raw space over a grid of time constants: a tracked-player pool
    cannot grow like a power law forever, and the early power-law fits
    badly overshoot a saturating pool when extrapolated over days.
    """
    if kind == 3:
        if len(snaps) < 4:
            return None
        best = None
        for tau in (18.0, 36.0, 72.0, 144.0, 288.0):
            rows = [[1.0, math.exp(-snap.age / tau)] for snap in snaps]
            ys = [snap.players for snap in snaps]
            beta = solve_ols(rows, ys)
            if beta is None or beta[1] >= 0 or beta[0] <= snaps[-1].players:
                continue
            sse = 0.0
            for snap, row in zip(snaps, rows):
                pred = beta[0] + beta[1] * row[1]
                if pred <= 0:
                    sse = float("inf")
                    break
                sse += (math.log(pred) - math.log(snap.players)) ** 2
            if best is None or sse < best[0]:
                best = (sse, tau, beta[0], beta[1])
        if best is None:
            return None
        return (3, best[1], best[2], best[3])
    if len(snaps) <= len(_player_features(kind, 1.0, 0.0)):
        return None
    t0 = snaps[0].when
    rows = []
    ys = []
    for snap in snaps:
        if snap.players <= 0:
            return None
        elapsed = (snap.when - t0).total_seconds() / 3600.0
        rows.append(_player_features(kind, snap.age, elapsed))
        ys.append(math.log(snap.players))
    beta = solve_ols(rows, ys)
    if beta is None:
        return None
    return (kind,) + tuple(beta)


def _predict_players(params, age, elapsed):
    kind = params[0]
    if kind == 3:
        _, tau, c0, c1 = params
        return c0 + c1 * math.exp(-age / tau)
    row = _player_features(kind, age, elapsed)
    return math.exp(sum(b * x for b, x in zip(params[1:], row)))


def forecast_players(snaps, target_age, target_when):
    last_p = snaps[-1].players
    t0 = snaps[0].when
    elapsed_t = (target_when - t0).total_seconds() / 3600.0
    kinds = [0, 1, 2, 3]
    weights = None
    if len(snaps) >= 6:
        cut = int(0.7 * len(snaps))
        cut = max(3, min(cut, len(snaps) - 2))
        train, test = snaps[:cut], snaps[cut:]
        raw = []
        for kind in kinds:
            params = _fit_players(train, kind)
            if params is None:
                raw.append(0.0)
                continue
            se = 0.0
            ok = True
            for snap in test:
                el = (snap.when - t0).total_seconds() / 3600.0
                pred = _predict_players(params, snap.age, el)
                if pred <= 0:
                    ok = False
                    break
                se += (math.log(pred) - math.log(snap.players)) ** 2
            raw.append(1.0 / (math.sqrt(se / len(test)) + 0.05) ** 2
                       if ok else 0.0)
        if sum(raw) > 0:
            weights = raw
    if weights is None:
        weights = [0.4, 0.2, 0.15, 0.25]
    total_w = 0.0
    log_pred = 0.0
    for kind, w in zip(kinds, weights):
        if w <= 0:
            continue
        params = _fit_players(snaps, kind)
        if params is None:
            continue
        pred = _predict_players(params, target_age, elapsed_t)
        if pred <= 0:
            continue
        log_pred += w * math.log(pred)
        total_w += w
    if total_w <= 0:
        return last_p
    p_target = math.exp(log_pred / total_w)
    # Tracked counts do not shrink in this data; growth is capped sanely.
    return min(max(p_target, last_p), last_p * 60.0)


# --------------------------------------------------------------------------
# Earners forecast: shared-shape saturating lognormal CDF
# --------------------------------------------------------------------------

_MU_GRID = [math.log(v) for v in (6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0)]
_S_GRID = [0.5, 0.8, 1.1, 1.4, 1.8, 2.3]


def earner_series(snaps, count):
    series = []
    for i in range(count):
        pts = []
        for snap in snaps:
            if i < len(snap.rates) and snap.rates[i] is not None:
                pts.append((snap.age, snap.rates[i] / 100.0 * snap.players))
        series.append(pts)
    return series


def _series_varies(pts):
    if len(pts) < 2:
        return False
    es = [e for _, e in pts]
    return max(es) - min(es) >= 1.0


def _fit_k(pts, mu, s):
    """Least-squares ceiling K for one trophy at a given (mu, s)."""
    fs = [norm_cdf((math.log(a) - mu) / s) for a, _ in pts]
    den = sum(f * f for f in fs)
    if den < 1e-9:
        return None, None, None
    k = sum(e * f for (_, e), f in zip(pts, fs)) / den
    if k <= 0:
        return None, None, None
    sse = sum((e - k * f) ** 2 for (_, e), f in zip(pts, fs))
    return k, sse, fs


def fit_shared_shape(series, last_rates):
    best = None
    for mu in _MU_GRID:
        for s in _S_GRID:
            total = 0.0
            used = 0
            for i, pts in enumerate(series):
                if last_rates[i] is not None and last_rates[i] >= 99.9:
                    continue
                if not _series_varies(pts):
                    continue
                _, sse, _ = _fit_k(pts, mu, s)
                if sse is None:
                    continue
                total += sse
                used += 1
            if used >= 3 and (best is None or total < best[0]):
                best = (total, mu, s)
    if best is None:
        return None
    return best[1], best[2]


def fit_earner_curves(series, last_rates, shape):
    """Per-trophy (mu_i, K_i) at the shared shape s.

    Trophies genuinely peak at different times (a chapter trophy in the
    first days, the platinum weeks later), so each trophy gets its own
    location mu_i, but the choice is shrunk toward the shared mu: the
    score penalises distance from mu_shared, so a trophy only moves its
    location when its own earners trajectory clearly demands it. The
    spread s stays shared, as do the shrinkage constants.
    """
    mu_shared, s = shape
    curves = []
    for i, pts in enumerate(series):
        if last_rates[i] is not None and last_rates[i] >= 99.9:
            curves.append(None)
            continue
        if not _series_varies(pts):
            curves.append(None)
            continue
        _, base_sse, _ = _fit_k(pts, mu_shared, s)
        if base_sse is None:
            curves.append(None)
            continue
        best = None
        for mu in _MU_GRID:
            k, sse, _ = _fit_k(pts, mu, s)
            if sse is None:
                continue
            penalty = 0.25 * (base_sse + 1.0) * ((mu - mu_shared) / 0.9) ** 2
            score = sse + penalty
            if best is None or score < best[0]:
                best = (score, mu, k)
        curves.append((best[1], best[2]) if best else None)
    return curves


def predict_medians(snaps, trophies, target_age, target_when):
    """Median earn rate (percent) per trophy at the target, plus P_target."""
    count = len(trophies)
    p_target = forecast_players(snaps, target_age, target_when)
    p_last = snaps[-1].players
    last_rates = []
    for i in range(count):
        rate, _ = last_known_rate(snaps, i)
        last_rates.append(rate)
    series = earner_series(snaps, count)
    shape = fit_shared_shape(series, last_rates) if len(snaps) >= 3 else None
    curves = fit_earner_curves(series, last_rates, shape) \
        if shape is not None else [None] * count
    medians = []
    for i in range(count):
        rate_last = last_rates[i]
        if rate_last is None:
            medians.append(1.0)
            continue
        if rate_last >= 99.9:
            medians.append(99.97)
            continue
        _, p_at = last_known_rate(snaps, i)
        e_last = rate_last / 100.0 * (p_at if p_at else p_last)
        e_pred = None
        if curves[i] is not None:
            mu_i, k_i = curves[i]
            e_pred = k_i * norm_cdf((math.log(target_age) - mu_i) / shape[1])
        if e_pred is None:
            # No measurable earners growth: earners are assumed to grow with
            # the square root of the player growth (partial dilution).
            e_pred = e_last * math.sqrt(max(1.0, p_target / max(p_last, 1.0)))
        e_pred = min(max(e_pred, e_last), p_target)
        medians.append(100.0 * e_pred / p_target if p_target > 0 else rate_last)
    apply_story_funnel(medians, trophies)
    return medians, p_target


# --------------------------------------------------------------------------
# Story funnel: later story beats can never exceed earlier ones
# --------------------------------------------------------------------------

def _find_by_description(trophies, wanted):
    for i, tr in enumerate(trophies):
        if (tr.get("description") or "").strip() == wanted:
            return i
    return None


def _find_hundred(trophies):
    for i, tr in enumerate(trophies):
        if "100% Completion" in (tr.get("description") or ""):
            return i
    return None


def story_chain(trophies):
    """Indices ordered from the earliest story beat to the latest."""
    chain = []
    for n in (1, 2, 3, 4):
        idx = _find_by_description(trophies, "Complete Chapter %d." % n)
        if idx is not None:
            chain.append(idx)
    idx = _find_by_description(trophies, "Complete the final mission.")
    if idx is not None:
        chain.append(idx)
    idx = _find_hundred(trophies)
    if idx is not None:
        chain.append(idx)
    for i, tr in enumerate(trophies):
        if tr.get("tier") == "platinum":
            chain.append(i)
            break
    return chain


def apply_story_funnel(medians, trophies):
    chain = story_chain(trophies)
    for earlier, later in zip(chain, chain[1:]):
        if medians[later] > medians[earlier]:
            medians[later] = medians[earlier]


# --------------------------------------------------------------------------
# Spread estimation by in-case pseudo-backtests
# --------------------------------------------------------------------------

def estimate_sigma(snaps, trophies, horizon_hours):
    if len(snaps) < 6:
        return None
    last = snaps[-1]
    errs = []
    pseudo_horizons = []
    for frac in (0.5, 0.65, 0.8):
        j = int(frac * len(snaps))
        j = max(3, min(j, len(snaps) - 1))
        prefix = snaps[: j + 1]
        pseudo_h = (last.when - prefix[-1].when).total_seconds() / 3600.0
        if pseudo_h <= 0:
            continue
        try:
            medians, _ = predict_medians(prefix, trophies, last.age, last.when)
        except Exception:
            continue
        pseudo_horizons.append(pseudo_h)
        for i in range(len(trophies)):
            if i < len(last.rates) and last.rates[i] is not None:
                errs.append(logit_pct(last.rates[i]) - logit_pct(medians[i]))
    if not errs or not pseudo_horizons:
        return None
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    mean_pseudo = sum(pseudo_horizons) / len(pseudo_horizons)
    scaled = rms * math.sqrt(max(horizon_hours, 0.01) / max(mean_pseudo, 0.01))
    return min(2.2, max(0.30, scaled))


def trophy_sigma(base, rate_last, median, tier):
    s = base
    if (rate_last is not None and (rate_last <= 1.0 or rate_last >= 99.0)) \
            or median <= 1.0 or median >= 99.0:
        s *= 1.3
    if tier == "platinum":
        s *= 1.25
    return min(2.8, max(0.30, s))


# --------------------------------------------------------------------------
# Forecast assembly
# --------------------------------------------------------------------------

def rows_from_medians(trophies, medians, sigmas):
    rows = []
    for tr, median, sigma in zip(trophies, medians, sigmas):
        centre = logit_pct(median)
        qs = [round4(inv_logit_pct(centre + z * sigma)) for z in Z]
        qs = sorted(min(100.0, max(0.0, q)) for q in qs)
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def degenerate_forecast(case, snaps):
    """Fewer than 3 distinct snapshots: no trend is identifiable."""
    trophies = case["trophies"]
    medians = []
    sigmas = []
    for i, tr in enumerate(trophies):
        rate, _ = last_known_rate(snaps, i)
        if rate is None:
            medians.append(1.0)
            sigmas.append(1.0)
        elif rate >= 99.9:
            medians.append(99.97)
            sigmas.append(0.30)
        else:
            medians.append(inv_logit_pct(logit_pct(rate) - 0.30))
            sigmas.append(1.15 if tr.get("tier") == "platinum" else 1.0)
    apply_story_funnel(medians, trophies)
    return rows_from_medians(trophies, medians, sigmas)


def model_forecast(case, snaps):
    trophies = case["trophies"]
    horizon_hours = float(case["horizon_minutes"]) / 60.0
    target_when = parse_time(case["target"])
    rel = release_time(case, snaps[0].when)
    target_age = max(0.25, (target_when - rel).total_seconds() / 3600.0)
    medians, _ = predict_medians(snaps, trophies, target_age, target_when)
    base_sigma = estimate_sigma(snaps, trophies, horizon_hours)
    if base_sigma is None:
        base_sigma = min(1.5, max(0.35, 0.5 * math.sqrt(horizon_hours / 24.0)))
    sigmas = []
    for i, tr in enumerate(trophies):
        rate, _ = last_known_rate(snaps, i)
        if rate is not None and rate >= 99.9:
            sigmas.append(0.30)
        else:
            sigmas.append(trophy_sigma(base_sigma, rate, medians[i],
                                       tr.get("tier")))
    return rows_from_medians(trophies, medians, sigmas)


def wide_persistence(case):
    """Last-resort forecast: last rates with a wide, honest spread."""
    trophies = case["trophies"]
    history = case.get("history") or []
    last_rates = (history[-1].get("rates") or []) if history else []
    medians = []
    sigmas = []
    for i, tr in enumerate(trophies):
        rate = last_rates[i] if i < len(last_rates) else None
        medians.append(rate if rate is not None else 1.0)
        sigmas.append(1.2)
    return rows_from_medians(trophies, medians, sigmas)


def forecast_case(case):
    snaps = distinct_snapshots(case)
    if not snaps:
        return wide_persistence(case)
    if len(snaps) < 3:
        return degenerate_forecast(case, snaps)
    return model_forecast(case, snaps)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = wide_persistence(case)
        payload = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(payload.encode("utf-8")) > 65536:
            forecast = wide_persistence(case)
            payload = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(payload,
                                                       encoding="utf-8")


if __name__ == "__main__":
    main()
