# makefun-gta6-trophy-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jesky1/makefun-gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jesky1/makefun-gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for the earn rate of every Grand Theft Auto VI PlayStation 5
trophy on Exophase. For each case it reads the house's recorded reads of the
game's trophy page and writes, for each of the 55 trophies, five quantiles
(5/25/50/75/95%) of the earn rate at the case's target, scored by mean
pinball loss on the logit scale. Pure Python 3.13 standard library, no
network, deterministic, well under the 8-second per-case cap (~0.07 s on
the practice cases, ~0.8 s on a synthetic 119-snapshot history).

## The model

Exophase refreshes a game's counts on its own schedule, so consecutive
house reads repeat verbatim between refreshes. Everything below works on
the history compressed to its **distinct snapshots** (dedupe on
`(players_tracked, rates)`); repeated reads carry no information.

**Players tracked at the target.** Four growth candidates are fitted in
log space and combined with weights from a backtest inside the case's own
history (fit the first ~70% of snapshots, score log-RMSE on the rest,
weight ∝ 1/(RMSE+0.05)²; fixed weights 0.4/0.2/0.15/0.25 when the history
is too short to backtest):

1. power law in age (hours since release),
2. power law in 1 + elapsed hours since the first snapshot,
3. quadratic in log-age (can bend downward, i.e. saturate),
4. a saturating exponential `P = c0 + c1·exp(-age/tau)`, `c1 < 0`, with
   `tau` over a grid — added after backtesting showed the pure power laws
   overshooting a saturating tracked-player pool by ~50% over the final
   horizon.

The combined forecast is clamped to `[P_last, 60 × P_last]`: the tracked
count is treated as non-decreasing, and unbounded growth is not credible
for a tracker pool.

**Earners per trophy.** A trophy's earners are `rate × players / 100`.
They are fitted with a saturating lognormal CDF over hours since release,
`E_i(age) = K_i · Phi((ln age − mu_i) / s)`:

- the spread `s` is shared by all trophies and chosen, with a shared
  location, by grid search (`mu ∈ ln{6…384 h}`, `s ∈ {0.5…2.3}`),
  minimising total least-squares error over trophies whose earners
  actually move;
- each trophy then gets its own location `mu_i` and ceiling `K_i`
  (least squares), with `mu_i` shrunk toward the shared `mu` by a penalty
  of `0.25 · (SSE_shared + 1) · ((mu_i − mu_shared)/0.9)²`, so a trophy
  only moves its timing when its own trajectory clearly demands it
  (chapter trophies peak in days, the platinum much later);
- trophies with no measurable earners movement fall back to earners
  growing with the square root of the player growth (partial dilution).

The median rate is `E_i(target) / P(target)`. Earners are clamped to
`[E_last, P_target]` — an earners count never decreases, only the rate
dilutes — so the pure-dilution outcome is the floor for every median.

**Story funnel.** True rates must satisfy platinum ≤ 100% Completion ≤
final mission ≤ Chapter 4 ≤ Chapter 3 ≤ Chapter 2 ≤ Chapter 1, because
each later beat requires the earlier ones. The chain is identified from
the case's own trophy descriptions (exact texts `Complete Chapter N.`,
`Complete the final mission.`, any description containing
`100% Completion`, and the platinum tier) and medians are clipped
downward along it. Only that chain is constrained.

**Spread.** Quantiles are built on the logit scale,
`q_tau = inv_logit(logit(median) + z_tau · sigma_i)` with the usual
z-values. `sigma` comes from pseudo-backtests inside the case: cut the
snapshot timeline at 50%, 65% and 80%, forecast the last snapshot with
the same model, pool the logit errors, take their RMS, and scale by
`sqrt(horizon / mean pseudo-horizon)`. It is floored at 0.30 and capped
at 2.2; per trophy it is widened ×1.3 when the rate sits near the 0/100
edges and ×1.25 for the platinum, with a final cap of 2.8. When no
backtest is possible the fallback is `0.5 · sqrt(horizon in days)`
clamped to [0.35, 1.5].

**Saturated trophies.** A trophy at ≥ 99.9% is pinned to a 99.97 median
with sigma 0.30. Basis: the one trophy at 100.00% on launch night stayed
at 100% through the first documented refresh (players 638 → 3,868, every
other rate fell), so it behaves like an automatic early-game trophy
whose earners track the player count.

**Degenerate histories.** With fewer than 3 distinct snapshots no trend
is identifiable — exactly the practice cases' shape (one frozen launch
night page, answer behind one unseen refresh). The forecast is then the
persistence median shifted down 0.30 on the logit scale with sigma 1.0
(1.15 for the platinum): the shift is the dilution prior for the refresh
that has not been seen yet, and the wide spread is the honest response
to a jump the history cannot size. (The house baseline's narrow
90-minute spread scored ≈ 1.51 on the practice with ~0 coverage; a wide
spread is the correct play there.)

**Safety.** Any internal failure degrades to a wide persistence forecast
(sigma 1.2) instead of crashing, and the output is size-checked against
the 65,536-byte cap: an invalid case is the only catastrophic outcome,
so the program is built to always emit a valid one.

**Calibration anchor (attempted, not used).** During the build, the
Exophase page was fetched once from the build session. It returned an
HTTP 403 Cloudflare challenge, so no live values were obtained and
nothing from any fetch is hardcoded anywhere. The sanity reference used
instead is the refresh documented in the competition rules themselves
(638 → 3,868 players with every non-100% rate falling), which is what
the dilution floor and the degenerate-history shift are based on.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

This reads every `*.in` case in `MAKEFUN_INPUT_DIR` (default `/input`)
and writes one `<id>.out` per case to `MAKEFUN_OUTPUT_DIR` (default
`/output`). Tests and backtests (practice-case validation, a synthetic
rich-history world with known truth, an in-history backtest, and a
runtime stress case) live in `tests/`:

```sh
python3 tests/test_forecast.py <kit input dir>
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ffhnz5an&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
