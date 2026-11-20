"""Tests and backtests for the GTA VI trophy forecaster.

Run from the repository root:

    python3 tests/test_forecast.py [kit_input_dir]

Checks, in order:
  1. The program runs on each practice case alone (as the grader runs it)
     and on both together, writes byte-identical output on a second run,
     and every output file passes a byte-level format validation.
  2. Runtime per practice case.
  3. A synthetic rich-history case (invented refreshes, invented truth in
     the saturating-earners family, clearly synthetic) is forecast and
     scored against its known target: this entry's model versus the house
     persistence baseline. The baseline below is a COMPARISON-ONLY
     reimplementation of the formula published in RULES.txt ("Baseline
     and prizes"); it is not part of the entry and must never be shipped
     as one.
  4. An in-history backtest on that synthetic case: history truncated at
     65% of its timeline, forecast to the last snapshot, scored the same
     way (model versus the same comparison baseline).
"""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

spec = importlib.util.spec_from_file_location("entry_main", MAIN)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


# ---------------------------------------------------------------------------
# Scoring (the formula from RULES.txt, "Scoring")
# ---------------------------------------------------------------------------

def logit_pct(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def pinball(forecast_rows, actual_rates):
    total = 0.0
    n = 0
    for row, y in zip(forecast_rows, actual_rates):
        ly = logit_pct(y)
        for tau, q in zip(LEVELS, row["q"]):
            d = ly - logit_pct(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return total / n


def mae_logit(forecast_rows, actual_rates):
    errs = [abs(logit_pct(y) - logit_pct(row["q"][2]))
            for row, y in zip(forecast_rows, actual_rates)]
    return sum(errs) / len(errs)


# ---------------------------------------------------------------------------
# COMPARISON ONLY: the house persistence baseline from RULES.txt.
# q_tau = inv_logit(logit(last rate) + z_tau * 0.5 * sqrt(horizon days)).
# ---------------------------------------------------------------------------

def house_baseline_rows(case):
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        rate = last["rates"][i]
        centre = logit_pct(rate if rate is not None else 0.0)
        qs = [100.0 / (1.0 + math.exp(-(centre + z * spread))) for z in Z]
        rows.append({"id": tr["id"], "q": qs})
    return rows


# ---------------------------------------------------------------------------
# Output validation (the format rules from RULES.txt, "Your forecast")
# ---------------------------------------------------------------------------

def validate_output(path, case):
    raw = path.read_bytes()
    assert len(raw) <= 65536, f"{path.name}: {len(raw)} bytes > 65536"
    doc = json.loads(raw.decode("utf-8"))
    assert set(doc.keys()) == {"trophies"}, f"{path.name}: top-level keys"
    rows = doc["trophies"]
    want_ids = [t["id"] for t in case["trophies"]]
    assert sorted(r["id"] for r in rows) == sorted(want_ids), \
        f"{path.name}: id set mismatch"
    for r in rows:
        assert set(r.keys()) == {"id", "q"}, f"{path.name}: row keys"
        q = r["q"]
        assert len(q) == 5, f"{path.name}: q length"
        for v in q:
            assert isinstance(v, (int, float)) and math.isfinite(v), \
                f"{path.name}: non-finite q"
            assert 0.0 <= v <= 100.0, f"{path.name}: q out of range"
        assert all(q[i] <= q[i + 1] for i in range(4)), \
            f"{path.name}: q not non-decreasing"
    return rows


def run_program(input_dir, output_dir):
    env = {"MAKEFUN_INPUT_DIR": str(input_dir),
           "MAKEFUN_OUTPUT_DIR": str(output_dir),
           "PATH": "/usr/bin:/bin:/usr/local/bin"}
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, str(MAIN)], env=env,
                          capture_output=True, text=True, timeout=60)
    elapsed = time.perf_counter() - start
    assert proc.returncode == 0, f"program failed: {proc.stderr}"
    return elapsed


def check_practice(kit_input):
    cases = {}
    for name in ("p00", "p01"):
        cases[name] = json.loads((kit_input / f"{name}.in").read_text())
    print("== practice cases ==")
    for name, case in cases.items():
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "in").mkdir()
            (td / "out").mkdir()
            (td / "in" / f"{name}.in").write_text(json.dumps(case))
            t1 = run_program(td / "in", td / "out")
            first = (td / "out" / f"{name}.out").read_bytes()
            t2 = run_program(td / "in", td / "out")
            second = (td / "out" / f"{name}.out").read_bytes()
            assert first == second, f"{name}: non-deterministic output"
            rows = validate_output(td / "out" / f"{name}.out", case)
            print(f"{name}: valid, {len(first)} bytes, "
                  f"runtime {t1:.3f}s / {t2:.3f}s, "
                  f"platinum median {rows[0]['q'][2]}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "in").mkdir()
        (td / "out").mkdir()
        for name, case in cases.items():
            (td / "in" / f"{name}.in").write_text(json.dumps(case))
        run_program(td / "in", td / "out")
        for name, case in cases.items():
            validate_output(td / "out" / f"{name}.out", case)
        print("both cases in one run: valid")


# ---------------------------------------------------------------------------
# Synthetic rich-history world (invented data; NOT Exophase data)
# ---------------------------------------------------------------------------

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def synth_world(frozen_rates):
    """Truth: players saturate; earners follow lognormal CDFs in age."""
    def players(age):
        return 638.0 + 80000.0 * (1.0 - math.exp(-max(age, 0.0) / 70.0))

    mus = []
    ceilings = []
    for i, base in enumerate(frozen_rates):
        if i == 0:
            mus.append(math.log(160.0))
            ceilings.append(900.0)
        elif base >= 99.9:
            mus.append(math.log(2.0))
            ceilings.append(None)  # earners == players at all times
        elif i in (2, 3, 4, 5, 6):
            mus.append(math.log(36.0 + 6.0 * i))
            ceilings.append(players(168.0) * min(0.85, base / 100.0 * 9.0))
        elif i == 13:
            mus.append(math.log(150.0))
            ceilings.append(players(168.0) * 0.05)
        else:
            mus.append(math.log(80.0 + (i * 37) % 90))
            ceilings.append(players(168.0) * min(0.6, base / 100.0 * 5.0))

    def earners(i, age):
        if ceilings[i] is None:
            return players(age)
        return ceilings[i] * entry.norm_cdf(
            (math.log(max(age, 0.25)) - mus[i]) / 1.1)

    def rates_at(age):
        p = players(age)
        return [round(100.0 * earners(i, age) / p, 2)
                for i in range(len(frozen_rates))]

    return players, rates_at


def synth_case(practice_case):
    frozen = practice_case["history"][-1]["rates"]
    players, rates_at = synth_world(frozen)
    refresh_ages = [0.0, 5.0, 11.0, 19.0, 30.0, 44.0, 59.0]
    history = []
    age = -1.0 / 6.0
    while age <= 59.5:
        state_age = max(r for r in refresh_ages if r <= max(age, 0.0))
        history.append({
            "slot": iso(RELEASE + timedelta(hours=age)),
            "at": iso(RELEASE + timedelta(hours=age)),
            "players_tracked": int(round(players(state_age))),
            "trophies_earned": 0,
            "rates": rates_at(state_age),
        })
        age += 0.5
    case = dict(practice_case)
    case["id"] = "synth"
    case["at"] = iso(RELEASE + timedelta(hours=59.5))
    case["horizon_minutes"] = 6510
    case["target"] = iso(RELEASE + timedelta(hours=168.0))
    case["history"] = history
    truth = rates_at(168.0)
    return case, truth


def check_synthetic(practice_case):
    print("== synthetic rich-history case (invented truth) ==")
    case, truth = synth_case(practice_case)
    rows = entry.forecast_case(case)["trophies"]
    rows = sorted(rows, key=lambda r: r["id"])
    base = house_baseline_rows(case)
    print(f"model    pinball {pinball(rows, truth):.4f}  "
          f"MAE-logit {mae_logit(rows, truth):.4f}")
    print(f"baseline pinball {pinball(base, truth):.4f}  "
          f"MAE-logit {mae_logit(base, truth):.4f}  (comparison only)")
    snaps = entry.distinct_snapshots(case)
    print(f"distinct snapshots in synthetic history: {len(snaps)}")
    sigma = entry.estimate_sigma(snaps, case["trophies"],
                                 case["horizon_minutes"] / 60.0)
    print(f"in-case backtest sigma estimate: {sigma}")

    print("== in-history backtest (truncate at 65%, target = last read) ==")
    hist = case["history"]
    cut = int(0.65 * len(hist))
    sub = dict(case)
    sub["history"] = hist[:cut]
    last_when = entry.parse_time(hist[-1]["at"])
    cut_when = entry.parse_time(hist[cut - 1]["at"])
    sub["horizon_minutes"] = int((last_when - cut_when).total_seconds() / 60)
    sub["target"] = hist[-1]["at"]
    sub_rows = sorted(entry.forecast_case(sub)["trophies"],
                      key=lambda r: r["id"])
    sub_base = house_baseline_rows(sub)
    actual = hist[-1]["rates"]
    print(f"model    pinball {pinball(sub_rows, actual):.4f}  "
          f"MAE-logit {mae_logit(sub_rows, actual):.4f}")
    print(f"baseline pinball {pinball(sub_base, actual):.4f}  "
          f"MAE-logit {mae_logit(sub_base, actual):.4f}  "
          f"(comparison only)")


def check_stress(practice_case):
    print("== stress: 10-minute reads, refresh every 30 minutes ==")
    frozen = practice_case["history"][-1]["rates"]
    players, rates_at = synth_world(frozen)
    history = []
    age = -1.0 / 6.0
    while age <= 59.5:
        state_age = max(0.0, int(age * 2) / 2)
        history.append({
            "slot": iso(RELEASE + timedelta(hours=age)),
            "at": iso(RELEASE + timedelta(hours=age)),
            "players_tracked": int(round(players(state_age))),
            "trophies_earned": 0,
            "rates": rates_at(state_age),
        })
        age += 1.0 / 6.0
    case = dict(practice_case)
    case["id"] = "stress"
    case["at"] = iso(RELEASE + timedelta(hours=59.5))
    case["horizon_minutes"] = 6510
    case["target"] = iso(RELEASE + timedelta(hours=168.0))
    case["history"] = history
    start = time.perf_counter()
    rows = entry.forecast_case(case)["trophies"]
    elapsed = time.perf_counter() - start
    assert len(rows) == 55
    assert elapsed < 8.0, f"stress forecast too slow: {elapsed:.2f}s"
    print(f"119 distinct snapshots forecast in {elapsed:.2f}s (cap 8s)")


def main():
    kit_input = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT.parent / "gta6-trophy-rarity-forecast-01-kit" / "input")
    check_practice(kit_input)
    p00 = json.loads((kit_input / "p00.in").read_text())
    check_synthetic(p00)
    check_stress(p00)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
