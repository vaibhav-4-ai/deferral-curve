"""Experiment: does knowing how code-mixed a question is make the hand-off decision better?

The calibrator turns 6 signals into "probability the answer is right". Here we add a 7th, the
question's CMI (from the sister project codemix-nlu), and measure both versions the SAME way the
main curve is measured: 5-fold held-out, 3 seeds, 10% error budget.

    with 6 signals  ->  ECE, share auto-resolved at the budget
    with 7 signals  ->  ECE, share auto-resolved at the budget
    difference      ->  per seed, so you can see whether it is consistent or noise

Also: are mixed questions answered worse at all? (accuracy by CMI band). If not, CMI cannot help.
Only 170 questions, so small differences are noise; the write-up says so.
"""

import json
import os

import numpy as np

from dc.cmi import WORDLISTS, CmiMeter, fetch
from dc.curve import (
    ERROR_BUDGET,
    N_FOLDS,
    OUT,
    ROOT,
    SEEDS,
    above,
    decision_point,
    ece,
    load_rows,
    pick_threshold,
)
from dc.policy import Calibrator, features

GOLDEN = ROOT / "evals" / "golden.jsonl"
BANDS = [("none", 0.0, 0.001), ("light", 0.001, 20.0), ("medium", 20.0, 35.0), ("heavy", 35.0, 51.0)]


def band_of(cmi):
    for name, low, high in BANDS:
        if low <= cmi < high:
            return name
    return "heavy"


def held_out(X, y, seed):
    """Every question scored by a calibrator fitted on the other 4 folds, plus its auto-resolve decision."""
    n = len(y)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), N_FOLDS)
    p_held = np.zeros(n)
    auto = np.zeros(n, dtype=bool)
    for fold in folds:
        train = np.setdiff1d(np.arange(n), fold)
        model = Calibrator().fit(X[train], y[train])
        p_held[fold] = model.predict(X[fold])
        threshold = pick_threshold(model.predict(X[train]), y[train], ERROR_BUDGET)
        auto[fold] = above(p_held[fold], threshold)
    return p_held, auto


def run():
    queries = {}
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                queries[row["id"]] = row["query"]
    rows = []
    for row in load_rows():
        if not row["degraded"]:
            rows.append(row)

    if not WORDLISTS.exists():
        try:
            fetch()
        except OSError as error:
            print(f"codemix-nlu has not published its word lists yet ({error}) - experiment skipped")
            return
    meter = CmiMeter()
    cmis = []
    for row in rows:
        cmis.append(meter.cmi(queries[row["id"]]))
    y = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    six = np.array([features(r) for r in rows])
    seven = np.column_stack([six, np.array(cmis) / 50])  # 0..1, like the other signals

    lines = ["# CMI as a hand-off signal (written by `python -m dc cmi-experiment`)\n"]
    lines.append(f"{len(rows)} questions; CMI from codemix-nlu's word lists.\n")
    lines.append("## Are more mixed questions answered worse?\n")
    lines.append("| CMI band | questions | answer right |")
    lines.append("|---|---|---|")
    by_band = {}
    for name, low, high in BANDS:
        picked = []
        for k, value in enumerate(cmis):
            if band_of(value) == name:
                picked.append(y[k])
        by_band[name] = {"n": len(picked), "right": float(np.mean(picked)) if picked else None}
        if picked:
            lines.append(f"| {name} | {len(picked)} | {np.mean(picked):.0%} |")

    lines.append("\n## The calibrator with and without CMI (5-fold held-out)\n")
    lines.append("| seed | ECE, 6 signals | ECE, + CMI | auto-resolved, 6 signals | auto-resolved, + CMI |")
    lines.append("|---|---|---|---|---|")
    per_seed = []
    for seed in SEEDS:
        p6, auto6 = held_out(six, y, seed)
        p7, auto7 = held_out(seven, y, seed)
        cov6, risk6 = decision_point(auto6, y)
        cov7, risk7 = decision_point(auto7, y)
        result = {"seed": seed, "ece_6": ece(p6, y), "ece_7": ece(p7, y), "coverage_6": cov6, "coverage_7": cov7,
                  "risk_6": risk6, "risk_7": risk7}  # fmt: skip
        per_seed.append(result)
        lines.append(f"| {seed} | {result['ece_6']:.3f} | {result['ece_7']:.3f} | {cov6:.0%} at {risk6:.0%} error | "
                     f"{cov7:.0%} at {risk7:.0%} error |")
    better = 0
    for result in per_seed:
        if result["coverage_7"] > result["coverage_6"] and result["risk_7"] <= ERROR_BUDGET:
            better += 1
    verdict = (f"CMI raised auto-resolved coverage within the budget in {better} of {len(SEEDS)} seeds. "
               "With 170 questions a change of a few points is within noise, so the live policy keeps "
               "6 signals unless this is consistent across all seeds.")
    lines.append(f"\n{verdict}\n")

    report = "\n".join(lines)
    print(report)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cmi_experiment.md").write_text(report, encoding="utf-8")
    summary = {"bands": by_band, "per_seed": per_seed, "seeds_better": better}
    (OUT / "cmi_experiment.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title=CMI experiment::{verdict}")
        for name, value in by_band.items():
            if value["n"]:
                print(f"::notice title=Right answers, {name} mixing::{value['right']:.0%} of {value['n']}")
