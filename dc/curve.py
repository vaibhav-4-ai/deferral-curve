import json
from pathlib import Path

import numpy as np

from dc.policy import GUESSED_THRESHOLD, Calibrator, features, raw_confidence

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "evals" / "results.jsonl"
OUT = ROOT / "artifacts"

# The ONE human-chosen number: the highest error rate we accept on auto-resolved answers.
ERROR_BUDGET = 0.10

# Cost assumptions (rupees). Stated, not measured: change them and the curve's cost axis moves with them.
COST_PER_LLM_CALL = 0.05
COST_PER_HUMAN_TICKET = 25.0

# How pessimistic the budget-safe threshold is: a one-sided 95% upper bound on the error rate.
SAFE_Z = 1.645

SEEDS = (0, 1, 2)
N_FOLDS = 5
N_BOOT = 1000


def load_rows(path=RESULTS):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def operating_point(conf, correct, t):
    # coverage: share of questions auto-resolved. risk: share of those auto-resolved that were wrong.
    if t is None:
        return 0.0, 0.0
    covered = conf >= t
    if not covered.any():
        return 0.0, 0.0
    return float(covered.mean()), float(1.0 - correct[covered].mean())


def risk_upper_bound(errors, n, z):
    # Wilson score upper bound on the true error rate, after seeing `errors` mistakes in `n` auto-resolved answers.
    # z=0 gives the measured rate itself; z=1.645 gives a one-sided 95% upper bound.
    if n == 0:
        return 1.0
    p = errors / n
    centre = p + z * z / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre + spread) / (1 + z * z / n))


def pick_threshold(conf, correct, budget, z=0.0):
    # The lowest threshold (= most questions auto-resolved) whose error rate stays within budget.
    # z=0 trusts the error rate measured on the calibration data. z>0 demands that even a pessimistic
    # estimate of it stays within budget, so the budget also holds on questions the policy hasn't seen.
    best_t, best_cov = None, 0.0
    for t in np.unique(conf):
        covered = conf >= t
        n = int(covered.sum())
        errors = n - float(correct[covered].sum())
        if risk_upper_bound(errors, n, z) <= budget and n / len(conf) > best_cov:
            best_t, best_cov = float(t), n / len(conf)
    return best_t


def cost_per_query(conf, calls, t):
    # Every question gets resolved: by the bot, or by a human if escalated.
    escalated = np.ones(len(conf), dtype=bool) if t is None else conf < t
    return float(np.mean(calls * COST_PER_LLM_CALL + escalated * COST_PER_HUMAN_TICKET))


def ece(conf, correct, bins=5):
    # Expected calibration error: average gap between "how sure it said it was" and "how often it was right".
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total / len(conf))


def reliability_bins(conf, correct, bins=5):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return [
        (float(conf[idx == b].mean()), float(correct[idx == b].mean()), int((idx == b).sum()))
        for b in range(bins)
        if (idx == b).any()
    ]


def decision_point(auto, correct):
    # Coverage and risk from per-question decisions (True = the bot answered on its own).
    if not auto.any():
        return 0.0, 0.0
    return float(auto.mean()), float(1.0 - correct[auto].mean())


def bootstrap_ci(auto, correct, seed):
    # 95% interval from resampling the questions. A resample where nothing is auto-resolved has no
    # error rate at all, so it's left out of the risk interval instead of counting as 0% error.
    rng = np.random.default_rng(seed)
    n = len(auto)
    coverage, risk = [], []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        cov, r = decision_point(auto[s], correct[s])
        coverage.append(cov)
        if cov > 0:
            risk.append(r)
    cov_lo, cov_hi = np.percentile(coverage, [2.5, 97.5])
    risk_lo, risk_hi = np.percentile(risk, [2.5, 97.5]) if risk else (0.0, 0.0)
    return {"coverage": [float(cov_lo), float(cov_hi)], "risk": [float(risk_lo), float(risk_hi)]}


def curve_points(conf, correct):
    return [operating_point(conf, correct, float(t)) for t in np.unique(conf)]


def above(conf, t):
    return np.zeros(len(conf), dtype=bool) if t is None else conf >= t


def evaluate(rows):
    # K-fold cross-fitting: every question is scored by a calibrator (and a threshold) fitted on the
    # OTHER folds, so all 170 questions are held-out test questions, and each fit still sees 80% of the data.
    X = np.array([features(r) for r in rows])
    y = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    raw = np.array([raw_confidence(r) for r in rows])
    calls = np.array([r["llm_calls"] for r in rows], dtype=float)
    n = len(rows)

    per_seed = []
    for seed in SEEDS:
        folds = np.array_split(np.random.default_rng(seed).permutation(n), N_FOLDS)
        p_held = np.zeros(n)
        auto = {name: np.zeros(n, dtype=bool) for name in ("raw_measured", "calibrated", "budget_safe")}
        auto["guessed"] = raw >= GUESSED_THRESHOLD  # nothing to fit: the guessed control
        for fold in folds:
            train = np.setdiff1d(np.arange(n), fold)
            model = Calibrator().fit(X[train], y[train])
            p_train = model.predict(X[train])
            p_held[fold] = model.predict(X[fold])
            auto["raw_measured"][fold] = above(raw[fold], pick_threshold(raw[train], y[train], ERROR_BUDGET))
            auto["calibrated"][fold] = above(p_held[fold], pick_threshold(p_train, y[train], ERROR_BUDGET))
            auto["budget_safe"][fold] = above(
                p_held[fold], pick_threshold(p_train, y[train], ERROR_BUDGET, z=SAFE_Z))
        per_seed.append(
            {
                "auto": auto,
                "points": {name: decision_point(a, y) for name, a in auto.items()},
                # Every question is resolved: by the bot, or by a human ticket if escalated.
                "cost": {name: float(np.mean(calls * COST_PER_LLM_CALL + ~a * COST_PER_HUMAN_TICKET))
                         for name, a in auto.items()},
                "ece_raw": ece(raw, y),
                "ece_calibrated": ece(p_held, y),
                "p_held": p_held,
            }
        )
    return X, y, raw, per_seed


def summarize(per_seed, y):
    def policy(name, with_ci=True):
        out = {
            "coverage": float(np.mean([s["points"][name][0] for s in per_seed])),
            "risk": float(np.mean([s["points"][name][1] for s in per_seed])),
            "cost_per_query": float(np.mean([s["cost"][name] for s in per_seed])),
        }
        if with_ci:
            # Each seed's own held-out point next to its own 95% interval.
            out["per_seed"] = [
                {"seed": seed, "coverage": s["points"][name][0], "risk": s["points"][name][1],
                 "ci": bootstrap_ci(s["auto"][name], y, seed=seed)}
                for seed, s in zip(SEEDS, per_seed, strict=True)
            ]
        return out

    return {
        "n_questions": len(y),
        "error_budget": ERROR_BUDGET,
        "method": f"{N_FOLDS}-fold cross-fitting, seeds {list(SEEDS)}",
        "assumptions": {"cost_per_llm_call": COST_PER_LLM_CALL, "cost_per_human_ticket": COST_PER_HUMAN_TICKET},
        "guessed": policy("guessed"),
        "raw_measured": policy("raw_measured", with_ci=False),
        "calibrated": policy("calibrated"),
        "budget_safe": {**policy("budget_safe"), "z": SAFE_Z},
        "ece": {"raw": float(np.mean([s["ece_raw"] for s in per_seed])),
                "calibrated": float(np.mean([s["ece_calibrated"] for s in per_seed]))},
    }


def plot(per_seed, y, raw, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = per_seed[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, conf in (("raw similarity", raw), ("calibrated (held-out)", first["p_held"])):
        pts = sorted(curve_points(conf, y))
        ax.plot([c for c, _ in pts], [r for _, r in pts], marker="o", markersize=3, label=name)
    for key, label, mark in (("guessed", "guessed threshold", "x"), ("calibrated", "chosen: measured error", "s"),
                             ("budget_safe", "chosen: budget-safe", "D")):
        cov, risk = first["points"][key]
        ax.scatter([cov], [risk], s=90, marker=mark, color="black", zorder=3, label=label)
    ax.axhline(ERROR_BUDGET, linestyle="--", color="grey", label="error budget")
    ax.set_xlabel("coverage (share auto-resolved)")
    ax.set_ylabel("risk (error rate on auto-resolved)")
    ax.set_title(f"Risk-coverage curve ({N_FOLDS}-fold held-out, seed 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    for name, conf in (("raw similarity", raw), ("calibrated (held-out)", first["p_held"])):
        bins = reliability_bins(conf, y)
        ax.plot([b[0] for b in bins], [b[1] for b in bins], marker="o", label=name)
        for x, acc, count in bins:
            ax.annotate(str(count), (x, acc), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("actual accuracy")
    ax.set_title(f"Reliability diagram ({N_FOLDS}-fold held-out, seed 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    plt.close(fig)


def calibrate(rows=None, out=OUT):
    rows = load_rows() if rows is None else rows
    skipped = [r for r in rows if r["degraded"]]
    if skipped:
        # Rows where the model was unreachable (usually the free quota) say nothing about the policy. Leave them out.
        print(f"leaving out {len(skipped)} questions where the model was unavailable; rerun the agent to fill them")
        rows = [r for r in rows if not r["degraded"]]
    X, y, raw, per_seed = evaluate(rows)
    summary = summarize(per_seed, y)

    # The deployed policy: fit on every labelled question; the live site uses the budget-safe threshold,
    # because "holds a 10% error budget" is the claim. The measured-error threshold is kept for comparison.
    final = Calibrator().fit(X, y)
    p_all = final.predict(X)
    threshold_measured = pick_threshold(p_all, y, ERROR_BUDGET)
    threshold_safe = pick_threshold(p_all, y, ERROR_BUDGET, z=SAFE_Z)

    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(
        json.dumps({"model": final.to_dict(), "threshold": threshold_safe,
                    "threshold_measured_error": threshold_measured, "error_budget": ERROR_BUDGET}, indent=1),
        encoding="utf-8",
    )
    (out / "curve.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot(per_seed, y, raw, out)

    g = summary["guessed"]
    print(f"{summary['n_questions']} questions, error budget {ERROR_BUDGET:.0%}, {N_FOLDS}-fold held-out, mean of {len(SEEDS)} seeds")
    print(f"  guessed threshold : auto-resolves {g['coverage']:.0%} at {g['risk']:.0%} error, Rs {g['cost_per_query']:.2f}/query")
    for key, label in (("calibrated", "calibrated, measured error"), ("budget_safe", "calibrated, budget-safe")):
        c = summary[key]
        print(f"  {label:26}: auto-resolves {c['coverage']:.0%} at {c['risk']:.0%} error, Rs {c['cost_per_query']:.2f}/query")
        for s in c["per_seed"]:
            cov, risk = s["ci"]["coverage"], s["ci"]["risk"]
            print(f"    seed {s['seed']}: auto-resolves {s['coverage']:.0%} [{cov[0]:.0%}, {cov[1]:.0%}]"
                  f" at {s['risk']:.0%} error [{risk[0]:.0%}, {risk[1]:.0%}]  (95% CI)")
    print(f"  ECE raw {summary['ece']['raw']:.3f} -> calibrated {summary['ece']['calibrated']:.3f}")
    return summary
