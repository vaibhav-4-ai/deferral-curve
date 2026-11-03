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

SEEDS = (0, 1, 2)
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


def pick_threshold(conf, correct, budget):
    # The lowest threshold (= most questions auto-resolved) whose error rate stays within budget.
    best_t, best_cov = None, 0.0
    for t in np.unique(conf):
        cov, risk = operating_point(conf, correct, t)
        if risk <= budget and cov > best_cov:
            best_t, best_cov = float(t), cov
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


def bootstrap_ci(conf, correct, t, seed):
    # 95% interval from resampling the test questions. A resample where nothing is auto-resolved has no
    # error rate at all, so it's left out of the risk interval instead of counting as 0% error.
    rng = np.random.default_rng(seed)
    n = len(conf)
    coverage, risk = [], []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        cov, r = operating_point(conf[s], correct[s], t)
        coverage.append(cov)
        if cov > 0:
            risk.append(r)
    cov_lo, cov_hi = np.percentile(coverage, [2.5, 97.5])
    risk_lo, risk_hi = np.percentile(risk, [2.5, 97.5]) if risk else (0.0, 0.0)
    return {"coverage": [float(cov_lo), float(cov_hi)], "risk": [float(risk_lo), float(risk_hi)]}


def curve_points(conf, correct):
    return [operating_point(conf, correct, float(t)) for t in np.unique(conf)]


def evaluate(rows):
    X = np.array([features(r) for r in rows])
    y = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    raw = np.array([raw_confidence(r) for r in rows])
    calls = np.array([r["llm_calls"] for r in rows], dtype=float)

    per_seed = []
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(rows))
        cal, test = order[: len(rows) // 2], order[len(rows) // 2 :]
        model = Calibrator().fit(X[cal], y[cal])
        p_test = model.predict(X[test])
        t_raw = pick_threshold(raw[cal], y[cal], ERROR_BUDGET)
        t_cal = pick_threshold(model.predict(X[cal]), y[cal], ERROR_BUDGET)
        per_seed.append(
            {
                "guessed": operating_point(raw[test], y[test], GUESSED_THRESHOLD),
                "raw_measured": operating_point(raw[test], y[test], t_raw),
                "calibrated": operating_point(p_test, y[test], t_cal),
                "cost_guessed": cost_per_query(raw[test], calls[test], GUESSED_THRESHOLD),
                "cost_calibrated": cost_per_query(p_test, calls[test], t_cal),
                "ece_raw": ece(raw[test], y[test]),
                "ece_calibrated": ece(p_test, y[test]),
                "test": test,
                "p_test": p_test,
                "t_cal": t_cal,
            }
        )
    return X, y, raw, per_seed


def summarize(per_seed, y):
    def mean_of(key, i=None):
        vals = [s[key] if i is None else s[key][i] for s in per_seed]
        return float(np.mean(vals))

    # Each seed's own held-out operating point next to its own interval, so a point always sits inside its CI.
    seeds = [
        {
            "seed": seed,
            "coverage": s["calibrated"][0],
            "risk": s["calibrated"][1],
            "ci": bootstrap_ci(s["p_test"], y[s["test"]], s["t_cal"], seed=seed),
        }
        for seed, s in zip(SEEDS, per_seed, strict=True)
    ]
    return {
        "n_questions": len(y),
        "error_budget": ERROR_BUDGET,
        "assumptions": {"cost_per_llm_call": COST_PER_LLM_CALL, "cost_per_human_ticket": COST_PER_HUMAN_TICKET},
        "guessed": {"coverage": mean_of("guessed", 0), "risk": mean_of("guessed", 1),
                    "cost_per_query": mean_of("cost_guessed")},
        "raw_measured": {"coverage": mean_of("raw_measured", 0), "risk": mean_of("raw_measured", 1)},
        "calibrated": {"coverage": mean_of("calibrated", 0), "risk": mean_of("calibrated", 1),
                       "cost_per_query": mean_of("cost_calibrated"), "per_seed": seeds},
        "ece": {"raw": mean_of("ece_raw"), "calibrated": mean_of("ece_calibrated")},
    }


def plot(per_seed, y, raw, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = per_seed[0]
    test = first["test"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, conf in (("raw similarity", raw[test]), ("calibrated", first["p_test"])):
        pts = sorted(curve_points(conf, y[test]))
        ax.plot([c for c, _ in pts], [r for _, r in pts], marker="o", label=name)
    ax.axhline(ERROR_BUDGET, linestyle="--", color="grey", label="error budget")
    ax.set_xlabel("coverage (share auto-resolved)")
    ax.set_ylabel("risk (error rate on auto-resolved)")
    ax.set_title("Risk-coverage curve (held-out split, seed 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    for name, conf in (("raw similarity", raw[test]), ("calibrated", first["p_test"])):
        bins = reliability_bins(conf, y[test])
        ax.plot([b[0] for b in bins], [b[1] for b in bins], marker="o", label=name)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("actual accuracy")
    ax.set_title("Reliability diagram (held-out split, seed 0)")
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

    # The deployed policy: fit on every labelled question, threshold chosen to hold the error budget.
    final = Calibrator().fit(X, y)
    threshold = pick_threshold(final.predict(X), y, ERROR_BUDGET)

    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(
        json.dumps({"model": final.to_dict(), "threshold": threshold, "error_budget": ERROR_BUDGET}, indent=1),
        encoding="utf-8",
    )
    (out / "curve.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot(per_seed, y, raw, out)

    g, c = summary["guessed"], summary["calibrated"]
    print(f"{summary['n_questions']} questions, error budget {ERROR_BUDGET:.0%}, mean of {len(SEEDS)} seeds")
    print(f"  guessed threshold : auto-resolves {g['coverage']:.0%} at {g['risk']:.0%} error, Rs {g['cost_per_query']:.2f}/query")
    print(f"  calibrated policy : auto-resolves {c['coverage']:.0%} at {c['risk']:.0%} error, Rs {c['cost_per_query']:.2f}/query")
    for s in c["per_seed"]:
        cov, risk = s["ci"]["coverage"], s["ci"]["risk"]
        print(f"    seed {s['seed']}: auto-resolves {s['coverage']:.0%} [{cov[0]:.0%}, {cov[1]:.0%}]"
              f" at {s['risk']:.0%} error [{risk[0]:.0%}, {risk[1]:.0%}]  (95% CI)")
    print(f"  ECE raw {summary['ece']['raw']:.3f} -> calibrated {summary['ece']['calibrated']:.3f}")
    return summary
