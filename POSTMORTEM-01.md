# Postmortem 01: the live Hinglish model answered "calendar_query" to everything

**Date:** 2026-11-20 (times IST) · **Service:** [`/codemix`](https://deferral-curve.onrender.com/codemix) on
Render's free plan · **Impact:** about 10 hours of wrong answers on `/codemix`. The support bot on `/` was not
affected (it doesn't use this model). No data was lost and nothing user-facing needed cleaning up.

## Symptom

Every request to `/codemix`, whatever it said, got the intent `calendar_query` at about 13% confidence, with
no slots. Before the page went live, the same int8 model had scored 83% on the Hinglish test set in CI.

## Timeline

| time | what happened |
|---|---|
| 03:17 | `/codemix` deployed with the int8 ONNX model from codemix-nlu (signed 8-bit weights, `QInt8`) |
| after deploy | First live requests: `calendar_query` for everything. Noticed by sending known sentences to the live page |
| 04:12 | Added `scripts/check_serving.py` to codemix-nlu's CI: it runs the model **exactly as the server does** (the light `tokenizers` library, the same memory options). Result: 83.2%, identical for all three ways of running it. So the code path was fine |
| 04:12–12:32 | Ruled out a stale or half-uploaded file (Hugging Face file times and sizes were as expected). What was left: the hardware |
| 12:32 | Re-exported with **unsigned** 8-bit weights (`QUInt8`) |
| 13:30 | Added `cpu_has_vnni` to `/stats`. The live server reports `false` |
| ~13:40 | Redeployed with the new model: 12 of 12 test requests right. Slot F1 of the int8 model also went up (63.7 → 68.2) |

## Cause

ONNX Runtime's int8 matrix multiply has two paths on x86 CPUs. CPUs with the **VNNI** instructions compute it
exactly. Older CPUs without VNNI use an instruction that adds pairs of products into a 16-bit number, and with
signed 8-bit weights that sum can **saturate** (hit the 16-bit limit and get clipped). The ONNX Runtime
quantization docs warn about this and suggest `reduce_range` or unsigned (`U8U8`) weights.

GitHub's CI runners have newer CPUs, where the same file worked. Render's free-plan CPU has no VNNI, so the
clipped sums wrecked every layer's output and the model fell back to one answer.

## Why the checks didn't catch it

Every check ran on GitHub's hardware: the export's accuracy check, the int8-vs-full agreement check and
`check_serving.py`. **No check ran on the machine that serves the traffic.** The safety gate added earlier
that night (export fails if int8 loses more than 10 points) works, but it also runs on GitHub.

## What changed so it can't recur

1. **Unsigned 8-bit weights** are the default in codemix-nlu's `cmx/export.py`, with a comment explaining why.
2. **`/stats` shows `cpu_has_vnni`**, so a hardware difference is visible at a glance.
3. **A daily check on the live site** ([`.github/workflows/live-check.yml`](.github/workflows/live-check.yml)):
   it sends three Hinglish requests with known answers to the live `/codemix` and fails (GitHub emails the
   owner) if any answer is wrong. This is the check that would have caught the bug on the first day.
4. `check_serving.py` stays in codemix-nlu's CI: it rules out the code path quickly next time.

## Lesson

Test on the hardware you serve from, not only on the hardware you build on. Quantized models are the
clearest case: the same file can give different answers on different CPUs.
