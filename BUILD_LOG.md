# Build log

A short record of the decisions I made while building deferral-curve, and why. The README has the results.

## Decisions

- **Licence:** Apache 2.0, which includes an explicit patent grant: anyone who contributes code also licenses any
  patents covering it, so nobody can contribute and later sue the people using the project.
- **Where things run:** tests, the evaluation and the calibration all run on GitHub Actions on every push. The
  live site runs on Render's free plan. Nothing heavy runs on my laptop.
- **Model:** Groq's free API with `openai/gpt-oss-120b`. The first model I picked had been shut down, which I only
  noticed after making every fallback record its error.
- **Every model reply is cached** in `evals/llm_cache.json`, so all the numbers can be recomputed without an API key.
- **Retrieval:** BM25 plus model2vec static embeddings, averaged after rescaling. Static embeddings instead of a
  transformer encoder keep the server inside 512 MB.
- **Grading:** a fixed list of required facts per question, no LLM judge. Questions that should go to a human never
  count as correctly auto-answered.
- **The only hand-tuned number is the 10% error budget.** The threshold is computed from it.
- **Two threshold rules, both reported:** the threshold at the measured 10% error drifted to 11–12% on new
  questions, so the live site uses a budget-safe threshold based on a 95% upper bound on the error.
- **Evaluation:** 5-fold cross-fitting with 3 seeds. I first split the questions in half, but 85 questions were too
  few for the budget-safe bound, so I switched.
- **Host:** Render instead of Hugging Face Spaces, because free Hugging Face accounts can no longer create Docker
  Spaces.

## Changes I made after reading results

These are in the git history; I list them so nobody has to find them:

- Widened accepted phrasings for 6 questions whose answers were right in different wording.
- Made the 13 order questions stricter: right tool *and* right facts, not just "a tool ran".
- Fixed the cancel tool so one question's cancel can't change the next question's answer.
- Reworded the cancel tool's message, which the model was misreading as "already cancelled".

No question's expected outcome was changed.

## Status

Done: retrieval, agent with tools and verification, grading, calibrated hand-off policy, risk–coverage curve with
confidence intervals, live site, README.
