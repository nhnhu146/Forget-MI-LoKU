---
name: kaggle-run-gotchas
description: Recurring Kaggle operational gotchas for running this project (secrets, 12h limit, notebook re-import, recovery)
metadata:
  type: project
---

The user runs experiments on **Kaggle across multiple accounts** (hoangnhu03, hoangnhu2, …). Recurring gotchas:

- **Per-account secrets:** auto-push (per-seed/per-config) needs `GITHUB_TOKEN`, `GIT_EMAIL`, `GIT_NAME` in **each** account's Add-ons → Secrets. If missing, push silently skips ("No user secrets exist…") and the run's results stay only in `/kaggle/working` → must recover manually. Remind the user to set secrets on every account before running.
- **12h session limit:** long runs time out (e.g. 3 baseline seeds ≈ 12.3h, or a full sweep). Mitigations already in code: per-seed push in `run_kaggle_baseline.ipynb` and (to add) per-config push in the sweep. Run ≤2 seeds / fewer configs per session; resume skips done work via the restored CSV.
- **Recover from timeout:** if a run finished work but didn't push, ask the user to download `/kaggle/working/results_summary.csv` (+ exp_*.md) from the Save Version Output; I merge into the repo CSV, regenerate the summary, and push. Done this several times (baseline 10% seeds, the 27-config sweep).
- **Notebook fixes need re-import:** Cell 1 does `git reset --hard origin/master` which updates the cloned **repo code** but NOT the **notebook cells** (those live in the Kaggle editor). After I change a notebook cell, the user must re-import the notebook from GitHub (or paste the cell) for the fix to take effect.
- **Dataset slugs:** MIMIC models = `forget-mi-models-full` (nesting `original_model/forgetme/training_original_model`, gold `model_retrained_<N>per/model_retrained_<N>per`); data = `forget-mi-data`. IU = `forget-mi-data-iu` / `forget-mi-models-iu`.

See [[thesis-status]], [[multi-tab-claude-collision]].
