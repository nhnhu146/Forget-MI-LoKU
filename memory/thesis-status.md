---
name: thesis-status
description: Current state of the Forget-MI-LoKU thesis — what's done, what we're waiting on, deadline, chosen config
metadata:
  type: project
---

Deadline: **~1 week from 2026-06-30**. Must include **IU-CXR 3% (reduced, 1–2 seeds)** for the cross-dataset story; full IU was cut.

**WAITING ON (current blocker):** the user is re-running the **baseline 3% (1 seed) on Kaggle** with the corrected paper-faithful config (weights 1/1/2/2). I should wait for that result and check **Df_AUC ≈ 0.735, MIA ≈ 0.571** (paper). If it matches → run 6%/10%; if Df_AUC still low → switch lr 1e-5→1e-4. See [[baseline-must-match-paper]].

**Done & on GitHub (`nhnhu146/Forget-MI-LoKU`, master):**
- LoKU code complete, runs on Kaggle (all bugs fixed: deps pin, forget-mi-models-full paths, read-only cache regen, study-id 's'-prefix resolve).
- LoKU MIMIC sweep: 27 configs (3% A–J, 6% A–J, 10% A–G), **seed 42 only** — chosen config **D_combo_aggressive** (IHL=1.25, img=0.5, 8ep, kappa=2.0): healthy forgetting (forget_ce≈test_ce) + beats paper. Needs multi-seed (123, 7).
- Baseline MIMIC 3/6/10% × 3 seeds — BUT with the WRONG weights (4/4/1/1, over-forget). Being redone with correct 1/1/2/2.
- LoKU 3% honest IHL0.75 multi-seed (Colab, older gentle config).

**Pending:** corrected baseline 3/6/10%; LoKU-D multi-seed; entire IU 3% pipeline (preprocess CPU → train og+re GPU → baseline IU → LoKU IU).

**LoKU vs paper (seed 42):** LoKU-D beats paper on MIA (0.143/0.385/0.364 vs 0.571/0.615/0.810) and utility (Dt_AUC higher) at all 3 forget%, ~12–17× faster, 0.45% params.

See [[kaggle-run-gotchas]], [[multi-tab-claude-collision]].
