---
name: baseline-must-match-paper
description: The Forget-MI baseline reproduction must faithfully reproduce the PAPER's numbers using the ORIGINAL code — never modify its logic or make it "better"
metadata:
  type: feedback
---

The reproduced Forget-MI baseline (`training/forgetmi_partial.py`) is the **controlled yardstick** used to judge how good LoKU is — NOT the paper's published numbers. Reason: a fair comparison needs LoKU and Forget-MI run under identical conditions (same machine/data/eval/MIA/seeds); comparing to the paper's published numbers is confounded by their different pipeline. BUT for the yardstick to be credible, the reproduction must land on the paper's numbers as closely as possible — **equal to paper, not better, not worse.**

**Why:** the user was firm about this after I repeatedly misframed it (suggested comparing to paper directly, or making the baseline forget harder). A baseline that's anomalously strong on some metric (e.g. Df_AUC 0.571 vs paper 0.735) is BROKEN as a reference — it blocks LoKU from winning there even though LoKU beats the paper on that same metric.

**How to apply:**
- Use the ORIGINAL paper code/config VERBATIM. Reference copy lives at `C:\Users\admin\Downloads\Forget-MI-main\Forget-MI-main` (original `config.yaml` + `training/forgetmi_partial.py`). The training logic in the working repo is already identical (data_split, unlearn loop, loss, hinge, optimizer.step() once per epoch).
- Do NOT modify training logic, do NOT "improve" or tune to beat the paper, do NOT invent epoch-selection schemes. Only minimal run-fixes (config loading, Kaggle read-only cache, torch.load weights_only).
- Root cause of the over-forget bug: loss weights were changed to Unimodal `4/4/1/1` (0.4/0.4/0.1/0.1); the paper's `config.yaml` uses `alpha=1,beta=1,theta=2,gamma=2` (0.17/0.17/0.33/0.33). High beta(UU) + `use_noise=false` (makes L_uu an unbounded push-away) → over-forget. Fixed in `config_baseline_kaggle.yaml`. lr: paper sweeps [1e-4,1e-5]; currently 1e-5, try 1e-4 if Df_AUC still low.

See [[thesis-status]], [[mia-computation]].
