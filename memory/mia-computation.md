---
name: mia-computation
description: How the two MIA metrics are computed in this project and what value is "perfect"
metadata:
  type: reference
---

MIA = membership inference attack. Feature = per-sample **cross-entropy on the IMAGE branch (`img_logits`)** only. Train `SVC(C=3, kernel='rbf', gamma='auto')` to separate **retain (member=1)** from **test (non-member=0)**, then predict on the **forget** set. **MIA = fraction of forget predicted as member** = `clf.predict(forget).mean()`. Code: `training/forgetmi_loku.py::run_mia` and original `evaluation/eval_unlearning.py`.

Two variants:
- **MIA_persample**: per-sample img-CE, retain:test balanced 1:1. Fine, stable.
- **MIA_paper**: per-BATCH-mean img-CE (bs=32), unbalanced — replicates the original Forget-MI eval exactly (verified line-by-line). Coarse: ~7 forget points → quantized ~1/7, high variance → needs multi-seed.

**Scale (both):** [0, 1], **lower = better forgetting.** Ideal ≈ **0.0** (the retrained-from-scratch gold gives ~0; Original/before-unlearn ≈ 1.0). **NOT 0.5** here — the 0.5="random" convention is a different (AUC) framing; this hard binary-classifier setup puts the perfect-unlearning reference at ~0. Compare to retrained gold, not 0.5.

**Why image-only:** the paper does this; radiology text reports are templated (weak membership signal) while the image carries the patient-specific signal. This is exactly why LoKU needed **Image-FILA** — the attack reads only img_logits, so the image branch must be unlearned.

**"Fake zero" caveat:** MIA→0 can happen two ways — healthy (`forget_ce ≈ test_ce`, forget looks like unseen data) OR over-forgetting (`forget_ce >> test_ce`, model broke on forget). Code warns when forget_ce > 1.3×test_ce. Always report MIA alongside forget_ce/test_ce to show forgetting is healthy. See [[baseline-must-match-paper]].
