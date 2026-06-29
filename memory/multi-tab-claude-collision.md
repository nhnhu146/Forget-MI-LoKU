---
name: multi-tab-claude-collision
description: User runs 3 Claude tabs on this repo at once — guard against file/index collisions
metadata:
  type: feedback
---

The user runs **3 concurrent Claude Code tabs on the same working directory** (`Forget-MI-main`) to save time before a thesis deadline. This causes recurring breakage: two tabs editing the same `.ipynb` mangle its JSON source array (symptom: `print("` split across a real newline → "unterminated string literal", or "source array was 1 mashed line"); and the shared git index means one tab's commit sweeps another tab's already-staged files (this happened: commit `e007787` bundled 8 unrelated files).

**Why:** parallelism for deadline speed — the user won't drop to a single tab.

**How to apply:**
- Commit with the **pathspec form** `git commit <explicit/path> -m "..."` so it captures ONLY your file regardless of what other tabs have staged. Never `git add .` / `git add -A` / `git add <dir>`.
- Before editing a notebook, assume another tab may also have it open; prefer the tab/file partition (Tab1=LoKU sweep + `forgetmi_loku.py`, Tab2=baseline + `forgetmi_partial.py`, Tab3=IU). Don't touch shared infra (`joint_img_txt/`, `evaluation/`) unless it's this tab's assigned area.
- After committing, verify with `git show --stat <sha>` that no foreign files got bundled.
- Kaggle notebooks `git reset --hard origin/master`, so everything must land on `master` — separate branches need merging back. See [[provide-commit-message-after-exp]].
