---
name: provide-commit-message-after-exp
description: User wants a ready-to-use git commit message after finishing edits for any experiment
metadata:
  type: feedback
---

After finishing code/config edits for a given experiment (exp08, exp09, exp10, ...), always proactively give the user a ready-to-paste git commit message.

**Why:** The user commits and pushes from local themselves, then pulls on Colab. They asked to receive the commit text each time without having to request it.

**How to apply:** Match the repo's existing commit style: `exp NN: <short description>` (see git log, e.g. "exp 09: implement Inverted Hinge Loss (IHL) from LoKU paper Sec 3.3"). Keep it concise; add a short body listing the key files/changes if the change is large. End with the `Co-Authored-By:` trailer per harness convention.
