"""
Experiment tracker — auto-fill MD files & INDEX after a training run.

Usage from training script:
    from scripts.exp_tracker import ExpTracker
    tracker = ExpTracker(name="forget_margin_20", hypothesis="...")
    # ... training loop ...
    tracker.log_epoch_line(f"[E00] loss=...")
    # ... after eval ...
    tracker.finalize(results_dict, elapsed_h)

Standalone CLI (for retroactive update from a CSV row):
    python scripts/exp_tracker.py finalize --exp forget_margin_20 --csv path.csv
"""
from __future__ import annotations
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiments"
TEMPLATE = EXP_DIR / "_TEMPLATE.md"
INDEX = EXP_DIR / "INDEX.md"
CONFIG_PATH = REPO_ROOT / "config.yaml"


# ---------- Git utils ----------

def _git(*args) -> str:
    try:
        return subprocess.check_output(['git', *args], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _git_info():
    commit = _git('rev-parse', '--short', 'HEAD') or "unknown"
    branch = _git('rev-parse', '--abbrev-ref', 'HEAD') or "unknown"
    if _git('status', '--porcelain'):
        commit += " (dirty)"
    return commit, branch


# ---------- Marker replacement ----------

def _replace_marker(text: str, marker: str, content: str) -> str:
    """Replace content between <!-- AUTO:marker --> ... <!-- /AUTO -->."""
    pattern = re.compile(
        rf"(<!-- AUTO:{re.escape(marker)} -->).*?(<!-- /AUTO -->)",
        re.DOTALL,
    )
    repl = rf"\1\n{content}\n\2"
    return pattern.sub(repl, text, count=1)


# ---------- ID/file lookup ----------

def _slug(s: str) -> str:
    s = re.sub(r'[\s\-]+', '_', s.strip().lower())
    return re.sub(r'[^a-z0-9_]', '', s)


def _find_or_alloc(name: str) -> tuple[int, Path, bool]:
    """Return (id, path, is_new). Reuse existing file if same name."""
    slug = _slug(name)
    pattern = re.compile(rf"^exp_(\d+)_{re.escape(slug)}\.md$")
    for f in EXP_DIR.glob("exp_*.md"):
        m = pattern.match(f.name)
        if m:
            return int(m.group(1)), f, False
    # Allocate new
    ids = []
    for f in EXP_DIR.glob("exp_*.md"):
        m = re.match(r"exp_(\d+)_", f.name)
        if m:
            ids.append(int(m.group(1)))
    new_id = (max(ids) + 1) if ids else 1
    return new_id, EXP_DIR / f"exp_{new_id:03d}_{slug}.md", True


# ---------- Config diff vs previous exp ----------

def _config_diff_vs_prev(prev_id: int | None) -> str:
    """Diff current config.yaml vs config snapshot in previous exp file."""
    if prev_id is None:
        return "_(no previous experiment to compare)_"
    prev_files = list(EXP_DIR.glob(f"exp_{prev_id:03d}_*.md"))
    if not prev_files:
        return "_(previous exp file not found)_"
    prev_text = prev_files[0].read_text(encoding='utf-8')
    # Extract config block
    m = re.search(r"```yaml\n(.*?)\n```", prev_text, re.DOTALL)
    if not m:
        return "_(no config snapshot in previous exp)_"
    prev_cfg = m.group(1)
    curr_cfg = CONFIG_PATH.read_text(encoding='utf-8')
    # Parse simple "key: value" lines
    prev_vals = _parse_simple_yaml(prev_cfg)
    curr_vals = _parse_simple_yaml(curr_cfg)
    changed = []
    for k, v in curr_vals.items():
        if k in prev_vals and prev_vals[k] != v:
            changed.append((k, prev_vals[k], v))
    if not changed:
        return "_(không thay đổi config so với exp trước)_"
    lines = ["| Param | Trước | Sau |", "|---|---|---|"]
    for k, old, new in changed:
        lines.append(f"| `{k}` | `{old}` | `{new}` |")
    return "\n".join(lines)


def _parse_simple_yaml(text: str) -> dict:
    """Extract { key: value } from YAML format with 'value:' sub-keys."""
    out = {}
    cur_key = None
    for line in text.splitlines():
        line = line.rstrip()
        m_key = re.match(r"^  ([\w]+):\s*$", line)
        if m_key:
            cur_key = m_key.group(1); continue
        m_val = re.match(r"^    value:\s*(.+)$", line)
        if m_val and cur_key:
            out[cur_key] = m_val.group(1).strip()
            cur_key = None
    return out


# ---------- Tracker class ----------

class ExpTracker:
    def __init__(self, name: str, hypothesis: str | None = None,
                 prev_id: int | None = None):
        self.name = _slug(name)
        self.hypothesis = hypothesis or "_(điền hypothesis vào đây)_"
        self.epoch_lines: list[str] = []
        self.exp_id, self.exp_path, self.is_new = _find_or_alloc(name)
        # Auto-detect prev_id if not given
        if prev_id is None and self.is_new and self.exp_id > 1:
            prev_id = self.exp_id - 1
        self.prev_id = prev_id
        if self.is_new:
            self._create_from_template()
        print(f"📝 Tracking exp {self.exp_id:03d} → {self.exp_path.relative_to(REPO_ROOT)}")

    def _create_from_template(self):
        if not TEMPLATE.exists():
            print(f"⚠️  Template not found at {TEMPLATE}"); return
        commit, branch = _git_info()
        text = TEMPLATE.read_text(encoding='utf-8')
        cfg_text = CONFIG_PATH.read_text(encoding='utf-8') if CONFIG_PATH.exists() else ""
        for k, v in {
            "{{ID}}":         f"{self.exp_id:03d}",
            "{{NAME}}":       self.name,
            "{{DATE}}":       datetime.now().strftime("%Y-%m-%d"),
            "{{COMMIT}}":     commit,
            "{{BRANCH}}":     branch,
            "{{HYPOTHESIS}}": self.hypothesis,
            "{{CONFIG}}":     cfg_text,
        }.items():
            text = text.replace(k, v)
        # Auto config diff vs prev exp
        diff_content = _config_diff_vs_prev(self.prev_id)
        text = _replace_marker(text, "config_diff", diff_content)
        self.exp_path.write_text(text, encoding='utf-8')

    def log_epoch_line(self, line: str):
        """Call this for every per-epoch print in training loop."""
        self.epoch_lines.append(line.rstrip())

    def finalize(self, results: dict, elapsed_h: float):
        """Update Results section + append row to INDEX."""
        text = self.exp_path.read_text(encoding='utf-8')

        # Section 5.1 — Training curves
        curves = "```\n" + ("\n".join(self.epoch_lines) or "(no epochs logged)") + "\n```"
        text = _replace_marker(text, "training_curves", curves)

        # Section 5.2 — Final metrics
        def fmt(k, decimals=3):
            v = results.get(k)
            if v is None or (isinstance(v, float) and (v != v)):  # NaN check
                return "N/A"
            return f"{v:.{decimals}f}"
        metrics_tbl = (
            "| Metric | Value | Δ vs Paper (3%) |\n"
            "|---|---|---|\n"
            f"| MIA_persample ↓ | **{fmt('MIA')}** | (LoKU per-sample SVM) |\n"
            f"| MIA_paper ↓ | **{fmt('MIA_paper')}** | {_delta(results.get('MIA_paper'), 0.571)} |\n"
            f"| Forget AUC ↓ | **{fmt('Df_AUC')}** | {_delta(results.get('Df_AUC'), 0.735)} |\n"
            f"| Forget F1 ↓ | **{fmt('Df_F1')}** | {_delta(results.get('Df_F1'), 0.393)} |\n"
            f"| Test AUC ↑ | **{fmt('Dt_AUC')}** | {_delta(results.get('Dt_AUC'), 0.625, higher=True)} |\n"
            f"| Test F1 ↑ | **{fmt('Dt_F1')}** | {_delta(results.get('Dt_F1'), 0.250, higher=True)} |\n"
            f"| 1−CosSim ↓ | **{fmt('dist_vs_re')}** | {_delta(results.get('dist_vs_re'), 0.45)} |\n"
            f"| Time (h) | **{fmt('time_h')}** | {_delta(results.get('time_h'), 5.0)} |\n"
            f"| GPU peak (GB) | **{fmt('gpu_gb', 2)}** | — |\n"
            f"| Trainable params | **{results.get('trainable_pct', 0):.3f}%** | (vs 100% paper) |\n"
        )
        text = _replace_marker(text, "metrics", metrics_tbl)

        # Duration field
        text = _replace_marker(text, "duration", f"{elapsed_h:.3f}h")

        # Update Status: 🔄 → ✅ Auto-tracked
        text = re.sub(r"\*\*Status\*\* \| 🔄 Running",
                      "**Status** | ✅ Auto-tracked (cần điền Observations + Conclusion)",
                      text, count=1)

        self.exp_path.write_text(text, encoding='utf-8')

        # Append row to INDEX
        self._append_index(results)

        print(f"\n✅ Auto-saved: {self.exp_path.relative_to(REPO_ROOT)}")
        print(f"✅ Index updated: {INDEX.relative_to(REPO_ROOT)}")
        print(f"\n📌 Còn lại CẦN ĐIỀN TAY (chỉ 3 section nhỏ):")
        print(f"   - Section 6: Observations (quan sát bất thường)")
        print(f"   - Section 7: Conclusion (giữ hay bỏ thay đổi này?)")
        print(f"   - Section 8: Next steps (exp tiếp theo)")

    def _append_index(self, results: dict):
        if not INDEX.exists():
            return
        text = INDEX.read_text(encoding='utf-8')
        date = datetime.now().strftime("%Y-%m-%d")
        row = (f"| {self.exp_id:03d} | {self.name} | {date} | 🔄 "
               f"| {results.get('MIA', 0):.3f} "
               f"| {results.get('Df_AUC', 0):.3f} "
               f"| {results.get('Df_F1', 0):.3f} "
               f"| {results.get('Dt_AUC', 0):.3f} "
               f"| {results.get('Dt_F1', 0):.3f} "
               f"| {results.get('time_h', 0):.3f} "
               f"| {results.get('trainable_pct', 0):.3f}% "
               f"| _(auto, cần kết luận)_ |\n")
        # Check if exp_id already in INDEX → replace that line
        existing = re.search(rf"^\| {self.exp_id:03d} \|.*$", text, re.MULTILINE)
        if existing:
            text = text.replace(existing.group(0), row.rstrip())
        else:
            # Insert before "**Reference" line
            marker = "\n**Reference"
            if marker in text:
                text = text.replace(marker, row + marker, 1)
            else:
                text += row
        INDEX.write_text(text, encoding='utf-8')


def _delta(val, ref, higher=False):
    """Format delta vs reference with arrow."""
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    d = val - ref
    sign = "+" if d >= 0 else ""
    if higher:
        ok = "✅" if d >= 0 else "❌"
    else:
        ok = "✅" if d <= 0 else "❌"
    return f"{sign}{d:.3f} {ok}"


# ---------- CLI (for retroactive use) ----------

if __name__ == "__main__":
    print(__doc__)
