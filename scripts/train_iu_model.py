#!/usr/bin/env python3
"""
Train the IU-CXR joint ImageText models — EXP7-I (THESIS_ROADMAP Section 6.8).

The Forget-MI paper only releases a pretrained model for MIMIC. To run unlearning
(baseline Forget-MI + LoKU) on Indiana University CXR we must first train:

  * model_og_IU       — trained on the FULL IU train fold                 (--mode og)
  * model_retrained_iu_Nper — trained on IU train fold MINUS the N% forget set (--mode re --forget_pct N)

`model_retrained_iu_Nper` is the *gold* "exactly retrained without the forget data"
reference used for the 1-CosSim metric.

Both are saved in the SAME HuggingFace-style layout as the MIMIC base model
(`forgetme/training_original_model/`: config.json + pytorch_model.bin + tokenizer
files) so `forgetmi_partial.py` / `forgetmi_loku.py` can load them via
`ImageTextModel.from_pretrained(...)` with no changes.

Why this is correct-by-construction: we reuse `forgetmi_partial.build_dataset`, so the
cached text features, the train/test split logic, the forget/retain carving and the
image loading are byte-identical to the unlearning stage. The "retain" split returned
by build_dataset is exactly (TRAIN fold - forget set):
  * --mode og  → we point --forget_set at an EMPTY csv  → retain = full TRAIN fold.
  * --mode re  → we point --forget_set at forget_set_Nper_iu.csv → retain = TRAIN - N%.

Supervised loss mirrors the classification term used inside forgetmi_loku.py:
    CE(img_logits, y) + CE(txt_logits, y)   with y = raw class index (0/1 for IU).

Warm start: by default we initialise from an existing ImageTextModel checkpoint
(`--init_from`, e.g. the MIMIC base model). Both datasets are chest X-rays + clinical
text, so this is plain chest-xray->chest-xray transfer learning and converges far
faster than cold training within Kaggle's 12h session limit. Pass `--init_from ""`
to cold-start (random ResNet + BERT initialised from `--bert_pretrained_dir`).

Usage (Kaggle):
    python scripts/train_iu_model.py --config config_baseline_iu_kaggle.yaml \
        --mode og --init_from /kaggle/input/.../base_model/training_original_model \
        --output_dir /kaggle/working/forget-mi-models-iu/base_model/model_og_IU \
        --epochs 20 --lr 2e-5 --batch_size 16

    python scripts/train_iu_model.py --config config_baseline_iu_kaggle.yaml \
        --mode re --forget_pct 3 --init_from /kaggle/input/.../base_model/training_original_model \
        --output_dir /kaggle/working/forget-mi-models-iu/retrained_model/model_retrained_iu_3per \
        --epochs 20 --lr 2e-5 --batch_size 16
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

# Make repo root importable when run as `python scripts/train_iu_model.py`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import BertTokenizer, PretrainedConfig

from joint_img_txt.model import ImageTextModel
from training.forgetmi_partial import build_dataset, get_model_inputs


# Tokenizer files copied next to the saved model so BertTokenizer.from_pretrained() works.
_TOKENIZER_FILES = (
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
)


def _load_config(path):
    """Flatten a wandb-style YAML (parameters: {k: {value: v}}) into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = {}
    for k, v in raw.get("parameters", {}).items():
        if isinstance(v, dict):
            if "value" in v:
                cfg[k] = v["value"]
            elif "values" in v:
                cfg[k] = v["values"][0]
        else:
            cfg[k] = v
    return cfg


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_empty_forget_csv(path):
    """forget_set with only the header → data_split() finds 0 forget ids → retain = full train."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["subject_id"])
    return path


def _build_args(cfg, cli):
    """Assemble the SimpleNamespace-like args object that build_dataset / get_model_inputs expect."""
    from types import SimpleNamespace

    forget_path = cli.forget_set
    if cli.mode == "og":
        # Empty forget set → retain split = full TRAIN fold.
        forget_path = _write_empty_forget_csv(
            os.path.join("/kaggle/working" if os.path.isdir("/kaggle/working") else ".",
                         "_empty_forget_iu.csv")
        )

    args = SimpleNamespace(
        id=cfg.get("id", f"train_iu_{cli.mode}"),
        # --- data location (config is source of truth; CLI may override) ---
        text_data_dir=cli.text_data_dir or cfg["text_data_dir"],
        img_data_dir=cli.img_data_dir or cfg["img_data_dir"],
        data_split_path=cli.data_split or cfg["data_split_path"],
        forget_set_path=forget_path,
        synonyms_dir=cli.synonyms or cfg.get("synonyms_dir", "./data_splits/Synonyms.csv"),
        # --- feature / label encoding ---
        output_channel_encoding=cfg.get("output_channel_encoding", "multiclass"),
        max_seq_length=int(cfg.get("max_seq_length", 320)),
        text_noise_level=float(cfg.get("text_noise_level", 0.5)),
        reprocess_input_data=bool(cfg.get("reprocess_input_data", False)),
        random_point_ratio=float(cfg.get("random_point_ratio", 0.1)),
        validation_ratio=float(cli.val_ratio),
        # --- runtime ---
        do_train=True,
        do_eval=False,
        num_cpu_workers=int(cli.num_workers),
        # --- model forward flags (must match unlearning defaults) ---
        bert_pool_last_hidden=bool(cfg.get("bert_pool_last_hidden", False)),
        bert_pool_use_img=bool(cfg.get("bert_pool_use_img", False)),
        bert_pool_img_lowerlevel=bool(cfg.get("bert_pool_img_lowerlevel", False)),
    )
    return args


def _copy_tokenizer(src_dir, dst_dir):
    """Copy tokenizer artefacts next to the saved model (save_pretrained only writes weights+config)."""
    n = 0
    for fname in _TOKENIZER_FILES:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst_dir, fname))
            n += 1
    return n


def _make_model(cli, num_labels, device):
    """Warm-start from an ImageTextModel checkpoint, or cold-start from a BERT config."""
    if cli.init_from:
        print(f"🔥 Warm start from ImageTextModel: {cli.init_from}")
        model = ImageTextModel.from_pretrained(cli.init_from)
    else:
        bert_dir = cli.bert_pretrained_dir
        if not bert_dir or not os.path.isdir(bert_dir):
            raise SystemExit(
                "Cold start requires --bert_pretrained_dir pointing to a BERT checkpoint "
                "(config.json + vocab.txt). Or pass --init_from <ImageTextModel dir>."
            )
        print(f"❄️  Cold start: ResNet random + BERT from {bert_dir}")
        config = PretrainedConfig.from_pretrained(bert_dir)
        config.num_labels = num_labels
        config.output_attentions = getattr(config, "output_attentions", False)
        config.output_hidden_states = True
        model = ImageTextModel(config, pretrained_bert_dir=bert_dir)
    return model.to(device)


@torch.no_grad()
def _evaluate(model, loader, args, device):
    model.eval()
    tot_loss, tot_correct, tot_n = 0.0, 0, 0
    for batch in loader:
        inputs, labels, _ = get_model_inputs(args, batch, device)
        labels = labels.long().view(-1)
        outputs = model(**inputs)
        img_logits, txt_logits = outputs[1], outputs[3]
        loss = F.cross_entropy(img_logits, labels) + F.cross_entropy(txt_logits, labels)
        tot_loss += loss.item() * labels.size(0)
        # Ensemble image+text logits for the accuracy read-out.
        preds = (img_logits + txt_logits).argmax(dim=-1)
        tot_correct += (preds == labels).sum().item()
        tot_n += labels.size(0)
    avg_loss = tot_loss / max(1, tot_n)
    acc = tot_correct / max(1, tot_n)
    return avg_loss, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_baseline_iu_kaggle.yaml")
    ap.add_argument("--mode", choices=["og", "re"], required=True,
                    help="og = full train (original); re = train minus forget set (gold retrained)")
    ap.add_argument("--forget_pct", type=int, default=3, help="Only for --mode re")
    ap.add_argument("--forget_set", default=None,
                    help="Explicit forget-set CSV path (overrides the --forget_pct default)")
    ap.add_argument("--init_from", default=None,
                    help="ImageTextModel checkpoint dir to warm-start from (recommended). "
                         "Empty string '' forces cold start.")
    ap.add_argument("--bert_pretrained_dir", default=None,
                    help="BERT checkpoint for cold start AND tokenizer source")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--resume_from", default="",
                    help="Dir chứa _train_state.json + model từ run trước (resume khi Kaggle timeout 12h)")
    # data path overrides (else taken from --config)
    ap.add_argument("--text_data_dir", default=None)
    ap.add_argument("--img_data_dir", default=None)
    ap.add_argument("--data_split", default=None)
    ap.add_argument("--synonyms", default=None)
    # training hyper-params
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_best", action="store_true",
                    help="Save the epoch with the lowest val loss (default: save the last epoch)")
    cli = ap.parse_args()

    if cli.mode == "re" and cli.forget_set is None:
        cli.forget_set = f"./data_iu/data_splits/forget_set_{cli.forget_pct}per_iu.csv"

    _set_seed(cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("🔴 WARNING: no GPU — training will be extremely slow.")

    cfg = _load_config(cli.config)
    args = _build_args(cfg, cli)

    # ---- Resume (Kaggle 12h timeout): nạp model train dở + train tiếp ----
    start_epoch, best_val = 1, float("inf")
    _resume_dir = None
    for _cand in ((cli.resume_from or ""), cli.output_dir):
        if _cand and os.path.exists(os.path.join(_cand, "_train_state.json")):
            _resume_dir = _cand
            break
    if _resume_dir:
        _st = json.load(open(os.path.join(_resume_dir, "_train_state.json")))
        start_epoch = int(_st.get("epoch", 0)) + 1
        best_val = float(_st.get("best_val", float("inf")))
        cli.init_from = _resume_dir   # nạp model đã train dở thay vì warm-start MIMIC
        print(f"♻️  RESUME: {_resume_dir} (đã xong epoch {_st.get('epoch')}) → train tiếp từ epoch {start_epoch}")

    # Tokenizer source: init_from (warm) or bert_pretrained_dir (cold). Falls back to config.
    tok_src = cli.init_from or cli.bert_pretrained_dir or cfg.get("bert_pretrained_dir")
    print(f"🔤 Tokenizer from: {tok_src}")
    tokenizer = BertTokenizer.from_pretrained(tok_src)

    print("\n📦 Building dataset (reuses forgetmi_partial.build_dataset)...")
    print(f"   mode={cli.mode}  forget_set={args.forget_set_path}")
    print(f"   text_data_dir={args.text_data_dir}")
    print(f"   img_data_dir ={args.img_data_dir}")
    print(f"   data_split   ={args.data_split_path}")
    dataset, num_labels = build_dataset(args, tokenizer)
    train_set = dataset["retain"]
    val_set = dataset["validation"]
    print(f"\n🎯 Training samples (retain) = {len(train_set)} | val = {len(val_set)} | num_labels = {num_labels}")
    if len(train_set) == 0:
        raise SystemExit("Retain set empty — check data_split / forget_set / cached features.")

    train_loader = DataLoader(train_set, sampler=RandomSampler(train_set),
                              batch_size=cli.batch_size, num_workers=cli.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, sampler=SequentialSampler(val_set),
                            batch_size=cli.batch_size, num_workers=cli.num_workers, pin_memory=True)

    model = _make_model(cli, num_labels, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cli.lr, weight_decay=cli.weight_decay)

    os.makedirs(cli.output_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*70}\n🚂 TRAIN {cli.mode.upper()} — epochs={cli.epochs} lr={cli.lr} bs={cli.batch_size}\n{'='*70}")

    for epoch in range(start_epoch, cli.epochs + 1):
        model.train()
        run_loss, n_batch = 0.0, 0
        pbar = tqdm(train_loader, desc=f"E{epoch:02d}/{cli.epochs}")
        for batch in pbar:
            inputs, labels, _ = get_model_inputs(args, batch, device)
            labels = labels.long().view(-1)
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs[1], labels) + F.cross_entropy(outputs[3], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cli.max_grad_norm)
            optimizer.step()
            run_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f"{run_loss / n_batch:.4f}")

        val_loss, val_acc = _evaluate(model, val_loader, args, device)
        elapsed = (time.time() - t0) / 60
        eta = elapsed / epoch * (cli.epochs - epoch)
        print(f"[E{epoch:02d}] train_loss={run_loss / max(1, n_batch):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"elapsed={elapsed:.1f}m ETA={eta:.1f}m")

        if val_loss < best_val:
            best_val = val_loss
        # Rolling checkpoint MỖI epoch → resume được khi Kaggle ngắt 12h.
        _save(model, cli.output_dir, tok_src)
        with open(os.path.join(cli.output_dir, "_train_state.json"), "w") as _f:
            json.dump({"epoch": epoch, "best_val": best_val}, _f)

    # Final save (cũng cover case start_epoch>epochs → loop rỗng, model đã train đủ từ run trước).
    _save(model, cli.output_dir, tok_src)
    with open(os.path.join(cli.output_dir, "_train_state.json"), "w") as _f:
        json.dump({"epoch": cli.epochs, "best_val": best_val}, _f)

    n_tok = _copy_tokenizer(tok_src, cli.output_dir)
    print(f"\n✅ DONE. Saved model + {n_tok} tokenizer files → {cli.output_dir}")
    print(f"   Total time: {(time.time() - t0) / 60:.1f} min")


def _save(model, output_dir, tok_src):
    os.makedirs(output_dir, exist_ok=True)
    to_save = model.module if hasattr(model, "module") else model
    to_save.save_pretrained(output_dir)
    _copy_tokenizer(tok_src, output_dir)


if __name__ == "__main__":
    main()
