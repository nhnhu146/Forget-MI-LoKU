"""Gold-free CE-crossing checkpoint selector — PILOT trên Forget-MI.

MỤC TIÊU: so sánh 4 cách chọn checkpoint — ĐỀU KHÔNG dùng GOLD/retrained, KHÔNG dùng final test:
    S1 First Crossing     : epoch đầu tiên forget_ce >= nm_val_ce
    S2 Closest CE         : epoch argmin |forget_ce - nm_val_ce|
    S3 First Stable Cross : crossing giữ 2 epoch liên tiếp (chống nhiễu 1 epoch)
    S4 CE-match + Utility  : trong {|forget_ce - nm_val_ce| <= δ}, chọn (AUC+F1)/2 cao nhất
  (CE trên forget set D_f vs non-member validation D_nm_val). S1/S3 có thể 'no crossing'.

KHÔNG sửa training/loss của Forget-MI. Script này chạy OFFLINE, đọc 30 checkpoint E0–E29
đã lưu bởi Forget-MI ở chế độ non-dual (output_dir/epoch_<e>/model_state_dict.pth), rồi:
  1) Tách test_full (seed 42, patient-level + stratified) -> 25% D_nm_val / 75% D_t_final,
     lưu manifest + sanity checks (disjoint patient/id).
  2) Với TỪNG checkpoint: load ở eval()/no_grad(), tính forget_ce (D_f) + nm_val_ce (D_nm_val)
     bằng CÙNG evaluator (per_sample_ce) + utility (AUC/F1 trên D_nm_val), KHÔNG noise/aug.
  3) Áp 4 selector S1–S4 lên trajectory (thuần logic, gold-free).
  4) SAU KHI chọn xong mới eval MỖI epoch-được-chọn trên D_t_final: Df-AUC/F1, Dt-AUC/F1, MIA.

Output (mặc định ./checkpoint_selection/):
  split_manifest.json, nonmember_val_25_s42.csv, final_test_75_s42.csv,
  forgetmi_selector.csv, selected_checkpoints.json
"""
import os, sys, csv, json, argparse
import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import BertTokenizer                                 # noqa: E402
from joint_img_txt.model import ImageTextModel                          # noqa: E402
from training.forgetmi_partial import _load_config, build_dataset       # noqa: E402
from training.forgetmi_loku import per_sample_ce, perf_metrics, run_mia, _subsample_dataset  # noqa: E402
from training.adv_common import _grouped_stratified_holdout            # noqa: E402


class Cfg(dict):
    """Config hỗ trợ cả attribute (cfg.x), item (cfg['x']) lẫn cfg.get('x', default)."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


def _mean_ce(model, dataset, device, cfg, bs):
    model.eval()
    with torch.no_grad():
        ce = per_sample_ce(model, dataset, device, cfg, batch_size=bs)
    return float(np.asarray(ce, dtype=np.float64).mean())


def _load_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'model_state' in obj:
        return obj['model_state']
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    return obj


def _split_test_by_patient(test_ds, split_csv, seed, val_ratio):
    """Tách test_ds -> (nm_val_idx, tfinal_idx) theo bệnh nhân + phân tầng nhãn.
    Dùng thứ tự KHÓA của test_ds.all_img_txt_ids (đã lọc) làm index dataset.
    LƯU Ý: KHÓA của all_img_txt_ids CHÍNH LÀ dicom_id (ảnh) — model_utils.__getitem__ so khóa
    với cột 'dicom_id' của CSV. VALUE mới là txt/study id. → tra subject bằng d2s[key]."""
    keys = list(test_ds.all_img_txt_ids.keys())                 # = dicom_id (đúng thứ tự dataset)
    txtid_of = test_ds.all_img_txt_ids                          # dicom_id -> txt/study id
    # dicom_id -> subject_id: đọc bằng csv.reader Y HỆT data_split (khóa khớp ký tự tuyệt đối)
    d2s = {}
    with open(split_csv, 'r') as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if len(row) < 3:
                continue
            d2s[row[2]] = row[0]                                # dicom_id (row[2]=khóa) -> subject_id (row[0])
    subjects, labels, keep = [], [], []
    for k in keys:
        if k not in d2s:
            continue
        subjects.append(d2s[k]); labels.append(int(float(test_ds.all_img_labels[k][0]))); keep.append(k)
    assert keep, (f"0 mẫu test khớp CSV — kiểm tra cột dicom_id của {split_csv} "
                  f"(sample key: {keys[:3]}; sample csv dicom: {list(d2s)[:3]})")
    n_splits = max(2, round(1.0 / val_ratio))                   # 0.25 -> 4
    nm_rids, tf_rids = _grouped_stratified_holdout(keep, subjects, labels, n_splits=n_splits, seed=seed)
    nm_set, tf_set = set(nm_rids), set(tf_rids)
    pos = {k: i for i, k in enumerate(keys)}
    nm_idx = [pos[k] for k in keys if k in nm_set]
    tf_idx = [pos[k] for k in keys if k in tf_set]
    subj_of = {k: d2s[k] for k in keep}
    lab_of = {k: int(float(test_ds.all_img_labels[k][0])) for k in keep}
    return nm_idx, tf_idx, nm_rids, tf_rids, subj_of, lab_of, txtid_of


def _dist(rids, lab_of, subj_of):
    from collections import Counter
    return dict(n=len(rids), class_dist=dict(Counter(lab_of[r] for r in rids)),
                n_patients=len(set(subj_of[r] for r in rids)))


def _write_manifest_csv(path, rids, dicom_of, subj_of, lab_of):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['report_id', 'dicom_id', 'subject_id', 'label'])
        for r in rids:
            w.writerow([r, dicom_of[r], subj_of[r], lab_of[r]])


def make_nmval_tfinal(test_ds, split_csv, out_dir, seed=42, val_ratio=0.25):
    """Tách test_ds -> (nm_val_ds, tfinal_ds) theo bệnh nhân + phân tầng; lưu manifest + sanity."""
    nm_idx, tf_idx, nm_rids, tf_rids, subj_of, lab_of, dicom_of = _split_test_by_patient(
        test_ds, split_csv, seed, val_ratio)
    nm_subj = set(subj_of[r] for r in nm_rids); tf_subj = set(subj_of[r] for r in tf_rids)
    assert set(nm_rids).isdisjoint(set(tf_rids)), 'D_nm_val & D_t_final CHUNG report_id!'
    assert nm_subj.isdisjoint(tf_subj), 'MỘT bệnh nhân ở CẢ D_nm_val và D_t_final!'
    os.makedirs(out_dir, exist_ok=True)
    dnm, dtf = _dist(nm_rids, lab_of, subj_of), _dist(tf_rids, lab_of, subj_of)
    print(f"[split] test={len(nm_rids) + len(tf_rids)}  D_nm_val(25%)={dnm['n']}(pat {dnm['n_patients']})  "
          f"D_t_final(75%)={dtf['n']}(pat {dtf['n_patients']})  patient/id disjoint OK")
    _write_manifest_csv(os.path.join(out_dir, f'nonmember_val_25_s{seed}.csv'), nm_rids, dicom_of, subj_of, lab_of)
    _write_manifest_csv(os.path.join(out_dir, f'final_test_75_s{seed}.csv'), tf_rids, dicom_of, subj_of, lab_of)
    json.dump({'split_seed': seed, 'nonmember_val_ratio': val_ratio, 'D_nm_val': dnm, 'D_t_final': dtf,
               'patient_disjoint': True, 'id_disjoint': True},
              open(os.path.join(out_dir, 'split_manifest.json'), 'w'), indent=2)
    return Subset(test_ds, nm_idx), Subset(test_ds, tf_idx)


class OnlineCESelector:
    """Chọn checkpoint gold-free NGAY TRONG lúc train — KHÔNG cần lưu 30 checkpoint.
    Mỗi epoch: eval CE (D_f + D_nm_val) + utility, cập nhật 4 selector S1–S4, snapshot ỨNG VIÊN
    của từng cách (tối đa ~4 file). finalize() eval các epoch được chọn trên D_t_final.
    RNG-safe (lưu/khôi phục RNG + train-mode + requires_grad) → KHÔNG lệch training/loss."""

    def __init__(self, args, dataset, device, out_dir, split_seed=42, val_ratio=0.25,
                 label='forgetmi', nm_val_ds=None, tfinal_ds=None,
                 ce_fn=None, perf_fn=None, mia_fn=None, subsample_fn=None,
                 state_fn=None, load_fn=None):
        """Inject để dùng cho NHIỀU pipeline:
          - Forget-MI (partial): mặc định (per_sample_ce/perf_metrics/run_mia của module này,
            snapshot full state_dict), tự tách sel/tfinal từ dataset['test'].
          - P3/P6 (adv_common): truyền nm_val_ds=datasets['sel'], tfinal_ds=datasets['test_final'],
            ce_fn/perf_fn/mia_fn/subsample_fn = của adv_common, state_fn=trainable_state (LoRA)."""
        self.args, self.device, self.out = args, device, out_dir
        self.delta = float(getattr(args, 's4_delta', 0.15))
        self.bs = int(getattr(args, 'eval_batch_size', 16))
        self.split_seed = split_seed; self.label = label
        self.forget_ds = dataset['forget']; self.retain_ds = dataset['retain']
        self.ce_fn = ce_fn or per_sample_ce
        self.perf_fn = perf_fn or perf_metrics
        self.mia_fn = mia_fn or run_mia
        self.subsample_fn = subsample_fn or _subsample_dataset
        self._snap = state_fn or (lambda m: {k: v.detach().cpu().clone()
                                             for k, v in (m.module if hasattr(m, 'module') else m).state_dict().items()})
        self._load = load_fn or (lambda m, s: (m.module if hasattr(m, 'module') else m).load_state_dict(s, strict=False))
        os.makedirs(out_dir, exist_ok=True)
        if nm_val_ds is not None and tfinal_ds is not None:
            self.nm_val_ds, self.tfinal_ds = nm_val_ds, tfinal_ds
            print(f"🧭 selector dùng sel/test_final có sẵn (nm_val={len(nm_val_ds)}, tfinal={len(tfinal_ds)})")
        else:
            self.nm_val_ds, self.tfinal_ds = make_nmval_tfinal(
                dataset['test'], args.data_split_path, out_dir, split_seed, val_ratio)
        self.rows = []; self.traj_csv = os.path.join(out_dir, f'{label}_selector.csv')
        self.s1 = None; self.s3 = None; self._prev_state = None; self._prev_crossed = False
        self.best_s2 = None; self.best_s4 = None
        print(f"🧭 OnlineCESelector bật (label={label}, out={out_dir}, s4_delta={self.delta})")

    def _state(self, model):
        return self._snap(model)

    def _flush_csv(self):
        with open(self.traj_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['method', 'epoch', 'forget_ce', 'nm_val_ce',
                                              'ce_gap', 'crossed', 'val_AUC', 'val_F1'])
            w.writeheader(); w.writerows(self.rows)

    def step(self, epoch, model):
        import random as _random
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        py_rng = _random.getstate(); np_rng = np.random.get_state()
        was_training = model.training
        grad = [p.requires_grad for p in model.parameters()]
        try:
            model.eval()
            with torch.no_grad():
                f_ce = float(np.asarray(self.ce_fn(model, self.forget_ds, self.device, self.args, self.bs)).mean())
                n_ce = float(np.asarray(self.ce_fn(model, self.nm_val_ds, self.device, self.args, self.bs)).mean())
                vm = self.perf_fn(model, self.nm_val_ds, self.device, self.args, batch_size=self.bs)
            gap = f_ce - n_ce; crossed = bool(gap >= 0.0)
            util = (float(vm['AUC']) + float(vm['Macro_F1'])) / 2.0
            self.rows.append({'method': self.label, 'epoch': epoch, 'forget_ce': round(f_ce, 4),
                              'nm_val_ce': round(n_ce, 4), 'ce_gap': round(gap, 4), 'crossed': int(crossed),
                              'val_AUC': round(float(vm['AUC']), 4), 'val_F1': round(float(vm['Macro_F1']), 4)})
            self._flush_csv()
            print(f"    🧭 [selector E{epoch:02d}] fce {f_ce:.4f} nm_ce {n_ce:.4f} gap {gap:+.4f} "
                  f"cross {crossed} | valAUC {vm['AUC']:.3f} valF1 {vm['Macro_F1']:.3f}")
            _cur = {}
            def S():
                if 'v' not in _cur:
                    _cur['v'] = self._state(model)
                return _cur['v']
            if crossed and self.s1 is None:
                self.s1 = epoch; torch.save(S(), os.path.join(self.out, 'cand_S1.pt'))
            if crossed and self._prev_crossed and self.s3 is None and self._prev_state is not None:
                self.s3 = epoch - 1; torch.save(self._prev_state, os.path.join(self.out, 'cand_S3.pt'))
            if self.best_s2 is None or abs(gap) < self.best_s2[1]:
                self.best_s2 = (epoch, abs(gap)); torch.save(S(), os.path.join(self.out, 'cand_S2.pt'))
            if abs(gap) <= self.delta and (self.best_s4 is None or util > self.best_s4[1]):
                self.best_s4 = (epoch, util); torch.save(S(), os.path.join(self.out, 'cand_S4.pt'))
            self._prev_crossed = crossed
            self._prev_state = None if self.s3 is not None else S()   # giữ cho S3 epoch sau
        finally:
            if was_training:
                model.train()
            for p, g in zip(model.parameters(), grad):
                p.requires_grad = g
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            _random.setstate(py_rng); np.random.set_state(np_rng)
            torch.cuda.empty_cache()

    def _eval_tfinal(self, model, cpath):
        self._load(model, torch.load(cpath, map_location='cpu'))
        model.eval()
        with torch.no_grad():
            fm = self.perf_fn(model, self.forget_ds, self.device, self.args, batch_size=self.bs)
            tm = self.perf_fn(model, self.tfinal_ds, self.device, self.args, batch_size=self.bs)
            member = self.subsample_fn(self.retain_ds, int(getattr(self.args, 'eval_max_retain', 512)), self.split_seed)
            try:
                mia = self.mia_fn(model, member, self.tfinal_ds, self.forget_ds, self.device, self.args,
                                  batch_size=self.bs, seed=self.split_seed,
                                  paper_batch_size=int(getattr(self.args, 'mia_paper_batch_size', 32)))
            except Exception as ex:
                print('  MIA lỗi:', ex); mia = {'persample': float('nan'), 'paper': float('nan')}
        return {'Df_AUC': round(fm['AUC'], 4), 'Df_F1': round(fm['Macro_F1'], 4),
                'Dt_AUC': round(tm['AUC'], 4), 'Dt_F1': round(tm['Macro_F1'], 4),
                'MIA': round(float(mia['persample']), 4), 'MIA_paper': round(float(mia['paper']), 4)}

    def finalize(self, model):
        picks = {'S1_first_crossing': self.s1,
                 'S2_closest_ce': self.best_s2[0] if self.best_s2 else None,
                 'S3_first_stable_crossing': self.s3,
                 'S4_ce_match_utility': self.best_s4[0] if self.best_s4 else (self.best_s2[0] if self.best_s2 else None)}
        candfile = {'S1_first_crossing': 'cand_S1.pt', 'S2_closest_ce': 'cand_S2.pt',
                    'S3_first_stable_crossing': 'cand_S3.pt',
                    'S4_ce_match_utility': 'cand_S4.pt' if self.best_s4 else 'cand_S2.pt'}
        print('\n===== EPOCH CHỌN THEO 4 CÁCH (gold-free, inline) =====')
        for k, v in picks.items():
            print(f"  {k:26} -> {('E' + str(v)) if v is not None else 'NO CROSSING'}")
        final_state = self._state(model)                    # giữ E-cuối để khôi phục
        results, cache = {}, {}
        for k, ep in picks.items():
            if ep is None:
                results[k] = {'epoch': None, 'note': 'no_crossing'}; continue
            cf = os.path.join(self.out, candfile[k])
            if not os.path.exists(cf):
                results[k] = {'epoch': ep, 'note': 'no_snapshot'}; continue
            if candfile[k] not in cache:
                cache[candfile[k]] = self._eval_tfinal(model, cf)
            r = dict(cache[candfile[k]]); r['epoch'] = ep
            sr = next((x for x in self.rows if x['epoch'] == ep), {})
            r['forget_ce'] = sr.get('forget_ce'); r['nm_val_ce'] = sr.get('nm_val_ce')
            results[k] = r
        self._load(model, final_state)                       # khôi phục E-cuối
        json.dump({'selectors_gold_free': True, 's4_delta': self.delta, 'split_seed': self.split_seed,
                   'results': results}, open(os.path.join(self.out, 'selected_checkpoints.json'), 'w'), indent=2)
        print('\n' + '=' * 92)
        print(f"{'selector':28}{'epoch':>6}{'Df-AUC':>9}{'Df-F1':>8}{'Dt-AUC':>9}{'Dt-F1':>8}{'MIA':>8}")
        print('-' * 92)
        for k, ep in picks.items():
            r = results[k]
            if ep is None or 'Df_AUC' not in r:
                print(f"{k:28}{('E' + str(ep)) if ep is not None else '—':>6}{'(no ckpt)':>34}"); continue
            print(f"{k:28}{('E' + str(ep)):>6}{r['Df_AUC']:>9}{r['Df_F1']:>8}{r['Dt_AUC']:>9}"
                  f"{r['Dt_F1']:>8}{r['MIA']:>8}")
        print('=' * 92)
        print('Output:', self.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--override', default='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--checkpoints_dir', required=True,
                    help='Thư mục chứa epoch_0/..epoch_29/model_state_dict.pth (Forget-MI non-dual)')
    ap.add_argument('--out_dir', default='./checkpoint_selection')
    ap.add_argument('--max_epochs', type=int, default=30)
    ap.add_argument('--nm_val_ratio', type=float, default=0.25)
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--s4_delta', type=float, default=0.15,
                    help='S4: nguong |forget_ce - nm_val_ce| <= delta de coi la "khop unseen"')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = Cfg(_load_config(cli))
    cfg.do_train = False; cfg.do_eval = True                    # CenterCrop -> KHÔNG random aug
    bs = int(cfg.get('eval_batch_size', 16))
    os.makedirs(cli.out_dir, exist_ok=True)

    print('=' * 70); print('CE-CROSSING SELECTOR (gold-free) — PILOT Forget-MI'); print('=' * 70)
    tok = BertTokenizer.from_pretrained(cfg.bert_pretrained_dir)
    model = ImageTextModel.from_pretrained(cfg.base_model_path).to(device)
    dataset, _ = build_dataset(cfg, tok)
    test_ds, forget_ds, retain_ds = dataset['test'], dataset['forget'], dataset['retain']

    # ---- 1) split test -> D_nm_val (25%) / D_t_final (75%) theo bệnh nhân + phân tầng ----
    nm_idx, tf_idx, nm_rids, tf_rids, subj_of, lab_of, dicom_of = _split_test_by_patient(
        test_ds, cfg.data_split_path, cli.split_seed, cli.nm_val_ratio)
    nm_val_ds = Subset(test_ds, nm_idx); tfinal_ds = Subset(test_ds, tf_idx)

    # ---- 9) sanity checks bắt buộc ----
    nm_subj = set(subj_of[r] for r in nm_rids); tf_subj = set(subj_of[r] for r in tf_rids)
    assert set(nm_rids).isdisjoint(set(tf_rids)), 'D_nm_val & D_t_final CHUNG report_id!'
    assert nm_subj.isdisjoint(tf_subj), 'MỘT bệnh nhân xuất hiện ở CẢ D_nm_val và D_t_final!'
    dnm, dtf = _dist(nm_rids, lab_of, subj_of), _dist(tf_rids, lab_of, subj_of)
    print(f"[split] test_full={len(nm_rids) + len(tf_rids)}  D_nm_val(25%)={dnm['n']}  D_t_final(75%)={dtf['n']}")
    print(f"[split] D_nm_val : class={dnm['class_dist']}  patients={dnm['n_patients']}")
    print(f"[split] D_t_final: class={dtf['class_dist']}  patients={dtf['n_patients']}")
    print(f"[split] patient disjoint OK · id disjoint OK · (test rows = og-unseen theo split gốc)")
    _write_manifest_csv(os.path.join(cli.out_dir, f'nonmember_val_25_s{cli.split_seed}.csv'),
                        nm_rids, dicom_of, subj_of, lab_of)
    _write_manifest_csv(os.path.join(cli.out_dir, f'final_test_75_s{cli.split_seed}.csv'),
                        tf_rids, dicom_of, subj_of, lab_of)
    json.dump({'split_seed': cli.split_seed, 'nonmember_val_ratio': cli.nm_val_ratio,
               'n_test_full': len(nm_rids) + len(tf_rids), 'D_nm_val': dnm, 'D_t_final': dtf,
               'patient_disjoint': True, 'id_disjoint': True},
              open(os.path.join(cli.out_dir, 'split_manifest.json'), 'w'), indent=2)

    # ---- 2) eval mỗi checkpoint E0..E(max-1): forget_ce (D_f) + nm_val_ce + utility (D_nm_val) ----
    traj_csv = os.path.join(cli.out_dir, 'forgetmi_selector.csv')
    rows = []
    print('\n[trajectory] epoch  forget_ce  nm_val_ce    gap    crossed  val_AUC  val_F1')
    for e in range(cli.max_epochs):
        ckpt = os.path.join(cli.checkpoints_dir, f'epoch_{e}', 'model_state_dict.pth')
        if not os.path.exists(ckpt):
            print(f"  E{e:02d}  (thiếu {ckpt}) — bỏ qua"); continue
        model.load_state_dict(_load_state(ckpt), strict=False)
        f_ce = _mean_ce(model, forget_ds, device, cfg, bs)
        n_ce = _mean_ce(model, nm_val_ds, device, cfg, bs)
        model.eval()
        with torch.no_grad():
            vm = perf_metrics(model, nm_val_ds, device, cfg, batch_size=bs)   # utility gold-free
        gap = f_ce - n_ce; crossed = bool(gap >= 0.0)
        rows.append({'method': 'forgetmi', 'epoch': e, 'forget_ce': round(f_ce, 4),
                     'nm_val_ce': round(n_ce, 4), 'ce_gap': round(gap, 4), 'crossed': int(crossed),
                     'val_AUC': round(float(vm['AUC']), 4), 'val_F1': round(float(vm['Macro_F1']), 4)})
        print(f"  E{e:02d}   {f_ce:7.4f}   {n_ce:7.4f}  {gap:+7.4f}    {str(crossed):5}  "
              f"{vm['AUC']:6.3f}  {vm['Macro_F1']:6.3f}")
    with open(traj_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'epoch', 'forget_ce', 'nm_val_ce', 'ce_gap',
                                          'crossed', 'val_AUC', 'val_F1'])
        w.writeheader(); w.writerows(rows)
    if not rows:
        print('KHÔNG có checkpoint nào để chấm. Dừng.'); return

    # ---- 3) áp 4 selector (đều gold-free, không dùng D_t_final) ----
    eps = [r['epoch'] for r in rows]
    gap = {r['epoch']: r['ce_gap'] for r in rows}
    util = {r['epoch']: (r['val_AUC'] + r['val_F1']) / 2.0 for r in rows}

    def s1_first_crossing():
        for e in eps:
            if gap[e] >= 0:
                return e
        return None

    def s2_closest_ce():
        return min(eps, key=lambda e: abs(gap[e]))

    def s3_first_stable_crossing():
        for i in range(len(eps) - 1):
            if gap[eps[i]] >= 0 and gap[eps[i + 1]] >= 0:
                return eps[i]
        return None

    def s4_ce_match_utility(delta):
        match = [e for e in eps if abs(gap[e]) <= delta]
        if not match:                       # rỗng -> lấy epoch CE gần nhất (khớp unseen nhất)
            match = [s2_closest_ce()]
        return max(match, key=lambda e: util[e]), sorted(match)

    s4_e, s4_match = s4_ce_match_utility(cli.s4_delta)
    picks = {'S1_first_crossing': s1_first_crossing(),
             'S2_closest_ce': s2_closest_ce(),
             'S3_first_stable_crossing': s3_first_stable_crossing(),
             'S4_ce_match_utility': s4_e}
    print('\n===== EPOCH CHỌN THEO 4 CÁCH (gold-free) =====')
    for k, v in picks.items():
        print(f"  {k:26} -> {('E' + str(v)) if v is not None else 'NO CROSSING'}")
    print(f"  (S4: E_match |gap|<= {cli.s4_delta} = {['E' + str(e) for e in s4_match]})")

    # ---- 4) SAU KHI chọn mới eval mỗi epoch-được-chọn trên D_t_final (cache theo epoch) ----
    def _final_eval(e):
        model.load_state_dict(_load_state(os.path.join(cli.checkpoints_dir, f'epoch_{e}',
                                                       'model_state_dict.pth')), strict=False)
        model.eval()
        with torch.no_grad():
            fm = perf_metrics(model, forget_ds, device, cfg, batch_size=bs)
            tm = perf_metrics(model, tfinal_ds, device, cfg, batch_size=bs)
            member = _subsample_dataset(retain_ds, int(cfg.get('eval_max_retain', 512)), cli.split_seed)
            try:
                mia = run_mia(model, member, tfinal_ds, forget_ds, device, cfg, batch_size=bs,
                              seed=cli.split_seed, paper_batch_size=int(cfg.get('mia_paper_batch_size', 32)))
            except Exception as ex:
                print('  MIA lỗi:', ex); mia = {'persample': float('nan'), 'paper': float('nan')}
        sr = next(r for r in rows if r['epoch'] == e)
        return {'epoch': e, 'forget_ce': sr['forget_ce'], 'nm_val_ce': sr['nm_val_ce'],
                'val_AUC': sr['val_AUC'], 'val_F1': sr['val_F1'],
                'Df_AUC': round(fm['AUC'], 4), 'Df_F1': round(fm['Macro_F1'], 4),
                'Dt_AUC': round(tm['AUC'], 4), 'Dt_F1': round(tm['Macro_F1'], 4),
                'MIA': round(float(mia['persample']), 4), 'MIA_paper': round(float(mia['paper']), 4)}

    print('\n[final] eval các epoch được chọn trên D_t_final (75%)...')
    cache = {}
    for e in picks.values():
        if e is not None and e not in cache:
            cache[e] = _final_eval(e)

    out = {'selectors_gold_free': True, 'max_epochs': cli.max_epochs, 'split_seed': cli.split_seed,
           'nonmember_val_ratio': cli.nm_val_ratio, 's4_delta': cli.s4_delta, 's4_match_epochs': s4_match,
           'results': {}}
    for k, e in picks.items():
        out['results'][k] = {'epoch': None, 'note': 'no_crossing'} if e is None else dict(cache[e])
    json.dump(out, open(os.path.join(cli.out_dir, 'selected_checkpoints.json'), 'w'), indent=2)

    # bảng so sánh
    print('\n' + '=' * 96)
    print(f"{'selector':28}{'epoch':>6}{'Df-AUC':>9}{'Df-F1':>8}{'Dt-AUC':>9}{'Dt-F1':>8}{'MIA':>8}")
    print('-' * 96)
    for k, e in picks.items():
        if e is None:
            print(f"{k:28}{'—':>6}{'(no crossing)':>25}"); continue
        r = cache[e]
        print(f"{k:28}{('E' + str(e)):>6}{r['Df_AUC']:>9}{r['Df_F1']:>8}{r['Dt_AUC']:>9}"
              f"{r['Dt_F1']:>8}{r['MIA']:>8}")
    print('=' * 96)
    uniq = sorted(set(e for e in picks.values() if e is not None))
    print(f"4 cách -> {len(uniq)} epoch khác nhau: {['E' + str(e) for e in uniq]}  "
          f"({'ĐỒNG THUẬN cao' if len(uniq) <= 2 else 'phân tán'})")
    print('Output:', cli.out_dir,
          '(split_manifest.json, *_25/75_s42.csv, forgetmi_selector.csv, selected_checkpoints.json)')


if __name__ == '__main__':
    main()
