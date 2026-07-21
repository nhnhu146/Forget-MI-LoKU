"""Generate thesis-ready figures for the fair Forget-MI/LoKU protocol."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_LABELS = {
    'baseline_partial': 'Forget-MI',
    'forgetmi': 'Forget-MI',
    'loku': 'LoKU',
}
CHECKPOINT_LABELS = {'last': 'Last', 'val_best': 'Val-best'}


def _save_show(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / f'{name}.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def _annotate_bars(ax, fmt='{:.3f}'):
    for container in ax.containers:
        try:
            ax.bar_label(container, fmt=fmt, padding=3, fontsize=8)
        except (AttributeError, ValueError):
            pass


def _read_run_rows(results_csv, run_id, seed):
    frame = pd.read_csv(results_csv)
    rows = frame[(frame['id'].astype(str) == str(run_id)) & (frame['seed'] == seed)].tail(2).copy()
    if set(rows.get('checkpoint', [])) != {'last', 'val_best'}:
        raise ValueError(f'Expected last and val_best rows for {run_id}, found {len(rows)} rows.')
    rows['method_label'] = rows['method'].map(METHOD_LABELS).fillna(rows['method'])
    rows['checkpoint_label'] = rows['checkpoint'].map(CHECKPOINT_LABELS).fillna(rows['checkpoint'])
    return rows


def _selection_figure(history, rows, out_dir, method_label):
    history = history.sort_values('epoch')
    best_row = rows[rows['checkpoint'] == 'val_best'].iloc[-1]
    last_row = rows[rows['checkpoint'] == 'last'].iloc[-1]
    best_epoch = int(best_row['selected_epoch'])
    last_epoch = int(last_row['selected_epoch'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(history['epoch'], history['val_ce'], color='#1565C0', marker='o', markersize=3, linewidth=1.8)
    best_point = history[history['epoch'] == best_epoch]
    last_point = history[history['epoch'] == last_epoch]
    if not best_point.empty:
        ax.scatter(best_point['epoch'], best_point['val_ce'], s=90, color='#2E7D32', zorder=5,
                   label=f'Val-best (epoch {best_epoch})')
    if not last_point.empty:
        ax.scatter(last_point['epoch'], last_point['val_ce'], s=90, color='#C62828', marker='X', zorder=5,
                   label=f'Last (epoch {last_epoch})')
    ax.set(title=f'{method_label}: checkpoint selection', xlabel='Epoch', ylabel='Validation CE')
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(history['epoch'], history['cumulative_train_hours'], label='Train', color='#6A1B9A', linewidth=2)
    ax.plot(history['epoch'], history['cumulative_monitor_hours'], label='Validation monitor',
            color='#EF6C00', linewidth=2)
    ax.set(title='Cumulative measured time', xlabel='Epoch', ylabel='Hours')
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_show(fig, out_dir, '01_checkpoint_selection')


def _loss_figure(history, out_dir, method_label):
    history = history.sort_values('epoch')
    component_candidates = [
        ('ukr_loss', 'UKR'), ('uu_loss', 'UU'), ('md_loss', 'MD'), ('mkr_loss', 'MKR'),
        ('cls_retain_loss', 'CLS retain'), ('cls_forget_loss', 'CLS forget'),
        ('distill_retain_loss', 'Distill retain'), ('distill_forget_loss', 'Distill forget'),
        ('ihl_forget_loss', 'IHL forget'),
    ]
    available = [(column, label) for column, label in component_candidates if column in history.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].plot(history['epoch'], history['total_loss'], color='#263238', linewidth=2.2)
    axes[0].axhline(0, color='#90A4AE', linewidth=0.8)
    axes[0].set(title=f'{method_label}: total training objective', xlabel='Epoch', ylabel='Mean loss')

    palette = sns.color_palette('tab10', n_colors=max(len(available), 1))
    for color, (column, label) in zip(palette, available):
        values = pd.to_numeric(history[column], errors='coerce')
        if values.notna().any() and not np.allclose(values.fillna(0), 0):
            axes[1].plot(history['epoch'], values, label=label, linewidth=1.7, color=color)
    axes[1].axhline(0, color='#90A4AE', linewidth=0.8)
    axes[1].set(title='Loss components', xlabel='Epoch', ylabel='Mean loss')
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    _save_show(fig, out_dir, '02_training_losses')


def _checkpoint_metrics_figure(rows, out_dir, method_label):
    metric_map = {
        'MIA_paper': 'MIA (paper)', 'Df_AUC': 'Forget AUC', 'Df_F1': 'Forget F1',
        'Dt_AUC': 'Test AUC', 'Dt_F1': 'Test F1', 'dist_vs_re': '1-CosSim',
    }
    gold_available = (rows['gold_retrained_available'].astype(str).str.lower()
                      .isin({'true', '1', 'yes'}).any()
                      if 'gold_retrained_available' in rows.columns else True)
    if not gold_available:
        metric_map.pop('dist_vs_re', None)
    metrics = [column for column in metric_map if column in rows.columns]
    long = rows.melt(id_vars=['checkpoint_label'], value_vars=metrics,
                     var_name='metric', value_name='value')
    long['metric'] = long['metric'].map(metric_map)
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    sns.barplot(data=long, x='metric', y='value', hue='checkpoint_label',
                hue_order=['Last', 'Val-best'], palette=['#C62828', '#2E7D32'], ax=ax)
    ax.set(title=f'{method_label}: final metrics by checkpoint', xlabel='', ylabel='Metric value')
    ax.set_ylim(0, max(1.0, float(long['value'].max()) * 1.15))
    ax.tick_params(axis='x', rotation=20)
    ax.legend(title='Checkpoint', frameon=False)
    _annotate_bars(ax)
    fig.tight_layout()
    _save_show(fig, out_dir, '03_last_vs_val_best_metrics')


def _ce_figure(rows, out_dir, method_label):
    columns = [column for column in ['forget_ce', 'test_ce'] if column in rows.columns]
    long = rows.melt(id_vars=['checkpoint_label'], value_vars=columns,
                     var_name='split', value_name='cross_entropy')
    long['split'] = long['split'].map({'forget_ce': 'Forget set', 'test_ce': 'Test set'})
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=long, x='split', y='cross_entropy', hue='checkpoint_label',
                hue_order=['Last', 'Val-best'], palette=['#C62828', '#2E7D32'], ax=ax)
    ax.set(title=f'{method_label}: forget/test cross-entropy', xlabel='', ylabel='Mean cross-entropy')
    ax.legend(title='Checkpoint', frameon=False)
    _annotate_bars(ax)
    fig.tight_layout()
    _save_show(fig, out_dir, '04_forget_test_cross_entropy')


def _efficiency_figure(rows, out_dir, method_label):
    row = rows[rows['checkpoint'] == 'val_best'].iloc[-1]
    time_parts = [
        ('fisher_hours', 'Fisher', '#5E35B1'),
        ('adapter_init_hours', 'Adapter init', '#00897B'),
        ('train_hours', 'Train', '#1E88E5'),
        ('monitor_hours', 'Validation', '#FB8C00'),
        ('ckpt_hours', 'Checkpoint I/O', '#6D4C41'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    left = 0.0
    for column, label, color in time_parts:
        value = float(row.get(column, 0) or 0)
        if value > 0:
            axes[0].barh([method_label], [value], left=left, label=label, color=color)
            if value >= 0.005:
                axes[0].text(left + value / 2, 0, f'{value:.3f}h', ha='center', va='center',
                             color='white', fontsize=8, fontweight='bold')
            left += value
    axes[0].set(title='Fair method-time breakdown', xlabel='Hours', ylabel='')
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    trainable = float(row['trainable_params'])
    total = float(row['total_params'])
    frozen = max(total - trainable, 0)
    axes[1].barh([method_label], [trainable], color='#D81B60', label='Trainable')
    axes[1].barh([method_label], [frozen], left=[trainable], color='#CFD8DC', label='Frozen')
    axes[1].set(title=f'Trainable parameters: {100 * trainable / total:.3f}%', xlabel='Parameters', ylabel='')
    axes[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save_show(fig, out_dir, '05_time_and_parameters')


def _load_cross_method_rows(paths, forget_token, seed):
    frames = []
    for path in paths:
        if path and Path(path).exists():
            frame = pd.read_csv(path)
            mask = frame['forget_pct'].astype(str).str.contains(forget_token, regex=False) & (frame['seed'] == seed)
            frames.append(frame[mask])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty or 'checkpoint' not in combined.columns:
        return pd.DataFrame()
    combined = combined.sort_values('timestamp').drop_duplicates(['method', 'checkpoint'], keep='last')
    combined['method_label'] = combined['method'].map(METHOD_LABELS).fillna(combined['method'])
    combined['checkpoint_label'] = combined['checkpoint'].map(CHECKPOINT_LABELS).fillna(combined['checkpoint'])
    return combined


def _cross_method_figures(combined, out_dir):
    if combined['method_label'].nunique() < 2:
        print('Cross-method figures skipped: run both Forget-MI and LoKU first, then rerun this cell.')
        return

    combined = combined.copy()
    combined['variant'] = combined['method_label'] + ' / ' + combined['checkpoint_label']
    metric_map = {
        'MIA_paper': 'MIA (paper)', 'Df_AUC': 'Forget AUC', 'Df_F1': 'Forget F1',
        'Dt_AUC': 'Test AUC', 'Dt_F1': 'Test F1',
    }
    long = combined.melt(id_vars=['variant'], value_vars=list(metric_map),
                         var_name='metric', value_name='value')
    long['metric'] = long['metric'].map(metric_map)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.barplot(data=long, x='metric', y='value', hue='variant', palette='colorblind', ax=ax)
    ax.set(title='Forget-MI vs LoKU: final metrics', xlabel='', ylabel='Metric value', ylim=(0, 1.05))
    ax.legend(title='', frameon=False, fontsize=8, ncol=2)
    _annotate_bars(ax)
    fig.tight_layout()
    _save_show(fig, out_dir, '06_cross_method_metrics')

    method_rows = combined[combined['checkpoint'] == 'val_best'].drop_duplicates('method')
    speed_note = ''
    if {'Forget-MI', 'LoKU'} <= set(method_rows['method_label']):
        fmi = method_rows[method_rows['method_label'] == 'Forget-MI'].iloc[-1]
        loku = method_rows[method_rows['method_label'] == 'LoKU'].iloc[-1]
        if float(loku['unlearn_total_hours']) > 0 and float(loku['trainable_params']) > 0:
            speedup = float(fmi['unlearn_total_hours']) / float(loku['unlearn_total_hours'])
            param_reduction = float(fmi['trainable_params']) / float(loku['trainable_params'])
            speed_note = f' | LoKU: {speedup:.2f}x faster, {param_reduction:.2f}x fewer trainable params'
            print(f'Cross-method efficiency: {speedup:.2f}x time speedup; '
                  f'{param_reduction:.2f}x trainable-parameter reduction.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=method_rows, x='method_label', y='unlearn_total_hours', hue='method_label',
                palette='colorblind', legend=False, ax=axes[0])
    axes[0].set(title='Total unlearning method time', xlabel='', ylabel='Hours')
    _annotate_bars(axes[0])
    sns.barplot(data=method_rows, x='method_label', y='trainable_params', hue='method_label',
                palette='colorblind', legend=False, ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set(title='Trainable parameters (log scale)', xlabel='', ylabel='Parameters')
    _annotate_bars(axes[1], fmt='%.2e')
    fig.suptitle(f'Cross-method efficiency{speed_note}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_show(fig, out_dir, '07_cross_method_efficiency')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=combined, x='MIA_paper', y='Dt_AUC', hue='method_label',
                    style='checkpoint_label', s=150, palette='colorblind', ax=axes[0])
    axes[0].set(title='Privacy-utility trade-off', xlabel='MIA (lower is better)',
                ylabel='Test AUC (higher is better)')
    sns.scatterplot(data=combined, x='Df_AUC', y='Dt_AUC', hue='method_label',
                    style='checkpoint_label', s=150, palette='colorblind', ax=axes[1])
    axes[1].set(title='Forgetting-utility trade-off', xlabel='Forget AUC',
                ylabel='Test AUC (higher is better)')
    for ax in axes:
        ax.legend(title='', frameon=False, fontsize=8)
    fig.tight_layout()
    _save_show(fig, out_dir, '08_tradeoff_scatter')


def _multiseed_figure(paths, forget_token, out_dir):
    frames = []
    for path in paths:
        if path and Path(path).exists():
            frame = pd.read_csv(path)
            mask = frame['forget_pct'].astype(str).str.contains(forget_token, regex=False)
            frames.append(frame[mask])
    if not frames:
        return
    all_rows = pd.concat(frames, ignore_index=True)
    if 'checkpoint' not in all_rows.columns or all_rows['seed'].nunique() < 2:
        print('Multi-seed figure skipped: at least two completed seeds are required.')
        return
    all_rows = all_rows.sort_values('timestamp').drop_duplicates(
        ['method', 'checkpoint', 'seed'], keep='last'
    )
    all_rows['method_label'] = all_rows['method'].map(METHOD_LABELS).fillna(all_rows['method'])
    all_rows['checkpoint_label'] = all_rows['checkpoint'].map(CHECKPOINT_LABELS).fillna(all_rows['checkpoint'])
    all_rows['variant'] = all_rows['method_label'] + ' / ' + all_rows['checkpoint_label']

    metrics = [
        ('MIA_paper', 'MIA (paper)'), ('Df_AUC', 'Forget AUC'),
        ('Dt_AUC', 'Test AUC'), ('unlearn_total_hours', 'Method time (h)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (column, title) in zip(axes.flat, metrics):
        sns.barplot(data=all_rows, x='variant', y=column, hue='variant',
                    palette='colorblind', errorbar='sd', capsize=0.12, legend=False, ax=ax)
        sns.stripplot(data=all_rows, x='variant', y=column, color='#212121',
                      size=4, jitter=0.08, alpha=0.65, ax=ax)
        ax.set(title=f'{title}: mean +/- SD', xlabel='', ylabel=title)
        ax.tick_params(axis='x', rotation=18)
    fig.suptitle(f'Multi-seed robustness (n={all_rows.seed.nunique()} seeds)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_show(fig, out_dir, '09_multiseed_robustness')


def generate_thesis_figures(results_csv, history_csv, run_id, seed, fig_dir,
                            comparison_csvs=()):
    """Display figures in the notebook and save each as 300-DPI PNG and vector PDF."""
    sns.set_theme(style='whitegrid', context='notebook')
    plt.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    rows = _read_run_rows(results_csv, run_id, seed)
    history = pd.read_csv(history_csv)
    method_label = rows['method_label'].iloc[0]
    forget_token = str(rows['forget_pct'].iloc[0])

    _selection_figure(history, rows, fig_dir, method_label)
    _loss_figure(history, fig_dir, method_label)
    _checkpoint_metrics_figure(rows, fig_dir, method_label)
    _ce_figure(rows, fig_dir, method_label)
    _efficiency_figure(rows, fig_dir, method_label)

    all_paths = list(dict.fromkeys(list(comparison_csvs) + [results_csv]))
    combined = _load_cross_method_rows(all_paths, forget_token, seed)
    _cross_method_figures(combined, fig_dir)
    _multiseed_figure(all_paths, forget_token, fig_dir)

    out_dir = Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_columns = [
        'method_label', 'checkpoint_label', 'selected_epoch', 'selection_value',
        'MIA_paper', 'Df_AUC', 'Df_F1', 'Dt_AUC', 'Dt_F1', 'forget_ce', 'test_ce',
        'unlearn_core_hours', 'selection_hours', 'unlearn_total_hours',
        'trainable_params', 'trainable_ratio', 'gpu_peak_GB', 'gpu_name',
    ]
    summary = combined if not combined.empty else rows
    summary = summary[[column for column in summary_columns if column in summary.columns]]
    summary.to_csv(out_dir / 'thesis_summary.csv', index=False)
    with open(out_dir / 'thesis_summary.tex', 'w', encoding='utf-8') as handle:
        handle.write(summary.to_latex(index=False, float_format=lambda value: f'{value:.4f}'))

    generated = sorted(path.name for path in out_dir.iterdir())
    print(f'Generated {len(generated)} thesis artifacts in {out_dir}:')
    for name in generated:
        print(' -', name)
    return rows, history, combined
