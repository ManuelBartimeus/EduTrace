"""Regenerate every manuscript figure from the actual locked run.

Palette: validated categorical slots (blue/orange/aqua/yellow) from the dataviz
reference instance. One axis per chart, legend whenever two or more series are
shown, recessive grid and axes, direct labels where they replace a lookup.
Contrast WARN on the aqua/yellow slots is relieved by visible labels + legend.
"""
import json, os, random, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score, confusion_matrix)
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from EduTrace_Revised_Pipeline import (load_and_prepare, run_models, SEEDS, SPLIT_YEAR, ATT_RULE_CUT,
                      set_all_seeds)
import shaptosms as S

R = json.load(open('results/results.json'))
OUT = 'figs'
os.makedirs(OUT, exist_ok=True)

C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100',
     'ink': '#0b0b0b', 'ink2': '#52514e', 'muted': '#8a8985', 'grid': '#e2e1dd',
     'surface': '#ffffff'}

plt.rcParams.update({
    'figure.facecolor': C['surface'], 'axes.facecolor': C['surface'],
    'savefig.facecolor': C['surface'], 'font.size': 9,
    'axes.labelcolor': C['ink2'], 'axes.edgecolor': C['grid'],
    'xtick.color': C['ink2'], 'ytick.color': C['ink2'],
    'text.color': C['ink'], 'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'lines.linewidth': 2.0, 'figure.dpi': 200,
})


def tidy(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


print('rebuilding seed-42 artefacts ...')
D = load_and_prepare('data/real_student_data_CLEANED_school.csv',
                     'data/synth_ctgan_s42.csv', use_lag=False)
M = run_models(D, 'figs', with_tabtransformer=False)
yte, Xte, fn = D['yte'].astype(int), D['Xte'], D['fnames']
p_eng, thr_eng = M['proba']['xgb_engineered'][42]
p_def, thr_def = M['proba']['xgb_default'][42]
p_dt, thr_dt = M['proba']['decision_tree'][42]
ai = D['feats_cont'].index('attendance_rate')
sc = D['scaler']
cut = (ATT_RULE_CUT - sc.data_min_[ai]) / (sc.data_max_[ai] - sc.data_min_[ai])
att = D['Xcont_te'][:, ai]
p_rank = 1.0 - att
bin_rule = (att < cut).astype(int)

# ------------------------------------------------------------------ Figure 4
series = [('Proposed XGBoost', p_eng, C['blue']),
          ('Default XGBoost', p_def, C['orange']),
          ('Decision Tree', p_dt, C['aqua']),
          ('Attendance ranker (continuous)', p_rank, C['yellow'])]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
for name, p, col in series:
    fpr, tpr, _ = roc_curve(yte, p)
    axes[0].plot(fpr, tpr, color=col, label=f'{name} (AUC {roc_auc_score(yte, p):.3f})')
    pr, rc, _ = precision_recall_curve(yte, p)
    axes[1].plot(rc, pr, color=col, label=f'{name} (AP {average_precision_score(yte, p):.3f})')
axes[0].plot([0, 1], [0, 1], color=C['muted'], lw=1, ls=(0, (4, 3)), label='Chance')
tn, fp, fnn, tp = confusion_matrix(yte, bin_rule, labels=[0, 1]).ravel()
axes[0].scatter([fp / (fp + tn)], [tp / (tp + fnn)], s=64, zorder=5, color=C['ink'],
                marker='D', edgecolor=C['surface'], linewidth=1.5,
                label='Attendance rule (binary) — operating point')
axes[1].axhline(yte.mean(), color=C['muted'], lw=1, ls=(0, (4, 3)),
                label=f'Prevalence floor ({yte.mean():.3f})')
axes[0].set_xlabel('False-positive rate'); axes[0].set_ylabel('True-positive rate')
axes[0].set_title('ROC'); axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision–Recall')
for a in axes:
    a.set_xlim(-0.02, 1.02); tidy(a)
axes[0].set_ylim(-0.02, 1.02); axes[1].set_ylim(0, max(0.35, yte.mean() * 6))
axes[0].legend(loc='lower right', fontsize=7.0)
axes[1].legend(loc='upper right', fontsize=7.0)
fig.tight_layout(); fig.savefig(f'{OUT}/figure4.png', bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ Figure 5
cv = R['cv']
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for k, (name, col) in enumerate([('proposed', C['blue']), ('default', C['orange'])]):
    v = cv[name]['per_fold']
    x = np.full(len(v), k) + np.linspace(-0.09, 0.09, len(v))
    ax.scatter(x, v, s=46, color=col, zorder=3, edgecolor=C['surface'], linewidth=1.5,
               label=f"{'Proposed' if name=='proposed' else 'Default'} XGBoost")
    ax.scatter([k], [cv[name]['mean']], marker='^', s=140, color=col, zorder=4,
               edgecolor=C['surface'], linewidth=1.5)
    ax.annotate(f"mean {cv[name]['mean']:.3f}\nSD {cv[name]['sd']:.3f}",
                (k + 0.16, cv[name]['mean']), fontsize=8, color=C['ink2'], va='center')
ax.axhline(cv['pool_prevalence'], color=C['muted'], lw=1, ls=(0, (4, 3)))
ax.annotate(f"prevalence floor {cv['pool_prevalence']:.3f}", (-0.42, cv['pool_prevalence']),
            fontsize=7.5, color=C['muted'], va='bottom')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Proposed XGBoost', 'Default XGBoost'])
ax.set_xlim(-0.45, 1.65); ax.set_ylabel('AUC-PR (per fold)')
ax.set_title('Five-fold cross-validation stability (train + validation pool)')
ax.legend(loc='upper center', ncol=2, fontsize=8); tidy(ax)
fig.tight_layout(); fig.savefig(f'{OUT}/figure5.png', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------- SHAP (Fig 6, S1-S3)
expl = shap.TreeExplainer(M['ref']['model_eng'])
sv_eng = expl.shap_values(Xte)
if isinstance(sv_eng, list): sv_eng = sv_eng[1]
sv_def = shap.TreeExplainer(M['ref']['model_def']).shap_values(Xte)
if isinstance(sv_def, list): sv_def = sv_def[1]
ma_e, ma_d = np.abs(sv_eng).mean(0), np.abs(sv_def).mean(0)
order = np.argsort(ma_e)[::-1]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
y = np.arange(len(order))[::-1]
h = 0.38
ax.barh(y + h / 2, ma_e[order], height=h, color=C['blue'], label='Proposed XGBoost')
ax.barh(y - h / 2, ma_d[order], height=h, color=C['orange'], label='Default XGBoost')
for yi, v in zip(y + h / 2, ma_e[order]):
    if v > 0.01:
        ax.annotate(f'{v:.3f}', (v, yi), xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=7.5, color=C['ink2'])
ax.set_yticks(y); ax.set_yticklabels([fn[i] for i in order], fontsize=8)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('SHAP feature importance — proposed vs default XGBoost')
ax.legend(loc='lower right', fontsize=8); ax.grid(axis='y', visible=False); tidy(ax)
fig.tight_layout(); fig.savefig(f'{OUT}/figure6.png', bbox_inches='tight'); plt.close(fig)

# Figure S1 — signed beeswarm for the top features (direction, not just magnitude)
top = order[:6]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
rng = np.random.default_rng(42)
for row, fi in enumerate(top):
    yv = len(top) - 1 - row + rng.normal(0, 0.055, len(yte))
    vals = Xte[:, fi].astype(float)
    rngv = np.ptp(vals)
    norm = (vals - vals.min()) / (rngv if rngv > 0 else 1)
    ax.scatter(sv_eng[:, fi], yv, c=norm, cmap='coolwarm', s=11, alpha=0.75,
               linewidths=0, rasterized=True)
ax.axvline(0, color=C['muted'], lw=1)
ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels([fn[i] for i in top], fontsize=8)
ax.set_xlabel('SHAP value  (negative = lowers predicted risk, positive = raises it)')
ax.set_title('Signed SHAP attributions, proposed XGBoost (full held-out test set)')
sm = plt.cm.ScalarMappable(cmap='coolwarm'); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
cb.set_ticks([0, 1]); cb.set_ticklabels(['low', 'high']); cb.set_label('Feature value', fontsize=8)
cb.outline.set_visible(False)
ax.grid(axis='y', visible=False); tidy(ax)
fig.tight_layout(); fig.savefig(f'{OUT}/figureS1.png', bbox_inches='tight'); plt.close(fig)

# Figures S2 / S3 — a real false negative and a real false positive at seed 42
pred = (p_eng >= thr_eng).astype(int)
fn_idx = np.where((yte == 1) & (pred == 0))[0]
fp_idx = np.where((yte == 0) & (pred == 1))[0]
CASES = []
if len(fn_idx): CASES.append(('S2', int(fn_idx[0]), 'false negative (missed dropout)', 1))
if len(fp_idx): CASES.append(('S3', int(fp_idx[int(np.argmax(p_eng[fp_idx]))]),
                              'false positive (incorrectly flagged)', 0))
LOCAL = {}
for tag, idx, label, truth in CASES:
    sid = str(D['te'].iloc[idx]['student_id'])
    contrib = sv_eng[idx]
    o = np.argsort(np.abs(contrib))[::-1][:8][::-1]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    cols = [C['orange'] if contrib[i] > 0 else C['blue'] for i in o]
    ax.barh(range(len(o)), contrib[o], color=cols, height=0.6)
    for k, i in enumerate(o):
        v = contrib[i]
        ax.annotate(f'{v:+.3f}', (v, k), xytext=(5 if v >= 0 else -5, 0),
                    textcoords='offset points', va='center',
                    ha='left' if v >= 0 else 'right', fontsize=7.5, color=C['ink2'])
    ax.axvline(0, color=C['muted'], lw=1)
    ax.set_yticks(range(len(o))); ax.set_yticklabels([fn[i] for i in o], fontsize=8)
    ax.set_xlabel('SHAP value (orange raises risk, blue lowers it)')
    ax.set_title(f'Local attribution — {label}: {sid}, p = {p_eng[idx]:.3f}, true label {truth}')
    pad = max(np.abs(contrib[o])) * 0.35
    ax.set_xlim(min(contrib[o]) - pad, max(contrib[o]) + pad)
    ax.grid(axis='y', visible=False); tidy(ax)
    fig.tight_layout(); fig.savefig(f'{OUT}/figure{tag}.png', bbox_inches='tight'); plt.close(fig)
    LOCAL[tag] = {'student_id': sid, 'p': round(float(p_eng[idx]), 4), 'true_label': truth}

# ------------------------------------------------------------------ Figure 7
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
for ax, (name, p, t) in zip(axes, [('Proposed XGBoost', p_eng, thr_eng),
                                   ('Default XGBoost', p_def, thr_def)]):
    cm = confusion_matrix(yte, (p >= t).astype(int), labels=[0, 1])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f'{cmn[r, c]:.2f}\n(n={cm[r, c]})', ha='center', va='center',
                    fontsize=9, color=C['surface'] if cmn[r, c] > 0.55 else C['ink'])
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Predicted\nnon-dropout', 'Predicted\ndropout'],
                                              fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Actual\nnon-dropout', 'Actual\ndropout'],
                                              fontsize=8)
    ax.set_title(f'{name}  (τ = {t})'); ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
fig.suptitle('Row-normalised confusion matrices, seed 42 (each row sums to 1.0)',
             fontsize=10, fontweight='bold')
fig.tight_layout(); fig.savefig(f'{OUT}/figure7.png', bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ Figure 8
ABL = R['shap_sms']['ablation']['proposed_own_threshold']
RP, DA = ABL['rank_preservation'], ABL['directional_agreement']
labels = ['RPS@1', 'RPS@2', 'DAS@1', 'DAS@2']
rank_v = [RP['rank_preserving_at1'], RP['rank_preserving_at2'],
          DA['rank_preserving_at1'], DA['rank_preserving_at2']]
naive_v = [RP['naive_fixed_order_at1'], RP['naive_fixed_order_at2'],
           DA['naive_fixed_order_at1'], DA['naive_fixed_order_at2']]
x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(7.0, 4.0))
b1 = ax.bar(x - w / 2 - 0.01, rank_v, w, color=C['blue'],
            label='Rank-preserving protocol (|SHAP|-sorted)')
b2 = ax.bar(x + w / 2 + 0.01, naive_v, w, color=C['orange'],
            label='Naive fixed canonical order (ablated)')
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f'{b.get_height():.3f}', (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8,
                    color=C['ink2'])
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.10); ax.set_ylabel('Score')
ax.set_title(f"SHAPtoSMS ordering ablation — proposed model's own seed-42 flagged "
             f"population (n = {ABL['n']})")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8)
ax.grid(axis='x', visible=False); tidy(ax)
fig.text(0.012, -0.13, 'RPS = rank preservation (packed order vs the model\'s own |SHAP| '
                       'ranking).  DAS = directional agreement (top-k attribution sign vs '
                       'observed outcome).', fontsize=7.2, color=C['ink2'])
fig.tight_layout(); fig.savefig(f'{OUT}/figure8.png', bbox_inches='tight'); plt.close(fig)

# ----------------------------------------------------------------- Figure S4
me = M['ref']['model_eng']
noise_raw = {'attendance_rate': 0.02, 'exam_score': 3.0, 'distance_km': 0.5}
span = {c: (sc.data_max_[i] - sc.data_min_[i]) for i, c in enumerate(D['feats_cont'])}
rng = np.random.default_rng(42); alld = []
for _ in range(100):
    Xp = Xte.copy()
    for c, nz in noise_raw.items():
        j = fn.index(c)
        Xp[:, j] = np.clip(Xp[:, j] + rng.normal(0, nz / span[c], len(yte)), 0, 1)
    alld.append(np.abs(me.predict_proba(Xp)[:, 1] - p_eng))
alld = np.concatenate(alld)
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.hist(alld, bins=48, color=C['blue'], edgecolor=C['surface'], linewidth=0.4)
ax.axvline(alld.mean(), color=C['orange'], lw=2, ls=(0, (5, 3)))
ax.annotate(f'mean {alld.mean():.4f}', (alld.mean(), ax.get_ylim()[1] * 0.92),
            xytext=(8, 0), textcoords='offset points', fontsize=8.5, color=C['orange'])
ax.annotate(f'max {alld.max():.4f}', (alld.max(), ax.get_ylim()[1] * 0.30),
            xytext=(-8, 0), textcoords='offset points', ha='right', fontsize=8.5,
            color=C['ink2'])
ax.set_xlabel('|Δ predicted probability| under register-error-scale noise')
ax.set_ylabel('Count (248 records × 100 trials)')
ax.set_title('Perturbation sensitivity, proposed XGBoost (seed 42, 100 trials)')
ax.grid(axis='x', visible=False); tidy(ax)
fig.tight_layout(); fig.savefig(f'{OUT}/figureS4.png', bbox_inches='tight'); plt.close(fig)

json.dump({'local_cases': LOCAL,
           'perturbation_fig': {'mean': round(float(alld.mean()), 4),
                                'max': round(float(alld.max()), 4)}},
          open('results/figure_meta.json', 'w'), indent=2)
print('figures written to', OUT, '->', sorted(os.listdir(OUT)))
print('local cases:', LOCAL)
