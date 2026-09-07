"""Compute the SMOTE Nearest-Neighbour Adversarial Accuracy on the LOCKED run.

The prior remediation run asserted that the retained SMOTE configuration breaches
the 0.60 NNAA ceiling but never computed it — the figure had been carried over
from the earlier, non-locked artefacts (final_summary_v3.json, which was produced
on a different pool). This closes that gap.

compute_nnaa and the escalation ladder are ported verbatim from
EduTrace_Revised_Pipeline.py Cell 2.1, so the numbers are comparable to what the
original pipeline would have produced on this pool.
"""
import json, os, random, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from EduTrace_Revised_Pipeline import load_and_prepare, SEEDS, SMOTE_STRATEGY, set_all_seeds

REAL = 'data/real_student_data_CLEANED_school.csv'
SYNTH = 'data/synth_ctgan_s42.csv'
CEILING = 0.60


def compute_nnaa(n_base, X_oversampled):
    """Verbatim from Cell 2.1: 1-NN adversary separating real from synthesised rows."""
    n_new = X_oversampled.shape[0] - n_base
    y_audit = np.array([0] * n_base + [1] * n_new)
    return float(cross_val_score(KNeighborsClassifier(n_neighbors=1),
                                 X_oversampled, y_audit, cv=3, scoring='accuracy').mean())


def dp_smote(X, y, k, strategy, seed, epsilon=1.0):
    """Verbatim from Cell 2.1."""
    sm = SMOTE(k_neighbors=k, sampling_strategy=strategy, random_state=seed)
    Xs, ys = sm.fit_resample(X, y)
    n0 = X.shape[0]
    delta = 1e-5
    frange = X.max(axis=0) - X.min(axis=0)
    frange = np.where(frange > 0, frange, 1.0)
    sens = np.mean(frange) / max(k, 1)
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * sens / epsilon
    rng = np.random.RandomState(seed)
    Xs = Xs.copy()
    Xs[n0:] = np.clip(Xs[n0:] + rng.normal(0, sigma, size=Xs[n0:].shape), 0.0, 1.0)
    return Xs, ys, float(sigma)


D = load_and_prepare(REAL, SYNTH, use_lag=False)
Xtv, ytv = D['Xtv'], D['ytv']

LADDER = [('SMOTE k=5', 5, False), ('SMOTE k=3', 3, False),
          ('SMOTE k=1', 1, False), ('DP-SMOTE k=1 (epsilon=1.0)', 1, True)]

per_seed = {}
for seed in SEEDS:
    set_all_seeds(seed)
    Xtr, _, ytr, _ = train_test_split(Xtv, ytv, test_size=0.15,
                                      stratify=ytv, random_state=seed)
    n0 = Xtr.shape[0]
    row = {}
    for label, k, dp in LADDER:
        if dp:
            Xs, ys, sigma = dp_smote(Xtr, ytr, k, SMOTE_STRATEGY, seed)
        else:
            sm = SMOTE(k_neighbors=k, sampling_strategy=SMOTE_STRATEGY, random_state=seed)
            Xs, ys = sm.fit_resample(Xtr, ytr)
            sigma = None
        nnaa = compute_nnaa(n0, Xs)
        row[label] = {'nnaa': round(nnaa, 4), 'rows_added': int(Xs.shape[0] - n0),
                      'passes_ceiling': bool(nnaa <= CEILING),
                      **({'dp_sigma': round(sigma, 6)} if sigma is not None else {})}
    per_seed[seed] = row

labels = [l for l, _, _ in LADDER]
summary = {}
for l in labels:
    vals = [per_seed[s][l]['nnaa'] for s in SEEDS]
    summary[l] = {'seed42': per_seed[42][l]['nnaa'],
                  'mean': round(float(np.mean(vals)), 4),
                  'std': round(float(np.std(vals)), 4),
                  'min': round(float(min(vals)), 4),
                  'max': round(float(max(vals)), 4),
                  'rows_added_seed42': per_seed[42][l]['rows_added'],
                  'any_passes_ceiling': bool(any(v <= CEILING for v in vals))}

# Which rung would the ladder retain? (lowest NNAA at seed 42, matching Cell 2.1's rule)
retained = min(labels, key=lambda l: per_seed[42][l]['nnaa'])

OUT = {'ceiling': CEILING, 'strategy': SMOTE_STRATEGY, 'seeds': SEEDS,
       'per_seed': per_seed, 'summary': summary,
       'configured_k': 3,
       'lowest_nnaa_rung_at_seed42': retained,
       'all_rungs_breach_ceiling': bool(all(not summary[l]['any_passes_ceiling'] for l in labels)),
       'note': ('NNAA computed on the locked pool, post 85/15 split, exactly as Cell 2.1 does. '
                'A value of 1.0 means a 1-NN adversary separates every synthesised row from the '
                'real rows; 0.60 is the ceiling adopted in M4.')}

print(f"{'configuration':<30} {'seed42':>8} {'mean±sd':>16} {'range':>18} {'rows':>6}")
print('-' * 84)
for l in labels:
    s = summary[l]
    print(f"{l:<30} {s['seed42']:>8.4f} {s['mean']:>8.4f}±{s['std']:<7.4f} "
          f"[{s['min']:.4f}, {s['max']:.4f}] {s['rows_added_seed42']:>6}")
print(f"\nceiling {CEILING}; all rungs breach = {OUT['all_rungs_breach_ceiling']}")
print(f"lowest-NNAA rung at seed 42 = {retained}  (configured: k=3)")
json.dump(OUT, open('results/smote_nnaa.json', 'w'), indent=2)
print('\nwrote results/smote_nnaa.json')
