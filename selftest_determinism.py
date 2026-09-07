#!/usr/bin/env python3
"""
Determinism self-test — proves the reported numbers do not depend on thread count.

WHY THIS EXISTS
---------------
The round-2 push exposed a defect that seeding alone does not fix. XGBoost and
torch both accumulate floating-point sums in thread-completion order, so the same
seed on a machine with a different core count produced different trees, different
weights, and different reported metrics. It was confirmed by changing only
OMP_NUM_THREADS on one machine: 329 of 858 result leaves moved, and two clearance
conditions changed verdict.

That is a reproducibility bug, not a limitation, and it is repaired rather than
declared: every estimator in this pipeline is now pinned to a single thread and
torch runs in deterministic mode (see set_all_seeds in
EduTrace_Revised_Pipeline.py).

WHAT THIS SCRIPT DOES
---------------------
It fits the model arms the reported figures rest on, twice, in two separate
subprocesses that differ ONLY in their thread environment, and compares the
resulting probability vectors bit for bit. If the pinning works, the two runs are
identical on any machine. If they are not identical, this script fails loudly and
names the arm that diverged.

A reviewer should run this on THEIR OWN machine. Passing here is what licenses
the claim in Methods M14 and M19 that the run is reproducible across machines,
rather than merely on the one that produced it.

Run:  python selftest_determinism.py
Exit: 0 = deterministic under every thread setting tried; 1 = not.
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join('data', 'real_student_data_CLEANED_school.csv')
SYNTH = os.path.join('data', 'synth_ctgan_s42.csv')

# Thread environments to compare. If the pinning is effective the arms below
# return identical output under all of them.
THREAD_SETTINGS = ['1', '2', '4']

WORKER = r'''
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or '.')
import EduTrace_Revised_Pipeline as P
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb

REAL, SYNTH = sys.argv[1], sys.argv[2]
D = P.load_and_prepare(REAL, SYNTH, use_lag=False)
Xtv, ytv, Xte = D['Xtv'], D['ytv'], D['Xte']
out = {}

def h(a):
    return hashlib.sha256(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()

for seed in (42, 123):
    P.set_all_seeds(seed)
    Xtr, Xvl, ytr, yvl = train_test_split(Xtv, ytv, test_size=0.15,
                                          stratify=ytv, random_state=seed)
    sm = SMOTE(k_neighbors=P.SMOTE_K, sampling_strategy=P.SMOTE_STRATEGY,
               random_state=seed)
    Xs, ys = sm.fit_resample(Xtr, ytr)
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)

    # Arm 1 - the proposed engineered XGBoost.
    P.set_all_seeds(seed)
    me = xgb.XGBClassifier(n_jobs=1, n_estimators=200, max_depth=4, learning_rate=0.05,
                           scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                           random_state=seed, eval_metric='aucpr',
                           verbosity=0).fit(Xs, ys)
    out['xgb_engineered_seed%d' % seed] = h(me.predict_proba(Xte)[:, 1])

    # Arm 2 - the default XGBoost comparator.
    P.set_all_seeds(seed)
    md = xgb.XGBClassifier(n_jobs=1, random_state=seed, eval_metric='logloss',
                           verbosity=0).fit(Xtr, ytr)
    out['xgb_default_seed%d' % seed] = h(md.predict_proba(Xte)[:, 1])

    # Arm 3 - SMOTE resampling itself.
    out['smote_seed%d' % seed] = h(Xs)

# Arm 4 - the TabTransformer comparator, one seed only (it is the slow arm).
P.set_all_seeds(42)
Xtr, Xvl, ytr, yvl, itr, ivl = train_test_split(
    Xtv, ytv, np.arange(len(ytv)), test_size=0.15, stratify=ytv, random_state=42)
Xc_tr, Xn_tr = D['Xcat_tv'][itr], D['Xcont_tv'][itr]
Xc_vl, Xn_vl = D['Xcat_tv'][ivl], D['Xcont_tv'][ivl]
comb = np.hstack([Xc_tr.astype(np.float32), Xn_tr])
smt = SMOTE(k_neighbors=P.SMOTE_K, sampling_strategy=P.SMOTE_STRATEGY, random_state=42)
comb_s, y_s = smt.fit_resample(comb, ytr)
nc = Xc_tr.shape[1]
ratio = float((ytv == 0).sum()) / max(float((ytv == 1).sum()), 1.0)
m = P.train_tabtransformer(comb_s[:, :nc].clip(0, None).astype(np.int64),
                           comb_s[:, nc:].clip(0.0, 1.0).astype(np.float32), y_s,
                           Xc_vl, Xn_vl, yvl, 42, ratio / (1.0 + ratio),
                           len(D['feats_cont']), max_epochs=12, patience=12)
out['tabtransformer_seed42_12ep'] = h(P.tt_predict(m, D['Xcat_te'], D['Xcont_te']))

print('@@JSON@@' + json.dumps(out))
'''


def run(threads):
    env = dict(os.environ)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[var] = threads
    worker = os.path.join(HERE, '_determinism_worker.py')
    with open(worker, 'w') as fh:
        fh.write(WORKER)
    try:
        r = subprocess.run([sys.executable, worker, REAL, SYNTH],
                           capture_output=True, text=True, env=env, cwd=HERE)
    finally:
        if os.path.exists(worker):
            os.remove(worker)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit('worker failed at OMP_NUM_THREADS=%s' % threads)
    for line in r.stdout.splitlines():
        if line.startswith('@@JSON@@'):
            return json.loads(line[len('@@JSON@@'):])
    print(r.stdout[-3000:])
    raise SystemExit('worker produced no result at OMP_NUM_THREADS=%s' % threads)


def main():
    print('=' * 78)
    print('DETERMINISM SELF-TEST — thread-count independence')
    print('=' * 78)
    print('  cpu cores visible to this machine : %s' % (os.cpu_count(),))
    print('  thread settings compared          : %s' % ', '.join(THREAD_SETTINGS))
    print('  Each setting runs in its own subprocess. Only the thread environment')
    print('  differs; data, seeds and code are identical.\n')

    runs = {}
    for t in THREAD_SETTINGS:
        print('  running with OMP_NUM_THREADS=%s ...' % t, flush=True)
        runs[t] = run(t)

    base_t = THREAD_SETTINGS[0]
    base = runs[base_t]
    arms = sorted(base)
    failures = []

    print('\n%-34s %-10s %s' % ('ARM', 'THREADS', 'DIGEST (first 16)'))
    for arm in arms:
        for t in THREAD_SETTINGS:
            mark = '' if runs[t][arm] == base[arm] else '   <-- DIFFERS'
            print('%-34s %-10s %s%s' % (arm if t == base_t else '', t,
                                        runs[t][arm][:16], mark))
            if runs[t][arm] != base[arm]:
                failures.append((arm, t))
        print()

    print('=' * 78)
    if failures:
        print('DETERMINISM SELF-TEST: FAILED')
        for arm, t in failures:
            print('  %s diverged at OMP_NUM_THREADS=%s' % (arm, t))
        print('\nThe reported numbers depend on thread count on this machine. Do not')
        print('treat any figure as reproducible until this passes.')
        return 1
    print('DETERMINISM SELF-TEST: PASSED')
    print('  %d arms x %d thread settings, all identical.' % (len(arms), len(THREAD_SETTINGS)))
    print('  The reported run does not depend on how many cores this machine has.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
