#!/usr/bin/env python3
"""
Leave-one-site-out validation (Q2 / M18).

Methods M18 names a school identifier as one of two ingredients required to move
beyond the Tier 1 claim ceiling, because it is what makes leave-one-site-out
validation possible: train on some schools, test on a school the model has never
seen. The identifier has now been recovered, so the analysis is run here.

DESIGN
------
Each fold holds out one school. The model trains on the TRAINING-PERIOD records
(academic_year < 2025) of the other three schools and is evaluated on the
TEST-PERIOD records (academic_year >= 2025) of the held-out school. The temporal
firewall is preserved inside every fold, so a fold tests site generalisation and
next-year prediction together rather than trading one for the other.

WHY THE SYNTHETIC SUPPLEMENT IS EXCLUDED FROM LOSO TRAINING
-----------------------------------------------------------
The supplement was fitted by CTGAN on the whole training-period pool, which
includes the records of every school. Any fold that trained on it would be
training on a generative summary of the held-out school, which is precisely the
leakage a site hold-out exists to prevent. Each fold therefore trains on real
records only. This is stated rather than quietly arranged, and it is consistent
with the leave-one-out result already reported in R3, where dropping the
supplement raised the primary metric.

WHAT THIS CAN AND CANNOT RETURN
-------------------------------
The script reports per fold and does not average over folds it cannot compute.
A held-out school with no dropout event in the test period yields undefined
AUC-PR and AUC-ROC, and that is reported as NOT COMPUTABLE rather than filled
with a zero, a prevalence floor or a pooled substitute. No cell is selected and
no fold is dropped for being unfavourable.
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, fbeta_score,
                             precision_score, recall_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import EduTrace_Revised_Pipeline as P

REAL_SCHOOL = os.path.join('data', 'real_student_data_CLEANED_school.csv')
OUT = os.path.join('results', 'loso_validation.json')
SEEDS = P.SEEDS
SCHOOL_NAME = {}


def design(train_df, eval_df):
    """Same preprocessing as the pipeline, fitted on the fold's training frame only."""
    tr, ev = train_df.copy(), eval_df.copy()
    ci = SimpleImputer(strategy='median').fit(tr[P.CONT])
    mi = SimpleImputer(strategy='most_frequent').fit(tr[P.CAT])
    tr[P.CONT] = ci.transform(tr[P.CONT]); ev[P.CONT] = ci.transform(ev[P.CONT])
    tr[P.CAT] = mi.transform(tr[P.CAT]);  ev[P.CAT] = mi.transform(ev[P.CAT])
    sc = MinMaxScaler().fit(tr[P.CONT])
    tr[P.CONT] = sc.transform(tr[P.CONT]); ev[P.CONT] = sc.transform(ev[P.CONT])
    a = pd.get_dummies(tr[P.CAT + P.CONT], columns=P.CAT)
    b = pd.get_dummies(ev[P.CAT + P.CONT], columns=P.CAT).reindex(columns=a.columns,
                                                                  fill_value=0)
    return (a.values.astype(np.float32), b.values.astype(np.float32),
            tr[P.TARGET].values.astype(int), ev[P.TARGET].values.astype(int), sc)


def main():
    df = pd.read_csv(REAL_SCHOOL)
    global SCHOOL_NAME
    SCHOOL_NAME = dict(df[['school_id', 'school_name']].drop_duplicates().values)
    sites = sorted(df['school_id'].unique())

    print('=' * 92)
    print('LEAVE-ONE-SITE-OUT VALIDATION   (Q2 / Methods M18)')
    print('=' * 92)
    print('  sites: %d — %s' % (len(sites), ', '.join(sites)))
    print('  each fold: train on academic_year < %d of the other sites (real records only),'
          % P.SPLIT_YEAR)
    print('             test on academic_year >= %d of the held-out site' % P.SPLIT_YEAR)
    print()

    print('%-7s %-36s %9s %9s %9s %9s %9s'
          % ('HELDOUT', 'SCHOOL', 'TRAIN_N', 'TRAIN_POS', 'TEST_N', 'TEST_POS', 'BASE'))
    frame = []
    for s in sites:
        tr = df[(df['school_id'] != s) & (df[P.YEAR] < P.SPLIT_YEAR)]
        te = df[(df['school_id'] == s) & (df[P.YEAR] >= P.SPLIT_YEAR)]
        base = float(te[P.TARGET].mean()) if len(te) else float('nan')
        frame.append((s, tr, te, base))
        print('%-7s %-36s %9d %9d %9d %9d %9.4f'
              % (s, SCHOOL_NAME[s][:36], len(tr), int(tr[P.TARGET].sum()),
                 len(te), int(te[P.TARGET].sum()), base))
    print()

    folds = []
    for s, tr, te, base in frame:
        row = dict(held_out_site=s, school=SCHOOL_NAME[s],
                   train_n=int(len(tr)), train_positives=int(tr[P.TARGET].sum()),
                   test_n=int(len(te)), test_positives=int(te[P.TARGET].sum()),
                   test_base_rate=None if np.isnan(base) else round(base, 4))
        print('--- held out: %s (%s)' % (s, SCHOOL_NAME[s]))
        if len(te) == 0:
            row.update(computable=False,
                       reason='the held-out site has no test-period record')
            print('    NOT COMPUTABLE — no test-period record for this site')
            folds.append(row); continue
        if int(te[P.TARGET].sum()) == 0:
            row.update(computable=False,
                       reason=('the held-out site has %d test-period records but 0 dropout '
                               'events, so AUC-PR and AUC-ROC are undefined on it'
                               % len(te)))
            print('    NOT COMPUTABLE — %d test records, 0 dropout events; ranking metrics '
                  'are undefined' % len(te))
            print('    Reported as not computable rather than filled with a floor value.')
            folds.append(row); continue

        per_seed = []
        for seed in SEEDS:
            P.set_all_seeds(seed)
            Xtr, Xte, ytr, yte, _ = design(tr, te)
            idx_tr, idx_val = train_test_split(np.arange(len(ytr)), test_size=0.15,
                                               stratify=ytr, random_state=seed)
            Xi, yi = Xtr[idx_tr], ytr[idx_tr]
            Xv, yv = Xtr[idx_val], ytr[idx_val]
            n_min = int((yi == 1).sum())
            if n_min > P.SMOTE_K:
                sm = SMOTE(k_neighbors=P.SMOTE_K, sampling_strategy=P.SMOTE_STRATEGY,
                           random_state=seed)
                Xi, yi = sm.fit_resample(Xi, yi)
            spw = float((yi == 0).sum()) / max(float((yi == 1).sum()), 1.0)
            P.set_all_seeds(seed)
            m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=spw, subsample=0.8,
                                  colsample_bytree=0.8, random_state=seed,
                                  eval_metric='aucpr', verbosity=0).fit(Xi, yi)
            thr, _, _ = P.f2_threshold_search(yv, m.predict_proba(Xv)[:, 1])
            p = m.predict_proba(Xte)[:, 1]
            pred = (p >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
            try:
                auc_roc = float(roc_auc_score(yte, p))
            except ValueError:
                auc_roc = float('nan')
            per_seed.append(dict(
                seed=seed, threshold=thr,
                auc_pr=float(average_precision_score(yte, p)), auc_roc=auc_roc,
                recall=float(recall_score(yte, pred, zero_division=0)),
                precision=float(precision_score(yte, pred, zero_division=0)),
                f2=float(fbeta_score(yte, pred, beta=2, zero_division=0)),
                flagged=int(pred.sum()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)))
            print('    seed %5d: AUC-PR=%.4f  AUC-ROC=%.4f  recall=%.4f  precision=%.4f  '
                  'flagged=%d  TP=%d' % (seed, per_seed[-1]['auc_pr'], auc_roc,
                                         per_seed[-1]['recall'], per_seed[-1]['precision'],
                                         int(pred.sum()), int(tp)))

        def agg(k):
            v = [d[k] for d in per_seed if d[k] == d[k]]
            return (dict(mean=round(float(np.mean(v)), 4), std=round(float(np.std(v)), 4))
                    if v else None)

        row.update(computable=True, per_seed=per_seed,
                   summary={k: agg(k) for k in ('auc_pr', 'auc_roc', 'recall',
                                                'precision', 'f2')},
                   prevalence_floor=round(base, 4))
        lift = (row['summary']['auc_pr']['mean'] / base) if base > 0 else None
        row['auc_pr_lift_over_prevalence'] = None if lift is None else round(lift, 2)
        print('    ACROSS SEEDS: AUC-PR %.4f ± %.4f  (prevalence floor %.4f, lift %.2fx) | '
              'recall %.4f ± %.4f'
              % (row['summary']['auc_pr']['mean'], row['summary']['auc_pr']['std'],
                 base, row['auc_pr_lift_over_prevalence'],
                 row['summary']['recall']['mean'], row['summary']['recall']['std']))
        folds.append(row)
        print()

    comp = [f for f in folds if f.get('computable')]
    non = [f for f in folds if not f.get('computable')]

    print('=' * 92)
    print('SUMMARY')
    print('=' * 92)
    print('  sites total                 : %d' % len(folds))
    print('  folds computable            : %d (%s)'
          % (len(comp), ', '.join(f['held_out_site'] for f in comp) or '—'))
    print('  folds NOT computable        : %d (%s)'
          % (len(non), ', '.join(f['held_out_site'] for f in non) or '—'))
    print('  test-period dropout events across all held-out sites: %d'
          % sum(f['test_positives'] for f in folds))
    if comp:
        aps = [f['summary']['auc_pr']['mean'] for f in comp]
        print('  AUC-PR across the computable folds: %s' % ', '.join('%.4f' % a for a in aps))
        print('  range %.4f to %.4f' % (min(aps), max(aps)))
    verdict = (
        'PARTIALLY COMPUTABLE. %d of %d sites can be held out; %d cannot, because they '
        'carry no dropout event in the test period. The %d computable folds rest on %d '
        'dropout events between them, so the result is reported as a direction and not as '
        'an estimate of site-transfer performance.'
        % (len(comp), len(folds), len(non), len(comp),
           sum(f['test_positives'] for f in comp)))
    print()
    print('  VERDICT: %s' % verdict)

    payload = dict(
        design=('train on academic_year < %d of the other sites, real records only; '
                'test on academic_year >= %d of the held-out site' % (P.SPLIT_YEAR,
                                                                      P.SPLIT_YEAR)),
        synthetic_excluded_reason=(
            'the CTGAN supplement was fitted on the whole training-period pool, including '
            'the held-out site, so training on it would leak the held-out site'),
        seeds=SEEDS, sites=len(folds), folds=folds,
        n_computable=len(comp), n_not_computable=len(non),
        computable_sites=[f['held_out_site'] for f in comp],
        not_computable_sites=[f['held_out_site'] for f in non],
        total_test_positives=sum(f['test_positives'] for f in folds),
        verdict=verdict)
    os.makedirs('results', exist_ok=True)
    json.dump(payload, open(OUT, 'w'), indent=2)
    print('\nwritten: %s' % OUT)


if __name__ == '__main__':
    main()
