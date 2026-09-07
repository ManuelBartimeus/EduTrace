#!/usr/bin/env python3
"""
Export the locked run's fitted artefacts (observation O3, and Methods M19).

TWO PROBLEMS THIS CLOSES
------------------------
O3 in the verification report: "app/ carries duplicate xgboost.pkl, scaler,
imputer and encoder objects; any refit in the notebook leaves those stale and the
interface serving an older model." That was true, and it stayed true through the
round-2 push: app/ and models/ still held artefacts fitted under the pre-
remediation configuration (SMOTE k=5, scale_pos_weight 13.48, threshold 0.62),
while the manuscript reported k=3, 7.49 and 0.34. The demonstrator was serving a
model no longer described anywhere in the paper.

Methods M19 also states serialised checkpoint sizes as a low-connectivity
deployability proxy. Those figures should be measured on the artefacts this run
actually produces, not carried over. This script prints them.

WHAT IT DOES
------------
Refits the seed-42 arms from the locked configuration, exactly as
EduTrace_Revised_Pipeline.run_models does, and writes them to models/. The
preprocessing objects are fitted on the same training frame the pipeline uses, so
a served prediction goes through the same transform as a reported one. app/ then
receives copies of only what app.py loads, so there is one source of truth rather
than two.

The pipeline is deterministic (see selftest_determinism.py), so these artefacts
correspond to the reported run by construction, not by coincidence.
"""
import json
import os
import pickle
import shutil
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import EduTrace_Revised_Pipeline as P

REAL = os.path.join('data', 'real_student_data_CLEANED_school.csv')
SYNTH = os.path.join('data', 'synth_ctgan_s42.csv')
MODELS = 'models'
APP = 'app'
SEED = 42

# Only these are loaded by app/app.py. Anything else in app/ is a stale duplicate.
APP_NEEDS = ['xgboost.pkl', 'minmax_scaler.pkl', 'cont_imputer.pkl',
             'cat_imputer.pkl', 'label_encoders.pkl']


def main():
    os.makedirs(MODELS, exist_ok=True)
    print('=' * 78)
    print('EXPORT FITTED ARTEFACTS FROM THE LOCKED RUN   (O3, M19)')
    print('=' * 78)

    D = P.load_and_prepare(REAL, SYNTH, use_lag=False)
    Xtv, ytv = D['Xtv'], D['ytv']

    P.set_all_seeds(SEED)
    Xtr, Xvl, ytr, yvl, itr, ivl = train_test_split(
        Xtv, ytv, np.arange(len(ytv)), test_size=0.15, stratify=ytv, random_state=SEED)
    sm = SMOTE(k_neighbors=P.SMOTE_K, sampling_strategy=P.SMOTE_STRATEGY, random_state=SEED)
    Xtr_s, ytr_s = sm.fit_resample(Xtr, ytr)
    spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    print('  seed %d | train %d -> %d after SMOTE | scale_pos_weight %.2f'
          % (SEED, len(ytr), len(ytr_s), spw))

    written = []

    def dump(obj, name):
        p = os.path.join(MODELS, name)
        with open(p, 'wb') as fh:
            pickle.dump(obj, fh)
        written.append(p)
        return p

    # ---- the proposed model -------------------------------------------------
    P.set_all_seeds(SEED)
    me = xgb.XGBClassifier(n_jobs=1, n_estimators=200, max_depth=4, learning_rate=0.05,
                           scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                           random_state=SEED, eval_metric='aucpr',
                           verbosity=0).fit(Xtr_s, ytr_s)
    dump(me, 'xgboost.pkl')

    # ---- comparators --------------------------------------------------------
    P.set_all_seeds(SEED)
    md = xgb.XGBClassifier(n_jobs=1, random_state=SEED, eval_metric='logloss',
                           verbosity=0).fit(Xtr, ytr)
    dump(md, 'xgboost_default.pkl')

    P.set_all_seeds(SEED)
    dt = DecisionTreeClassifier(max_depth=5, random_state=SEED,
                                class_weight='balanced').fit(Xtr_s, ytr_s)
    dump(dt, 'decision_tree.pkl')

    # ---- preprocessing, refitted on the same training frame -----------------
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    import pandas as pd
    real = pd.read_csv(REAL); real['data_source'] = 'real'
    synth = pd.read_csv(SYNTH)
    if 'data_source' not in synth:
        synth['data_source'] = 'synthetic'
    pooled = pd.concat([real, synth], ignore_index=True)
    tv = pooled[pooled[P.YEAR] < P.SPLIT_YEAR].copy()
    dump(SimpleImputer(strategy='median').fit(tv[P.CONT]), 'cont_imputer.pkl')
    dump(SimpleImputer(strategy='most_frequent').fit(tv[P.CAT]), 'cat_imputer.pkl')
    tv_imp = tv.copy()
    tv_imp[P.CONT] = SimpleImputer(strategy='median').fit_transform(tv[P.CONT])
    dump(MinMaxScaler().fit(tv_imp[P.CONT]), 'minmax_scaler.pkl')
    encs = {}
    for c in P.CAT:
        le = LabelEncoder(); le.classes_ = np.array(P.CAT_VOCAB[c]); encs[c] = le
    dump(encs, 'label_encoders.pkl')

    # ---- the TabTransformer comparator, seed 42 -----------------------------
    P.set_all_seeds(SEED)
    Xc_tr, Xn_tr = D['Xcat_tv'][itr], D['Xcont_tv'][itr]
    Xc_vl, Xn_vl = D['Xcat_tv'][ivl], D['Xcont_tv'][ivl]
    comb = np.hstack([Xc_tr.astype(np.float32), Xn_tr])
    smt = SMOTE(k_neighbors=P.SMOTE_K, sampling_strategy=P.SMOTE_STRATEGY, random_state=SEED)
    comb_s, y_s = smt.fit_resample(comb, ytr)
    nc = Xc_tr.shape[1]
    ratio = float((ytv == 0).sum()) / max(float((ytv == 1).sum()), 1.0)
    tt = P.train_tabtransformer(comb_s[:, :nc].clip(0, None).astype(np.int64),
                                comb_s[:, nc:].clip(0.0, 1.0).astype(np.float32), y_s,
                                Xc_vl, Xn_vl, yvl, SEED, ratio / (1.0 + ratio),
                                len(D['feats_cont']))
    ttp = os.path.join(MODELS, 'tabtransformer_seed%d.pt' % SEED)
    torch.save(tt.state_dict(), ttp)
    written.append(ttp)

    # ---- sync app/ to the canonical artefacts -------------------------------
    print('\n  syncing app/ so the demonstrator serves the reported model (O3)')
    os.makedirs(APP, exist_ok=True)
    for name in APP_NEEDS:
        src = os.path.join(MODELS, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(APP, name))
            print('    app/%-22s <- models/%s' % (name, name))

    # ---- report sizes for M19 ----------------------------------------------
    print('\n%-42s %12s' % ('ARTEFACT', 'SIZE'))
    sizes = {}
    for p in sorted(written):
        kb = os.path.getsize(p) / 1024.0
        sizes[os.path.basename(p)] = round(kb, 1)
        print('%-42s %9.1f KB' % (p, kb))
    xgb_mb = sizes.get('xgboost.pkl', 0) / 1024.0
    tt_mb = sizes.get('tabtransformer_seed42.pt', 0) / 1024.0
    print('\n  M19 deployability proxy, measured on this run:')
    print('    proposed XGBoost checkpoint : %.2f MB' % xgb_mb)
    print('    TabTransformer checkpoint   : %.2f MB' % tt_mb)
    print('    both within the sub-10 MB offline-transfer target: %s'
          % (xgb_mb < 10 and tt_mb < 10))

    json.dump({'seed': SEED, 'scale_pos_weight': round(spw, 2),
               'artefact_sizes_kb': sizes,
               'xgboost_checkpoint_mb': round(xgb_mb, 2),
               'tabtransformer_checkpoint_mb': round(tt_mb, 2),
               'app_synced': APP_NEEDS,
               'note': ('Fitted from the locked run under the pinned deterministic '
                        'settings, so these artefacts correspond to the reported '
                        'figures by construction. Closes O3.')},
              open(os.path.join('results', 'model_artefacts.json'), 'w'), indent=2)
    print('\nwritten: results/model_artefacts.json')


if __name__ == '__main__':
    main()
