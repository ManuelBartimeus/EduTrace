"""
Generate the CTGAN synthetic supplement DETERMINISTICALLY, with the 4-level
audit from the original Cell 1.3.

Fixes the reproducibility defect in EduTrace_Revised_Pipeline.py Cell 1.2:
the original constructed CTGANSynthesizer with no seed and called
set_all_seeds(42) only AFTER .sample(), so the supplement differed on every
run and no reported model number was reproducible.

Usage: python gen_synth.py <out_csv> [--epochs 600] [--batch 50] [--seed 42]
"""
import argparse, json, os, random, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist

CAT = ['fee_payment_status', 'grade_level', 'gender']
CONT = ['attendance_rate', 'exam_score', 'distance_km']
TARGET, ID, YEAR = 'dropout', 'student_id', 'academic_year'
SPLIT = 2025
N_SYNTH = 350


def set_all_seeds(s):
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.use_deterministic_algorithms(False)
    os.environ['PYTHONHASHSEED'] = str(s)


def generate(real_csv, epochs, batch, seed):
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    df_real = pd.read_csv(real_csv)
    tv = df_real[df_real[YEAR] < SPLIT].copy()
    # Identifier columns are dropped before fitting. school_id/school_name are
    # record-level identifiers, not register variables, so the synthesiser must not
    # model them; dropping them keeps this regeneration byte-identical to the
    # committed supplement, which was generated before the identifier existed.
    drop_cols = [c for c in [ID, YEAR, 'school_id', 'school_name'] if c in tv.columns]
    fit_df = tv.drop(columns=drop_cols).reset_index(drop=True)

    md = SingleTableMetadata()
    md.detect_from_dataframe(fit_df)
    for c in CONT:
        md.update_column(c, sdtype='numerical', computer_representation='Float')
    for c in CAT + [TARGET]:
        md.update_column(c, sdtype='categorical')

    # Seed BEFORE construction and fit — this is the fix.
    set_all_seeds(seed)
    synth = CTGANSynthesizer(md, epochs=epochs, batch_size=batch, verbose=False)
    synth.fit(fit_df)

    set_all_seeds(seed)
    raw = synth.sample(N_SYNTH)

    # academic_year conditioned exclusively on the pre-split training period.
    years = sorted([y for y in df_real[YEAR].unique() if y < SPLIT])
    set_all_seeds(seed)
    raw[YEAR] = np.random.choice(years, size=N_SYNTH, p=[1.0 / len(years)] * len(years))
    raw[ID] = [f'SYN_{i:04d}' for i in range(N_SYNTH)]
    raw[TARGET] = raw[TARGET].astype(int)
    raw['data_source'] = 'synthetic'
    out = raw[[ID, YEAR] + CONT + CAT + [TARGET, 'data_source']]

    assert (out[YEAR] >= SPLIT).sum() == 0, 'CRITICAL: split-year rows in synthetic data'
    return df_real, out


def audit(df_real, df_synth):
    """4-level synthetic validation, ported verbatim in substance from Cell 1.3."""
    res = {}
    tv = df_real[df_real[YEAR] < SPLIT]
    te = df_real[df_real[YEAR] >= SPLIT]

    # Level a — statistical fidelity
    res['level_a'] = {}
    for c in CONT:
        r, s = tv[c].dropna().values, df_synth[c].dropna().values
        res['level_a'][c] = {'real_mean': round(float(r.mean()), 4), 'real_std': round(float(r.std()), 4),
                             'synth_mean': round(float(s.mean()), 4), 'synth_std': round(float(s.std()), 4)}
    for c in CAT:
        rd = tv[c].value_counts(normalize=True)
        sd = df_synth[c].value_counts(normalize=True)
        res['level_a'][c] = {str(k): {'real': round(float(rd.get(k, 0)), 4),
                                      'synth': round(float(sd.get(k, 0)), 4)}
                             for k in sorted(set(rd.index) | set(sd.index))}

    # Level b — TSTR vs TRTR
    def prep(tr, teX):
        cols = CAT + CONT + [TARGET]
        a, b = tr[cols].copy(), teX[cols].copy()
        imp = SimpleImputer(strategy='median')
        a[CONT] = imp.fit_transform(a[CONT]); b[CONT] = imp.transform(b[CONT])
        ao = pd.get_dummies(a.drop(columns=[TARGET]), columns=CAT)
        bo = pd.get_dummies(b.drop(columns=[TARGET]), columns=CAT).reindex(columns=ao.columns, fill_value=0)
        return ao.values.astype(np.float32), a[TARGET].values, bo.values.astype(np.float32), b[TARGET].values

    Xtr, ytr, Xte, yte = prep(tv, te)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced').fit(Xtr, ytr)
    trtr = average_precision_score(yte, lr.predict_proba(Xte)[:, 1])
    Xs, ys, Xt, yt = prep(df_synth, te)
    lr2 = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced').fit(Xs, ys)
    tstr = average_precision_score(yt, lr2.predict_proba(Xt)[:, 1])
    res['level_b'] = {'trtr_auc_pr': round(float(trtr), 4), 'tstr_auc_pr': round(float(tstr), 4),
                      'gap': round(float(abs(trtr - tstr)), 4), 'pass': bool(abs(trtr - tstr) < 0.10)}

    # Level c — NNAA privacy audit
    rn, sn = tv[CAT + CONT].copy(), df_synth[CAT + CONT].copy()
    imp = SimpleImputer(strategy='median')
    rn[CONT] = imp.fit_transform(rn[CONT]); sn[CONT] = imp.transform(sn[CONT])
    ro = pd.get_dummies(rn, columns=CAT)
    so = pd.get_dummies(sn, columns=CAT).reindex(columns=ro.columns, fill_value=0)
    Xr, Xsy = ro.values.astype(np.float32), so.values.astype(np.float32)
    Xa = np.vstack([Xr, Xsy]); ya = np.array([0] * len(Xr) + [1] * len(Xsy))
    nnaa = cross_val_score(KNeighborsClassifier(n_neighbors=1), Xa, ya, cv=3, scoring='accuracy').mean()
    res['level_c'] = {'nnaa': round(float(nnaa), 4), 'ceiling': 0.60, 'pass': bool(nnaa <= 0.60)}

    # Level d — membership inference (min NN distance)
    md_ = cdist(Xsy, Xr, metric='euclidean').min(axis=1)
    res['level_d'] = {'near_copies_lt_1e-3': int((md_ < 1e-3).sum()), 'n_synth': int(len(Xsy)),
                      'min_nn_dist': round(float(md_.min()), 6), 'mean_nn_dist': round(float(md_.mean()), 4)}
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--batch', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--real', default='data/real_student_data_CLEANED_school.csv')
    a = ap.parse_args()

    real, syn = generate(a.real, a.epochs, a.batch, a.seed)
    syn.to_csv(a.out, index=False)
    rep = audit(real, syn)
    rep['config'] = {'generator': f'CTGAN (SDV, {a.epochs} epochs, batch={a.batch}, seed={a.seed})',
                     'n_synth': N_SYNTH, 'synth_positives': int(syn[TARGET].sum())}
    with open(a.out.replace('.csv', '_audit.json'), 'w') as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
