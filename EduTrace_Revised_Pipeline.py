"""
EduTrace remediation pipeline.

Base: EduTrace_Revised_Pipeline.py (preprocessing, split logic, threshold search
and model configurations preserved verbatim in substance).

Changes, each tied to a supervisor checklist item:
  #1/#2  Attendance baseline split into a binary rule row and a continuous
         ranker row, each scored on its own score vector.
  #3/#4  SHAPtoSMS ablation re-run on the PROPOSED model's own flagged
         population with the gold ranking from the same model.
  #7     seed-42 single-run reference metrics printed for a single reference
         frame in the isotonic comparison.
  #10    SMOTE isolation experiment (on / off / raised strategy).
  #11    Layer-2 isolation (scale_pos_weight vs F2 threshold).
  #15    Student-disjoint test subset (reports n=0 — a data limitation).
  #18    Signed SHAP for the top three features.
  #20    Year-on-year lag/delta features + re-evaluation.
  #22    Error analysis on consistently-missed dropout cases across five seeds.
  #23    Perturbation analysis re-run pre/post lag features.
  Table 5 built from a real SHAPtoSMS implementation (Algorithm 1).

Synthetic supplement is loaded from a LOCKED, seeded artefact rather than
regenerated unseeded on every run.
"""
import argparse, copy, json, os, random, sys, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             fbeta_score, precision_score, recall_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
from statsmodels.stats.multitest import multipletests
import shap
from tab_transformer_pytorch import TabTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shaptosms as S

SEEDS = [42, 123, 777, 2024, 9999]
CAT = ['fee_payment_status', 'grade_level', 'gender']
CONT = ['attendance_rate', 'exam_score', 'distance_km']
TARGET, ID, YEAR = 'dropout', 'student_id', 'academic_year'
CAT_VOCAB = {'fee_payment_status': ['Fully Paid', 'Partially Paid', 'Unpaid'],
             'grade_level': ['Grade 7', 'Grade 8', 'Grade 9'],
             'gender': ['Male', 'Female']}
SPLIT_YEAR = 2025
SMOTE_K, SMOTE_STRATEGY = 3, 0.20
ATT_RULE_CUT = 0.70
DEVICE = torch.device('cpu')
R = {}   # results accumulator


def set_all_seeds(s):
    """Seed every source of randomness AND remove thread-order dependence.

    Seeding alone is not enough for reproducibility across machines. XGBoost and
    torch both accumulate floating-point sums in thread-completion order, so the
    same seed on a machine with a different core count produces different trees
    and different weights. The round-2 push confirmed this directly: changing
    only OMP_NUM_THREADS altered 329 of 858 result leaves and moved two
    clearance conditions. Every estimator in this pipeline is therefore pinned
    to a single thread, and torch is put into deterministic mode.

    This costs wall-clock time and buys the property Q4 actually asks for: a
    reader who clones the repository lands on the reported numbers.
    """
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s); random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.set_num_threads(1)
    try:                                    # only settable before parallel work starts
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- evaluation
def evaluate(y, proba, binary=None, threshold=None):
    if binary is None:
        binary = (proba >= (threshold if threshold is not None else 0.60)).astype(int)
    cm = confusion_matrix(y, binary, labels=[0, 1])
    return dict(auc_roc=float(roc_auc_score(y, proba)),
                auc_pr=float(average_precision_score(y, proba)),
                f2=float(fbeta_score(y, binary, beta=2, zero_division=0)),
                macro_f1=float(f1_score(y, binary, average='macro', zero_division=0)),
                precision=float(precision_score(y, binary, zero_division=0)),
                recall=float(recall_score(y, binary, zero_division=0)),
                cm=cm.tolist(), threshold=threshold)


def f2_threshold_search(y, p, lo=0.05, hi=0.95, step=0.01, min_p=0.10, min_s=0.50):
    ths = np.arange(lo, hi + step, step)
    best_f2, best_t, constrained = -1.0, 0.60, True
    for t in ths:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / max(tp + fp, 1); spec = tn / max(tn + fp, 1)
        if prec >= min_p and spec >= min_s:
            f2 = fbeta_score(y, pred, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2, best_t = f2, round(float(t), 2)
    if best_f2 < 0:
        constrained = False
        f2s = [fbeta_score(y, (p >= t).astype(int), beta=2, zero_division=0) for t in ths]
        i = int(np.argmax(f2s)); best_f2, best_t = f2s[i], round(float(ths[i]), 2)
    return best_t, float(best_f2), constrained


def agg(vals):
    return {'mean': round(float(np.mean(vals)), 4), 'std': round(float(np.std(vals)), 4)}


def summarise(per_seed, keys=('auc_pr', 'auc_roc', 'f2', 'macro_f1', 'precision', 'recall')):
    return {k: agg([d[k] for d in per_seed]) for k in keys}


# ------------------------------------------------- NF-1 / Q9 provenance guard
class SyntheticInTestPartition(AssertionError):
    """Raised when a generated record would be scored as a school record."""


NF1_LOG = []


def assert_test_partition_is_real(te, source_col='data_source', label='test'):
    """NF-1 / Q9 — load-time provenance firewall for the temporal split.

    This is the runtime check Methods M10 names. It is deliberately placed in
    the load path, so it runs on every construction of the evaluation partition
    and cannot be bypassed by a caller. It RAISES; it does not warn.

    Two conditions, both fatal:
      1. no row whose data_source is 'synthetic' may appear in the partition;
      2. the partition must carry at least one positive, or every minority
         metric computed on it is undefined rather than merely poor.

    It also prints the run-log line the verification report asks to see, and
    appends it to NF1_LOG so the line can be written to the results archive.
    """
    n = int(len(te))
    n_syn = int((te[source_col] == 'synthetic').sum())
    n_real = n - n_syn
    n_pos = int(te[TARGET].sum())
    if n_syn:
        raise SyntheticInTestPartition(
            'NF-1 GUARD FAILED: %d synthetic row(s) in the %s partition of %d. '
            'Every metric computed on this partition is contaminated. The '
            'synthesiser must be bounded to academic_year < %d (Q9).'
            % (n_syn, label, n, SPLIT_YEAR))
    if n_pos == 0:
        raise AssertionError(
            'NF-1 GUARD FAILED: %s partition of %d rows carries no positive '
            'case; minority-class metrics are undefined on it.' % (label, n))
    line = ('NF-1 GUARD: %s partition n=%d | synthetic rows=%d | real rows=%d | '
            'positives=%d (%.2f%%) — PASS'
            % (label, n, n_syn, n_real, n_pos, 100.0 * n_pos / n))
    print(line)
    NF1_LOG.append(line)
    return dict(n=n, n_synthetic=n_syn, n_real=n_real, n_positive=n_pos, passed=True)


# ------------------------------------------------------------------- loading
def load_and_prepare(real_csv, synth_csv, use_lag=False):
    real = pd.read_csv(real_csv); real['data_source'] = 'real'
    synth = pd.read_csv(synth_csv)
    if 'data_source' not in synth: synth['data_source'] = 'synthetic'

    feats_cont = list(CONT)
    if use_lag:
        real = add_lag_features(real)
        for c in LAG_COLS:
            synth[c] = 0.0                      # synthetic rows have no panel history
        feats_cont = feats_cont + LAG_COLS

    pooled = pd.concat([real, synth], ignore_index=True)
    tv = pooled[pooled[YEAR] < SPLIT_YEAR].copy().reset_index(drop=True)
    te = pooled[pooled[YEAR] >= SPLIT_YEAR].copy().reset_index(drop=True)

    # NF-1 / Q9 — the load-time provenance firewall. This is the runtime check
    # M10 names. It runs on EVERY load, it raises rather than warns, and it
    # prints the line the verification report asks to see. The test partition is
    # built from the POOLED frame above precisely so that the guard can observe
    # a contaminated partition if one ever exists; filtering synthetic rows out
    # before the check would make the check unfalsifiable.
    assert_test_partition_is_real(te)

    ci = SimpleImputer(strategy='median').fit(tv[feats_cont])
    mi = SimpleImputer(strategy='most_frequent').fit(tv[CAT])
    tv[feats_cont] = ci.transform(tv[feats_cont]); te[feats_cont] = ci.transform(te[feats_cont])
    tv[CAT] = mi.transform(tv[CAT]); te[CAT] = mi.transform(te[CAT])

    for c in CAT:
        le = LabelEncoder(); le.classes_ = np.array(CAT_VOCAB[c])
        tv[c + '_enc'] = le.transform(tv[c]); te[c + '_enc'] = le.transform(te[c])

    sc = MinMaxScaler().fit(tv[feats_cont])
    tv[feats_cont] = sc.transform(tv[feats_cont]); te[feats_cont] = sc.transform(te[feats_cont])

    ohe_tv = pd.get_dummies(tv[CAT + feats_cont], columns=CAT)
    ohe_te = pd.get_dummies(te[CAT + feats_cont], columns=CAT).reindex(columns=ohe_tv.columns, fill_value=0)

    return dict(
        tv=tv, te=te, feats_cont=feats_cont, scaler=sc,
        Xtv=ohe_tv.values.astype(np.float32), Xte=ohe_te.values.astype(np.float32),
        fnames=list(ohe_tv.columns),
        Xcat_tv=tv[[c + '_enc' for c in CAT]].values.astype(np.int64),
        Xcat_te=te[[c + '_enc' for c in CAT]].values.astype(np.int64),
        Xcont_tv=tv[feats_cont].values.astype(np.float32),
        Xcont_te=te[feats_cont].values.astype(np.float32),
        ytv=tv[TARGET].values.astype(np.float32), yte=te[TARGET].values.astype(np.float32))


LAG_COLS = ['att_delta', 'exam_delta', 'prior_risk_flag', 'terms_observed']


def add_lag_features(df):
    """Checklist #20 — year-on-year lag/delta features from the existing panel.

    No new data collection: derived entirely from the student-year panel the
    study already holds. Rows with no prior-year record get 0 (first
    observation), which is disclosed rather than imputed from other students.
    """
    d = df.copy()
    d['_sid'] = d[ID].str.split('_').str[0]
    d = d.sort_values(['_sid', YEAR])
    d['att_delta'] = d.groupby('_sid')['attendance_rate'].diff()
    d['exam_delta'] = d.groupby('_sid')['exam_score'].diff()
    d['prior_risk_flag'] = d.groupby('_sid')[TARGET].shift(1)
    d['terms_observed'] = d.groupby('_sid').cumcount()
    for c in ['att_delta', 'exam_delta', 'prior_risk_flag']:
        d[c] = d[c].fillna(0.0)
    d['terms_observed'] = d['terms_observed'].astype(float)
    return d.sort_index().drop(columns=['_sid'])


# ------------------------------------------------------------- TabTransformer
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, x, t):
        bce = nn.functional.binary_cross_entropy_with_logits(x, t, reduction='none')
        pt = torch.exp(-bce); at = self.alpha * t + (1 - self.alpha) * (1 - t)
        return (at * (1 - pt) ** self.gamma * bce).mean()


def train_tabtransformer(Xc_tr, Xn_tr, y_tr, Xc_vl, Xn_vl, y_vl, seed, alpha,
                         n_cont, max_epochs=200, bs=32, lr=1e-4, patience=20):
    set_all_seeds(seed)
    m = TabTransformer(categories=(3, 3, 2), num_continuous=n_cont, dim=32, depth=6,
                       heads=8, attn_dropout=0.2, ff_dropout=0.2,
                       mlp_hidden_mults=(4, 2), mlp_act=nn.ReLU(), dim_out=1).to(DEVICE)
    lf = FocalLoss(alpha, 2.0)
    opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    ct = torch.tensor(Xc_tr, dtype=torch.long); nt = torch.tensor(Xn_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    cv = torch.tensor(Xc_vl, dtype=torch.long); nv = torch.tensor(Xn_vl, dtype=torch.float32)
    yv = torch.tensor(y_vl, dtype=torch.float32)
    _g = torch.Generator(); _g.manual_seed(seed)   # shuffle order must not
    dl = DataLoader(TensorDataset(ct, nt, yt), batch_size=bs, shuffle=True,
                    generator=_g, num_workers=0)   # depend on global RNG state
    best, wait, state = float('inf'), 0, None
    for _ in range(max_epochs):
        m.train()
        for cb, nb, yb in dl:
            opt.zero_grad(); loss = lf(m(cb, nb).squeeze(-1), yb); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lf(m(cv, nv).squeeze(-1), yv).item()
        sch.step()
        if vl < best - 1e-5: best, wait, state = vl, 0, copy.deepcopy(m.state_dict())
        else: wait += 1
        if wait >= patience: break
    if state is not None: m.load_state_dict(state)
    return m


def tt_predict(m, Xc, Xn):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(Xc, dtype=torch.long),
                               torch.tensor(Xn, dtype=torch.float32)).squeeze(-1)).cpu().numpy()


# =============================================================== main harness
def run_models(D, tag, with_tabtransformer=True):
    """Train all comparators across five seeds. Returns per-seed artefacts."""
    Xtv, Xte, ytv, yte = D['Xtv'], D['Xte'], D['ytv'], D['yte']
    n_cont = len(D['feats_cont'])
    out = {k: [] for k in ['xgb_default', 'xgb_engineered', 'decision_tree', 'tabtransformer']}
    ref = {}
    proba_by_seed = {'xgb_engineered': {}, 'xgb_default': {}, 'decision_tree': {},
                     'tabtransformer': {}}

    pool_neg, pool_pos = int((ytv == 0).sum()), int((ytv == 1).sum())
    ratio = pool_neg / max(pool_pos, 1)
    alpha_focal = ratio / (1.0 + ratio)

    for seed in SEEDS:
        set_all_seeds(seed)
        Xtr, Xvl, ytr, yvl, itr, ivl = train_test_split(
            Xtv, ytv, np.arange(len(ytv)), test_size=0.15, stratify=ytv, random_state=seed)
        sm = SMOTE(k_neighbors=SMOTE_K, sampling_strategy=SMOTE_STRATEGY, random_state=seed)
        Xtr_s, ytr_s = sm.fit_resample(Xtr, ytr)
        spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)

        # Default XGBoost
        set_all_seeds(seed)
        md = xgb.XGBClassifier(n_jobs=1, random_state=seed, eval_metric='logloss', verbosity=0).fit(Xtr, ytr)
        t, _, _ = f2_threshold_search(yvl, md.predict_proba(Xvl)[:, 1])
        p = md.predict_proba(Xte)[:, 1]
        out['xgb_default'].append(evaluate(yte, p, threshold=t)); proba_by_seed['xgb_default'][seed] = (p, t)

        # Proposed (engineered) XGBoost
        set_all_seeds(seed)
        me = xgb.XGBClassifier(n_jobs=1, n_estimators=200, max_depth=4, learning_rate=0.05,
                               scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                               random_state=seed, eval_metric='aucpr', verbosity=0).fit(Xtr_s, ytr_s)
        pv = me.predict_proba(Xvl)[:, 1]
        t, _, _ = f2_threshold_search(yvl, pv)
        p = me.predict_proba(Xte)[:, 1]
        out['xgb_engineered'].append(evaluate(yte, p, threshold=t)); proba_by_seed['xgb_engineered'][seed] = (p, t)

        # Decision Tree
        set_all_seeds(seed)
        dt = DecisionTreeClassifier(max_depth=5, random_state=seed,
                                    class_weight='balanced').fit(Xtr_s, ytr_s)
        p = dt.predict_proba(Xte)[:, 1]
        out['decision_tree'].append(evaluate(yte, p, threshold=0.60)); proba_by_seed['decision_tree'][seed] = (p, 0.60)

        # TabTransformer
        if with_tabtransformer:
            Xc_tr, Xn_tr = D['Xcat_tv'][itr], D['Xcont_tv'][itr]
            Xc_vl, Xn_vl = D['Xcat_tv'][ivl], D['Xcont_tv'][ivl]
            comb = np.hstack([Xc_tr.astype(np.float32), Xn_tr])
            smt = SMOTE(k_neighbors=SMOTE_K, sampling_strategy=SMOTE_STRATEGY, random_state=seed)
            comb_s, y_s = smt.fit_resample(comb, ytr)
            nc = Xc_tr.shape[1]
            m = train_tabtransformer(comb_s[:, :nc].clip(0, None).astype(np.int64),
                                     comb_s[:, nc:].clip(0.0, 1.0).astype(np.float32), y_s,
                                     Xc_vl, Xn_vl, yvl, seed, alpha_focal, n_cont)
            t, _, _ = f2_threshold_search(yvl, tt_predict(m, Xc_vl, Xn_vl))
            p = tt_predict(m, D['Xcat_te'], D['Xcont_te'])
            out['tabtransformer'].append(evaluate(yte, p, threshold=t))
            proba_by_seed['tabtransformer'][seed] = (p, t)

        if seed == 42:
            ref = dict(model_eng=me, model_def=md, Xtr=Xtr, ytr=ytr, Xvl=Xvl, yvl=yvl,
                       Xtr_s=Xtr_s, ytr_s=ytr_s, spw=spw, pv_eng=pv)

    if not with_tabtransformer:
        out.pop('tabtransformer'); proba_by_seed.pop('tabtransformer')
    return dict(per_seed=out, ref=ref, proba=proba_by_seed,
                pool=dict(n=len(ytv), pos=pool_pos, neg=pool_neg, ratio=round(ratio, 2),
                          pct_pos=round(100 * pool_pos / len(ytv), 1)))


def attendance_rows(D):
    """Checklist #1/#2 — split the attendance baseline into two labelled rows."""
    ai = D['feats_cont'].index('attendance_rate')
    att = D['Xcont_te'][:, ai]
    y = D['yte']
    sc = D['scaler']
    cut = (ATT_RULE_CUT - sc.data_min_[ai]) / (sc.data_max_[ai] - sc.data_min_[ai])
    binary = (att < cut).astype(int)
    ranker = 1.0 - att

    tn, fp, fn, tp = confusion_matrix(y, binary, labels=[0, 1]).ravel()
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    prev = float(y.mean())
    row_binary = dict(
        label='Attendance rule (binary, threshold 0.70)',
        precision=float(precision_score(y, binary, zero_division=0)),
        recall=float(tpr), f2=float(fbeta_score(y, binary, beta=2, zero_division=0)),
        macro_f1=float(f1_score(y, binary, average='macro', zero_division=0)),
        single_point_auc_roc=float((1 + tpr - fpr) / 2),
        single_point_auc_pr=float(tp / (tp + fp) * tpr + prev * (1 - tpr)),
        flagged=int(tp + fp), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        fpr=float(fpr), scaled_cut=float(cut))
    row_ranker = dict(
        label='Attendance ranker (continuous, univariate)',
        auc_roc=float(roc_auc_score(y, ranker)),
        auc_pr=float(average_precision_score(y, ranker)),
        score_vector='1.0 - scaled(attendance_rate)')
    return row_binary, row_ranker


def student_disjoint(D):
    """Checklist #15 — student-disjoint test subset. Reports n=0 honestly."""
    real = pd.read_csv(ARGS.real)
    real['_sid'] = real[ID].str.split('_').str[0]
    tr_sids = set(real[real[YEAR] < SPLIT_YEAR]['_sid'])
    te = real[real[YEAR] >= SPLIT_YEAR]
    disj = te[~te['_sid'].isin(tr_sids)]
    return dict(n_test_records=int(len(te)), n_test_students=int(te['_sid'].nunique()),
                n_train_students=int(len(tr_sids)),
                n_overlap_students=int(te['_sid'].isin(tr_sids).sum() and te[te['_sid'].isin(tr_sids)]['_sid'].nunique()),
                n_disjoint_records=int(len(disj)), n_disjoint_positives=int(disj[TARGET].sum()),
                computable=bool(len(disj) > 0))


def smote_ablation(D):
    """Checklist #10 — isolate Layer 1 (SMOTE). Reports its ACTUAL effect size."""
    Xtv, ytv, Xte, yte = D['Xtv'], D['ytv'], D['Xte'], D['yte']
    arms = {'smote_0.20_asconfigured': 0.20, 'smote_off': None,
            'smote_0.40': 0.40, 'smote_0.50': 0.50}
    res = {}
    for name, strat in arms.items():
        per_seed, added = [], []
        for seed in SEEDS:
            set_all_seeds(seed)
            Xtr, Xvl, ytr, yvl = train_test_split(Xtv, ytv, test_size=0.15,
                                                  stratify=ytv, random_state=seed)
            if strat is None:
                Xtr_s, ytr_s = Xtr, ytr
            else:
                sm = SMOTE(k_neighbors=SMOTE_K, sampling_strategy=strat, random_state=seed)
                Xtr_s, ytr_s = sm.fit_resample(Xtr, ytr)
            added.append(int(len(ytr_s) - len(ytr)))
            spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
            set_all_seeds(seed)
            m = xgb.XGBClassifier(n_jobs=1, n_estimators=200, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                                  random_state=seed, eval_metric='aucpr', verbosity=0).fit(Xtr_s, ytr_s)
            t, _, _ = f2_threshold_search(yvl, m.predict_proba(Xvl)[:, 1])
            per_seed.append(evaluate(yte, m.predict_proba(Xte)[:, 1], threshold=t))
        res[name] = dict(summary=summarise(per_seed),
                         synthetic_rows_added_per_seed=added,
                         train_n=int(len(ytr)), train_pos=int((ytr == 1).sum()))
    return res


def layer2_ablation(D):
    """Checklist #11 — isolate Layer 2 (scale_pos_weight) against Layer 3 (F2 threshold)."""
    Xtv, ytv, Xte, yte = D['Xtv'], D['ytv'], D['Xte'], D['yte']
    arms = {'spw_full__f2_threshold': ('full', 'f2'),      # as published (both layers)
            'spw_1__f2_threshold':    (1.0, 'f2'),         # Layer 3 only
            'spw_full__fixed_0.5':    ('full', 0.50),      # Layer 2 only
            'spw_1__fixed_0.5':       (1.0, 0.50)}         # neither
    res = {}
    for name, (spw_mode, thr_mode) in arms.items():
        per_seed = []
        for seed in SEEDS:
            set_all_seeds(seed)
            Xtr, Xvl, ytr, yvl = train_test_split(Xtv, ytv, test_size=0.15,
                                                  stratify=ytv, random_state=seed)
            sm = SMOTE(k_neighbors=SMOTE_K, sampling_strategy=SMOTE_STRATEGY, random_state=seed)
            Xtr_s, ytr_s = sm.fit_resample(Xtr, ytr)
            spw = (float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
                   if spw_mode == 'full' else 1.0)
            set_all_seeds(seed)
            m = xgb.XGBClassifier(n_jobs=1, n_estimators=200, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                                  random_state=seed, eval_metric='aucpr', verbosity=0).fit(Xtr_s, ytr_s)
            t = (f2_threshold_search(yvl, m.predict_proba(Xvl)[:, 1])[0]
                 if thr_mode == 'f2' else thr_mode)
            per_seed.append(evaluate(yte, m.predict_proba(Xte)[:, 1], threshold=t))
        res[name] = dict(summary=summarise(per_seed),
                         thresholds=[d['threshold'] for d in per_seed])
    return res


def shap_and_sms(D, M, tag):
    """Checklist #3/#4/#18 + Table 5.

    Population is the PROPOSED model's own seed-42 flagged set, and the gold
    ranking is taken from the same model whose attributions are packed.
    """
    me = M['ref']['model_eng']
    p_eng, thr_eng = M['proba']['xgb_engineered'][42]
    yte, Xte, fn = D['yte'], D['Xte'], D['fnames']

    sv = shap.TreeExplainer(me).shap_values(Xte)
    if isinstance(sv, list): sv = sv[1]

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    global_importance = [{'feature': fn[i], 'mean_abs_shap': round(float(mean_abs[i]), 4)}
                         for i in order]

    # --- Checklist #18: signed SHAP for the top three features -------------
    signed = []
    for i in order[:3]:
        col = sv[:, i]
        xcol = Xte[:, i]
        nz = np.abs(col) > 1e-12
        corr = float(np.corrcoef(xcol[nz], col[nz])[0, 1]) if nz.sum() > 2 else float('nan')
        signed.append({
            'feature': fn[i],
            'mean_abs_shap': round(float(mean_abs[i]), 4),
            'mean_signed_shap': round(float(col.mean()), 4),
            'pct_positive_attribution': round(float((col > 0).mean() * 100), 1),
            'corr_feature_value_vs_shap': None if np.isnan(corr) else round(corr, 4),
            'direction_high_value_raises_risk': None if np.isnan(corr) else bool(corr > 0),
            'mean_shap_when_true_dropout': round(float(col[yte == 1].mean()), 4),
            'mean_shap_when_non_dropout': round(float(col[yte == 0].mean()), 4)})

    # --- Populations -------------------------------------------------------
    own_mask = p_eng >= thr_eng
    pops = {'proposed_own_threshold': own_mask}
    if 'tabtransformer' in M['proba'] and 42 in M['proba']['tabtransformer']:
        p_tab, thr_tab = M['proba']['tabtransformer'][42]
        pops['tabtransformer_threshold_ORIGINAL'] = p_tab >= thr_tab

    cont_idx = [fn.index(c) for c in D['feats_cont'] if c in fn]
    bands = S.severity_bands(sv, cont_idx)
    raw_te = D['te']

    abl = {}
    for pname, mask in pops.items():
        n = int(mask.sum())
        if n == 0:
            abl[pname] = {'n': 0, 'note': 'empty population'}
            continue
        sub_sv, sub_y = sv[mask], yte[mask].astype(int)
        sub_rows = raw_te[mask].reset_index(drop=True)
        rp1 = rp2 = da1 = da2 = nda1 = nda2 = 0
        nrp1 = nrp2 = 0
        lens, nfactors, msgs = [], [], []
        for j in range(n):
            row = {c: sub_rows.loc[j, c] for c in CAT}
            sid = sub_rows.loc[j, ID]
            msg, inc, gold = S.build_alert(sid, sub_sv[j], fn, row, bands, 'rank')
            nmsg, ninc, ngold = S.build_alert(sid, sub_sv[j], fn, row, bands, 'canonical')
            rp1 += S.rank_preservation_at_k(inc, gold, 1)
            rp2 += S.rank_preservation_at_k(inc, gold, 2)
            nrp1 += S.rank_preservation_at_k(ninc, gold, 1)
            nrp2 += S.rank_preservation_at_k(ninc, gold, 2)
            da1 += S.directional_agreement_at_k(sub_y[j], sub_sv[j], 1)
            da2 += S.directional_agreement_at_k(sub_y[j], sub_sv[j], 2)
            fixed = [fn.index(g) for g in ngold]
            nda1 += S.directional_agreement_at_k_fixed(sub_y[j], sub_sv[j], fixed, 1)
            nda2 += S.directional_agreement_at_k_fixed(sub_y[j], sub_sv[j], fixed, 2)
            lens.append(len(msg)); nfactors.append(len(inc)); msgs.append(msg)
        abl[pname] = {
            'n': n, 'true_dropouts_in_population': int(sub_y.sum()),
            'rank_preservation': {'rank_preserving_at1': round(rp1 / n, 4),
                                  'rank_preserving_at2': round(rp2 / n, 4),
                                  'naive_fixed_order_at1': round(nrp1 / n, 4),
                                  'naive_fixed_order_at2': round(nrp2 / n, 4)},
            'directional_agreement': {'rank_preserving_at1': round(da1 / n, 4),
                                      'rank_preserving_at2': round(da2 / n, 4),
                                      'naive_fixed_order_at1': round(nda1 / n, 4),
                                      'naive_fixed_order_at2': round(nda2 / n, 4)},
            'table5': {'alerts': n, 'within_160': int(sum(l <= 160 for l in lens)),
                       'len_min': int(min(lens)), 'len_max': int(max(lens)),
                       'len_mean': round(float(np.mean(lens)), 1),
                       'factors_min': int(min(nfactors)), 'factors_max': int(max(nfactors)),
                       'factors_mean': round(float(np.mean(nfactors)), 2)},
            'example_alerts': msgs[:3]}
    return dict(global_importance=global_importance, signed_top3=signed,
                ablation=abl, seed42_threshold=float(thr_eng))


def nf1_real_only_rescore(D, M):
    """NF-1 — the real-only-subset re-scoring, RESTORED and computed.

    An earlier draft of M10 retired this analysis as "redundant" on the strength
    of the load-time guard. The verification report logged that removal as a
    SILENT WITHDRAWAL — a safeguard analysis replaced by a guarantee. It is
    restored here, and it is COMPUTED rather than asserted: the full evaluation
    partition and its real-only subset are scored independently and the two are
    printed side by side.

    Under a correctly bounded synthesiser the two rows are identical, because
    the subset IS the partition. That identity is the evidence; asserting
    redundancy without computing it is what the report objected to.
    """
    te = D['te']
    yte = D['yte'].astype(int)
    p, thr = M['proba']['xgb_engineered'][42]
    mask_real = (te['data_source'] == 'real').to_numpy()
    rows = {}
    for label, mask in (('full_test_partition', np.ones(len(yte), dtype=bool)),
                        ('real_only_subset', mask_real)):
        yy, pp = yte[mask], p[mask]
        binary = (pp >= thr).astype(int)
        cm = confusion_matrix(yy, binary, labels=[0, 1])
        rows[label] = dict(
            n=int(mask.sum()), positives=int(yy.sum()),
            base_rate=round(float(yy.mean()), 4),
            auc_pr=round(float(average_precision_score(yy, pp)), 4),
            auc_roc=round(float(roc_auc_score(yy, pp)), 4),
            precision=round(float(precision_score(yy, binary, zero_division=0)), 4),
            recall=round(float(recall_score(yy, binary, zero_division=0)), 4),
            f2=round(float(fbeta_score(yy, binary, beta=2, zero_division=0)), 4),
            cm=cm.tolist(), threshold=thr)
    identical = rows['full_test_partition'] == rows['real_only_subset']
    return dict(rows=rows, subsets_identical=bool(identical),
                n_synthetic_removed=int((~mask_real).sum()),
                interpretation=(
                    'The real-only subset is the whole evaluation partition; removing '
                    'synthetic rows removes 0 records, so every reported figure is '
                    'already a real-records-only figure. Computed, not asserted.'
                    if identical else
                    'The two rows DIFFER: the evaluation partition is contaminated and '
                    'every headline figure computed on it must be regenerated.'))


def write_rps_artefact(R, path):
    """Q21 — the standalone contribution-evidence artefact.

    FF2 asks for a committed results file that a reviewer can open in a browser
    and read four numbers off. results.json is too large for that to be a
    thirty-second check, so the ablation is additionally written out on its own,
    with RPS and DAS under separate names (M17) and the cross-model sensitivity
    population beside the corrected one.
    """
    s = R['shap_sms']
    own = s['ablation']['proposed_own_threshold']
    cross = s['ablation'].get('tabtransformer_threshold_ORIGINAL', {})
    payload = {
        'artefact': 'SHAPtoSMS ablation — contribution evidence anchor (Q21)',
        'population': 'proposed model\'s own seed-42 flagged records',
        'threshold_used': s['seed42_threshold'],
        'n_evaluated': own['n'],
        'true_dropouts_in_population': own['true_dropouts_in_population'],
        'rank_preservation_RPS': {
            'rps_at_1': own['rank_preservation']['rank_preserving_at1'],
            'rps_at_2': own['rank_preservation']['rank_preserving_at2'],
            'naive_fixed_order_at_1': own['rank_preservation']['naive_fixed_order_at1'],
            'naive_fixed_order_at_2': own['rank_preservation']['naive_fixed_order_at2'],
            'definition': ('does the packed alert name the model\'s own top-k SHAP '
                           'features, in order (M17)')},
        'directional_agreement_DAS': {
            'das_at_1': own['directional_agreement']['rank_preserving_at1'],
            'das_at_2': own['directional_agreement']['rank_preserving_at2'],
            'naive_fixed_order_at_1': own['directional_agreement']['naive_fixed_order_at1'],
            'naive_fixed_order_at_2': own['directional_agreement']['naive_fixed_order_at2'],
            'definition': ('does a top-k attribution carry a sign agreeing with the '
                           'record\'s true label — a DIFFERENT quantity from RPS, and '
                           'the one the original pipeline mislabelled as RPS')},
        'cross_model_sensitivity': {
            'population': 'TabTransformer-threshold flagged records (the earlier, '
                          'cross-model definition)',
            'n_evaluated': cross.get('n'),
            'rps_at_1': cross.get('rank_preservation', {}).get('rank_preserving_at1'),
            'rps_at_2': cross.get('rank_preservation', {}).get('rank_preserving_at2'),
            'naive_fixed_order_at_1': cross.get('rank_preservation', {}).get('naive_fixed_order_at1'),
            'naive_fixed_order_at_2': cross.get('rank_preservation', {}).get('naive_fixed_order_at2'),
            'das_at_1': cross.get('directional_agreement', {}).get('rank_preserving_at1'),
            'das_at_2': cross.get('directional_agreement', {}).get('rank_preserving_at2')},
        'table5_character_compliance': own['table5'],
        'example_alerts': own.get('example_alerts', []),
        'note': ('RPS and DAS are reported under separate names because M17 describes '
                 'rank preservation while the original implementation computed '
                 'directional agreement. Neither is 1.0 by construction.')}
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    return payload


def stats_tests(D, M):
    """McNemar with DIRECTION (checklist #6), paired bootstrap, Cohen's h."""
    yte = D['yte'].astype(int)
    pe, te_ = M['proba']['xgb_engineered'][42]
    prop = (pe >= te_).astype(int)
    comps, rows = {}, []
    for key, label in [('xgb_default', 'Default XGBoost'),
                       ('decision_tree', 'Decision Tree'),
                       ('tabtransformer', 'TabTransformer')]:
        if key not in M['proba'] or 42 not in M['proba'][key]:
            continue
        p, t = M['proba'][key][42]
        comps[label] = (p >= t).astype(int)
    ai = D['feats_cont'].index('attendance_rate')
    sc = D['scaler']
    cut = (ATT_RULE_CUT - sc.data_min_[ai]) / (sc.data_max_[ai] - sc.data_min_[ai])
    comps['Attendance rule (binary)'] = (D['Xcont_te'][:, ai] < cut).astype(int)

    for label, other in comps.items():
        # b = proposed correct & other wrong; c = proposed wrong & other correct
        pc, oc = (prop == yte), (other == yte)
        b = int((pc & ~oc).sum()); c = int((~pc & oc).sum())
        tab = [[int((pc & oc).sum()), b], [c, int((~pc & ~oc).sum())]]
        r = mcnemar_test(tab, exact=True)
        p = float(r.pvalue)
        if b > c:   direction = 'Favours proposed'
        elif c > b: direction = 'Favours comparator'
        else:       direction = 'No directional difference'
        rows.append({'comparison': label, 'p': p, 'b': b, 'c': c, 'direction': direction})
    padj = multipletests([r['p'] for r in rows], alpha=0.05, method='fdr_bh')[1]
    for r, pa in zip(rows, padj):
        r['p_bh'] = float(pa)
        r['decision'] = 'Reject H₀' if pa < 0.05 else 'Retain H₀'

    # Paired bootstrap on AUC-PR delta (proposed - default), 10,000 resamples
    pd_, td = M['proba']['xgb_default'][42]
    rng = np.random.default_rng(42); deltas = []
    n = len(yte)
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        if yte[idx].sum() == 0: continue
        deltas.append(average_precision_score(yte[idx], pe[idx]) -
                      average_precision_score(yte[idx], pd_[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    rec_p = recall_score(yte, prop, zero_division=0)
    rec_d = recall_score(yte, (pd_ >= td).astype(int), zero_division=0)
    h = float(abs(2 * np.arcsin(np.sqrt(rec_p)) - 2 * np.arcsin(np.sqrt(rec_d))))
    return dict(mcnemar=rows,
                bootstrap=dict(ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                               mean_delta=round(float(np.mean(deltas)), 4),
                               spans_zero=bool(lo < 0 < hi), n_resamples=10000),
                cohens_h_recall=round(h, 4),
                recall_proposed=round(float(rec_p), 4), recall_default=round(float(rec_d), 4))


def cv_diagnostic(D):
    Xtv, ytv = D['Xtv'], D['ytv']
    out = {}
    for name, mk in [('proposed', 'eng'), ('default', 'def')]:
        folds = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, va in skf.split(Xtv, ytv):
            set_all_seeds(42)
            if mk == 'eng':
                sm = SMOTE(k_neighbors=SMOTE_K, sampling_strategy=SMOTE_STRATEGY, random_state=42)
                Xs, ys = sm.fit_resample(Xtv[tr], ytv[tr])
                spw = float((ytv[tr] == 0).sum()) / max(float((ytv[tr] == 1).sum()), 1.0)
                m = xgb.XGBClassifier(n_jobs=1, n_estimators=200, max_depth=4, learning_rate=0.05,
                                      scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                                      random_state=42, eval_metric='aucpr', verbosity=0).fit(Xs, ys)
            else:
                m = xgb.XGBClassifier(n_jobs=1, random_state=42, eval_metric='logloss',
                                      verbosity=0).fit(Xtv[tr], ytv[tr])
            folds.append(round(float(average_precision_score(ytv[va], m.predict_proba(Xtv[va])[:, 1])), 4))
        out[name] = {'per_fold': folds, 'mean': round(float(np.mean(folds)), 4),
                     'sd': round(float(np.std(folds)), 4)}
    out['pool_prevalence'] = round(float(ytv.mean()), 4)
    return out


def isotonic_check(D, M):
    """Checklist #7 + supervisor Finding B: is the isotonic AUC-ROC reachable?"""
    me = M['ref']['model_eng']; Xvl, yvl = M['ref']['Xvl'], M['ref']['yvl']
    pe, thr = M['proba']['xgb_engineered'][42]; yte = D['yte']
    iso = IsotonicRegression(out_of_bounds='clip').fit(me.predict_proba(Xvl)[:, 1], yvl)
    pc = iso.predict(pe)
    t, _, _ = f2_threshold_search(yvl, iso.predict(me.predict_proba(Xvl)[:, 1]))
    unc = evaluate(yte, pe, threshold=thr)
    cal = evaluate(yte, pc, threshold=t)
    return dict(seed42_uncalibrated=unc, seed42_isotonic=cal,
                delta_auc_pr=round(cal['auc_pr'] - unc['auc_pr'], 4),
                delta_auc_roc=round(cal['auc_roc'] - unc['auc_roc'], 4),
                auc_roc_rose=bool(cal['auc_roc'] > unc['auc_roc'] + 1e-9),
                note=('isotonic is a monotone transform: AUC-ROC can fall through tie '
                      'creation but cannot rise; a rise indicates a scoring error'))


def perturbation(D, M, trials=100):
    """Checklist #23 — perturbation sensitivity under register-error-scale noise."""
    me = M['ref']['model_eng']
    pe, thr = M['proba']['xgb_engineered'][42]
    Xte, yte, fn = D['Xte'].copy(), D['yte'], D['fnames']
    sc = D['scaler']; fc = D['feats_cont']
    noise_raw = {'attendance_rate': 0.02, 'exam_score': 3.0, 'distance_km': 0.5,
                 'att_delta': 0.02, 'exam_delta': 3.0, 'prior_risk_flag': 0.0,
                 'terms_observed': 0.0}
    scale_span = {c: (sc.data_max_[i] - sc.data_min_[i]) for i, c in enumerate(fc)}
    rng = np.random.default_rng(42)
    base = pe; alld = []
    pos = yte == 1
    base_cls = (base >= thr).astype(int)
    flips_per_case = np.zeros(len(yte), dtype=int)
    flips_per_trial_dropouts = []
    delta_dropouts = []
    for _ in range(trials):
        Xp = Xte.copy()
        for c in fc:
            if c not in fn or noise_raw.get(c, 0.0) == 0.0: continue
            j = fn.index(c)
            sd = noise_raw[c] / max(scale_span[c], 1e-9)
            Xp[:, j] = np.clip(Xp[:, j] + rng.normal(0, sd, len(yte)), 0.0, 1.0)
        pp = me.predict_proba(Xp)[:, 1]
        d = np.abs(pp - base); alld.append(d)
        delta_dropouts.append(d[pos])
        f = ((pp >= thr).astype(int) != base_cls).astype(int)
        flips_per_case += f
        flips_per_trial_dropouts.append(int(f[pos].sum()))
    alld = np.concatenate(alld)
    return dict(trials=trials, mean_abs_delta=round(float(alld.mean()), 4),
                max_abs_delta=round(float(alld.max()), 4),
                dropout_cases=int(pos.sum()),
                dropout_cases_ever_flipped=int((flips_per_case[pos] > 0).sum()),
                max_dropout_flips_in_a_single_trial=int(max(flips_per_trial_dropouts)),
                mean_dropout_flips_per_trial=round(float(np.mean(flips_per_trial_dropouts)), 2),
                mean_abs_delta_dropouts=round(float(np.concatenate(delta_dropouts).mean()), 4))


def error_analysis(D, M):
    """Checklist #22 — which dropout cases are missed across ALL five seeds?"""
    yte = D['yte'].astype(int); te = D['te']
    pos_idx = np.where(yte == 1)[0]
    miss = {int(i): 0 for i in pos_idx}
    for seed in SEEDS:
        p, t = M['proba']['xgb_engineered'][seed]
        for i in pos_idx:
            if p[i] < t: miss[int(i)] += 1
    rows = []
    for i, m in sorted(miss.items(), key=lambda kv: -kv[1]):
        r = te.iloc[i]
        rows.append({'student_id': str(r[ID]), 'year': int(r[YEAR]),
                     'seeds_missed_of_5': m,
                     'attendance_rate_scaled': round(float(D['Xcont_te'][i, 0]), 4),
                     'exam_score_scaled': round(float(D['Xcont_te'][i, 1]), 4),
                     'fee_payment_status': str(r['fee_payment_status']),
                     'grade_level': str(r['grade_level'])})
    return dict(per_case=rows,
                missed_by_all_5=[r['student_id'] for r in rows if r['seeds_missed_of_5'] == 5],
                caught_by_all_5=[r['student_id'] for r in rows if r['seeds_missed_of_5'] == 0])


# ===================================================================== main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', default='data/real_student_data_CLEANED_school.csv')
    ap.add_argument('--synth', default='data/synth_ctgan_s42.csv')
    ap.add_argument('--out', default='results/results.json')
    ap.add_argument('--no-tt', action='store_true')
    ARGS = ap.parse_args()
    globals()['ARGS'] = ARGS

    print('=' * 72); print('EduTrace remediation run'); print('=' * 72)

    # ---------------- baseline feature set (six raw register variables) -----
    D = load_and_prepare(ARGS.real, ARGS.synth, use_lag=False)
    print(f"pool n={len(D['ytv'])} pos={int(D['ytv'].sum())} | test n={len(D['yte'])} pos={int(D['yte'].sum())}")
    M = run_models(D, 'base', with_tabtransformer=not ARGS.no_tt)
    R['pool'] = M['pool']
    R['test'] = {'n': int(len(D['yte'])), 'pos': int(D['yte'].sum()),
                 'neg': int((D['yte'] == 0).sum()),
                 'pct_pos': round(100 * float(D['yte'].mean()), 2)}
    R['table3'] = {k: summarise(v) for k, v in M['per_seed'].items()}
    R['table3_seed42'] = {k: {m: round(v[0][m], 4) for m in
                              ['auc_pr', 'auc_roc', 'f2', 'macro_f1', 'precision', 'recall']}
                          for k, v in M['per_seed'].items()}
    R['seed42_thresholds'] = {k: M['proba'][k][42][1] for k in M['proba']}
    R['seed42_flags'] = {k: int((M['proba'][k][42][0] >= M['proba'][k][42][1]).sum())
                         for k in M['proba']}
    R['seed42_confusion'] = {k: v[0]['cm'] for k, v in M['per_seed'].items()}
    R['scale_pos_weight_seed42'] = round(float(M['ref']['spw']), 2)

    rb, rr = attendance_rows(D)
    R['attendance_binary_rule'] = rb
    R['attendance_continuous_ranker'] = rr
    print(f"  attendance ranker AUC-ROC={rr['auc_roc']:.4f} | binary single-point={rb['single_point_auc_roc']:.4f}")

    R['student_disjoint'] = student_disjoint(D)
    print(f"  student-disjoint test records: {R['student_disjoint']['n_disjoint_records']}")

    R['smote_ablation'] = smote_ablation(D)
    R['layer2_ablation'] = layer2_ablation(D)
    R['stats'] = stats_tests(D, M)
    R['cv'] = cv_diagnostic(D)
    R['isotonic'] = isotonic_check(D, M)
    R['shap_sms'] = shap_and_sms(D, M, 'base')
    R['perturbation_base'] = perturbation(D, M)
    R['error_analysis'] = error_analysis(D, M)

    # ---- NF-1: the restored real-only re-scoring, computed not asserted -----
    R['nf1_real_only_rescore'] = nf1_real_only_rescore(D, M)
    print(f"  NF-1 real-only re-score: subsets identical = "
          f"{R['nf1_real_only_rescore']['subsets_identical']} "
          f"({R['nf1_real_only_rescore']['n_synthetic_removed']} synthetic rows removed)")

    # ---------------- lag-feature arm (checklist #20, #23) ------------------
    print('\n--- lag-feature arm ---')
    D2 = load_and_prepare(ARGS.real, ARGS.synth, use_lag=True)
    M2 = run_models(D2, 'lag', with_tabtransformer=False)
    R['table3_with_lag'] = {k: summarise(v) for k, v in M2['per_seed'].items()}
    R['perturbation_with_lag'] = perturbation(D2, M2)
    R['error_analysis_with_lag'] = error_analysis(D2, M2)
    R['lag_coverage'] = {
        'n_test_records_with_prior_year': int((D2['te']['terms_observed'] > 0).sum()),
        'n_test_records': int(len(D2['te'])),
        'features': LAG_COLS}
    sv2 = shap.TreeExplainer(M2['ref']['model_eng']).shap_values(D2['Xte'])
    if isinstance(sv2, list): sv2 = sv2[1]
    ma2 = np.abs(sv2).mean(axis=0); o2 = np.argsort(ma2)[::-1]
    R['shap_with_lag'] = [{'feature': D2['fnames'][i], 'mean_abs_shap': round(float(ma2[i]), 4)}
                          for i in o2]

    # ---------------- lag-feature FEASIBILITY diagnostic --------------------
    # Under the declared 2025 cut, the whole training pool is academic year
    # 2024, which is every student's FIRST observation. Lag features are
    # therefore identically zero in training and non-zero only at test: a
    # model cannot learn from a constant. Recorded explicitly rather than
    # reported as an improvement.
    _real_lag = add_lag_features(pd.read_csv(ARGS.real))
    _tr = _real_lag[_real_lag[YEAR] < SPLIT_YEAR]
    _te = _real_lag[_real_lag[YEAR] >= SPLIT_YEAR]
    R['lag_feasibility'] = {
        'train_partition_years': sorted(_tr[YEAR].unique().tolist()),
        'train_variance': {c: {'nunique': int(_tr[c].nunique()),
                               'std': round(float(_tr[c].std()), 6)} for c in LAG_COLS},
        'test_variance': {c: {'nunique': int(_te[c].nunique()),
                              'std': round(float(_te[c].std()), 4)} for c in LAG_COLS},
        'learnable_under_2025_cut': False,
        'reason': ('training pool is academic year 2024 only, which is every '
                   'student\'s first panel observation; all lag/delta features '
                   'are identically zero there')}

    # Sensitivity arm: move the temporal cut to 2026 so that 2025 records
    # (which DO have a prior year) enter training and the lag features carry
    # variance. Reported as a sensitivity analysis, not as the main result.
    print('\n--- lag feasibility arm: temporal cut moved to 2026 ---')
    _orig_split = SPLIT_YEAR
    try:
        SPLIT_YEAR = 2026
        D3 = load_and_prepare(ARGS.real, ARGS.synth, use_lag=False)
        M3 = run_models(D3, 'cut2026_nolag', with_tabtransformer=False)
        D4 = load_and_prepare(ARGS.real, ARGS.synth, use_lag=True)
        M4 = run_models(D4, 'cut2026_lag', with_tabtransformer=False)
        _tr4 = D4['tv']
        R['lag_sensitivity_cut2026'] = {
            'train_n': int(len(D4['ytv'])), 'train_pos': int(D4['ytv'].sum()),
            'test_n': int(len(D4['yte'])), 'test_pos': int(D4['yte'].sum()),
            'lag_std_in_training': {c: round(float(_tr4[c].std()), 4) for c in LAG_COLS},
            'without_lag': {k: summarise(v) for k, v in M3['per_seed'].items()},
            'with_lag': {k: summarise(v) for k, v in M4['per_seed'].items()},
            'perturbation_without_lag': perturbation(D3, M3),
            'perturbation_with_lag': perturbation(D4, M4),
            'error_without_lag': error_analysis(D3, M3),
            'error_with_lag': error_analysis(D4, M4)}
    except Exception as e:
        R['lag_sensitivity_cut2026'] = {'error': str(e)}
    finally:
        SPLIT_YEAR = _orig_split

    # ---------------- data-property diagnostics -----------------------------
    real = pd.read_csv(ARGS.real)
    R['data_properties'] = {
        'grade_by_year': pd.crosstab(real[YEAR], real['grade_level']).to_dict(),
        'fee_by_year': pd.crosstab(real[YEAR], real['fee_payment_status']).to_dict(),
        'dropout_by_year': pd.crosstab(real[YEAR], real[TARGET]).to_dict(),
        'n_unique_students': int(real[ID].str.split('_').str[0].nunique()),
        'records_per_student': real[ID].str.split('_').str[0].value_counts()
                                   .value_counts().to_dict(),
        'missing': real.isna().sum().to_dict(),
        'school_identifier_present': bool('school' in ' '.join(real.columns).lower())}

    R['nf1_guard_log'] = list(NF1_LOG)

    with open(ARGS.out, 'w') as f:
        json.dump(R, f, indent=2, default=str)
    print(f'\nWrote {ARGS.out}')

    # ---- Q21: the standalone contribution-evidence artefact ----------------
    rps_path = os.path.join(os.path.dirname(ARGS.out) or '.', 'rps_results.json')
    write_rps_artefact(R, rps_path)
    print(f'Wrote {rps_path}')

    # ---- NF-1 run log, as a committed artefact -----------------------------
    log_path = os.path.join(os.path.dirname(ARGS.out) or '.', 'nf1_guard_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(NF1_LOG) + '\n')
    print(f'Wrote {log_path}')
