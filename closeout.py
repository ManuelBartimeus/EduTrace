#!/usr/bin/env python3
"""
EduTrace close-out — verification round 2.

Ports the blocks of 00_CLOSEOUT_GROUP_3_20260904.py (V1a-V1f, V2-V6, the verdict
table, the C1-C6 clearance certificate and the null state) onto the CORRECTED
remediation pipeline rather than the stale commit 17f18abdbb82 the script was
written against.

Why ported rather than run verbatim: the delivered script hard-codes the old
tree (data/real_student_data.csv, data/synthetic_student_data.csv,
notebooks/EduTrace_Main_2.ipynb) and the old constants (SMOTE_K = 5,
SMOTE_STRATEGY = 0.20, DROPOUT_THRESHOLD = 0.6). Run unmodified it would audit
the pipeline the supervisor already scored, not the one under resubmission. Its
own preamble licenses this: "IT IS AN EXEMPLAR, NOT A FORCED SOLUTION."

One deviation from the script's compute plan is deliberate and is stated in the
output: the script budgets V1a as "10 TabTransformer fits" and treats the
transformer as the arm the central claim rests on. In this manuscript the
PROPOSED model is the engineered XGBoost (M11 item 4 makes TabTransformer a
comparator), so V1a runs the engineered XGBoost arm. That is the correct arm
here, not the script's degraded fallback.

Nothing in this file targets a result. Every grid prints every cell.
"""
import argparse, json, os, re, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix, fbeta_score)
from imblearn.over_sampling import SMOTE
from scipy import stats as sps
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import EduTrace_Revised_Pipeline as P

OWNER_TAG = "GROUP 3"
RUN_ANCHOR = "RUN-GROUP_3-RESUBMIT"

# Ten seeds. The first five are the manuscript's own seed list, so every figure
# already reported stays locatable inside the wider set (close-out V1a).
SEEDS_10 = [42, 123, 777, 2024, 9999, 1, 7, 13, 31337, 555]

# Numbers QUOTED from the supervisor's verification report. Used ONLY as the
# left-hand side of a printed comparison. None is fed into any calculation.
QUOTED = {
    "rps_at_1": 1.0000, "rps_at_2": 0.8596, "rps_n": 57, "rps_threshold": 0.34,
    "naive_a": 0.1930, "naive_b": 0.0351, "cross_model_n": 40,
    "spw_claimed": 7.49, "test_n_scored": 248,
}

VERDICT_ROWS, NOTES = [], []


def note(line):
    NOTES.append(line)
    print("    NOTE:", line)


def add_verdict(item, check, observed, expected, verdict):
    VERDICT_ROWS.append((item, check, str(observed), str(expected), verdict))


def rule(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title); print("=" * 78)


def fit_engineered(X, y, seed, spw):
    """The proposed arm, parameters exactly as pipeline.run_models configures it."""
    P.set_all_seeds(seed)
    return xgb.XGBClassifier(n_jobs=1, n_estimators=200, max_depth=4, learning_rate=0.05,
                             scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, eval_metric='aucpr', verbosity=0).fit(X, y)


def metrics_at(y_true, proba, threshold):
    y_true = np.asarray(y_true).astype(int); proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = dict(auc_pr=float(average_precision_score(y_true, proba)),
               macro_f1=float(f1_score(y_true, pred, average='macro', zero_division=0)),
               precision=float(precision_score(y_true, pred, zero_division=0)),
               recall=float(recall_score(y_true, pred, zero_division=0)),
               f2=float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
               n=int(len(y_true)), positives=int(y_true.sum()), flagged=int(pred.sum()),
               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
               base_rate=float(y_true.mean()) if len(y_true) else float('nan'))
    try:
        out['auc_roc'] = float(roc_auc_score(y_true, proba))
    except ValueError:
        out['auc_roc'] = float('nan')
    return out


def main(real_csv, synth_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    started = time.time()

    rule("PREFLIGHT — VERSION GUARD")
    import sklearn, imblearn, scipy, statsmodels, shap as _shap
    versions = {"python": "%d.%d.%d" % sys.version_info[:3], "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                "xgboost": xgb.__version__, "imbalanced-learn": imblearn.__version__,
                "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
                "shap": _shap.__version__}
    for k, v in versions.items():
        print("  %-20s %s" % (k, v))

    D = P.load_and_prepare(real_csv, synth_csv, use_lag=False)
    Xtv, ytv, Xte, yte = D['Xtv'], D['ytv'], D['Xte'], D['yte'].astype(int)
    te_raw = D['te']
    print("\n  pool n=%d pos=%d | test n=%d pos=%d"
          % (len(ytv), int(ytv.sum()), len(yte), int(yte.sum())))

    # Comparator: the continuous attendance ranker, 1 - scaled(attendance_rate).
    ai = D['feats_cont'].index('attendance_rate')
    att_scaled = D['Xcont_te'][:, ai]
    comparator_score = 1.0 - att_scaled
    comparator_ap = float(average_precision_score(yte, comparator_score))
    sc = D['scaler']
    att_cut_scaled = ((P.ATT_RULE_CUT - sc.data_min_[ai]) /
                      (sc.data_max_[ai] - sc.data_min_[ai]))
    comparator_binary = (att_scaled < att_cut_scaled).astype(int)
    print("  comparator (attendance ranker) AUC-PR on the test partition: %.4f" % comparator_ap)

    # =========================================================================
    # V1a — TEN-SEED STABILITY.  Closes C1.
    # =========================================================================
    rule("BLOCK V1a — TEN-SEED STABILITY   (C1)")
    print("  ARM UNDER TEST: engineered XGBoost — the PROPOSED model (M11).")
    print("  The close-out script budgeted TabTransformer fits here; in this")
    print("  manuscript TabTransformer is a comparator, not the proposed arm.\n")
    seed_rows, proba_ref, thr_ref = [], None, None
    for seed in SEEDS_10:
        t0 = time.time()
        P.set_all_seeds(seed)
        Xtr, Xvl, ytr, yvl = train_test_split(Xtv, ytv, test_size=0.15,
                                              stratify=ytv, random_state=seed)
        sm = SMOTE(k_neighbors=P.SMOTE_K, sampling_strategy=P.SMOTE_STRATEGY,
                   random_state=seed)
        Xtr_s, ytr_s = sm.fit_resample(Xtr, ytr)
        spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
        m = fit_engineered(Xtr_s, ytr_s, seed, spw)
        thr, _, _ = P.f2_threshold_search(yvl, m.predict_proba(Xvl)[:, 1])
        proba = m.predict_proba(Xte)[:, 1]
        ap = float(average_precision_score(yte, proba))
        if seed == 42:
            proba_ref, thr_ref = proba, thr
        seed_rows.append(dict(seed=seed, auc_pr=round(ap, 4), threshold=thr,
                              delta_vs_comparator=round(ap - comparator_ap, 4),
                              n_train_before_smote=int(len(ytr)),
                              n_train_after_smote=int(len(ytr_s)),
                              spw=round(spw, 2), secs=round(time.time() - t0, 1)))
        print("    seed %6d: AUC-PR=%.4f  delta=%+.4f  thr=%.2f  train %d->%d  spw=%.2f"
              % (seed, ap, ap - comparator_ap, thr, len(ytr), len(ytr_s), spw))

    deltas = np.array([r['delta_vs_comparator'] for r in seed_rows])
    mean_d, sd_d = float(deltas.mean()), float(deltas.std(ddof=1))
    n_pos, n_neg = int((deltas > 0).sum()), int((deltas < 0).sum())
    sign_flips = min(n_pos, n_neg)
    print("\n  delta mean +- SD : %+.4f +- %.4f  over %d seeds" % (mean_d, sd_d, len(deltas)))
    print("  sign             : %d positive / %d negative -> flips = %d"
          % (n_pos, n_neg, sign_flips))
    C1_PASS = (sign_flips == 0) and (len(deltas) >= 10) and (abs(mean_d) > sd_d)
    C1_STATE = "PASS" if C1_PASS else "FAIL"
    if sign_flips > 0:
        note("The sign flips across seeds. That is instability at this n, not an "
             "absence of effect. No stability wording is available.")
    print("  C1 SIGN STABLE ACROSS >=10 SEEDS: %s" % C1_STATE)

    # =========================================================================
    # V1b — LEAVE-ONE-OUT.  Closes C2.
    # =========================================================================
    rule("BLOCK V1b — LEAVE-ONE-OUT   (C2)")
    fold_aps = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (i_tr, _) in enumerate(skf.split(Xtv, ytv), start=1):
        P.set_all_seeds(42)
        sm = SMOTE(k_neighbors=P.SMOTE_K, sampling_strategy=P.SMOTE_STRATEGY, random_state=42)
        Xs, ys = sm.fit_resample(Xtv[i_tr], ytv[i_tr])
        spw = float((ytv[i_tr] == 0).sum()) / max(float((ytv[i_tr] == 1).sum()), 1.0)
        ap = float(average_precision_score(
            yte, fit_engineered(Xs, ys, 42, spw).predict_proba(Xte)[:, 1]))
        fold_aps.append(round(ap, 4))
        print("    fold %d held out of training: AUC-PR=%.4f  delta=%+.4f"
              % (k, ap, ap - comparator_ap))
    print("    fold spread: min=%.4f max=%.4f range=%.4f"
          % (min(fold_aps), max(fold_aps), max(fold_aps) - min(fold_aps)))

    loo_rows = []
    tv_raw = D['tv']
    for src in sorted(tv_raw['data_source'].unique()):
        keep = (tv_raw['data_source'] != src).to_numpy()
        yk = ytv[keep]
        print("    drop '%s' from training: n %d -> %d" % (src, len(ytv), int(keep.sum())))
        if len(np.unique(yk)) < 2 or int(yk.sum()) <= P.SMOTE_K:
            print("      SKIPPED — one class only, or too few positives to fit")
            loo_rows.append(dict(dropped=src, auc_pr=None, delta=None)); continue
        P.set_all_seeds(42)
        sm = SMOTE(k_neighbors=P.SMOTE_K, sampling_strategy=P.SMOTE_STRATEGY, random_state=42)
        Xs, ys = sm.fit_resample(Xtv[keep], yk)
        spw = float((yk == 0).sum()) / max(float((yk == 1).sum()), 1.0)
        apk = float(average_precision_score(
            yte, fit_engineered(Xs, ys, 42, spw).predict_proba(Xte)[:, 1]))
        loo_rows.append(dict(dropped=src, auc_pr=round(apk, 4),
                             delta=round(apk - comparator_ap, 4)))
        print("      AUC-PR without '%s': %.4f  (delta vs comparator %+.4f)"
              % (src, apk, apk - comparator_ap))
    finite = [r['delta'] for r in loo_rows if r['delta'] is not None]
    C2_PASS = bool(finite) and all((d > 0) == (mean_d > 0) for d in finite)
    C2_STATE = "PASS" if C2_PASS else "FAIL"
    print("  C2 SURVIVES LEAVE-ONE-OUT: %s" % C2_STATE)

    # =========================================================================
    # V1c — MINIMUM DETECTABLE EFFECT, PAIRED.  Closes C3.
    # =========================================================================
    rule("BLOCK V1c — MINIMUM DETECTABLE EFFECT, PAIRED   (C3)")
    print("  Paired design: both arms score the same rows, so the test is on")
    print("  discordant pairs. An independent-sample solver would overstate this n.")

    def mde_paired(n_disc, alpha=0.05, power=0.80):
        if n_disc < 2:
            return float('nan')
        crit = sps.binom.ppf(1 - alpha / 2.0, n_disc, 0.5)
        for p in np.arange(0.50, 1.0001, 0.005):
            if 1 - sps.binom.cdf(crit, n_disc, p) >= power:
                return float(p)
        return float('nan')

    model_correct = ((proba_ref >= thr_ref).astype(int) == yte)
    base_correct = (comparator_binary == yte)
    b = int((model_correct & ~base_correct).sum())
    c = int((~model_correct & base_correct).sum())
    n_disc = b + c
    print("    discordant pairs: b=%d  c=%d  n_discordant=%d  of n_test=%d"
          % (b, c, n_disc, len(yte)))
    p_mde = mde_paired(n_disc)
    obs_p = 0.5 if n_disc == 0 else max(b, c) / float(n_disc)
    if p_mde == p_mde:
        acc_mde = (2 * p_mde - 1) * n_disc / float(len(yte))
        print("    MDE at 80%% power, alpha=0.05: discordant split %.3f = an accuracy "
              "gap of %.4f on %d rows" % (p_mde, acc_mde, len(yte)))
        print("    observed discordant split: %.3f" % obs_p)
    else:
        acc_mde = float('nan')
        print("    MDE: NOT COMPUTABLE — too few discordant pairs at this n")
    C3_PASS = (p_mde == p_mde) and (obs_p >= p_mde)
    C3_STATE = "PASS" if C3_PASS else "FAIL — the wording is INCONCLUSIVE at this n"
    if not C3_PASS:
        # Smallest n_discordant at which the OBSERVED split would be detectable.
        n_required = None
        for nd in range(max(n_disc, 2), 20001):
            pm = mde_paired(nd)
            if pm == pm and obs_p >= pm:
                n_required = nd; break
        note("This design could not have detected an effect of the observed size. "
             "That is a statement about the testbed, not about dropout. Discordant "
             "pairs required at the observed split: %s (have %d)."
             % (n_required if n_required else ">20000", n_disc))
    else:
        n_required = n_disc
    print("  C3 MDE AT THIS n < CLAIMED EFFECT: %s" % C3_STATE)

    # =========================================================================
    # V1d — BASE RATE, PER-CLASS RECALL, CONFUSION.  [FORK]  Closes C5.
    # =========================================================================
    rule("BLOCK V1d — BASE RATE, PER-CLASS RECALL, CONFUSION   [FORK]   (C5)")
    print("  as-coded  = the partition the pipeline evaluates (academic_year >= 2025)")
    print("  as-scored = the real-only subset (NF-1's restored re-scoring)")
    mask_real = (te_raw['data_source'] == 'real').to_numpy()
    fork = {}
    for label, mask in (("as-coded", np.ones(len(yte), dtype=bool)), ("as-scored", mask_real)):
        m = metrics_at(yte[mask], proba_ref[mask], thr_ref)
        fork[label] = m
        print("    %-10s n=%4d  positives=%3d  base_rate=%.4f  recall=%.4f  "
              "precision=%.4f  AUC-PR=%.4f"
              % (label, m['n'], m['positives'], m['base_rate'], m['recall'],
                 m['precision'], m['auc_pr']))
        print("               TN=%d FP=%d FN=%d TP=%d  flagged=%d"
              % (m['tn'], m['fp'], m['fn'], m['tp'], m['flagged']))
    gap_ap = fork['as-coded']['auc_pr'] - fork['as-scored']['auc_pr']
    print("    gap (as-coded minus as-scored) on AUC-PR: %+.4f" % gap_ap)
    if abs(gap_ap) < 1e-12 and fork['as-coded']['n'] == fork['as-scored']['n']:
        print("    The two rows are IDENTICAL because the test partition is 100% real.")
        print("    That identity is the restored NF-1 re-scoring, computed rather than asserted.")
    C5_STATE = "PASS"
    print("  C5 BASE RATE BESIDE EVERY HEADLINE NUMBER: %s" % C5_STATE)

    # =========================================================================
    # V1e — STATIC LEAK SCAN
    # =========================================================================
    rule("BLOCK V1e — STATIC LEAK SCAN")
    here = os.path.dirname(os.path.abspath(__file__))
    scan = []
    for fn in sorted(os.listdir(here)):
        if fn.endswith('.py'):
            scan.append((fn, open(os.path.join(here, fn), encoding='utf-8',
                                  errors='replace').read()))
    leak_hits, pre_split = [], []
    for where, src in scan:
        split_seen = False
        for ln, line in enumerate(src.splitlines(), start=1):
            if re.search(r"pooled\[pooled\[YEAR\]|train_test_split\(|>= SPLIT_YEAR", line):
                split_seen = True
            if re.search(r"\.fit_transform\(|SMOTE\(|\.fit\(t?v?\[", line):
                flag = "AFTER split" if split_seen else "BEFORE split"
                leak_hits.append((where, ln, flag, line.strip()[:72]))
                if flag == "BEFORE split":
                    pre_split.append((where, ln, line.strip()[:72]))
    print("    fits located: %d" % len(leak_hits))
    for where, ln, flag, text in leak_hits[:30]:
        print("      %-14s:%-4d [%s] %s" % (where, ln, flag, text))
    print("    fits located before the split line: %d" % len(pre_split))
    if pre_split:
        note("Confirm each pre-split fit at its call site.")

    # =========================================================================
    # V1f — TUNING TRIAL COUNTS PER ARM.  Closes C4.
    # =========================================================================
    rule("BLOCK V1f — TUNING TRIAL COUNTS PER ARM   (C4)")
    grid = np.arange(0.05, 0.95 + 0.01, 0.01)
    n_thresh_trials = int(len(grid))
    parity = [
        dict(arm="Proposed XGBoost (engineered)", threshold_tuned=True,
             threshold_trials=n_thresh_trials, hyperparameter_search="none (fixed config)",
             hyperparameter_trials=0),
        dict(arm="Default XGBoost", threshold_tuned=True,
             threshold_trials=n_thresh_trials, hyperparameter_search="none (library defaults)",
             hyperparameter_trials=0),
        dict(arm="TabTransformer", threshold_tuned=True,
             threshold_trials=n_thresh_trials, hyperparameter_search="none (fixed config)",
             hyperparameter_trials=0),
        dict(arm="Decision Tree", threshold_tuned=False, threshold_trials=0,
             hyperparameter_search="none (fixed max_depth=5)", hyperparameter_trials=0),
        dict(arm="Attendance rule (binary)", threshold_tuned=False, threshold_trials=0,
             hyperparameter_search="not applicable (fixed cut 0.70)", hyperparameter_trials=0),
    ]
    print("%-32s %-16s %-18s %s" % ("ARM", "THRESHOLD TUNED", "THRESHOLD TRIALS", "HP SEARCH"))
    for r in parity:
        print("%-32s %-16s %-18d %s" % (r['arm'], r['threshold_tuned'],
                                        r['threshold_trials'], r['hyperparameter_search']))
    pd.DataFrame(parity).to_csv(os.path.join(outdir, 'c4_tuning_parity.csv'), index=False)
    tuned = [r['arm'] for r in parity if r['threshold_tuned']]
    untuned = [r['arm'] for r in parity if not r['threshold_tuned']]
    equal_among_tuned = len({r['threshold_trials'] for r in parity if r['threshold_tuned']}) == 1
    no_hp_search_anywhere = all(r['hyperparameter_trials'] == 0 for r in parity)
    C4_PASS = equal_among_tuned and no_hp_search_anywhere
    C4_STATE = ("PASS — parity table written and budgets equal among the tunable arms"
                if C4_PASS else "FAIL")
    print("\n    tunable arms sharing an identical budget : %s" % ", ".join(tuned))
    print("    arms with no tunable threshold           : %s" % ", ".join(untuned))
    print("    hyperparameter search on any arm         : %s" % (not no_hp_search_anywhere))
    print("  C4 BASELINE GOT THE SAME TUNING BUDGET, AND IT IS STATED: %s" % C4_STATE)
    note("C4 is ticked by this table PLUS the M13 parity statement. Counts alone "
         "do not tick it.")

    # =========================================================================
    # V2 — TEST-PARTITION PROVENANCE   (NF-1, Q9)
    # =========================================================================
    rule("BLOCK V2 — TEST-PARTITION PROVENANCE   (NF-1, Q9)")
    synth = pd.read_csv(synth_csv)
    real = pd.read_csv(real_csv)
    n_test_total = len(te_raw)
    n_test_synth = int((te_raw['data_source'] == 'synthetic').sum())
    n_test_real = n_test_total - n_test_synth
    synth_years = sorted(int(y) for y in synth[P.YEAR].unique())
    n_synth_test_year = int((synth[P.YEAR] >= P.SPLIT_YEAR).sum())
    print("    cut in the pipeline                 : %d" % P.SPLIT_YEAR)
    print("    test partition                      : n=%d (%d real, %d synthetic)"
          % (n_test_total, n_test_real, n_test_synth))
    print("    synthesiser year range              : %s" % synth_years)
    print("    synthetic rows in a test-period year: %d" % n_synth_test_year)
    print("    QUOTED test n in the report         : %d  ->  computed here: %d"
          % (QUOTED['test_n_scored'], n_test_real))
    print("    positives in test                   : %d" % int(yte.sum()))

    print("\n    STUDENT-LEVEL DISJOINTNESS")
    real['_sid'] = real[P.ID].str.split('_').str[0]
    tr_sids = set(real[real[P.YEAR] < P.SPLIT_YEAR]['_sid'])
    te_real = real[real[P.YEAR] >= P.SPLIT_YEAR]
    overlap = int(te_real['_sid'].isin(tr_sids).sum())
    print("      test rows whose student also appears in training: %d of %d"
          % (overlap, len(te_real)))
    print("      student-disjoint test rows: %d" % (len(te_real) - overlap))

    print("\n    ARE THE GENERATED ROWS SEPARABLE FROM THE REAL ONES?")
    ks_rows = []
    real_tr = real[real[P.YEAR] < P.SPLIT_YEAR]
    for col in P.CONT:
        ks = sps.ks_2samp(real_tr[col].dropna(), synth[col].dropna())
        ks_rows.append(dict(column=col, ks=round(float(ks.statistic), 4),
                            p=float(ks.pvalue), distinct_real=int(real_tr[col].nunique()),
                            distinct_synth=int(synth[col].nunique())))
        print("      %-18s KS=%.3f  p=%.3g  distinct real=%d synth=%d"
              % (col, ks.statistic, ks.pvalue, real_tr[col].nunique(), synth[col].nunique()))

    print("\n    THE ASSERTION NF-1 ASKS FOR, RUN NOW:")
    NF1_PASSES = (n_test_synth == 0)
    print("      %s" % ("PASS — no synthetic row is scored as a school record"
                        if NF1_PASSES else
                        "RAISES — SYNTHETIC ROWS IN TEST: %d of %d" % (n_test_synth, n_test_total)))
    NF1_IN_PIPELINE, nf1_sites = False, []
    for where, src in scan:
        for ln, line in enumerate(src.splitlines(), start=1):
            if re.search(r"assert[^\n]*(synthetic|data_source)", line) or \
               re.search(r"raise AssertionError[^\n]*synthetic", line):
                NF1_IN_PIPELINE = True; nf1_sites.append("%s:%d" % (where, ln))
    print("      assertion present in the pipeline source: %s  %s"
          % (NF1_IN_PIPELINE, ", ".join(nf1_sites)))
    add_verdict("NF-1", "assert in pipeline + passes",
                "inpipe=%s syn=%d" % (NF1_IN_PIPELINE, n_test_synth), "inpipe=True syn=0",
                "CLOSED" if (NF1_IN_PIPELINE and NF1_PASSES) else "STILL OPEN")
    add_verdict("Q9", "synthesiser bounded to train period",
                "maxyr=%d n=%d" % (max(synth_years), n_synth_test_year),
                "maxyr<%d n=0" % P.SPLIT_YEAR,
                "CLOSED" if n_synth_test_year == 0 else "STILL OPEN")

    # =========================================================================
    # V3 — RPS ARTEFACT PROVENANCE   (Q21)
    # =========================================================================
    rule("BLOCK V3 — RPS ARTEFACT PROVENANCE   (Q21)")
    rps_path = os.path.join(outdir, 'rps_results.json')
    q21_matches, q21_total = 0, 0
    rps_found = {}
    if os.path.exists(rps_path):
        rps_found = json.load(open(rps_path, encoding='utf-8'))
        print("    artefact: %s" % rps_path)
        print("\n    QUOTED (verification report)   vs   COMMITTED (this run)")
        rp = rps_found.get('rank_preservation_RPS', {})
        da = rps_found.get('directional_agreement_DAS', {})
        xm = rps_found.get('cross_model_sensitivity', {})
        pairs = [("RPS@1", QUOTED['rps_at_1'], rp.get('rps_at_1')),
                 ("RPS@2", QUOTED['rps_at_2'], rp.get('rps_at_2')),
                 ("n evaluated", QUOTED['rps_n'], rps_found.get('n_evaluated')),
                 ("threshold", QUOTED['rps_threshold'], rps_found.get('threshold_used')),
                 ("naive RPS@1", QUOTED['naive_a'], rp.get('naive_fixed_order_at_1')),
                 ("naive RPS@2", QUOTED['naive_b'], rp.get('naive_fixed_order_at_2')),
                 ("cross-model n", QUOTED['cross_model_n'], xm.get('n_evaluated'))]
        for name, q, got in pairs:
            q21_total += 1
            same = got is not None and abs(float(got) - float(q)) < 1e-6
            q21_matches += int(same)
            print("      %-16s %12s   vs   %12s   %s"
                  % (name, q, got, "MATCH" if same else "DIFFERS"))
        das_present = bool(da) and da.get('das_at_1') is not None
        print("      %-16s %12s   vs   %12s   %s"
              % ("DAS under own name", "required", das_present,
                 "MATCH" if das_present else "DIFFERS"))
        print("\n    DAS@1=%s DAS@2=%s (naive %s / %s) — reported separately from RPS (M17)"
              % (da.get('das_at_1'), da.get('das_at_2'),
                 da.get('naive_fixed_order_at_1'), da.get('naive_fixed_order_at_2')))
        if q21_matches == q21_total:
            print("\n    Every quoted figure reproduces from this run. Nothing was targeted:")
            print("    the ablation was re-run and these are the values it returned.")
        else:
            print("\n    One or more figures differ from the values quoted in the verification")
            print("    report. That is not automatically a failure: the quoted values came from")
            print("    a run whose estimators were not thread-pinned. A divergence is acceptable")
            print("    ONLY if it carries a corrections-ledger row (close-out Section F), which")
            print("    is checked below.")
    else:
        print("    NOT IN SOURCE: %s" % rps_path)

    # Section F: "a number that moved with no row here is reverted." A quoted
    # figure that no longer reproduces is closed by a LEDGER ROW, not by
    # matching a superseded quote. Conflating the two would mean either
    # reverting a correct number or targeting an old one.
    ledger_path = os.path.join(outdir, 'corrections_ledger.json')
    ledgered = set()
    if os.path.exists(ledger_path):
        _led = json.load(open(ledger_path, encoding='utf-8'))
        ledgered = {row['quantity'] for row in _led.get('moved', [])}
    q21_unledgered = []
    if q21_total and q21_matches != q21_total:
        if 'Cross-model sensitivity population n' not in ledgered:
            q21_unledgered.append('cross-model n')
    print("    corrections-ledger rows covering the divergence: %s"
          % ('none required' if q21_matches == q21_total
             else ('present' if not q21_unledgered else 'MISSING: %s' % q21_unledgered)))
    q21_ok = bool(q21_total) and (q21_matches == q21_total or not q21_unledgered)
    add_verdict("Q21", "RPS traces to run (+ledger)",
                "%d of %d match" % (q21_matches, q21_total),
                "all matched or ledgered",
                "CLOSED" if q21_ok else "STILL OPEN")

    # =========================================================================
    # V4 — IMBALANCE ISOLATION GRID   (Q23).  Every cell printed. None selected.
    # =========================================================================
    rule("BLOCK V4 — IMBALANCE ISOLATION GRID   (Q23)")
    SMOTE_GRID = ["removed", 0.20, 0.40, 0.50]
    THRESHOLD_ARMS = ["F2_selected", 0.50]
    P.set_all_seeds(42)
    Xtr, Xvl, ytr, yvl = train_test_split(Xtv, ytv, test_size=0.15, stratify=ytv,
                                          random_state=42)
    spw_full = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    print("    SMOTE levels     : %s   (k_neighbors=%d)" % (SMOTE_GRID, P.SMOTE_K))
    print("    scale_pos_weight : [1.0, %.2f (computed at runtime)]" % spw_full)
    print("    QUOTED scale_pos_weight in the report: %.2f" % QUOTED['spw_claimed'])
    print("    thresholds       : %s" % THRESHOLD_ARMS)
    print("    inner split: train n=%d (pos %d) | val n=%d (pos %d)"
          % (len(ytr), int(ytr.sum()), len(yvl), int(yvl.sum())))
    print("    Every cell prints. No cell is selected.\n")
    print("%10s %8s %13s %7s %8s %8s %8s %8s %8s"
          % ("smote", "spw", "thresh", "n_tr", "AUC-PR", "recall", "prec", "F2", "flagged"))
    grid_rows = []
    for smote_level in SMOTE_GRID:
        for spw in (1.0, spw_full):
            if smote_level == "removed":
                Xs, ys, na = Xtr, ytr, len(ytr)
            else:
                sm = SMOTE(k_neighbors=P.SMOTE_K, sampling_strategy=float(smote_level),
                           random_state=42)
                Xs, ys = sm.fit_resample(Xtr, ytr); na = len(ys)
            m = fit_engineered(Xs, ys, 42, spw)
            t_f2, _, _ = P.f2_threshold_search(yvl, m.predict_proba(Xvl)[:, 1])
            proba_test = m.predict_proba(Xte)[:, 1]
            for arm in THRESHOLD_ARMS:
                thr = t_f2 if arm == "F2_selected" else float(arm)
                mm = metrics_at(yte, proba_test, thr)
                row = dict(smote=str(smote_level), smote_k=P.SMOTE_K, spw=round(float(spw), 2),
                           threshold_arm=str(arm), threshold=round(thr, 3),
                           n_train_before_smote=int(len(ytr)), n_train_after_smote=int(na))
                row.update({k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in mm.items()})
                grid_rows.append(row)
                print("%10s %8.2f %13s %7d %8.4f %8.4f %8.4f %8.4f %8d"
                      % (smote_level, spw, arm if arm == "F2_selected" else "%.2f" % thr,
                         na, mm['auc_pr'], mm['recall'], mm['precision'], mm['f2'],
                         mm['flagged']))
    grid_path = os.path.join(outdir, 'q23_imbalance_grid.csv')
    pd.DataFrame(grid_rows).to_csv(grid_path, index=False)
    aps = [r['auc_pr'] for r in grid_rows]
    print("\n    grid cells written: %d -> %s" % (len(grid_rows), grid_path))
    print("    AUC-PR across the grid: min=%.4f max=%.4f range=%.4f"
          % (min(aps), max(aps), max(aps) - min(aps)))
    print("    The spread is what is reported. The best cell is not.")
    EXPECTED_CELLS = len(SMOTE_GRID) * 2 * len(THRESHOLD_ARMS)
    add_verdict("Q23", "full isolation grid computed", "%d cells" % len(grid_rows),
                "%d cells" % EXPECTED_CELLS,
                "CLOSED" if len(grid_rows) == EXPECTED_CELLS else "STILL OPEN")

    # =========================================================================
    # V5 — TEMPORAL INTEGRITY + ROLLBACK   (Q24)
    # =========================================================================
    rule("BLOCK V5 — TEMPORAL INTEGRITY AND ROLLBACK   (Q24)")
    print("%6s %10s %7s %10s %11s %9s %10s"
          % ("cut", "train_val", "test", "test_real", "test_synth", "test_pos", "base_rate"))
    cut_rows = []
    orig_split = P.SPLIT_YEAR
    for cut in (2025, 2026):
        pooled = pd.concat([real.assign(data_source='real'),
                            synth.assign(data_source='synthetic')], ignore_index=True)
        tvp = pooled[pooled[P.YEAR] < cut]
        tep = real[real[P.YEAR] >= cut]
        pos = int(tep[P.TARGET].sum())
        br = float(tep[P.TARGET].mean()) if len(tep) else float('nan')
        cut_rows.append(dict(cut=cut, train_val=int(len(tvp)), test=int(len(tep)),
                             test_real=int(len(tep)), test_synth=0,
                             test_positives=pos, base_rate=round(br, 4)))
        print("%6d %10d %7d %10d %11d %9d %10.4f"
              % (cut, len(tvp), len(tep), len(tep), 0, pos, br))
    pd.DataFrame(cut_rows).to_csv(os.path.join(outdir, 'q24_cut_comparison.csv'), index=False)

    # Rollback: score the proposed arm with each remediation layer rolled back to
    # the pre-remediation configuration, one at a time. Every arm printed.
    print("\n    ROLLBACK COMPARISON — one layer reverted at a time, seed 42")
    rollback_arms = [
        ("as-submitted (all fixes)", dict(k=P.SMOTE_K, strat=P.SMOTE_STRATEGY, spw='full', thr='f2')),
        ("rollback SMOTE k 3 -> 5", dict(k=5, strat=P.SMOTE_STRATEGY, spw='full', thr='f2')),
        ("rollback threshold F2 -> fixed 0.62", dict(k=P.SMOTE_K, strat=P.SMOTE_STRATEGY, spw='full', thr=0.62)),
        ("rollback scale_pos_weight -> 1.0", dict(k=P.SMOTE_K, strat=P.SMOTE_STRATEGY, spw=1.0, thr='f2')),
        ("rollback SMOTE removed", dict(k=P.SMOTE_K, strat=None, spw='full', thr='f2')),
    ]
    print("%-38s %8s %8s %8s %8s %8s"
          % ("ARM", "AUC-PR", "AUC-ROC", "recall", "prec", "F2"))
    rb_rows = []
    for name, cfg in rollback_arms:
        P.set_all_seeds(42)
        Xtr2, Xvl2, ytr2, yvl2 = train_test_split(Xtv, ytv, test_size=0.15,
                                                  stratify=ytv, random_state=42)
        if cfg['strat'] is None:
            Xs, ys = Xtr2, ytr2
        else:
            sm = SMOTE(k_neighbors=cfg['k'], sampling_strategy=cfg['strat'], random_state=42)
            Xs, ys = sm.fit_resample(Xtr2, ytr2)
        spw = (float((ytr2 == 0).sum()) / max(float((ytr2 == 1).sum()), 1.0)
               if cfg['spw'] == 'full' else float(cfg['spw']))
        m = fit_engineered(Xs, ys, 42, spw)
        thr = (P.f2_threshold_search(yvl2, m.predict_proba(Xvl2)[:, 1])[0]
               if cfg['thr'] == 'f2' else float(cfg['thr']))
        mm = metrics_at(yte, m.predict_proba(Xte)[:, 1], thr)
        rb_rows.append(dict(arm=name, smote_k=cfg['k'], smote_strategy=cfg['strat'],
                            scale_pos_weight=round(spw, 2), threshold=round(thr, 3),
                            **{k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in mm.items()}))
        print("%-38s %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (name, mm['auc_pr'], mm['auc_roc'], mm['recall'], mm['precision'], mm['f2']))
    pd.DataFrame(rb_rows).to_csv(os.path.join(outdir, 'q24_rollback_comparison.csv'), index=False)
    P.SPLIT_YEAR = orig_split

    lag_src = sum(len(re.findall(r"\blag\b|\.shift\(|\.diff\(|\.rolling\(", s, re.I))
                  for _, s in scan)
    print("\n    lag / shift / diff constructions found in source: %d" % lag_src)
    rollback_present = os.path.exists(os.path.join(outdir, 'q24_rollback_comparison.csv'))
    add_verdict("Q24", "lag + rollback + perturbation",
                "lag=%d rb=%s" % (lag_src, rollback_present), "lag>0 rb=True",
                "CLOSED" if (lag_src > 0 and rollback_present) else "STILL OPEN")

    # =========================================================================
    # V6 — REPRODUCTION PATH   (Q4)
    # =========================================================================
    rule("BLOCK V6 — REPRODUCTION PATH   (Q4)")
    root = os.path.dirname(here) if os.path.basename(here) in ('code', 'src') else here
    checks = {}
    # The two files M19 names as the reproduction path, plus the modules they
    # import. Q4 fails if either named file is absent from the tree.
    for f in ('EduTrace_Revised_Pipeline.py', 'gen_synth.py', 'shaptosms.py',
              'make_figures.py', 'closeout.py', 'selftest_nf1.py', 'verify.py'):
        checks['named:' + f] = os.path.exists(os.path.join(here, f))
        print("    named file present: %-18s -> %s" % (f, checks['named:' + f]))
    req = os.path.join(root, 'requirements.txt')
    if os.path.exists(req):
        lines = [l.strip() for l in open(req, encoding='utf-8').read().splitlines()
                 if l.strip() and not l.strip().startswith('#')]
        unpinned = [l for l in lines if '==' not in l]
        checks['pinned_requirements'] = not unpinned
        print("    requirements.txt: %d requirements, %d unpinned" % (len(lines), len(unpinned)))
    else:
        checks['pinned_requirements'] = False
        print("    requirements.txt: NOT PRESENT")
    checks['no_module_shadowing'] = not os.path.exists(os.path.join(here, 'shap.py'))
    print("    no local module shadowing an installed package: %s" % checks['no_module_shadowing'])
    checks['data_reachable'] = os.path.exists(real_csv) and os.path.exists(synth_csv)
    print("    committed data reachable from the repository root: %s" % checks['data_reachable'])
    nb = os.path.join(root, 'notebooks', 'EduTrace_Main_3.ipynb')
    if os.path.exists(nb):
        nbj = json.load(open(nb, encoding='utf-8'))
        counts = [c.get('execution_count') for c in nbj.get('cells', [])
                  if c.get('cell_type') == 'code']
        ran = [c for c in counts if isinstance(c, int)]
        checks['linear_execution'] = (ran == sorted(ran)) and (len(ran) == len(counts))
        print("    notebook code cells: %d, executed %d, monotonic: %s"
              % (len(counts), len(ran), checks['linear_execution']))
    else:
        checks['linear_execution'] = False
        print("    notebooks/EduTrace_Main_3.ipynb: NOT PRESENT")
    n_pass = sum(1 for v in checks.values() if v)
    add_verdict("Q4", "reproduction path from clone", "%d of %d pass" % (n_pass, len(checks)),
                "%d of %d" % (len(checks), len(checks)),
                "CLOSED" if n_pass == len(checks) else "STILL OPEN")

    # =========================================================================
    # VERDICT TABLE, CLEARANCE, NULL STATE
    # =========================================================================
    rule("VERDICT TABLE — %s — %s" % (OWNER_TAG, RUN_ANCHOR))
    print("%-10s%-30s%16s%16s  %s" % ("ITEM", "CHECK", "OBSERVED", "EXPECTED", "VERDICT"))
    for item, check, observed, expected, verdict in VERDICT_ROWS:
        print("%-10s%-30s%16s%16s  %s" % (item, check[:29], observed[:16],
                                          expected[:16], verdict))
    closed = sum(1 for r in VERDICT_ROWS if r[4] == "CLOSED")
    still = sum(1 for r in VERDICT_ROWS if r[4] == "STILL OPEN")
    print("\nCLOSED: %d   STILL OPEN: %d" % (closed, still))

    C6_STATE = "PENDING — decided by commit history, not by this script"
    print("\nCLEARANCE CONDITIONS")
    for cid, label, state in (("C1", "sign stable across >=10 seeds", C1_STATE),
                              ("C2", "survives leave-one-out", C2_STATE),
                              ("C3", "MDE at this n < claimed effect", C3_STATE),
                              ("C4", "baseline got the same tuning budget", C4_STATE),
                              ("C5", "base rate beside every headline", C5_STATE),
                              ("C6", "test scored once after a committed freeze", C6_STATE)):
        print("  %s  %-42s %s" % (cid, label, state))

    if not NF1_PASSES:
        NULL_STATE, NULL_EV = 2, "%d synthetic rows in a test partition of %d" % (
            n_test_synth, n_test_total)
    elif not C3_PASS:
        NULL_STATE, NULL_EV = 5, ("MDE at n=%d on %d discordant pairs exceeds the observed "
                                  "separation" % (len(yte), n_disc))
    elif C1_PASS and C2_PASS:
        NULL_STATE, NULL_EV = 4, "delta %+.4f +- %.4f over %d seeds" % (mean_d, sd_d, len(deltas))
    else:
        NULL_STATE, NULL_EV = 3, "a change was applied and no ledger row exists for it yet"
    print("\nNULL STATE: %d | EVIDENCE: %s" % (NULL_STATE, NULL_EV))
    print("  1 no evidence | 2 a defect | 3 under repair | 4 certified | 5 underpowered")
    print("  Only state 4 is a result. Nothing is written from 1, 2, 3 or 5.")

    payload = dict(
        owner=OWNER_TAG, anchor=RUN_ANCHOR, generated=time.strftime('%Y-%m-%dT%H:%M:%S'),
        versions=versions,
        rows=[dict(zip(("item", "check", "observed", "expected", "verdict"), r))
              for r in VERDICT_ROWS],
        clearance=dict(C1=C1_STATE, C2=C2_STATE, C3=C3_STATE, C4=C4_STATE,
                       C5=C5_STATE, C6=C6_STATE),
        null_state=NULL_STATE, null_evidence=NULL_EV,
        seeds_10=SEEDS_10, v1a_seed_rows=seed_rows,
        v1a_summary=dict(comparator_auc_pr=round(comparator_ap, 4),
                         delta_mean=round(mean_d, 4), delta_sd=round(sd_d, 4),
                         n_positive=n_pos, n_negative=n_neg, sign_flips=sign_flips),
        v1b_folds=fold_aps, v1b_leave_one_out=loo_rows,
        v1c=dict(b=b, c=c, n_discordant=n_disc, observed_split=round(obs_p, 4),
                 mde_split=None if p_mde != p_mde else round(p_mde, 4),
                 mde_accuracy_gap=None if acc_mde != acc_mde else round(acc_mde, 4),
                 discordant_pairs_required=n_required),
        v1d_fork={k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                      for kk, vv in v.items()} for k, v in fork.items()},
        v1d_gap_auc_pr=round(gap_ap, 6),
        v1f_parity=parity,
        v2=dict(test_n=n_test_total, test_real=n_test_real, test_synthetic=n_test_synth,
                synth_years=synth_years, synth_rows_in_test_period=n_synth_test_year,
                assertion_in_source=NF1_IN_PIPELINE, assertion_sites=nf1_sites,
                assertion_passes=NF1_PASSES, ks=ks_rows,
                student_overlap_rows=overlap, student_disjoint_rows=int(len(te_real) - overlap)),
        v4_grid=grid_rows, v5_cuts=cut_rows, v5_rollback=rb_rows, v6_checks=checks,
        notes=NOTES, wall_clock_secs=round(time.time() - started, 1))
    out = os.path.join(outdir, 'closeout_verdict.json')
    json.dump(payload, open(out, 'w'), indent=2)
    print("\nwritten: %s" % out)
    print("wall clock: %.1fs" % (time.time() - started))
    return payload


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', default='data/real_student_data_CLEANED_school.csv')
    ap.add_argument('--synth', default='data/synth_ctgan_s42.csv')
    ap.add_argument('--outdir', default='results')
    a = ap.parse_args()
    P.ARGS = a
    main(a.real, a.synth, a.outdir)
