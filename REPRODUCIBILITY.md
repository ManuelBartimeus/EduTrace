# Reproducibility note — verification round 2

Recorded 2026-09-07, at the time of the round-2 push. This note documents what
was independently re-run before this branch was pushed, what reproduced exactly,
and what did not. It is disclosure, not a change to any result: nothing in
`results/`, `figs/` or `docs_out/` was regenerated, and every artefact in this
commit is the original locked run.

## Environment used for the re-run

Python 3.11.15, matching the `python` field recorded in
`results/closeout_verdict.json`, with every pin in `requirements.txt` installed
at its exact version (numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy
1.17.1, xgboost 3.2.0, imbalanced-learn 0.14.2, statsmodels 0.15.0, shap 0.51.0,
torch 2.14.0, tab-transformer-pytorch 0.6.1, matplotlib 3.10.9, python-docx
1.2.0). No pin was relaxed or substituted.

## What passed

- `MANIFEST.sha256`: 62 of 62 files verified byte-exact.
- `add_school_identifier.py`: 4 sites, 180 students, 428 records.
- `selftest_nf1.py`: all checks passed, including the negative control — the
  provenance guard raises on a deliberately contaminated test partition.
- `verify.py`: 31/31 checks pass programmatically.
- `consistency_pass.py`: 0 failures.

## What did not reproduce

The pipeline was re-run start to finish from a clean clone into a scratch
directory and compared leaf by leaf against `results/results.json` at a
tolerance of 1e-9. Of 859 shared leaves, 349 differed.

The divergence partitions cleanly:

- Reproduced byte-exact: every data-level, partition-level and rule-based
  quantity — `data_properties`, `pool`, `test`, `student_disjoint`,
  `nf1_guard_log`, `attendance_binary_rule`, `attendance_continuous_ranker`,
  `lag_coverage`, `lag_feasibility`, `scale_pos_weight_seed42`. The test
  partition is n=248 with 7 positives and 0 synthetic rows, as reported.
- Did not reproduce: every quantity derived from a fitted XGBoost or
  TabTransformer model.

## Cause

The model estimators are seeded (`random_state`) but their thread count is not
pinned: `XGBClassifier` is constructed without `n_jobs`/`nthread` in
`EduTrace_Revised_Pipeline.py` and in `closeout.py`, so XGBoost uses all
available cores and accumulates histogram floating-point sums in thread
completion order. The torch-based TabTransformer arm has the same exposure.
Results are therefore reproducible only on a machine with the same effective
thread count as the one that produced the locked run.

This was confirmed directly. Re-running on the same machine, same environment,
same seeds and same data, changing only `OMP_NUM_THREADS`, altered 329 of 858
leaves — and moved two clearance conditions:

| Run                        | C1   | C2   | C3   | C4   | C5   | C6      | Null |
|----------------------------|------|------|------|------|------|---------|------|
| Locked (committed here)    | FAIL | FAIL | FAIL | PASS | PASS | PENDING | 5    |
| Re-run, 12 threads         | FAIL | FAIL | PASS | PASS | PASS | PENDING | 3    |
| Re-run, 1 thread           | FAIL | PASS | FAIL | PASS | PASS | PENDING | 5    |

C1 fails in every configuration. C2 and C3 are not stable across thread counts
at this sample size.

## Consequence, stated plainly

A verifier re-running this pipeline on different hardware should expect the
data-level and provenance results to match exactly, and the model-derived
figures and the C2/C3 clearance outcomes to differ. The numbers reported in the
manuscript trace to the committed artefacts in `results/`; they are not
currently reproducible across machines with differing core counts.

## Recommended remediation

Pin `n_jobs=1` on every XGBoost estimator, set the torch determinism flags, and
regenerate the locked run once under that configuration, updating the reported
figures accordingly. This was deliberately not done as part of this push,
because it changes every reported quantity and is a decision for the authors and
the supervisor rather than a mechanical fix.
