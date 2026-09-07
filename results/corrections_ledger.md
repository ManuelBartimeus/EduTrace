### Ledger scope

Leaf-level differences across all regenerated artefacts: **58**.

- `closeout_verdict.json`: 10 changed leaves
- `loso_validation.json`: 0 changed leaves
- `results.json`: 42 changed leaves
- `rps_results.json`: 6 changed leaves
- `school_structure.json`: 0 changed leaves
- `smote_nnaa.json`: 0 changed leaves

The complete leaf-by-leaf diff is committed as `results/corrections_ledger.json`. The table below is not a sample of it: it is every quantity that appears in the Methods or Results sections, which is the set a reader can check against the manuscript.

**14 of 45 manuscript-facing quantities moved; 31 held.**

| WHERE | QUANTITY | CAUSE | CONFIRMED BY | FIX APPLIED | BEFORE | AFTER |
|---|---|---|---|---|---|---|
| R2 / Table 3 | TabTransformer AUC-PR | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `0.2387` | `0.2533` |
| R2 / Table 3 | TabTransformer recall | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `0.5143` | `0.4857` |
| R5 | Cross-model sensitivity population n | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `40` | `83` |
| R4 / Table 4 | McNemar vs TabTransformer: discordant b | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `22` | `53` |
| R4 / Table 4 | McNemar vs TabTransformer: discordant c | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `41` | `31` |
| R4 / Table 4 | McNemar vs TabTransformer: BH-adjusted p | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `0.0451` | `0.0428` |
| R4 / Table 4 / R10 | McNemar vs TabTransformer: DIRECTION | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `Favours comparator` | `Favours proposed` |
| Table 3 | TabTransformer AUC-ROC | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `0.7248` | `0.6964` |
| Table 3 | TabTransformer F2 | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `0.2572` | `0.2039` |
| Table 3 | TabTransformer precision | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `0.0901` | `0.1499` |
| R7 / Figure 7 | TabTransformer seed-42 threshold | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `0.5` | `0.49` |
| R5 | TabTransformer seed-42 flagged count | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `40` | `83` |
| M19 | TabTransformer checkpoint size (MB) | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `1.25` | `1.19` |
| M19 | Proposed XGBoost checkpoint size (MB) | Unpinned estimator thread count (non-deterministic reduction order) | OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped | n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once | `0.29` | `0.27` |


### Quantities that did NOT move

- R2 / Table 3 — Proposed XGBoost AUC-PR (5-seed mean): `0.1167`
- R2 / Table 3 — Proposed XGBoost AUC-PR (SD): `0.0444`
- R2 / Table 3 — Default XGBoost AUC-PR: `0.1469`
- R2 / Table 3 — Decision Tree AUC-PR: `0.0774`
- R2 / Table 3 — Proposed XGBoost recall: `0.4`
- R2 / R4 / R10 — Proposed seed-42 AUC-PR: `0.1033`
- R2 / R10 — Proposed seed-42 AUC-ROC: `0.5969`
- R5 / R7 / Table 6 — Seed-42 operating threshold (proposed): `0.34`
- R5 / Table 5 — Records flagged at the seed-42 threshold: `57`
- M8 / M12 — scale_pos_weight at seed 42: `7.49`
- R5 / Figure 8 — RPS@1 (rank-preserving): `1.0`
- R5 / Figure 8 — RPS@2 (rank-preserving): `0.8596`
- R5 / Figure 8 — RPS@1 (naive fixed order): `0.193`
- R5 / Figure 8 — DAS@1 (rank-preserving): `0.7368`
- R5 / Table 5 — Ablation population n (proposed own threshold): `57`
- R10 — SMOTE 0.50 arm AUC-PR: `0.1343`
- R10 — Layer 2 spw=1 arm AUC-PR: `0.1302`
- R10 — SMOTE k=3 NNAA at seed 42: `0.9118`
- R3 — Ten-seed delta vs attendance ranker (mean): `-0.0047`
- R3 — Ten-seed delta sign changes: `2`
- R3 — Leave-one-out, synthetic dropped (AUC-PR): `0.1447`
- R4 — Discordant pairs vs the attendance rule: `69`
- R4 — Discordant pairs required at 80% power: `786`
- R8 — LOSO SCH01 AUC-PR: `0.1016`
- R8 — LOSO SCH02 AUC-PR: `0.4172`
- R4 / Table 4 — McNemar vs Default XGBoost: discordant b/c: `17/20`
- R4 / Table 4 — McNemar vs Attendance rule: discordant b/c: `31/38`
- R12 — C1 verdict: `FAIL`
- R12 — C2 verdict: `FAIL`
- R12 — C3 verdict: `FAIL — the wording is INCONCLUSIVE at this n`
- R12 — Null state: `5`

### Cause

Estimator thread count was not pinned. XGBoost and torch accumulate floating-point sums in thread-completion order, so the same seed on a machine with a different core count produced different trees, weights and metrics. Confirmed by changing only OMP_NUM_THREADS: 329 of 858 leaves moved and two clearance conditions changed verdict.

### Fix

n_jobs=1 on every XGBoost and scikit-learn estimator; torch pinned to one thread with deterministic algorithms enabled and a seeded DataLoader generator; the run regenerated once under the pinned settings. selftest_determinism.py verifies thread-count independence on any machine.
