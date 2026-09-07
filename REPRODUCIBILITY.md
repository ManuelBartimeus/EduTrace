# Reproducibility note

Supersedes the note recorded at the round-2 push, which reported a defect this
document reports as repaired. The defect was a bug, it has been fixed, and the
run has been regenerated once under the fix. One genuine limitation survives the
repair and is recorded in full under *Limits of what has been established*:
bit-exact reproduction is bound to the platform that produced the run.

## What was wrong

The round-2 push verified byte-exactly against its manifest and passed every
checklist audit, but the pipeline did not reproduce its own results on a second
machine. Re-running from a clean clone produced **349 of 859 differing leaves**.

The divergence partitioned cleanly. Everything data-level, partition-level and
rule-based reproduced exactly — `data_properties`, `pool`, `test`,
`student_disjoint`, `nf1_guard_log`, both attendance-baseline rows,
`lag_feasibility`, `scale_pos_weight_seed42`. Everything derived from a fitted
XGBoost or TabTransformer did not.

**Cause.** The estimators were seeded but their thread count was not pinned.
`XGBClassifier` was constructed without `n_jobs`, so XGBoost used every available
core and accumulated histogram floating-point sums in thread-completion order;
the torch-based TabTransformer had the same exposure, and its DataLoader shuffle
drew from global RNG state. The same seed on a machine with a different core
count therefore produced different trees, different weights, and different
reported metrics.

This was confirmed by a single-variable experiment: same machine, same
environment, same seeds, same data, changing only `OMP_NUM_THREADS`. **329 of 858
leaves moved, and two clearance conditions changed verdict** — C3 flipped from
FAIL to PASS and the null state moved from 5 to 3.

That last point is why this could not be left as a disclosure. A reviewer running
the clean-clone self-test on their own hardware could have obtained a *different
clearance certificate* from the one the manuscript reports. Close-out checklist
item 27 requires every reported number to trace to one locked run; a run that
depends on the core count of the machine that produced it does not.

## What was done

Repair, not declaration. The close-out is explicit that a defect is not a
limitation, and Q4 asks whether a reader who clones the repository lands on the
reported numbers — a question a disclosure note cannot answer.

- `n_jobs=1` on every XGBoost estimator in the pipeline, the close-out and the
  leave-one-site-out validation.
- `n_jobs=1` on the scikit-learn estimators that fan out (`RandomForestClassifier`
  and `cross_val_score` in `school_structure.py`, the 1-NN adversary in
  `smote_nnaa.py`).
- torch pinned to a single thread with `use_deterministic_algorithms` enabled, and
  the training DataLoader given its own seeded `torch.Generator` so the shuffle
  order no longer depends on global RNG state. All of this lives in
  `set_all_seeds()`, so it applies everywhere the pipeline seeds itself.
- The run regenerated **once**, under the pinned settings, and every downstream
  artefact and figure regenerated from it.

`selftest_determinism.py` verifies the property rather than asserting it: it fits
the model arms in separate subprocesses that differ only in their thread
environment and compares the probability vectors bit for bit. **A reviewer should
run it on their own machine** — that is the check that licenses the claim.

```
python selftest_determinism.py      # 7 arms x 3 thread settings, all identical
```

## What moved, and what did not

The repair changed the TabTransformer arm and nothing else. Its seed-42
F2-selected threshold moved from 0.50 to 0.49, which moved its flagged count from
40 to 83 records, which moved everything downstream of it.

**Every XGBoost figure is bit-identical** between the pre-repair and post-repair
runs on the machine that produced them, including every quantity the proposed
model contributes: AUC-PR 0.1167 ± 0.0444, the seed-42 threshold of 0.34, the
57-record ablation population, RPS@1 1.0000 and RPS@2 0.8596, and all six
clearance verdicts.

Fourteen manuscript-facing quantities moved and thirty-one held. Every one of the
fourteen has a row in `results/corrections_ledger.md`, and the complete
leaf-by-leaf diff is committed as `results/corrections_ledger.json`.

**One of the fourteen is substantive and is flagged here rather than left to the
table.** The McNemar comparison against TabTransformer reversed direction: it ran
b = 22, c = 41 in the comparator's favour, and now runs b = 53, c = 31 in the
proposed model's favour, remaining significant after Benjamini-Hochberg
correction (p = 0.0428). Two sentences asserted that *every* comparison favoured
the comparator; both are corrected, in R4 and in R10.

Correcting R10 is a deliberate departure from forward-fix block FF6, which said
R10 should stay exactly as it is. FF6's reason was that R10 was well written and
nothing had asked for changes to it. That reason does not extend to leaving a
sentence standing that the corrected run has made untrue. The change is the
minimum required for accuracy and is recorded in `CHANGELOG.md`.

The direction of this movement happens to favour the study. That makes the
disclosure more important, not less, and it is why the change is stated in the
manuscript text as well as in the ledger.

## Limits of what has been established

The determinism self-test passed on the machine that produced this run. That
machine has two cores, so its `OMP_NUM_THREADS=4` setting does not exercise four
real threads. The evidence that the pinning works on wider hardware is the
mechanism — every reduction is now single-threaded, so there is no completion
order to vary — and the self-test is provided precisely so the property can be
confirmed on hardware that can vary it. **Run it on a many-core machine before
relying on the claim.**

### The self-test has now been run on wider hardware, and it passed

It was run on a 12-core machine (Windows 11, x86-64, Python 3.11.15 and also
3.14.6, under the pins in `requirements.txt`): **7 arms x 3 thread settings, all
identical**. Thread-count independence holds where the originating machine could
not exercise it. The full pipeline repeated under those two interpreters produced
a byte-identical `results.json`, so the pinning is not sensitive to the
interpreter either.

### Bit-exact reproduction is nevertheless bound to the platform

That same machine does **not** reproduce the locked run. A full re-run from a
clean clone, under the pinned environment, differs from `results/results.json` in
**343 of 859 leaves**.

The divergence partitions exactly as the round-2 defect did. Everything
data-level, partition-level and rule-based reproduces bit-exactly —
`data_properties`, `pool`, `test`, `student_disjoint`, `nf1_guard_log`,
`lag_coverage`, `lag_feasibility`, `scale_pos_weight_seed42` and both
attendance-baseline rows, 104 leaves in all. Everything derived from a fitted
estimator does not.

The cause is not the one repaired above, and not any of the obvious candidates:

- **Not thread count.** `selftest_determinism.py` passes on this machine.
- **Not the interpreter.** Python 3.11.15 and 3.14.6 give byte-identical output.
- **Not the `n_jobs` pinning.** Removing it moves the seed-42 threshold further
  from the reported 0.34, to 0.64, rather than closer.
- **Not `tree_method`.** None of `auto`, `exact`, `hist` or `approx` reproduces
  the committed model.

It is the XGBoost build. Refitting the proposed model exactly as
`export_models.py` does, from inputs verified identical by SHA-256 — the same
450 to 476 SMOTE frame, the same `scale_pos_weight` of 7.49, the same seed —
produces a structurally different model: **all 200 trees differ, beginning at
tree 0**, whose root split moves from `f0 < 0.910700023` to `f0 < 0.571399987`.
That is a different greedy split choice, not floating-point drift. With
`subsample=0.8` and `colsample_bytree=0.8` the per-tree row and column samples
are drawn inside XGBoost, and `random_state` does not pin that draw across
builds.

The committed artefacts are internally consistent, and a reader can check that
directly: `models/xgboost.pkl` from the locked run scores AUC-PR 0.1033 and
AUC-ROC 0.5969 on the test partition, which is exactly what `results/results.json`
reports. The locked run is a real run, and its model, its results file and its
logs agree with one another. What does not hold is the stronger claim that a
reader on different hardware re-derives it.

**So Q4 has a narrower answer than this document previously gave.** A reader who
clones the repository lands on the reported numbers only on the platform that
produced them. Elsewhere they land on the reported partition and the reported
data exactly, and on model-derived numbers that differ. The locked environment
records the library pins and Python 3.11.15; it does not record the operating
system, the CPU or the XGBoost wheel, and it should. That omission is a defect in
this note rather than in the run, and it is why the platform cannot simply be
restated here.

### How to check a reproduction on other hardware

Bit-exact comparison of `results.json` is the right check only on the recorded
platform. Elsewhere, check instead that:

- every leaf under the ten data- and partition-level sections named above is
  bit-identical. Those sections carry the sampling frame, the temporal split and
  the NF-1 guard, and any difference there is a real defect rather than a build
  difference;
- `selftest_nf1.py` and `selftest_determinism.py` both pass;
- model-derived metrics agree in magnitude and direction rather than to the last
  digit. On the 12-core machine above, the median absolute difference across the
  343 moved leaves is 0.020.

One consequence is stated here rather than left to the reader to discover:
because the clearance conditions are discrete, they can flip. On that machine C2
moved from FAIL to PASS and Q21 re-opened, giving CLOSED 5 and STILL OPEN 1
against the reported CLOSED 6. **A reviewer on different hardware can therefore
obtain a different clearance certificate.** The repair in this document removed
the core-count dependence that caused this within a machine. It did not, and
could not, remove it across machines.

Exact bit-level reproduction still assumes the pinned library versions in
`requirements.txt`. Floating-point results can differ across CPU architectures and
BLAS builds even single-threaded; the pins narrow that, and the evidence above
shows they do not close it.
