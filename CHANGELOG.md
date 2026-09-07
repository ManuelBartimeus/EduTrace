# EduTrace — remediation changelog, verification round 2

Checklist item → what changed → where it is reflected. Item numbers refer to `CHECKLIST.md`,
extracted from `GROUP 3_VERIFICATION 2 FEEDBACK.docx` and the close-out script.

---

## Section F — CORRECTIONS LEDGER (numbers that moved)

| CAUSE | CONFIRMED BY | FIX APPLIED | BEFORE | AFTER |
|---|---|---|---|---|
| The register carried no school identifier, so the site count could not be stated and leave-one-site-out validation was blocked (Q2, M3, M18) | The group recovered the identifier from the four participating schools while the HuSSREC approval and site access were live | `add_school_identifier.py` attaches `school_id` / `school_name` to every record, asserting first that the supplied ranges are contiguous, non-overlapping and partition the register exactly | `data_properties.school_identifier_present = False` | `= True` |
| Same | Same | `data_properties.missing` gains the two identifier columns | key absent | `school_id: 0`, `school_name: 0` |

**Three leaves moved, all of them metadata. No model-bearing number moved.**

The pipeline was re-run on the school-augmented register and diffed leaf-by-leaf against the
run immediately preceding it: **3 differing leaves out of the entire artefact**, listed above,
and **0 of them model-bearing**. Every metric, threshold, confusion matrix, ablation cell, SHAP
value, RPS/DAS figure and perturbation statistic is bit-identical.

That is by construction, not by luck: the identifier is attached as a record-level attribute and
the feature set in M7 is an explicit list of six register variables, so the model never sees it.
Adding school membership as a predictor would have been a modelling change nobody asked for, and
on this panel it would amount to fitting a school-level intercept on 20–80 students per site.

**The earlier rounds' ledger remains empty, and that still matters.** Before the identifier
arrived, the pipeline was re-run unchanged against the cleaned register and diffed against the
`results.json` delivered in the remediation bundle: **0 differing leaves**. It was re-run again
after the NF-1 fix: **0 differing leaves**, plus two added keys. So every figure the supervisor
reconciled to the digit still stands, and the only thing that has changed about the numbers in
this entire remediation is that one boolean flipped from False to True.

Per F12, a number that moved with no row here would be reverted. The three that moved have rows.

---

## A. Fatal weight (5)

### 1. NF-1 — the retired safeguard

| Req | Change | Reflected in |
|---|---|---|
| 1.1 | `assert_test_partition_is_real()` added to the load path of `EduTrace_Revised_Pipeline.py`. Raises `SyntheticInTestPartition`; never warns. Cannot be bypassed by a caller. | `EduTrace_Revised_Pipeline.py`; **Methods M10** |
| 1.1a | **The evaluation partition is now built from the pooled frame, not from the real frame alone.** Previously `te` was sliced from `real`, so the assertion was tautological — it tested for a condition its own construction had already excluded. That is the unfalsifiable guarantee the report objected to. | `EduTrace_Revised_Pipeline.py` `load_and_prepare()`; **Methods M10** |
| 1.2 | The guard prints and archives its run-log line on every run: `NF-1 GUARD: test partition n=248 \| synthetic rows=0 \| real rows=248 \| positives=7 (2.82%) — PASS` | `results/nf1_guard_log.txt`; quoted verbatim in **Methods M10** and **Results R1** |
| 1.3 | `selftest_nf1.py` — three checks: the clean path prints 248 rows, the Q9 year bound holds, and a **negative control** injects one synthetic record into the test period and confirms the guard raises. | `selftest_nf1.py`; **Methods M10**; notebook §1 |
| 1.4 / 1.5 | **The real-only-subset re-scoring is restored**, and computed rather than asserted. `nf1_real_only_rescore()` scores the full partition and its real-only subset independently. Both return n=248, AUC-PR 0.1033, AUC-ROC 0.5969, precision 0.0351, recall 0.2857 — identical, because 0 records are removed. The "redundant and has been retired" sentence is deleted. | `results.json → nf1_real_only_rescore`; **Methods M10**; **Results R1** |

> **On FF3's literal wording.** FF3 asks to confirm that "deleting the assertion line makes the
> run fail". That cannot hold and the discrepancy is stated rather than papered over: on a
> correctly bounded synthesiser there is nothing for the guard to catch, so deleting it changes
> nothing. The negative control in `selftest_nf1.py` establishes what FF3 is actually after —
> that the guard is load-bearing rather than decorative.

### 2. Q21 — contribution evidence anchor

| Req | Change | Reflected in |
|---|---|---|
| 2.1–2.5 | `results/rps_results.json` written as a **standalone artefact** — a reviewer reads the four anchor figures off one file rather than hunting inside the consolidated archive. | `results/rps_results.json`; **Results R5**; **Methods M21** |
| 2.2 | RPS@1 **1.0000**, RPS@2 **0.8596**, n **57**, threshold **0.34** | verified 7/7 by close-out block V3 |
| 2.3 | naive fixed-order baseline **0.1930** / **0.0351** | same artefact |
| 2.4 | DAS reported under its own name: DAS@1 0.7368, DAS@2 0.9825 (naive 0.0877 / 0.0877) | same artefact; **Methods M17** |
| 2.5 | cross-model sensitivity population, n **40** | same artefact |

> **On the "no result was targeted" tension** (flagged as F1 in `CHECKLIST.md`). Q21 named the
> figures the artefact must report, while Section G item 24 forbids targeting a result. The
> tension resolved itself: the ablation was re-run honestly, with nothing tuned toward those
> values, and **all seven quoted figures reproduced exactly**. Had any differed, the difference
> would have been reported as a finding with a ledger row.

### 3. Q22 — negative-result process gate
No separate action; lifts when Q10, Q21, Q23 close on a pushed commit. All three are closed in
code here. **No Discussion of the negative result has been drafted, no Conclusion has been drafted,
and the "attendance rule outperformed machine learning" sentence appears nowhere** — the
consistency pass checks for it by regex in both documents (`consistency_pass.py`, check C).
**R10 was not touched.** New unfavourable findings were placed in a new R12 rather than edited
into R10.

### 4. Q2 — sampling and generalisability
Student-side work was already complete. A **countersignature block** for Dr Osei has been added at
the end of Methods, naming the three declaration sentences (M9 — 0 of 248; R8 — no substitute
reported; M18 — the inference boundary) with a dated signature line. The consistency pass verifies
that no unqualified "held-out test performance" appears in either document and that the required
"next-year performance for students with prior-year register history" qualifier is present.
**This item cannot be closed by the group — it needs Dr Osei's signature.**

### 5. Q10 — inflated-baseline illusion
The M20 Baseline-currency defence sentence *"All received identical data access and tuning budgets
(M11, M13)"* is **deleted**. It is replaced with the corrected statement, matching M11 and M13:
identical data access for all five; an identical 91-point threshold grid for the three learners
with a tunable threshold (proposed XGBoost, default XGBoost, TabTransformer); and the two that
could not be tuned (Decision Tree at its fixed default, Attendance rule fixed by construction).
The per-arm budget is tabulated in `results/c4_tuning_parity.csv`.
**FF3's self-test executed:** every sentence in Methods containing "identical" was read; none now
claims parity across all five comparators.

---

## B. Weight 3

### 6. Q23 — imbalance isolation
| Req | Change | Reflected in |
|---|---|---|
| 6.2–6.4 | SMOTE isolation, the Layer 2 × Layer 3 2×2, and the corrected-population ablation all run and committed. | `results.json`; `results/q23_imbalance_grid.csv` |
| 6.5 | Committed code shows **SMOTE k = 3, scale_pos_weight 7.49 (computed at runtime), threshold 0.34** — the values the revision reports, not the stale commit's k=5 / 13.48 / 0.62. | close-out V1a and V4 print the runtime `spw` per seed |
| 6.6 | **Full 16-cell grid** (4 SMOTE levels × 2 `scale_pos_weight` × 2 threshold arms), every cell printed and written to CSV before any commentary. AUC-PR spans 0.0760 to 0.1534. **No cell selected.** | `results/q23_imbalance_grid.csv`; **Results R12**; notebook §6 |

### 7. Q24 — remediation protocol
| Req | Change | Reflected in |
|---|---|---|
| 7.2 | **Rollback table**: each layer reverted one at a time at the declared 2025 cut, seed 42, every arm reported — including two that run against the submitted configuration (reverting `scale_pos_weight` to 1.0 raises AUC-PR to 0.1233 from 0.1033; reverting SMOTE k to 5 raises recall to 0.7143 from 0.2857). | `results/q24_rollback_comparison.csv`; **Results R12** |
| 7.2 | **Lag-feature run** at the declared cut, plus the 2026 sensitivity arm. | `results.json → lag_feasibility`, `lag_sensitivity_cut2026`; **Results R8** |
| 7.2 | **Perturbation re-run** on both arms. | `results.json → perturbation_base`, `perturbation_with_lag`; **Results R8** |
| — | Cut comparison, 2025 vs 2026. | `results/q24_cut_comparison.csv` |

---

## C. Weight 1 — regressions

### 8. Q4 — reproduction path
| Req | Change |
|---|---|
| 8.1 | Both files M19 names now exist: `EduTrace_Revised_Pipeline.py` at the root (renamed from `pipeline.py`, so M10/M19/M21 are correct as written) and `notebooks/EduTrace_Main_3.ipynb`. |
| 8.2 | Every path is relative to the repository root. No `PROJECT_DIR`, no mounted Drive. `gen_synth.py`, `make_figures.py`, `smote_nnaa.py`, `verify.py` and `closeout.py` all repointed from `repro/` and `codebase/model_codebase/` to `data/` and `results/`. |
| 8.3 | `requirements.txt` — **11 requirements, 0 unpinned**, all `==`. SDV split into `requirements-synthesis.txt` because it constrains pandas to a different major version; regenerating the supplement is not needed to reproduce any reported figure. |
| 8.4 | No module shadows an installed package; there is no local `shap.py`. |
| 8.5 | The notebook is committed with a **linear execution_count over 9 code cells** (1…9, none unexecuted), executed in order by `build_notebook.py`. |
| 8.6 | No cell mutates a results store in place; the notebook holds no mutable global state and reads committed artefacts. |
| 8.7 | Figures 4–8 and S1–S4 regenerated into `figs/`. |

Close-out block V6: **11 of 11 checks pass**.

### 9. Q9 — provenance firewall
The synthesiser is bounded to the training period (`gen_synth.py` conditions `academic_year`
exclusively on pre-split years and asserts it). The committed supplement carries **max
academic_year 2024** and **0 rows in a test-period year**, against the 52 contaminating rows at the
stale commit. Verified by close-out block V2 and by `selftest_nf1.py` check (b).

---

## D. Clearance certificate C1–C6 and the null state

Run by `closeout.py`; archived in `results/closeout_verdict.json`; reported in **Results R12**.

| | Condition | Verdict | Evidence |
|---|---|---|---|
| **C1** | sign stable across ≥10 seeds | **FAIL** | 10 seeds run; delta vs the attendance ranker −0.0047 ± 0.0437; 8 negative / 2 positive; **2 sign changes**. No stability wording is available, and none is used. |
| **C2** | survives leave-one-out | **FAIL** | Dropping the synthetic supplement *raises* AUC-PR to 0.1447; dropping the real records collapses it to 0.0213. The 180 real records carry the result. |
| **C3** | MDE at this n < claimed effect | **FAIL** | 69 discordant pairs; observed split 0.551; detectable split at 80% power 0.680. **≈786 discordant pairs would be required.** Reported as INCONCLUSIVE at this n, never as absence. |
| **C4** | equal tuning budget, stated | **PASS** | 91-point grid shared by the three tunable arms; no hyperparameter search on any arm; parity table at `results/c4_tuning_parity.csv`; stated in M13 and M20. |
| **C5** | base rate beside every headline | **PASS** | Test base rate 0.0282 printed beside every level; the as-coded/as-scored fork prints identically. |
| **C6** | scored once after a committed freeze | **PENDING** | Settled by repository history, not by any script. Closes when the tagged release is pushed. |

**Null state: 5 — UNDERPOWERED** (was **2 — a confirmed defect**). The contamination that produced
state 2 is not present in this pipeline, and the guard makes its absence checkable on every run.
State 5 licenses no positive finding — and equally no claim of equivalence or absent effect.
Reported in R12.

---

## D2. Q2 — the school identifier (recovered after the round-2 submission)

The site count was the one outstanding recommendation with an expiry: recoverable only while the
HuSSREC approval and site access were live, and a permanent limitation once the manuscript is
submitted. It was recovered. Four sites:

| Code | School | ID range | Students | Records | Dropouts | Base rate | Test records | Test events |
|---|---|---|---|---|---|---|---|---|
| SCH01 | Al Huda Islamic and JHS | S1–S80 | 80 | 225 | 11 | 0.0489 | 145 | 5 |
| SCH02 | Tawjeed Islamic and JHS | S81–S130 | 50 | 109 | 9 | 0.0826 | 59 | 2 |
| SCH03 | Ayaarno M/A Primary and JHS | S131–S150 | 20 | 37 | 3 | 0.0811 | 17 | **0** |
| SCH04 | Buokrom Block A M/A Primary and JHS | S151–S180 | 30 | 57 | 2 | 0.0351 | 27 | **0** |

> **Transcription note.** The second range was supplied as "S801 to S130". S801 does not exist —
> student numbers run S1 to S180 with no gaps — and the four ranges are contiguous and sum to
> exactly 180 students (80 + 50 + 20 + 30). It is read as **S81**, and `add_school_identifier.py`
> asserts contiguity, non-overlap and exact partition before writing anything, so a wrong reading
> would have failed loudly rather than silently mislabelling records.

### What it unlocked

**Leave-one-site-out validation (`loso_validation.py`)** — the analysis M18 named as one of two
ingredients for Tier 2. Each fold trains on the training-period records of the other sites and
tests on the test-period records of the held-out site, so the temporal firewall holds inside every
fold. The synthetic supplement is excluded from LOSO training, because CTGAN was fitted on the
whole training-period pool including the held-out site and training on it would leak that site.

- **SCH01 held out** — AUC-PR 0.1016 ± 0.0140 against a 0.0345 prevalence floor (2.95× lift)
- **SCH02 held out** — AUC-PR 0.4172 ± 0.1684 against a 0.0339 floor (12.31× lift)
- **SCH03, SCH04 held out** — **NOT COMPUTABLE.** Neither site records a dropout event in the
  test period, so AUC-PR and AUC-ROC are undefined. Reported as not computable rather than filled
  with a floor value, a zero, or a pooled substitute.

Computable for 2 of 4 sites, resting on 7 events between them, with within-fold seed spread
(±0.1684) of the same order as the between-fold gap. Reported in R8 as a direction, not an
estimate. Nothing in it contradicts the within-site result, and it is far too thin to lift the
claim tier.

### Three findings only the identifier could expose

1. **Two of four sites have no test-period dropout event at all.** This is why LOSO is half
   computable, and it is a sampling fact the manuscript could not previously state.
2. **Site is confounded with gender** (Cramér's V = 0.4396). Ayaarno M/A is single-sex, and two
   sites are 100% Unpaid on `fee_payment_status`. `gender_Female` is the fourth-ranked feature by
   mean |SHAP| (0.322), so that attribution is **partly a site effect** and cannot be read as a
   behavioural property of students. Recorded in Table 2b and R6.
3. **Site is NOT materially recoverable from the six register features** — 0.6707 cross-validated
   accuracy against a 0.5257 majority-class baseline, a lift of 1.28. This one is favourable: it
   is what keeps the LOSO folds meaningful, because a model denied the site identifier cannot
   readily infer it.

Site base rates span 0.0351 to 0.0826, a 2.4-fold spread across four sites of 20 to 80 students,
so the sites are not interchangeable replicates and pooled figures are dominated by the largest.

### Where it is reflected

| Location | Change |
|---|---|
| **M3** | Four named sites replace the UNRECORDED declaration, with the recovery route, the identifier's non-feature status, and the site-size and base-rate spread |
| **Table 1b** | "N of sites" row rewritten; a new "Site composition" row carries per-site counts, the confounding statistics and the recoverability diagnostic |
| **Table 2b** | The `gender` row records the site confound |
| **M18** | LOSO reported as available and partially computable; the Tier-2 ingredient list updated, and a **third** requirement named that only this analysis could expose — more test-period dropout events per site |
| **R1** | The evaluation partition's composition across the four sites |
| **R6** | The gender attribution qualified as partly a site effect |
| **R8** | The full LOSO result, including both non-computable folds and why |
| **R12** | Q2's change of status: what it closes and what it does not |
| **Countersignature** | **Narrowed** — the site count is recovered, so the not-computable declaration now covers the student-disjoint re-score alone |

### What it does not close

The claim tier stays **Tier 1**. The second Tier-2 ingredient — an independent out-of-region
dataset — is still absent, and a third is now visible: the sites yield too few test-period dropout
events for a site hold-out to estimate anything. **Q2 still needs Dr Osei's countersignature**,
now on a narrower declaration.

---

## E. Close-out procedural requirements

| # | Requirement | Status |
|---|---|---|
| 17 | Corrections ledger | Above — **empty, with the leaf-diff evidence for why** |
| 18 | Every block carries its item number | `closeout.py` block headers |
| 19 | Version guard / preflight clean | `results/run_closeout.log` preflight |
| 20 | Verdict table returned in full | below, and `results/closeout_verdict.json` |
| 21 | No column name invented | every column read from the committed CSVs |
| 22 | Every numeric literal carries a source | source comments in `closeout.py`; manuscript numbers **read from JSON at edit time**, never typed |
| 23 | No data silently dropped | `n` printed before and after every filter and SMOTE call |
| 24 | No result targeted | see the Q21 note above |
| 25 | Full grids, never the best point | all 16 grid cells and all 5 rollback arms reported |
| 26 | Unfavourable results disclosed | C1/C2/C3 failures, the two rollback arms that beat the submitted config, and the leave-one-out finding are all in R3/R4/R12 |
| 27 | Every number traces to ONE locked run | R11 / Table 7; verified by `consistency_pass.py` check A (45/45) |
| 28 | Commit hash stated, postdating every fix | **slot left in M21 — you fill it at push** |
| 29 | No lab identifier in the code | checked |

---

## F. Enhancement block (optional)

| # | Item | Status |
|---|---|---|
| 30 | Tag a release, quote the hash in M21 | **slot added in M21**; you cut the tag |
| 31 | Give R11's table its number | Done — Table 7 caption added |
| 32 | Label the two reference frames in R2 | Done — R2 now states that −0.0302 is a difference of five-seed means and +0.0170 a seed-42 resampling quantity, and explains why the signs differ |

---

## VERDICT TABLE — GROUP 3 — RUN-GROUP_3-RESUBMIT

```
ITEM      CHECK                                 OBSERVED        EXPECTED  VERDICT
NF-1      assert in pipeline + passes   inpipe=True syn=0  inpipe=True syn=0  CLOSED
Q9        synthesiser bounded to train     maxyr=2024 n=0     maxyr<2025 n=0  CLOSED
Q21       RPS artefact traces to run         7 of 7 match             7 of 7  CLOSED
Q23       full isolation grid computed           16 cells           16 cells  CLOSED
Q24       lag + rollback + perturbation    lag=21 rb=True      lag>0 rb=True  CLOSED
Q4        reproduction path from clone      11 of 11 pass           11 of 11  CLOSED

CLOSED: 6   STILL OPEN: 0
```

**All six code-dependent items are closed in code.** They close *on the scorecard* when the tagged
release is pushed — that is the one thing this work cannot do for you.

---

## What remains, and who owns it

| Item | Owner | Action |
|---|---|---|
| Q2 countersignature | **Dr Osei** | Sign and date the block at the end of Methods |
| C6 / the push | **You** | `git push` the tagged release; see `PUSH_INSTRUCTIONS.md` |
| M21 commit hash | **You** | Replace the bracketed slot with the tag and hash once pushed |
| Q22 gate | lifts on its own | Once Q10, Q21, Q23 are visible on the pushed commit |
| School identifier | ~~The group~~ | **DONE** — recovered, attached, and leave-one-site-out validation run (D2) |
