# EduTrace — reproducible run (resubmission, verification round 2)

Every number in the revised Methods and Results sections comes from this repository.
One locked run, one path, pinned versions.

## Reproduce end to end

```bash
git clone https://github.com/ManuelBartimeus/EduTrace.git
cd EduTrace
pip install -r requirements.txt

python add_school_identifier.py         # attach the school identifier (~5 s)
python selftest_nf1.py                  # NF-1 guard self-test         (~10 s)
python EduTrace_Revised_Pipeline.py     # the locked run               (~25 min CPU)
python closeout.py                      # C1-C6 certificate + verdict  (~1 min)
python school_structure.py              # sampling frame + confounding (~10 s)
python loso_validation.py               # leave-one-site-out           (~1 min)
python make_figures.py                  # Figures 4-8, S1-S4           (~3 min)
python smote_nnaa.py                    # SMOTE NNAA privacy ladder    (~1 min)
python verify.py                        # checklist audit
python consistency_pass.py              # manuscript <-> artefact check
```

All entry points default to `data/` and write to `results/`. No path needs editing,
and nothing reads a mounted Drive. `notebooks/EduTrace_Main_3.ipynb` is the same run as a
notebook, executed top to bottom.

Regenerating the synthetic supplement is a separate, optional step and is **not** part of
reproducing the reported numbers — the supplement is committed as a locked data artefact:

```bash
pip install -r requirements-synthesis.txt
python gen_synth.py data/synth_ctgan_s42.csv --epochs 600 --batch 50 --seed 42   # ~15 min
```

`gen_synth.py` is deterministic: two runs produce a byte-identical supplement
(`DataFrame.equals` == True). This fixes the defect in the original pipeline Cell 1.2, where
`CTGANSynthesizer` was constructed with no seed and `set_all_seeds(42)` ran only *after*
`.sample()`, so the 350-row supplement — 350 of the 530 training records — differed on every
run and no reported model number was reproducible.

## The provenance firewall (NF-1, Q9)

`assert_test_partition_is_real()` in `EduTrace_Revised_Pipeline.py` is the load-time check
Methods M10 names. It sits in the load path, runs on every construction of the evaluation
partition, and raises rather than warns. Its run-log line is written to
`results/nf1_guard_log.txt` on every run:

```
NF-1 GUARD: test partition n=248 | synthetic rows=0 | real rows=248 | positives=7 (2.82%) — PASS
```

The test partition is built from the **pooled** frame, not from the real frame alone. That is
deliberate: filtering synthetic rows out before checking for them makes the check
unfalsifiable, which is the defect the verification report identified. `selftest_nf1.py`
proves the guard is load-bearing by injecting a synthetic row into the test period and
confirming the guard raises.

## Files

| File | What it is |
|---|---|
| `EduTrace_Revised_Pipeline.py` | The locked run. Named in M10, M19 and M21. Contains the NF-1 guard. |
| `add_school_identifier.py` | Attaches the school identifier, asserting the ranges partition the register |
| `school_structure.py` | Sampling frame by site, site/variable confounding, site recoverability |
| `loso_validation.py` | Leave-one-site-out validation (Q2, M18) |
| `consistency_pass.py` | Checks every manuscript number against the artefact it comes from |
| `selftest_nf1.py` | NF-1 self-test: clean path, Q9 bound, and the negative control |
| `closeout.py` | V1a–V6, the C1–C6 clearance certificate, the verdict table and the null state |
| `shaptosms.py` | Algorithm 1 (M12); RPS and DAS defined and reported separately |
| `gen_synth.py` | Seeded CTGAN generation + the 4-level synthetic audit |
| `make_figures.py` | Regenerates Figures 4–8 and S1–S4 from the run |
| `smote_nnaa.py` | SMOTE nearest-neighbour adversarial accuracy ladder |
| `verify.py` | Checks each checklist item against the output documents |
| `data/real_student_data_CLEANED.csv` | The register as collected: 428 records, 180 students |
| `data/real_student_data_CLEANED_school.csv` | **The analysed register** — the above plus the recovered school identifier. Every entry point defaults to this file. |
| `data/synth_ctgan_s42.csv` | **The locked synthetic supplement** — 350 rows, all academic_year 2024 |
| `data/synth_ctgan_s42_audit.json` | Its 4-level fidelity/privacy audit |
| `results/results.json` | Every reported quantity |
| `results/rps_results.json` | The Q21 contribution-evidence anchor, on its own |
| `results/closeout_verdict.json` | The C1–C6 certificate, verdict table and null state |
| `results/q23_imbalance_grid.csv` | The full 16-cell imbalance isolation grid, every cell |
| `results/q24_rollback_comparison.csv` | Layer-by-layer rollback, one layer reverted at a time |
| `results/q24_cut_comparison.csv` | 2025 (declared) vs 2026 (sensitivity) partition shapes |
| `results/c4_tuning_parity.csv` | Tuning budget per arm — the C4 parity table |
| `results/nf1_guard_log.txt` | The guard's run-log line for this run |
| `results/school_structure.json` | Per-site sampling frame, confounding statistics, recoverability |
| `results/loso_validation.json` | Leave-one-site-out, including the folds that are not computable |
| `notebooks/EduTrace_Main_3.ipynb` | The same run as a notebook, executed linearly. Named in M19. |
| `figs/` | The nine regenerated manuscript figures |

## Known limitations, stated rather than worked around

- **Student-disjoint re-score is not computable.** 0 of 248 test records belong to a student
  unseen in training; all 155 test-year students appear in the 2024 training partition. The
  reported figures are next-year performance for students with prior-year register history,
  not performance on new intakes.
- **Lag features are not learnable under the declared 2025 cut.** The training pool is
  academic year 2024 only — every student's first observation — so all four derived features
  have zero training variance. A 2026-cut sensitivity arm is reported instead, and it rests
  on 3 dropout events, so it is reported as a direction and not as a result.
- **`grade_level` is perfectly confounded with `academic_year`** (2024 = all Grade 7, 2025 =
  all Grade 8, 2026 = all Grade 9), so the temporal hold-out is simultaneously a grade
  hold-out and `grade_level` carries zero training variance.
- **Leave-one-site-out validation is only half computable.** The school identifier has been
  recovered and the analysis run, but 2 of the 4 sites record no dropout event in the test period,
  so AUC-PR and AUC-ROC are undefined on those folds. They are reported as not computable rather
  than filled with a floor value. The 2 computable folds rest on 7 events between them and are
  reported as a direction, not an estimate.
- **Site is confounded with gender.** Cramer's V = 0.4396; one school is single-sex and two are
  100% Unpaid on fee_payment_status. The `gender_Female` attribution is therefore partly a site
  effect and carries no behavioural reading.
- **The claim tier remains Tier 1.** The school identifier was one of two ingredients named for
  Tier 2. The second — an independent out-of-region dataset — is still absent, and a third is now
  visible: too few test-period dropout events per site for a site hold-out to estimate anything.
