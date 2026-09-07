# Prompt for Claude Code — round 3, push to master

Copy the fenced block below into Claude Code, running in the folder where you unzipped
`edutrace_round3_bundle.zip`. Prerequisites are unchanged from last time, with two additions noted
at the bottom.

---

```
You are pushing a corrected thesis-project repository to master. The previous round is already on
master; this round repairs three defects that round introduced or left behind. Accuracy matters far
more than speed. If a check disagrees with what I say it should print, STOP and report it — do not
adjust code, seeds, thresholds or data to make anything pass.

WHAT THIS ROUND FIXES
1. Non-determinism. The pushed pipeline did not reproduce its own results across machines: 349 of
   859 leaves differed, and changing only OMP_NUM_THREADS moved 329 of 858 leaves and flipped two
   clearance conditions. Every estimator is now pinned to a single thread and the run has been
   regenerated once. Only the TabTransformer arm moved; every XGBoost figure is bit-identical.
2. Superseded artefacts. The merge left 46 files from commit 17f18abdbb82 in the tree. Two of them
   re-opened closed items: results/experiment_reproducibility_report.json still asserted SMOTE k=5 /
   scale_pos_weight 13.48 / threshold 0.62 (the configuration Q23 required be corrected), and
   data/synthetic_student_data.csv still carried 52 synthetic rows in a test-period year (the Q9
   defect itself). Fifteen files are removed, each recorded in results/superseded_artefacts.json,
   all still retrievable at 17f18abdbb82.
3. A broken reproduction path. EduTrace_Revised_Pipeline.py defaulted --synth to
   repro/synth_ctgan_s42.csv; there is no repro/ directory, so the command the README gives a
   reviewer failed instantly from a clean clone. Both defaults now point at data/ and results/.

THE RULE THAT OVERRIDES EVERYTHING
Do NOT regenerate anything in results/, figs/, models/, app/ or docs_out/. This bundle IS the
regenerated locked run. Your job is to verify it and push it, not to re-run it into place. The one
exception is the read-only scratch verification in step 6, which must write to verify_run/ only.

STEP 0 — Orient
Confirm the folder contains MANIFEST.sha256, EduTrace_Revised_Pipeline.py, selftest_determinism.py,
cleanup_superseded.py, export_models.py, ledger_generator.py, REPRODUCIBILITY.md, and the
directories data/ results/ results_pre_determinism/ figs/ models/ app/ notebooks/ docs_in/ docs_out/.
Print `python --version`.

STEP 1 — Integrity
Verify every file against MANIFEST.sha256 (sha256 + two spaces + path, relative to this folder).
Report matched/mismatched/missing. If anything mismatches, STOP.

STEP 2 — Clone master fresh
Clone https://github.com/ManuelBartimeus/EduTrace into a NEW sibling folder EduTrace_round3.
Print `git log -1 --format='%H %ad %s'`. Expect the merge commit b73859af6726ec2e7b5c541689ba9497c83f9adb
("Merge pull request #1"). If HEAD is something else, STOP and show me — someone has pushed since.
Create branch `resubmit/round-3-determinism`.

STEP 3 — Apply the bundle, including the deletions
Copy the bundle contents over the clone, replacing existing files.

Then stage the removals. These files must NOT survive the push — they are superseded artefacts that
contradict the locked run. Use `git rm -f`:

  results/experiment_reproducibility_report.json
  results/master_results_table.csv
  results/mcnemar_results.json
  results/tabtransformer_summary_default_threshold.json
  results/tabtransformer_summary_optimal_threshold.json
  results/shap_values_test_subset.npy
  results/shap_values_xgboost_default.npy
  results/shap_values_xgboost_engineered.npy
  results/roc_pr_curves.png
  results/shap_summary_beeswarm.png
  results/shap_waterfall_false_negative_missed_dropout.png
  results/shap_waterfall_false_positive_incorrectly_flagged_student.png
  results/threshold_optimisation_curve.png
  results/attention_heatmap_note.txt
  data/synthetic_student_data.csv

Do NOT remove data/real_student_data.csv, app/ or models/ — those are kept deliberately, and app/
and models/ are replaced by the bundle's regenerated versions.

Cross-check your deletions against results/superseded_artefacts.json: the `removed` array lists
exactly these fifteen paths. If the two lists disagree, STOP.

Run `git status` and show me the summary.

STEP 4 — Environment
Create a venv in the clone and install requirements.txt EXACTLY as pinned. If any package fails,
STOP and show the error; do not substitute a version. Do not install requirements-synthesis.txt.

STEP 5 — Verify without regenerating
Run these five in the clone, in order:

  python add_school_identifier.py
  python selftest_nf1.py
  python selftest_determinism.py
  python verify.py
  python consistency_pass.py

Expected, all five:
  1. 4 sites, 180 students, 428 records — and `git status --porcelain data/` shows NO modification.
  2. "NF-1 SELF-TEST: ALL CHECKS PASSED", 248-row partition, and check (c) shows the guard RAISING
     on a contaminated partition.
  3. "DETERMINISM SELF-TEST: PASSED", 7 arms x 3 thread settings all identical. THIS ONE MATTERS
     MOST — it was written because your machine has more cores than the one that produced the run,
     so it is the first real test of the repair. Takes about 6 minutes.
  4. "31/31 checks pass programmatically."
  5. "CONSISTENCY PASS: 0 failure(s)" with 60 of 60 numeric claims located.

If the determinism self-test FAILS, STOP immediately and show me every diverging arm and the thread
setting it diverged at. Do not push. That would mean the repair is incomplete on your hardware and I
need to know before anything reaches master.

STEP 6 — Full re-run into scratch, and the leaf diff
This is the check that failed last round. Run:

  python EduTrace_Revised_Pipeline.py --out verify_run/results.json
  python closeout.py --outdir verify_run

Note it takes longer than before — roughly 40 minutes — because single-threading is the fix.

Then diff verify_run/results.json against results/results.json leaf by leaf (recurse dicts and
lists; floats at 1e-9 tolerance). Report the count and list every differing leaf.

EXPECTED: zero differing leaves. Last round this was 349. If it is not zero now, STOP, do not push,
and show me the leaves and your library versions.

Also confirm verify_run/closeout_verdict.json shows CLOSED: 6, STILL OPEN: 0, and clearance
C1 FAIL, C2 FAIL, C3 FAIL, C4 PASS, C5 PASS, C6 PENDING, null state 5. C1/C2/C3 are SUPPOSED to
fail — they are honest measurement outcomes and must not be "fixed".

Add verify_run/ to .gitignore.

STEP 7 — Commit and push the branch
Only if steps 5 and 6 are clean.

  git add -A
  git commit -m "Round 3: repair non-determinism, remove superseded artefacts, fix the reproduction path

  - Determinism: n_jobs=1 on every XGBoost and scikit-learn estimator; torch
    pinned single-thread with deterministic algorithms and a seeded DataLoader;
    run regenerated once. selftest_determinism.py verifies thread-count
    independence. Only the TabTransformer arm moved; every XGBoost figure is
    bit-identical. Full detail in REPRODUCIBILITY.md.
  - McNemar vs TabTransformer reversed direction (b=22,c=41 -> b=53,c=31) and
    remains significant; R4 and R10 corrected, disclosed in the manuscript text
    and in the corrections ledger.
  - Removed 15 superseded artefacts that contradicted the locked run, including
    experiment_reproducibility_report.json (asserted the pre-remediation
    configuration) and synthetic_student_data.csv (52 synthetic rows in a
    test-period year). Recorded in results/superseded_artefacts.json; all remain
    retrievable at 17f18abdbb82.
  - O3 closed: export_models.py exports the locked run's fitted artefacts to
    models/ and syncs app/, so the demonstrator serves the reported model.
  - Fixed --synth and --out defaults pointing at a non-existent repro/ directory,
    which made the documented reproduction command fail from a clean clone.
  - Corrections ledger generated from the artefacts (Section F)."

  git push -u origin resubmit/round-3-determinism

STEP 8 — Merge to master
I have asked for this to go to master directly. Merge the branch into master with a merge commit and
push:

  git checkout master
  git pull --ff-only
  git merge --no-ff resubmit/round-3-determinism -m "Merge round 3: determinism repair, superseded-artefact removal, reproduction-path fix"
  git push origin master

If the merge conflicts, STOP and show me the conflicting paths. Do not resolve conflicts by guessing
which side is correct.

STEP 9 — Re-tag
The v1.0-resubmit tag points at a commit whose numbers have since been corrected. Leave it in place
as history and add a new one:

  git tag -a v1.1-round3 -m "Round 3 — deterministic locked run"
  git push origin v1.1-round3
  git rev-parse v1.1-round3^{commit}

Print that hash on its own line, labelled.

STEP 10 — Update M21
In docs_out/Group 3_Methods Section.docx, M21 currently names the v1.0-resubmit tag and its hash.
Replace that tag name and hash with v1.1-round3 and the hash from step 9, using python-docx, editing
only the substring inside the paragraph that contains it. Do not reformat the document or touch any
other text in either .docx.

Then re-run `python consistency_pass.py` (must still be 0 failures), commit as
"docs: point M21 at the round-3 release tag", and push to master.

STEP 11 — Report back
Give me: the merge commit hash, the new tag, the leaf-diff count from step 6, the determinism
self-test result including how many cores your machine has, the result of each check in step 5,
anything you stopped on, and the GitHub URLs for results/corrections_ledger.md and
REPRODUCIBILITY.md on master.

Do not open a pull request — merge directly, as instructed above.
```

---

## Prerequisites — what is different from last time

Everything from the round-2 prompt still applies (write access, non-interactive git auth,
Python 3.11, ~2 GB free, machine awake). Two changes:

**1. Budget about 60 minutes, not 40.** Single-threading is the fix, so the pipeline re-run in step 6
takes roughly 40 minutes rather than 25, and the determinism self-test adds about 6.

**2. Your token.** If you revoked the one from last round — and you should have — you will need a
fresh one, or `gh auth login`. If you have not revoked it yet, do that first regardless.

**One thing I want you to watch for.** Step 5's determinism self-test is the single most important
check in this prompt, and it has never been run on hardware that can actually vary the thread count.
My machine has two cores, so its "4 threads" setting was not really four threads. Yours has twelve.
If that test fails on your machine, the repair is incomplete and I need to see the output before
anything is pushed — the prompt tells Claude Code to stop there, and it should.
