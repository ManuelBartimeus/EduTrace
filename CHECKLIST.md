# EduTrace — Supervisor Checklist, Verification Round 2

Extracted from `GROUP 3_VERIFICATION 2 FEEDBACK.docx` (per-item verdicts, scorecard, forward-fix
blocks, Sections B/C/D/F/G) and the close-out script `00_CLOSEOUT_GROUP_3_20260904.py` (Section E).

Standing at submission: **NOT CLEARED — 54.5 / 79 = 69.0% weighted.** Nine items outstanding
(Q2, Q4, Q9, Q10, Q21, Q22, Q23, Q24, NF-1), of which five carry fatal weight, two are
regressions, and one is newly introduced.

The supervisor's own summary of the shape of the list: *"Seven of the nine outstanding entries —
NF-1, Q21, Q22, Q23, Q24, Q4, Q9 — have one root cause: a repository with a single commit
predating the submission... it is measuring a push that never happened rather than analysis that
was never done."*

---

## A. Fatal weight (5) — outstanding

### 1. NF-1 — the retired safeguard (new fatal, weight 5)
1.1 A load-time assertion that **raises on any synthetic row in the test partition** must exist in
    a file that is actually in the repository and is **named in M10**.
1.2 The run log must carry a line showing that check **passing on the 248-record partition**.
1.3 Self-test: clone the repository into an empty folder and run start to finish; confirm (a) the
    test partition prints **248 rows**, and (b) **deleting the assertion line makes the run fail**.
1.4 Manuscript consequence: while this stands, R1's sentence that the partition was "verified by a
    per-run assertion" may not be carried, and M10's retirement of the real-only re-scoring has no
    justification behind it. **Either restore that re-scoring or push the assertion.**
1.5 Logged additionally as a **SILENT WITHDRAWAL**: no item asked for the removal of the
    real-only-subset re-scoring analysis.
1.6 No DECLARE route. Repair is the only route. Not fixable after submission.

### 2. Q21 — contribution evidence anchor (weight 5, failed two consecutive rounds)
2.1 Commit the results file from the ablation re-run on the proposed model's own **57 flagged
    records at threshold 0.34**.
2.2 That file must report **RPS@1 1.0000, RPS@2 0.8596, n = 57, threshold 0.34**.
2.3 It must carry the **naive baseline at 0.1930 and 0.0351** alongside.
2.4 It must carry the **DAS figures under their own name**, separately from RPS.
2.5 It must carry the **40-record cross-model sensitivity run**.
2.6 Self-test: read the four numbers off the repository page in a browser and check them against
    R5 and Figure 8 word by word.
2.7 Manuscript consequence: no Abstract or Discussion sentence may claim SHAPtoSMS as a validated
    contribution, and **Figure 8 may not be cited as evidence for it**. Table 5's compliance result
    is the only contribution evidence currently writable.
2.8 No DECLARE route.

### 3. Q22 — negative-result process gate (weight 5, PARTIAL)
3.1 No separate action. The gate lifts when **Q10, Q21 and Q23 all close**, plus the corrected M20
    sentence.
3.2 Self-test: can a reader confirm from the repository that the ML was correctly built, correctly
    compared and correctly evaluated? Today: no — the committed run uses a different split, three
    seeds and a different index model.
3.3 Manuscript consequence: **no Discussion of the negative result may be drafted.** Specifically,
    the sentence that a simple attendance rule outperformed machine learning in a low-resource
    context **may not be written in any section**, and no Conclusion may be drafted around it.
3.4 **R10 stays exactly as it is** — explicitly, nothing here asks for it to be touched.

### 4. Q2 — sampling and generalisability (weight 5, PARTIAL)
4.1 Student-side work is **finished**. Site count, Table 1b and the M18 boundary sentence are all
    delivered and the 0-of-248 disjoint claim was independently verified.
4.2 Outstanding: a **dated countersignature from Dr Osei** on the not-computable declaration,
    against three sentences — M9 (0 of 248 test records belong to an unseen student), R8 (no
    substitute is reported in its place), M18 (the inference-boundary sentence).
4.3 The countersignature must be attached to the **submitted version**, not to a draft.
4.4 Manuscript consequence: until countersigned, no Results or Discussion sentence may describe the
    test partition as held-out without the qualifier. Write **"next-year performance for students
    with prior-year history"**, never "held-out test performance" unqualified.
4.5 Optional but window-limited: recover a **school identifier** from the schools while HuSSREC
    approval is live (to 6 July 2027). After submission it is a permanent limitation.

### 5. Q10 — inflated-baseline illusion (weight 5, PARTIAL)
5.1 In **M20**, delete *"All received identical data access and tuning budgets (M11, M13)"* and
    replace it with the corrected parity statement already written in M11.
5.2 The replacement must name **the three learners that shared the threshold-tuning budget and the
    two that could not**, matching M11 and M13 word for word on the point.
5.3 Self-test: search the Methods file for the word **"identical"** and read every sentence it
    appears in.
5.4 Manuscript consequence: while M20 and M11 disagree, no Discussion sentence may attribute the
    comparative result to fair resourcing across all comparators.

---

## B. Weight 3 — outstanding

### 6. Q23 — F/M/R/S mismatch (PARTIAL)
6.1 NNAA consistency requirement is **fully met** (M8 and R10 carry the SMOTE breach beside the
    CTGAN figure with the majority-baseline qualification). No further work.
6.2 Push a commit containing the **SMOTE isolation run**.
6.3 Push the **Layer 2 × Layer 3 2×2 run**.
6.4 Push the **corrected-population ablation**.
6.5 The committed code must show **SMOTE k = 3, scale_pos_weight 7.49, threshold 0.34** — the
    stale commit shows k = 5, 13.48 and 0.62.
6.6 Close-out V4 additionally requires the **full grid printed, every cell, no cell selected**, and
    written to CSV before any commentary.

### 7. Q24 — remediation protocol (PARTIAL)
7.1 Step 2 (error analysis on consistently-missed cases) is **delivered and confirmed**.
7.2 Steps 1, 3 and 4 wait on the same commit: the **rollback comparison table**, the **lag-feature
    run**, and the **perturbation re-run**, at the cut the Method states.

---

## C. Weight 1 — regressions (currently scoring −1 each)

### 8. Q4 — reproduction path
8.1 M19 names `EduTrace_Revised_Pipeline.py` and `EduTrace_Main_3.ipynb`; the tree holds
    `EduTrace_Main_2.ipynb` only. **Both named files must be present.**
8.2 `PROJECT_DIR = '/content/drive/MyDrive/EduTrace_Project/'` means the notebook never reads the
    repository's own `data/`. **One path, reading the repository's own data.**
8.3 `requirements.txt` pins nothing (`>=` throughout). **Pinned versions (`==`).**
8.4 `notebooks/shap.py` shadows the installed `shap` package, so the SHAP values feeding Q21's RPS
    depend on which directory the notebook starts in. **Remove the shadowing module.**
8.5 Stored `execution_count` runs 1, 2, 1, 2, 3, 4, 6 … 12, 37, 38, 40, 39, 15, 16, 13, 14 …, with
    two cells never executed. **Linear execution, top to bottom.**
8.6 CELL 4.4 mutates `results_store` in place before CELL 8.1 writes `master_results_table.csv`, so
    contents depend on execution order. **Remove the order dependence.**
8.7 Also absent from the tree and named in the manuscript: `final_summary_v3.json`; Figures 1–8 and
    S1–S4 image files.

### 9. Q9 — provenance firewall
9.1 The firewall M10 calls structural is asserted in a file that does not exist. (Shared root cause
    with NF-1 — *"they are not two independent failures."*)
9.2 Cell 1.2 / Cell 5 draws synthetic years from `[2022, 2023, 2024, 2025]`, putting **52 synthetic
    rows into the test partition**. The synthesiser must be **bounded to the training period**.
9.3 Evidence required: a year-range bound on the synthesiser, and a printed **count of synthetic
    rows in test = 0**.

---

## D. Clearance certificate C1–C6 and the null state (close-out script)

10. **C1** — effect sign stable across **≥ 10 seeds**. The study stands at 3 seeds in the commit
    (5 in the unpushed run). *"No wording about stability is available at any point until V1a
    returns."*
11. **C2** — the result survives **leave-one-out** (drop each contributor in turn).
12. **C3** — the **minimum detectable effect at this n**, computed on the paired (discordant-pairs)
    design, is smaller than the claimed effect. If not, the claim is INCONCLUSIVE at this n — not
    absent — and the n that would have been required must be stated.
13. **C4** — the baseline received the **same tuning budget, and it is stated**. Needs both a
    parity table in the Methods **and** the trial counts per arm. *"Counts alone do not tick it."*
14. **C5** — the **base rate printed beside every headline number**.
15. **C6** — the test set scored **once, after a committed freeze**. Decided by commit history, not
    by any script.
16. **Null state must reach 4 (certified).** It currently stands at **2 — a confirmed defect**.
    *"Only state 4 is a result. Nothing is written from 1, 2, 3 or 5."* And explicitly:
    *"DO NOT write a limitation for this. It is not a limitation, it is a bug."*

---

## E. Close-out procedural requirements (Sections F and G)

17. **Corrections ledger** (Section F): one row per number that moves —
    `CAUSE | CONFIRMED BY | FIX APPLIED | BEFORE | AFTER`. *"A number that moved with no row here
    is reverted."*
18. Every block carries its item and cause number.
19. Version guard and preflight ran clean, or the assert that fired is reported as a finding.
20. The verdict table is returned in full, including the failing rows.
21. No column name was invented.
22. Every numeric literal carries a source.
23. No data is silently dropped.
24. **No result was targeted.**
25. Full grids reported, **never the best point**.
26. Unfavourable results disclosed.
27. Every number in the manuscript traces to **ONE locked run**.
28. The commit hash is stated and **postdates every fix**.
29. No lab identifier appears anywhere in the code.

---

## F. Enhancement block (11B — optional, not scored, not outstanding)

30. **Tag a release** and quote the hash in M21.
31. **Give R11's table its number** — the prose says "Table 7 reconciles each headline figure" and
    the table beneath carries no caption.
32. **Label the two reference frames in R2's own sentences.** R2 gives Δ AUC-PR as −0.0302 and, two
    sentences later, a bootstrap mean delta of +0.0170 — opposite signs. R11 assigns each to its
    frame correctly, but a reader hits R2 first.

---

## G. Carried forward — no action

Q1, Q6, Q8, Q11, Q13, Q18, Q19 cleared in round 1. Q15 resolved. Q20, Q25, Q26 cleared as scope
boundaries. Q3, Q5, Q7, Q12, Q14, Q16 FIXED and closed. **Q17 UPGRADED** — the lag-feature
diagnosis was verified independently and credited beyond the ask.

---

# FLAGGED — ambiguities and contradictions

These are raised before proceeding rather than resolved by guessing.

### ⚠️ F1. Q21 names the numbers the re-run must produce — but Section G forbids targeting a result
FF2 requires *"a committed results file reporting RPS@1 1.0000, RPS@2 0.8596, n = 57, threshold
0.34"*. Section G item 24 requires that **"no result was targeted"**, and the close-out preamble
says *"It can return a result you did not want. Report what returns."*

These pull in opposite directions. **My reading, which I am acting on:** the quoted figures are the
supervisor's record of what your unpushed run reported, and the requirement is that the *artefact
matches the manuscript* — not that the run be steered to hit them. So I will re-run honestly and
report whatever returns; if a figure differs from the quoted one, I will report the difference as a
finding with a corrections-ledger row and update the manuscript to the new value, rather than
tuning anything to reproduce 1.0000 / 0.8596 / 57 / 0.34. **Tell me if you read this differently.**

### ⚠️ F2. C1 requires ≥10 seeds; the manuscript reports 5 (and the stale commit, 3)
Regenerating all of Table 3, R3, R4, R5, R7 at ten seeds would move every number the supervisor
independently verified and reconciled to the digit — the one thing 11C singles out as the
submission's strength. **My reading:** C1 is a certificate on the *sign stability of the central
claim's delta*, not an instruction to re-base the whole results section. I will keep Table 3 at
five seeds and add a dedicated ten-seed stability block (the close-out's V1a) reporting the delta's
sign across ten seeds. Both will be reported explicitly, with the seed list in Methods.
**Say the word if you want the full section re-based at ten seeds instead** — it is a much larger
change and every verified figure moves.

### ⚠️ F3. Q24's cut year is stated two ways
The open-issue register says *"the committed cut is 2025, not the 2026 the item names"*, while FF
text says the re-runs go *"at the cut the Method states"* (= 2025), and Q17 credits the 2026-cut
counterfactual as a sensitivity arm. **My reading:** run the rollback, lag and perturbation work at
**2025** as primary, and carry the **2026** arm as the declared sensitivity analysis it already is.
Both reported.

### ⚠️ F4. NF-1 offers two routes and the safest answer is both
FF6: *"Either restore that re-scoring or push the assertion."* But NF-1 is also logged as a silent
withdrawal — an analysis was removed. **My reading:** do both. Push a named, logging assertion
**and** restore the real-only re-score as an explicitly computed line (which, once the assertion
holds, is an identity — the test partition is 100% real — and saying so in one sentence is
stronger than asserting redundancy).

### ⚠️ F5. The close-out script cannot run unmodified against your remediation bundle
`00_CLOSEOUT_GROUP_3_20260904.py` is written against the **stale repository at 17f18abdbb82**: it
expects `data/real_student_data.csv`, `data/synthetic_student_data.csv` and
`notebooks/EduTrace_Main_2.ipynb`, and it hard-codes `SMOTE_K = 5`, `SMOTE_STRATEGY = 0.20` and
`DROPOUT_THRESHOLD = 0.6` from the old cells. Your remediation bundle has different paths, a CTGAN
supplement, k = 3, and an F2-selected threshold. Running it verbatim would audit the *old* pipeline.
**My reading:** port its blocks (V1a–V1f, V2–V6), its verdict table, its clearance conditions and
its null-state logic onto the corrected pipeline, and return the verdict table in the format it
specifies. The script's own preamble licenses this — *"IT IS AN EXEMPLAR, NOT A FORCED SOLUTION."*

### ⚠️ F6. The close-out's compute estimate misidentifies the arm under test
It budgets *"V1a — 10 TabTransformer fits"* and treats the transformer as the arm the central claim
rests on. In your manuscript the **proposed model is the engineered XGBoost**; TabTransformer is a
comparator (M11 item 4). The script's XGBoost fallback is therefore the *correct* arm here, not a
degraded substitute. I will run V1a on the engineered XGBoost and say so.

### ⚠️ F7. Which build is of record is unresolved by the supervisor
The audit footer records: *"two earlier builds under this same filename were already present in the
output folder. Neither was written in this session and neither has been verified here... Rule on
which build is of record before this goes to the student. (H13)"* This is an open instruction to
the supervisor, not to you. **Flagging only** — no action available at your end.

### ⚠️ F8. Two items are not student work and cannot be closed from here
- **Q2** needs Dr Osei's dated countersignature (4.2). I will insert a marked countersignature
  block in Methods for him to sign.
- **C6** and every "push the commit" requirement need a push to
  `github.com/ManuelBartimeus/EduTrace`, which I cannot do from this session. You will get a
  ready-to-commit tree and the exact git commands.

### ⚠️ F9. Blocked sections stay unwritten
Per 10E and FF6, the Abstract, the Discussion of the negative result, the contribution claim and
the Conclusion remain blocked until Q21/Q22 clear on a pushed commit. I am **not** drafting them,
and I will not carry any sentence forward that the blocks forbid.
