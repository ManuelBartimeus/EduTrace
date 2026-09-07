# Pushing the resubmission commit

Seven of the nine outstanding items score as failures for one reason: the repository is stale at
commit `17f18abdbb82`, dated 2026-07-15, and the remediation work was never pushed. Everything in
this bundle is ready to commit; the push is the step that closes them on the scorecard.

Run these on your own machine, from a clean clone.

## 1. Clone fresh and confirm what is actually there

```bash
cd ~/Desktop
git clone https://github.com/ManuelBartimeus/EduTrace.git EduTrace_resubmit
cd EduTrace_resubmit
git log -1 --format='%H %ad %s'          # expect 17f18abdbb82, 2026-07-15
git checkout -b resubmit/verification-round-2
```

## 2. Drop the bundle in

Unzip `edutrace_resubmission_bundle.zip` and copy its contents over the clone, replacing what is
there. Then remove the files the verification report named as defects:

```bash
git rm -r --cached notebooks/EduTrace_Main_2.ipynb notebooks/shap.py 2>/dev/null
rm -f notebooks/EduTrace_Main_2.ipynb notebooks/shap.py
```

`notebooks/shap.py` is the module that shadowed the installed `shap` package (Q4, item 8.4) — the
SHAP values feeding Q21's RPS depended on which directory the notebook started in. It must not
survive the push.

## 3. Verify from the clean clone before committing

This is FF3's self-test, and it is worth doing properly — if it does not run here, it will not run
for a reviewer either.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python add_school_identifier.py        # must print 4 sites, 180 students, 428 records
python selftest_nf1.py                 # must print ALL CHECKS PASSED, and 248 rows
python EduTrace_Revised_Pipeline.py    # ~25 min; must print the NF-1 GUARD line
python closeout.py                     # must print CLOSED: 6   STILL OPEN: 0
python school_structure.py             # site frame and confounding
python loso_validation.py              # must print 2 computable, 2 not computable
python verify.py                       # must print 31/31 checks pass
python consistency_pass.py             # must print CONSISTENCY PASS: 0 failure(s)
```

Check two things by eye in that output:

- the test partition prints **248 rows**, and
- `selftest_nf1.py` check (c) shows the guard **raising** on the contaminated partition.

## 4. Commit and push

```bash
git add -A
git commit -m "Verification round 2: close NF-1, Q4, Q9, Q21, Q23, Q24

- NF-1: falsifiable load-time provenance guard in the load path; real-only
  re-scoring restored and computed; negative-control self-test
- Q9:  synthesiser bounded to the training period; 0 synthetic rows in test
- Q21: standalone rps_results.json; RPS and DAS reported under separate names
- Q23: full 16-cell imbalance isolation grid, every cell committed
- Q24: rollback comparison, lag arm and perturbation re-run at the declared cut
- Q4:  both named reproduction files present, pinned requirements, linear
       notebook execution, module shadowing removed
- Q2:  school identifier recovered and attached; site count now exact (4);
       leave-one-site-out validation run and reported in R8
- C1-C6 clearance certificate and null state committed"

git push -u origin resubmit/verification-round-2
```

## 5. Tag the release and put the hash in M21

Enhancement 11B asks for this and it takes two minutes:

```bash
git tag -a v1.0-resubmit -m "Verification round 2 resubmission — locked run"
git push origin v1.0-resubmit
git rev-parse v1.0-resubmit          # copy this hash
```

Then open `Group 3_Methods Section.docx`, find M21, and replace

> `[RELEASE TAG AND COMMIT HASH TO BE INSERTED AT PUSH — see CHANGELOG]`

with `v1.0-resubmit (<hash>)`. **The commit must postdate every fix** (close-out checklist item
28), which it will if you tag after the commit above.

## 6. Read the four Q21 numbers off the repository page

This is FF3 for Q21, and the report is specific that it be done **in a browser, on the repository
page, not on your own machine**:

Open `results/rps_results.json` on GitHub and confirm:

| field | value |
|---|---|
| `n_evaluated` | 57 |
| `threshold_used` | 0.34 |
| `rank_preservation_RPS.rps_at_1` | 1.0 |
| `rank_preservation_RPS.rps_at_2` | 0.8596 |
| `rank_preservation_RPS.naive_fixed_order_at_1` | 0.193 |
| `rank_preservation_RPS.naive_fixed_order_at_2` | 0.0351 |
| `cross_model_sensitivity.n_evaluated` | 40 |

Check them word by word against R5 and Figure 8. The file currently on the repository reports five
records at threshold 0.62, so you will see immediately whether the push worked.

## 7. The countersignature

Take the block at the end of `Group 3_Methods Section.docx` to Dr Osei for a **dated** signature,
and make sure it is attached to the version you submit rather than to a draft. Q2 cannot close
without it, and it is his action, not the group's.

Note that the declaration has **narrowed** since the last round. It no longer covers the number of
sites — that is now recovered and reported exactly as 4 — and covers only the student-disjoint
re-score, which remains not computable. Point that out when you hand it over: he is signing a
smaller claim than the one he was asked to sign before.

---

## What this closes, and what it does not

Closed by the push: **NF-1, Q4, Q9, Q21, Q23, Q24** — and **Q22** lifts on its own once Q10, Q21
and Q23 are visible on the pushed commit. **Q10** is closed already, in the document.

Not closed by the push: **Q2** (needs the countersignature) and **C6** (needs the commit history the
push creates — it is satisfied by pushing, then confirmed by the hash in M21).

Still failing after the push, and correctly so: **C1, C2 and C3**. These are honest measurement
outcomes, not defects to be fixed — the study is underpowered at this n, and the manuscript now
says so and states the n that would have been required. Do not try to make them pass.
