# Prompt for Claude Code

Copy everything inside the fenced block below and paste it into Claude Code, running in the folder
where you unzipped `edutrace_resubmission_bundle.zip`.

Read the **"What you need in place first"** section underneath before you run it. Four of those
items are things only you can supply, and the push will stall without them.

---

```
You are helping me push a thesis-project resubmission to GitHub. Accuracy matters far more
than speed here: this repository is the evidence for a supervisor's verification report, and a
figure that changes silently is worse than a task that stops and asks me.

CONTEXT
The repository is https://github.com/ManuelBartimeus/EduTrace. It is stale at commit
17f18abdbb82 (2026-07-15). My remediation work was never pushed, and seven of nine outstanding
items on my supervisor's report score as failures for that single reason. The bundle in this
folder contains the corrected repository tree, the locked results artefacts, and the revised
manuscript. Your job is to get it committed, pushed and tagged, and to verify it from a clean
clone before you do.

THE ONE RULE THAT OVERRIDES EVERYTHING ELSE
Do NOT regenerate, overwrite or "fix" anything in results/, figs/ or docs_out/. Those are a
locked run. Every number in my manuscript traces to them, and my supervisor has independently
reconciled them to four decimal places. If any check below disagrees with what I say it should
print, STOP and report it to me. Do not adjust code, thresholds, seeds or data to make a check
pass, and do not commit anything you could not verify. A stopped task is a good outcome; a green
check I cannot trust is not.

STEP 0 — Establish where things are
- Confirm this folder contains: EduTrace_Revised_Pipeline.py, closeout.py, selftest_nf1.py,
  verify.py, consistency_pass.py, add_school_identifier.py, loso_validation.py,
  school_structure.py, requirements.txt, MANIFEST.sha256, and the directories data/, results/,
  figs/, notebooks/, docs_in/, docs_out/.
- Tell me the Python version available (`python --version`). Record it; I will need it later.
- Do not install anything yet.

STEP 1 — Integrity check before anything is touched
Verify every file against MANIFEST.sha256 (it lists sha256 + path, one per line, relative to
this folder). Report the count that matched and list any that did not. If anything mismatches,
stop and tell me — the bundle is not intact and nothing should be pushed.

STEP 2 — Clone the repository fresh
Clone https://github.com/ManuelBartimeus/EduTrace into a NEW sibling folder named
EduTrace_resubmit. Do not clone into this folder.
- Print `git log -1 --format='%H %ad %s'` and confirm the HEAD commit is 17f18abdbb82 dated
  2026-07-15. If it is not, STOP and show me what is actually there — someone else may have
  pushed and I need to know before we overwrite anything.
- Create and switch to a branch named `resubmit/verification-round-2`.

STEP 3 — Apply the bundle
- Copy the full contents of this bundle folder over the clone, replacing files that already
  exist. Do not delete anything in the clone that the bundle does not mention, except the two
  files in the next line.
- Delete `notebooks/EduTrace_Main_2.ipynb` and `notebooks/shap.py` from the clone, using
  `git rm` so the deletion is staged. `notebooks/shap.py` shadows the installed `shap` package
  and is a named defect in the report (item Q4); it must not survive the push.
- Run `git status` and show me the summary before going further.

STEP 4 — Set up an isolated environment
In the clone, create a virtual environment and install requirements.txt EXACTLY as pinned. Do
not relax, upgrade or substitute any pin. If any package fails to install on this machine, STOP
and show me the exact error — do not install a different version. The pins are what make the
numbers reproducible, and a substituted version silently changes results.

Note: requirements.txt is the reproduction environment. Do NOT install requirements-synthesis.txt
— it pins a different pandas major version and is only needed to regenerate the synthetic
supplement, which is already committed and must not be regenerated.

STEP 5 — Verify WITHOUT overwriting the locked artefacts
Run these four, in order, in the clone. All four are safe: they either write nothing or write
only files that are already deterministic.

  python add_school_identifier.py
  python selftest_nf1.py
  python verify.py
  python consistency_pass.py

Expected, and all four must hold:
  1. add_school_identifier.py prints 4 sites, 180 students, 428 records. Then run
     `git status --porcelain data/` — it must show NO modification. That file is committed and
     the script is deterministic, so any diff means something is wrong.
  2. selftest_nf1.py prints "NF-1 SELF-TEST: ALL CHECKS PASSED", shows a test partition of
     248 rows, and — this one matters most — its check (c) shows the guard RAISING on a
     deliberately contaminated partition.
  3. verify.py prints "31/31 checks pass programmatically."
  4. consistency_pass.py prints "CONSISTENCY PASS: 0 failure(s)".

If any of these fails, STOP and show me the full output.

STEP 6 — The full re-run, into a scratch directory only
This is my supervisor's required self-test: the pipeline must run start to finish from a clean
clone. It takes roughly 25 minutes on CPU. Run it so that it writes NOWHERE NEAR results/:

  python EduTrace_Revised_Pipeline.py --out verify_run/results.json
  python closeout.py --outdir verify_run

Then compare, without changing anything:
  - Load verify_run/results.json and results/results.json and diff them leaf by leaf (recurse
    through nested dicts and lists; compare floats with a tolerance of 1e-9). Report the number
    of differing leaves and list every one.
  - Confirm verify_run/closeout_verdict.json shows "CLOSED: 6, STILL OPEN: 0" and clearance
    C1 FAIL, C2 FAIL, C3 FAIL, C4 PASS, C5 PASS, C6 PENDING, null state 5.

EXPECTED: zero differing leaves, and the clearance exactly as listed.

C1, C2 and C3 are SUPPOSED to fail. They are honest measurement outcomes — the study is
underpowered at this sample size and the manuscript says so explicitly. Do not try to make them
pass. Do not treat them as bugs.

If there ARE differing leaves, STOP. Do not overwrite results/. Show me every differing leaf and
the Python and library versions you used. That would mean this machine's environment produces
different numbers from the locked run, and I need to decide what to do — it is not yours to
resolve.

Add `verify_run/` to .gitignore so the scratch output is never committed.

STEP 7 — Commit and push
Only if STEP 5 and STEP 6 both came back clean.

Stage everything and commit with this message:

  Verification round 2: close NF-1, Q2, Q4, Q9, Q21, Q23, Q24

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
  - C1-C6 clearance certificate and null state committed

Push the branch to origin. If the push is rejected for permissions, STOP and tell me exactly
what the error said — do not try another remote, another account, or a fork.

STEP 8 — Tag the release
  git tag -a v1.0-resubmit -m "Verification round 2 resubmission - locked run"
  git push origin v1.0-resubmit
  git rev-parse v1.0-resubmit^{commit}

Print that commit hash on its own line, clearly labelled. I need to paste it into the manuscript.

STEP 9 — Insert the hash into the manuscript
In `docs_out/Group 3_Methods Section.docx`, section M21 contains this exact placeholder text:

  [RELEASE TAG AND COMMIT HASH TO BE INSERTED AT PUSH — see CHANGELOG]

Replace it with:  v1.0-resubmit (<the full commit hash from step 8>)

Use python-docx and replace the substring inside the paragraph that contains it, preserving the
rest of the paragraph. Do not rewrite the paragraph, do not reformat the document, and do not
touch any other text anywhere in either .docx file.

Then re-run `python consistency_pass.py` and confirm it still prints 0 failures. Commit the
document with the message "docs: record the release tag and commit hash in M21" and push.

STEP 10 — Report back
Give me a short summary containing:
  - the commit hash and the tag
  - the branch name and its URL
  - the result of each check in steps 5 and 6, including the leaf-diff count
  - the Python and library versions used
  - anything you had to stop on
  - the direct GitHub URL to results/rps_results.json on the pushed branch

Do not open a pull request, do not merge to main, and do not modify main. I will do that myself
after my supervisor has looked at the branch.
```

---

# What you need in place first

Four of these are yours alone. The rest are quick.

## 1. Push rights to `ManuelBartimeus/EduTrace` — **check this first**

The repository is under **ManuelBartimeus**, not your account. Before anything else, confirm you
can actually write to it. Run this in a terminal:

```bash
git ls-remote https://github.com/ManuelBartimeus/EduTrace.git
```

If that works you can read it. Writing is the separate question. You need **one** of:

- a **collaborator invitation** accepted on that repository, or
- the credentials for the account that owns it, or
- to be in an organisation team with write access.

If none of those is true, ask Manuel to add you as a collaborator with **Write** permission
before you start. Claude Code cannot resolve a permissions failure for you, and the prompt
deliberately tells it to stop rather than improvise a fork — a fork would not close the items,
because your supervisor checks the named repository.

## 2. Git authentication that actually works non-interactively

Claude Code cannot type a password into a prompt. Set one of these up first and test it:

- **GitHub CLI** (easiest): `gh auth login`, then `gh auth status` to confirm.
- **SSH key**: test with `ssh -T git@github.com`. If you use SSH, tell Claude Code to clone the
  SSH URL instead — add a line to the prompt saying so.
- **Personal access token** with `repo` scope, stored in a credential helper.

Also set your identity if you never have:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email"
```

## 3. Python 3.11, and a tolerance for the pins not installing

The locked run used **Python 3.11.15**. `requirements.txt` pins exact versions (numpy 2.4.4,
pandas 3.0.2, torch 2.14.0, xgboost 3.2.0 and so on).

**Be aware of this**: you are on Windows, and the run those numbers came from was on Linux. Some
of those pins may not have a Windows wheel for your Python version. If that happens, Claude Code
is instructed to stop rather than substitute a version — which is the right behaviour, because a
different library version can silently change the numbers your supervisor already reconciled.

**If step 4 or step 6 fails, that is not a disaster.** Steps 5, 7, 8 and 9 do not need the
pipeline to run. Come back and tell me, and we will either adjust the pins or skip the local
re-run and note in the commit that the artefacts are carried from the locked Linux run. What you
must not do is let anything overwrite `results/`.

## 4. The bundle, unzipped, in a folder you can find

Download `edutrace_resubmission_bundle.zip`, unzip it, and start Claude Code **in that folder**.
Check that `MANIFEST.sha256` is sitting next to `EduTrace_Revised_Pipeline.py` — if it is, you
are in the right place.

## 5. Time and machine

- About **40 minutes** total, of which ~25 is the pipeline re-run on CPU.
- Do not let the machine sleep during it. On Windows, set the power plan to never sleep, or run
  it plugged in with the lid open.
- Roughly 2 GB free disk for the virtual environment (torch is large).
- A network connection for the clone, the pip install and the push.

## 6. Two things to have open in a browser afterwards

Your supervisor asks for both of these to be checked **on the repository page, not on your
machine** — so do them yourself, after the push:

- `results/rps_results.json` — read four numbers off the screen and check them against R5 and
  Figure 8 word by word: `n_evaluated` 57, `threshold_used` 0.34, `rps_at_1` 1.0, `rps_at_2`
  0.8596. The file currently on the repository reports 5 records at threshold 0.62, so you will
  see immediately whether the push landed.
- The branch page, to confirm the tag `v1.0-resubmit` appears.

---

# What this does not do, and cannot

**The countersignature.** Q2 still needs Dr Osei's dated signature on the block at the end of the
Methods document. No amount of pushing closes it. Note when you hand it to him that the
declaration has **narrowed** since he last saw it — it no longer covers the number of sites, now
that you have recovered them, and covers only the student-disjoint re-score.

**Merging to main.** The prompt deliberately stops at a branch. Let your supervisor look at it
first; merging is a decision, not a step.

**Making C1, C2 or C3 pass.** They fail because the study is underpowered at this sample size,
and the manuscript now states exactly how underpowered — roughly 786 discordant pairs would have
been needed against the 69 available. That honesty is worth more to your score than a green tick
would be. If Claude Code offers to "fix" them, say no.
