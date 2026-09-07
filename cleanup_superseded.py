#!/usr/bin/env python3
"""
Remove superseded artefacts that contradict the locked run (O2, Q9, Q23, item 27).

WHY THIS IS NECESSARY
---------------------
The round-2 branch added the corrected pipeline and its artefacts, but the merge
left the pre-remediation files from commit 17f18abdbb82 in place beside them. The
repository therefore carried two mutually contradictory runs at once, and two of
the survivors re-open items the report had asked to close:

  results/experiment_reproducibility_report.json
      Still asserts SMOTE k=5, scale_pos_weight 13.48 and threshold 0.62 — the
      exact configuration Q23 required be corrected to k=3, 7.49 and 0.34. A
      reviewer checking Q23 against the tree could open this file and read the
      superseded values back.

  data/synthetic_student_data.csv
      Still carries academic_year drawn from [2022, 2023, 2024, 2025], putting 52
      synthetic rows in a test-period year. That is the Q9 defect itself, and
      Q9's evidence requirement is a count of zero. Pointing the pipeline at this
      file reproduces the defect.

Close-out checklist item 27 requires that every number in the manuscript trace to
ONE locked run, and observation O2 applied to the whole results directory. Two
runs in one directory cannot satisfy either.

REMOVAL IS DISCLOSURE, NOT CONCEALMENT
--------------------------------------
Nothing is destroyed. Every file below remains permanently retrievable at commit
17f18abdbb82 — which is the commit the supervisor already examined — and this
script writes results/superseded_artefacts.json recording exactly what was
removed, why, and where to find it. That record is committed alongside the
removal.

Run with --dry-run to list without deleting.
"""
import argparse
import hashlib
import json
import os
import sys

STALE_COMMIT = '17f18abdbb82'

# path -> reason it must not sit beside the locked run
SUPERSEDED = {
    'results/experiment_reproducibility_report.json':
        'Asserts the pre-remediation configuration (SMOTE k=5, scale_pos_weight '
        '13.48, threshold 0.62) that Q23 required be corrected. Contradicts '
        'results/results.json and results/closeout_verdict.json.',
    'results/master_results_table.csv':
        'The results table CELL 4.4 mutated in place before CELL 8.1 wrote it (Q4). '
        'Superseded by results/results.json; contradicts Table 3.',
    'results/mcnemar_results.json':
        'Pre-remediation McNemar output. Superseded by results.json -> stats; '
        'contradicts Table 4.',
    'results/tabtransformer_summary_default_threshold.json':
        'Pre-remediation TabTransformer summary. Superseded by results.json -> table3.',
    'results/tabtransformer_summary_optimal_threshold.json':
        'Pre-remediation TabTransformer summary. Superseded by results.json -> table3.',
    'results/shap_values_test_subset.npy':
        'SHAP values from the pre-remediation run, computed on a subsample rather '
        'than the full 248-row partition M17 specifies.',
    'results/shap_values_xgboost_default.npy':
        'SHAP values from the pre-remediation run. Superseded by results.json -> shap_sms.',
    'results/shap_values_xgboost_engineered.npy':
        'SHAP values from the pre-remediation run. Superseded by results.json -> shap_sms.',
    'results/roc_pr_curves.png':
        'Named in the report as an image file that does NOT correspond to the '
        'current run. Superseded by figs/figure4.png.',
    'results/shap_summary_beeswarm.png':
        'Named in the report as not corresponding to the current run. Superseded '
        'by figs/figure6.png and figs/figureS1.png.',
    'results/shap_waterfall_false_negative_missed_dropout.png':
        'Named in the report as not corresponding to the current run. Superseded '
        'by figs/figureS2.png.',
    'results/shap_waterfall_false_positive_incorrectly_flagged_student.png':
        'Named in the report as not corresponding to the current run. Superseded '
        'by figs/figureS3.png.',
    'results/threshold_optimisation_curve.png':
        'Threshold curve from the pre-remediation run at threshold 0.62. The '
        'reported operating point is 0.34.',
    'results/attention_heatmap_note.txt':
        'Note on a diagnostic M17 states was not produced. The deviation is stated '
        'in M17 itself; a loose note in results/ implies an artefact that does not exist.',
    'data/synthetic_student_data.csv':
        'THE Q9 DEFECT. academic_year drawn from [2022, 2023, 2024, 2025], placing '
        '52 synthetic rows in a test-period year. Superseded by '
        'data/synth_ctgan_s42.csv, which is bounded to the training period.',
}

# Kept deliberately, with the reason recorded so the choice is visible.
KEPT = {
    'data/real_student_data.csv':
        'The register as collected, before cleaning. Retained as provenance for '
        'data/real_student_data_CLEANED.csv. No entry point reads it.',
    'app/':
        'The demonstrator. Its model artefacts are regenerated from the locked run '
        'by export_models.py rather than deleted, so the interface serves the '
        'model the manuscript reports (O3).',
    'models/':
        'Regenerated from the locked run by export_models.py.',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    print('=' * 78)
    print('SUPERSEDED ARTEFACT REMOVAL   (O2, Q9, Q23, checklist item 27)')
    print('=' * 78)
    print('  Every file below stays retrievable at commit %s.\n' % STALE_COMMIT)

    removed, absent = [], []
    for path, reason in sorted(SUPERSEDED.items()):
        if not os.path.exists(path):
            absent.append(path)
            print('  [absent ] %s' % path)
            continue
        sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        size = os.path.getsize(path)
        removed.append(dict(path=path, sha256=sha, bytes=size, reason=reason,
                            retrievable_at=STALE_COMMIT))
        print('  [%s] %-64s %8d B' % ('would ' if a.dry_run else 'REMOVE', path, size))
        print('             %s' % reason)
        if not a.dry_run:
            os.remove(path)

    print('\n  removed: %d | already absent: %d' % (len(removed), len(absent)))
    print('\n  KEPT DELIBERATELY:')
    for p, why in KEPT.items():
        print('    %-34s %s' % (p, why))

    if not a.dry_run:
        os.makedirs('results', exist_ok=True)
        json.dump(dict(
            removed=removed, already_absent=absent, kept=KEPT,
            retrievable_at_commit=STALE_COMMIT,
            rationale=('The repository carried two mutually contradictory runs. '
                       'Checklist item 27 requires every reported number to trace '
                       'to one locked run, and observation O2 applied to the whole '
                       'results directory. These files are superseded, not '
                       'suppressed: they remain in git history at the commit the '
                       'verification report examined.')),
            open(os.path.join('results', 'superseded_artefacts.json'), 'w'), indent=2)
        print('\nwritten: results/superseded_artefacts.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
