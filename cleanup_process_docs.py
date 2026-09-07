#!/usr/bin/env python3
"""
Remove internal remediation documents from the published tree (close-out
Section G, "no lab identifier appears anywhere in the code").

WHY
---
The analysis code passes the naming scrub. The tree around it does not. The
repository is the data-and-code-availability link in Methods M21, so a reviewer
opens it expecting the study; what is currently at the root is the remediation
process — the working prompts that drove two push rounds, the extracted
supervisor checklist, and push instructions. Two separate problems:

  1. They name the supervisor and reproduce his verification feedback verbatim,
     including per-item verdicts and the weighted score. That is correspondence
     about the student's work, published without his agreement, in a repository
     the manuscript invites reviewers to open.

  2. They describe how the submission was assembled rather than what was
     measured. Section G's naming rule exists so that the artefact a reviewer
     reads is the study.

The app/ directory carries a third class: four superseded copies of an extracted
script, three text dumps of a proposal, and the proposal PDF itself. None is
reachable from app.py and none is an input to any reported figure.

REMOVAL IS DISCLOSURE, NOT CONCEALMENT
--------------------------------------
Nothing is destroyed and nothing that evidences a result is touched. Every file
below stays permanently retrievable at commit 01b22e3 — the round-3 release
state — and this script records what was removed, its sha256, its size and the
reason, in results/superseded_artefacts.json beside the round-3 record. The
verification evidence itself (verify.py, closeout.py, consistency_pass.py,
CHANGELOG.md, REPRODUCIBILITY.md and every artefact under results/) is untouched:
the checks a reviewer needs to re-run all remain.

Run with --dry-run to list without deleting.
"""
import argparse
import hashlib
import json
import os
import sys

RELEASE_COMMIT = '01b22e3'

PROCESS_DOCS = {
    'CLAUDE_CODE_PUSH_PROMPT.md':
        'Working instructions for the round-2 push. Describes how the submission '
        'was assembled, not what was measured, and names the supervisor.',
    'CLAUDE_CODE_ROUND3_PROMPT.md':
        'Working instructions for the round-3 push. Same class as above.',
    'CHECKLIST.md':
        'The supervisor\'s verification feedback transcribed in full — per-item '
        'verdicts, the weighted score, the forward-fix blocks and his quoted '
        'words. Correspondence about the student\'s work, published in a public '
        'repository. The items it tracks are checked in code by verify.py, which '
        'stays.',
    'PUSH_INSTRUCTIONS.md':
        'Git commands for a specific push that has already happened. Superseded '
        'by README.md, which documents the reproduction path a reader needs.',
    'app/Group3_Vanguard_Proposal.pdf':
        'The group\'s project proposal. Not an input to any reported figure and '
        'not read by app.py.',
    'app/proposal_extract.txt':
        'Text dump of the proposal PDF, kept while the demonstrator was drafted.',
    'app/proposal_extract2.txt':
        'Second text dump of the same proposal.',
    'app/proposal_extract3.txt':
        'Third text dump of the same proposal.',
    'app/proposal_mid.txt':
        'Partial text dump of the same proposal.',
    'app/extracted.js':
        'Superseded copy of the demonstrator script. app.py and index.html do not '
        'reference it.',
    'app/extracted_script.js':
        'Superseded copy of the demonstrator script.',
    'app/extracted_script_new.js':
        'Superseded copy of the demonstrator script.',
    'app/extracted_script_final.js':
        'Superseded copy of the demonstrator script.',
}

KEPT = {
    'verify.py':
        'The item-by-item gate. It is the machine-checkable form of the checklist '
        'and it stays, which is why removing the transcribed document loses no '
        'evidence.',
    'consistency_pass.py':
        'The manuscript gate, including the clearance, ledger and numeric-sweep '
        'checks added in round 4.',
    'CHANGELOG.md':
        'The record of what changed in each round and why. Substantive, not '
        'process scaffolding.',
    'REPRODUCIBILITY.md':
        'The determinism defect, its repair and the surviving platform boundary.',
    'results/':
        'Every artefact a reported figure traces to.',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    print('=' * 78)
    print('INTERNAL PROCESS DOCUMENT REMOVAL   (close-out Section G, naming rule)')
    print('=' * 78)
    print('  Every file below stays retrievable at commit %s.\n' % RELEASE_COMMIT)

    removed, absent = [], []
    for path, reason in sorted(PROCESS_DOCS.items()):
        if not os.path.exists(path):
            absent.append(path)
            print('  [absent ] %s' % path)
            continue
        sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        size = os.path.getsize(path)
        removed.append(dict(path=path, sha256=sha, bytes=size, reason=reason,
                            retrievable_at=RELEASE_COMMIT))
        print('  [%s] %-46s %9d B' % ('would ' if a.dry_run else 'REMOVE', path, size))
        print('             %s' % reason)
        if not a.dry_run:
            os.remove(path)

    print('\n  removed: %d | already absent: %d' % (len(removed), len(absent)))
    print('\n  KEPT DELIBERATELY:')
    for p, why in KEPT.items():
        print('    %-22s %s' % (p, why))

    if not a.dry_run:
        path = os.path.join('results', 'superseded_artefacts.json')
        rec = json.load(open(path)) if os.path.exists(path) else {}
        rec['process_documents_removed'] = removed
        rec['process_documents_already_absent'] = absent
        rec['process_documents_retrievable_at_commit'] = RELEASE_COMMIT
        rec['process_documents_rationale'] = (
            'Close-out Section G requires that no lab identifier appear in the '
            'published code. These files are the remediation process rather than '
            'the study, and CHECKLIST.md additionally reproduced the supervisor\'s '
            'verification feedback verbatim in a public repository. They are '
            'superseded, not suppressed: all remain in git history at commit %s. '
            'No file that evidences a reported figure was touched, and verify.py '
            'still checks every item CHECKLIST.md tracked.' % RELEASE_COMMIT)
        json.dump(rec, open(path, 'w'), indent=2)
        print('\nwritten: %s' % path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
