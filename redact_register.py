#!/usr/bin/env python3
"""
Withdraw or de-identify the field register in the published repository.

NOT RUN AUTOMATICALLY. Nothing in the reproduction sequence calls this, and it
changes nothing until it is invoked deliberately. It exists so that whichever
option the supervisor and the ethics committee choose can be applied exactly,
recorded, and re-run by anyone checking the repository.

THE PROBLEM
-----------
`data/real_student_data*.csv` publishes 428 real student-year records —
pseudonymous student identifiers, but named schools beside gender, grade level,
distance to school, fee-payment status, attendance, examination score and dropout
outcome. Methods M21 states the opposite: that the field institutional records
cannot be shared, consistent with the Ghana Data Protection Act, 2012 and the
HuSSREC approval. A reviewer can check that in one click. Separately, four named
schools with 20 to 80 students each, published beside those quasi-identifiers,
are a re-identification surface regardless of what M21 says.

THE TRADE-OFF, STATED PLAINLY
-----------------------------
The committed register is what makes the clean-clone reproduction work. A full
re-run currently reproduces 859 of 859 leaves of results.json bit-identically,
which is the evidence that answers Q4. Withdrawing the register removes that
evidence. De-identifying it does not.

    --mode names      Remove the school_name column only. Keeps SCH01-SCH04, so
                      every model number, every partition figure and the whole
                      leave-one-site-out analysis still reproduce exactly. The
                      four institutions are no longer identified.
                      COST: the manuscript names the schools in R8 and in Methods
                      Table 1b, so those sentences must be edited to carry the
                      site codes alone. This script lists them and does not touch
                      the documents.

    --mode withdraw   Remove the three register files entirely and publish
                      data/register_schema.json in their place: the column
                      schema, the partition shape, per-site counts and the base
                      rates, which is everything the partition-level checks need.
                      COST: no model-derived quantity reproduces from a clean
                      clone any more. Q4's answer narrows to the partition-level
                      checks plus the committed model artefacts, and M21, M19 and
                      REPRODUCIBILITY.md all have to say so.

Both modes write results/register_disclosure.json recording what was done, and
both leave the files retrievable in git history — so if the decision is to
withdraw, the history must be rewritten as well. That step is deliberately NOT
automated here: it is irreversible, it invalidates every published commit hash
including the one M21 cites, and it needs the supervisor's instruction in
writing.

    python redact_register.py --mode names --dry-run
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

REGISTER = ['data/real_student_data.csv',
            'data/real_student_data_CLEANED.csv',
            'data/real_student_data_CLEANED_school.csv']

MANUSCRIPT_FOLLOW_UPS = {
    'names': [
        'R8 — the leave-one-site-out paragraph names each school beside its code '
        '("SCH01 (Al Huda Islamic and JHS, 145 test records ...)"). Cut to the code.',
        'Methods Table 1b / M3 — the sampling frame table names the four schools. '
        'Cut to the codes, and state that the names are withheld under the HuSSREC '
        'approval.',
        'M21 — replace "the field institutional records cannot be shared" with a '
        'sentence describing what is shared: a de-identified register with the '
        'institution names withheld, sufficient to reproduce every reported figure.',
        'add_school_identifier.py:88 writes school_name from a mapping held in the '
        'script itself, so the names would return on the next full reproduction and '
        'the mapping would still be published in the source. Remove the mapping and '
        'the column from that script in the same commit, or the redaction is cosmetic.',
        'school_structure.py and loso_validation.py read school_name for their labels '
        'and would raise once the column is gone. Both fall back to school_id.',
    ],
    'withdraw': [
        'M21 — state that the register is held under controlled access, name the '
        'contact, and say that data/register_schema.json plus the committed model '
        'artefacts are what a reader can reproduce from.',
        'M19 / README — the reproduction path no longer reaches the model-derived '
        'quantities from a clean clone. Say so rather than leaving the sequence '
        'reading as though it does.',
        'REPRODUCIBILITY.md — the "how to check a reproduction on other hardware" '
        'section assumes the register is present. Rewrite around the schema and '
        'the committed artefacts.',
        'R8 and Table 1b — per-site counts survive in the schema, so they can stay, '
        'but they must cite the schema rather than the register.',
    ],
}


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def schema_from(path):
    d = pd.read_csv(path)
    per_site = (d.groupby('school_id')
                 .apply(lambda g: dict(records=int(len(g)),
                                       students=int(g.student_id.str.split('_').str[0].nunique()),
                                       dropouts=int(g.dropout.sum()),
                                       base_rate=round(float(g.dropout.mean()), 4)),
                        include_groups=False)
                 .to_dict())
    test = d[d.academic_year >= 2025]
    train = d[d.academic_year < 2025]
    return dict(
        note=('Schema and partition shape of the field register, published in place '
              'of the record-level data. Sufficient for every partition-level check '
              'in REPRODUCIBILITY.md; not sufficient to refit the models.'),
        columns={c: str(d[c].dtype) for c in d.columns},
        n_records=int(len(d)),
        n_students=int(d.student_id.str.split('_').str[0].nunique()),
        years=sorted(int(y) for y in d.academic_year.unique()),
        train_partition=dict(n=int(len(train)), positives=int(train.dropout.sum())),
        test_partition=dict(n=int(len(test)), positives=int(test.dropout.sum()),
                            base_rate=round(float(test.dropout.mean()), 4)),
        per_site=per_site,
        students_first_observed_2024=int(
            (d.groupby(d.student_id.str.split('_').str[0]).academic_year.min() == 2024).sum()),
        withheld=('Record-level values and institution names are withheld under the '
                  'Ghana Data Protection Act, 2012 and the HuSSREC approval.'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('names', 'withdraw'), required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    present = [p for p in REGISTER if os.path.exists(p)]
    if not present:
        sys.exit('no register files found — nothing to do')

    print('=' * 78)
    print('REGISTER DISCLOSURE — mode: %s%s' % (a.mode, '  (dry run)' if a.dry_run else ''))
    print('=' * 78)

    record = dict(mode=a.mode, files=[])
    for p in present:
        record['files'].append(dict(path=p, sha256_before=sha(p), bytes=os.path.getsize(p)))

    if a.mode == 'names':
        for p in present:
            d = pd.read_csv(p)
            if 'school_name' not in d.columns:
                print('  [skip   ] %s — no school_name column' % p)
                continue
            print('  [%s] %s — dropping school_name, keeping %s'
                  % ('would  ' if a.dry_run else 'REDACT ', p,
                     ', '.join(sorted(d.school_id.unique()))))
            if not a.dry_run:
                d.drop(columns=['school_name']).to_csv(p, index=False)
        record['effect'] = ('school_name removed; school_id retained, so every reported '
                            'figure still reproduces from a clean clone')
    else:
        src = 'data/real_student_data_CLEANED_school.csv'
        schema = schema_from(src) if os.path.exists(src) else None
        for p in present:
            print('  [%s] %s' % ('would  ' if a.dry_run else 'REMOVE ', p))
            if not a.dry_run:
                os.remove(p)
        if not a.dry_run and schema:
            json.dump(schema, open('data/register_schema.json', 'w'), indent=2)
            print('  written: data/register_schema.json')
        record['effect'] = ('register withdrawn; data/register_schema.json published in its '
                            'place. Model-derived quantities no longer reproduce from a '
                            'clean clone.')

    print('\n  MANUSCRIPT EDITS THIS MODE REQUIRES (not applied by this script):')
    for line in MANUSCRIPT_FOLLOW_UPS[a.mode]:
        print('    - %s' % line)
    print('\n  GIT HISTORY: the files remain retrievable at every published commit. If the '
          '\n  decision is that they must not be recoverable at all, the history has to be '
          '\n  rewritten, which invalidates the commit hash M21 cites. That step is not '
          '\n  automated here and needs the supervisor\'s instruction in writing.')

    if not a.dry_run:
        for f in record['files']:
            f['sha256_after'] = sha(f['path']) if os.path.exists(f['path']) else None
        record['manuscript_edits_required'] = MANUSCRIPT_FOLLOW_UPS[a.mode]
        os.makedirs('results', exist_ok=True)
        json.dump(record, open('results/register_disclosure.json', 'w'), indent=2)
        print('\nwritten: results/register_disclosure.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
