#!/usr/bin/env python3
"""
Q2 — attaches the school identifier to the analysed register.

The register as collected carried no school-identifier column, so Methods M3 had
to report the number of sites as UNRECORDED and Methods M18 listed a school
identifier as one of two ingredients required to move beyond the Tier 1 claim
ceiling. The identifier was recovered from the schools while the HuSSREC
approval and site access were live, and is applied here.

SOURCE OF THE MAPPING
---------------------
Supplied by the research group from the participating schools, as contiguous
student-number ranges:

    S1   - S80    Al Huda Islamic and JHS
    S81  - S130   Tawjeed Islamic and JHS
    S131 - S150   Ayaarno M/A Primary and JHS
    S151 - S180   Buokrom Block A M/A Primary and JHS

TRANSCRIPTION NOTE. The second range was supplied as "S801 to S130". S801 does
not exist in the register — student numbers run S1 to S180 with no gaps — and
the four ranges are contiguous and sum to exactly 180 students (80 + 50 + 20 +
30). It is therefore read as S81, and this script asserts both properties rather
than assuming them: every student number in the register must fall in exactly
one range, and the ranges must partition the register with no gap and no
overlap. If either assertion fails, nothing is written.

WHAT THIS DOES NOT DO
---------------------
school_id is attached as a RECORD-LEVEL IDENTIFIER, not as a model feature. The
feature set in EduTrace_Revised_Pipeline.py is an explicit list of six register
variables and is unchanged, so no reported model figure moves. The identifier is
used for two things only: reporting the sampling frame (M3, Table 1b), and
enabling the leave-one-site-out validation in loso_validation.py that Methods
M18 names.

Adding school membership as a predictor would be a modelling change nobody
asked for, and on this panel it would be close to fitting a school-level
intercept on 20-80 students per site.
"""
import os
import pandas as pd

SRC = os.path.join('data', 'real_student_data_CLEANED.csv')
DST = os.path.join('data', 'real_student_data_CLEANED_school.csv')

# (first student number, last student number, school code, school name)
SCHOOLS = [
    (1,   80,  'SCH01', 'Al Huda Islamic and JHS'),
    (81,  130, 'SCH02', 'Tawjeed Islamic and JHS'),
    (131, 150, 'SCH03', 'Ayaarno M/A Primary and JHS'),
    (151, 180, 'SCH04', 'Buokrom Block A M/A Primary and JHS'),
]
SCHOOL_NAME = {c: n for _, _, c, n in SCHOOLS}


def school_of(num):
    for lo, hi, code, _ in SCHOOLS:
        if lo <= num <= hi:
            return code
    return None


def main():
    df = pd.read_csv(SRC)
    base = df['student_id'].str.split('_').str[0]
    num = base.str[1:].astype(int)

    # --- assertions on the mapping, before anything is written ---------------
    ranges = [(lo, hi) for lo, hi, _, _ in SCHOOLS]
    for (lo1, hi1), (lo2, hi2) in zip(ranges, ranges[1:]):
        assert lo2 == hi1 + 1, 'ranges are not contiguous: %d then %d' % (hi1, lo2)
    span = set()
    for lo, hi in ranges:
        seg = set(range(lo, hi + 1))
        assert not (span & seg), 'ranges overlap'
        span |= seg
    observed = set(num.unique())
    assert observed <= span, ('student numbers outside every supplied range: %s'
                              % sorted(observed - span))
    unused = span - observed
    if unused:
        print('  NOTE: %d numbers in the supplied ranges have no register record: %s'
              % (len(unused), sorted(unused)[:10]))

    df['school_id'] = num.map(school_of)
    df['school_name'] = df['school_id'].map(SCHOOL_NAME)
    assert df['school_id'].notna().all(), 'unmapped records remain'

    # Column order: identifier columns first, then the register, target last.
    cols = ['student_id', 'school_id', 'school_name', 'academic_year',
            'attendance_rate', 'exam_score', 'fee_payment_status', 'distance_km',
            'grade_level', 'gender', 'dropout']
    df = df[cols]
    df.to_csv(DST, index=False)

    print('wrote %s' % DST)
    print('  records %d | students %d | sites %d'
          % (len(df), base.nunique(), df['school_id'].nunique()))
    print()
    print('%-7s %-38s %-12s %8s %8s %9s %10s'
          % ('CODE', 'SCHOOL', 'ID RANGE', 'STUDENTS', 'RECORDS', 'DROPOUTS', 'BASE RATE'))
    for lo, hi, code, name in SCHOOLS:
        d = df[df['school_id'] == code]
        sid = d['student_id'].str.split('_').str[0]
        print('%-7s %-38s %-12s %8d %8d %9d %10.4f'
              % (code, name[:38], 'S%d-S%d' % (lo, hi), sid.nunique(), len(d),
                 int(d['dropout'].sum()), float(d['dropout'].mean())))


if __name__ == '__main__':
    main()
