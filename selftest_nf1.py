#!/usr/bin/env python3
"""
NF-1 self-test — proves the load-time provenance guard is load-bearing.

The forward-fix block FF3 asks for a self-test confirming two things after a
clean clone:

  (a) the test partition prints 248 rows;
  (b) "deleting the assertion line makes the run fail".

(a) is checked directly below.

(b) cannot be satisfied literally, and saying so is more useful than pretending
otherwise: on a correctly bounded synthesiser there are no synthetic rows in the
test period, so DELETING the guard changes nothing and the run still succeeds.
An assertion that never fires on clean data is exactly the unfalsifiable
guarantee the report objected to. What FF3 is actually asking — that the guard
be demonstrably load-bearing rather than decorative — is established by the
negative control in test (c): a single synthetic row is injected into the test
period, and the guard must raise. If it does not raise, the guard is decoration
and NF-1 is not closed.

Run:  python selftest_nf1.py
Exit: 0 = all three checks pass; 1 = a check failed.
"""
import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import EduTrace_Revised_Pipeline as P

REAL = os.path.join('data', 'real_student_data_CLEANED_school.csv')
SYNTH = os.path.join('data', 'synth_ctgan_s42.csv')
EXPECTED_TEST_N = 248

failures = []


def check(label, ok, detail=''):
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', label,
                           (' — ' + detail) if detail else ''))
    if not ok:
        failures.append(label)


print('=' * 74)
print('NF-1 SELF-TEST — provenance guard')
print('=' * 74)

# --- (a) the clean path -----------------------------------------------------
print('\n(a) clean load: the guard runs and the partition is what M9 states')
D = P.load_and_prepare(REAL, SYNTH, use_lag=False)
n_test = len(D['te'])
n_syn = int((D['te']['data_source'] == 'synthetic').sum())
n_pos = int(D['yte'].sum())
check('test partition prints %d rows' % EXPECTED_TEST_N, n_test == EXPECTED_TEST_N,
      'got %d' % n_test)
check('test partition holds 0 synthetic rows', n_syn == 0, 'got %d' % n_syn)
check('test partition holds at least one positive', n_pos > 0, 'got %d' % n_pos)
check('the guard emitted its run-log line', len(P.NF1_LOG) > 0,
      P.NF1_LOG[-1] if P.NF1_LOG else 'no line emitted')

# --- (b) the synthesiser bound (Q9) ----------------------------------------
print('\n(b) Q9: the synthesiser is bounded to the training period')
synth = pd.read_csv(SYNTH)
max_year = int(synth[P.YEAR].max())
n_in_test_period = int((synth[P.YEAR] >= P.SPLIT_YEAR).sum())
check('synthetic max academic_year < %d' % P.SPLIT_YEAR, max_year < P.SPLIT_YEAR,
      'max = %d' % max_year)
check('synthetic rows in a test-period year == 0', n_in_test_period == 0,
      'got %d' % n_in_test_period)

# --- (c) the negative control: the guard must actually fire -----------------
print('\n(c) negative control: inject one synthetic row into the test period')
real = pd.read_csv(REAL)
contaminated = pd.read_csv(SYNTH)
contaminated.loc[contaminated.index[0], P.YEAR] = P.SPLIT_YEAR   # 2024 -> 2025
tmp = os.path.join('results', '_nf1_contaminated_tmp.csv')
os.makedirs('results', exist_ok=True)
contaminated.to_csv(tmp, index=False)
raised = False
try:
    P.load_and_prepare(REAL, tmp, use_lag=False)
except P.SyntheticInTestPartition as exc:
    raised = True
    print('      guard raised as required:')
    print('      %s' % str(exc).replace('\n', ' '))
except AssertionError as exc:
    raised = True
    print('      guard raised (AssertionError): %s' % exc)
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
check('guard RAISES on a contaminated test partition', raised,
      'the guard is decoration if this fails')

print('\n' + '=' * 74)
if failures:
    print('NF-1 SELF-TEST: FAILED — %d check(s): %s' % (len(failures), '; '.join(failures)))
    sys.exit(1)
print('NF-1 SELF-TEST: ALL CHECKS PASSED')
print('The guard is present in the load path, it passes on the %d-record real'
      % EXPECTED_TEST_N)
print('partition, and it demonstrably raises when the partition is contaminated.')
sys.exit(0)
