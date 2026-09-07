#!/usr/bin/env python3
"""
Final consistency pass.

Three separate checks, run against the REVISED documents:

  A. Every numeric claim in the prose that this pass knows how to locate is
     compared against the artefact it is supposed to come from. A figure that
     appears in the text but not in the run is a failure, not a rounding note.
  B. Methods and Results are checked against each other for model names, metric
     names and terminology.
  C. The manuscript is checked for language the forward-fix blocks forbid while
     Q21, Q22 and Q2 remain open.

Prints a report and exits non-zero if any check fails.
"""
import json, re, sys
import docx

R = json.load(open('results/results.json'))
V = json.load(open('results/closeout_verdict.json'))
RPS = json.load(open('results/rps_results.json'))
NN = json.load(open('results/smote_nnaa.json'))

M = docx.Document('docs_out/Group 3_Methods Section.docx')
RS = docx.Document('docs_out/Group 3_Results and Analysis Section.docx')


def alltext(doc):
    parts = [p.text for p in doc.paragraphs]
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                parts.append(c.text)
    return '\n'.join(parts)


MT, RT = alltext(M), alltext(RS)
BOTH = MT + '\n' + RT

fails, warns, oks = [], [], []


def expect(label, needle, haystack, where):
    if needle in haystack:
        oks.append((label, needle, where))
    else:
        fails.append((label, needle, where))


def expect_absent(label, needle, haystack, where):
    if needle not in haystack:
        oks.append((label, 'ABSENT: ' + needle, where))
    else:
        fails.append((label, 'SHOULD BE ABSENT: ' + needle, where))


# =========================================================== A. numeric claims
print('=' * 78)
print('A. NUMERIC CLAIMS vs THE RUN ARTEFACTS')
print('=' * 78)

t3 = R['table3']
checks = [
    # partition
    ('R1 test n', str(R['test']['n']), RT),
    ('R1 test positives', str(R['test']['pos']), RT),
    ('R1 base rate', '%.2f%%' % R['test']['pct_pos'], RT),
    ('R1/M9 pool n', str(R['pool']['n']), BOTH),
    ('R1 pool positives', str(R['pool']['pos']), RT),
    ('R1 pool ratio', '%.2f:1' % R['pool']['ratio'], RT),
    # Table 3 headline levels
    ('R2 proposed AUC-PR', '%.4f' % t3['xgb_engineered']['auc_pr']['mean'], RT),
    ('R2 default AUC-PR', '%.4f' % t3['xgb_default']['auc_pr']['mean'], RT),
    ('R2 tabtransformer AUC-PR', '%.4f' % t3['tabtransformer']['auc_pr']['mean'], RT),
    ('R2 decision tree AUC-PR', '%.4f' % t3['decision_tree']['auc_pr']['mean'], RT),
    ('R2 attendance ranker AUC-PR', '%.4f' % R['attendance_continuous_ranker']['auc_pr'], RT),
    ('R2 proposed recall', '%.4f' % t3['xgb_engineered']['recall']['mean'], RT),
    ('R2 attendance rule recall', '%.4f' % R['attendance_binary_rule']['recall'], RT),
    ('R2 rule single-point AUC-ROC',
     '%.4f' % R['attendance_binary_rule']['single_point_auc_roc'], RT),
    # Q21 anchor
    ('R5 RPS@1', '%.4f' % RPS['rank_preservation_RPS']['rps_at_1'], RT),
    ('R5 RPS@2', '%.4f' % RPS['rank_preservation_RPS']['rps_at_2'], RT),
    ('R5 naive RPS@1', '%.4f' % RPS['rank_preservation_RPS']['naive_fixed_order_at_1'], RT),
    ('R5 naive RPS@2', '%.4f' % RPS['rank_preservation_RPS']['naive_fixed_order_at_2'], RT),
    ('R5 DAS@1', '%.4f' % RPS['directional_agreement_DAS']['das_at_1'], RT),
    ('R5 DAS@2', '%.4f' % RPS['directional_agreement_DAS']['das_at_2'], RT),
    ('R5 population n', str(RPS['n_evaluated']), RT),
    ('R5 threshold', str(RPS['threshold_used']), RT),
    ('R5 cross-model n', str(RPS['cross_model_sensitivity']['n_evaluated']), RT),
    # imbalance / privacy
    ('R10 SMOTE 0.50 arm',
     '%.4f' % R['smote_ablation']['smote_0.50']['summary']['auc_pr']['mean'], RT),
    ('R10 spw=1 arm',
     '%.4f' % R['layer2_ablation']['spw_1__f2_threshold']['summary']['auc_pr']['mean'], RT),
    ('R10 SMOTE k=3 NNAA', '%.4f' % NN['summary']['SMOTE k=3']['seed42'], RT),
    ('R10 SMOTE k=5 NNAA', '%.4f' % NN['summary']['SMOTE k=5']['seed42'], RT),
    # student disjointness
    ('M9/R8 disjoint records', str(R['student_disjoint']['n_disjoint_records']), BOTH),
    ('M9 test students', str(R['student_disjoint']['n_test_students']), MT),
    # close-out certificate
    ('R3 ten-seed delta mean', '%+.4f' % V['v1a_summary']['delta_mean'], RT),
    ('R3 ten-seed delta SD', '%.4f' % V['v1a_summary']['delta_sd'], RT),
    ('R3 comparator AUC-PR', '%.4f' % V['v1a_summary']['comparator_auc_pr'], RT),
    ('R3 LOO real-only', '%.4f' % V['v1b_leave_one_out'][0]['auc_pr'], RT),
    ('R3 LOO synthetic-dropped', '%.4f' % V['v1b_leave_one_out'][1]['auc_pr'], RT),
    ('R4 discordant pairs', str(V['v1c']['n_discordant']), RT),
    ('R4 MDE split', '%.3f' % V['v1c']['mde_split'], RT),
    ('R4 required discordant pairs', str(V['v1c']['discordant_pairs_required']), RT),
    ('R12 grid cell count', str(len(V['v4_grid'])), RT),
    ('M13/M20 threshold grid points', str(V['v1f_parity'][0]['threshold_trials']), MT),
    # NF-1
    ('M10 guard function name', 'assert_test_partition_is_real', MT),
    ('M10/R1 guard log line', R['nf1_guard_log'][0], BOTH),
    ('M19 named file 1', 'EduTrace_Revised_Pipeline.py', MT),
    ('M19 named file 2', 'EduTrace_Main_3.ipynb', MT),
    ('M21 rps artefact', 'rps_results.json', MT),
    ('R5 rps artefact', 'rps_results.json', RT),
]
for label, needle, hay in checks:
    expect(label, needle, hay, 'prose')

# --- school-identifier claims (Q2) -----------------------------------------
SS = json.load(open('results/school_structure.json'))
LO = json.load(open('results/loso_validation.json'))
COMPF = [f for f in LO['folds'] if f.get('computable')]
school_checks = [
    ('M3 site count', str(SS['n_sites']), MT),
    ('Table 1b / M3 site names', 'Al Huda Islamic and JHS', MT),
    ('M3 site name 2', 'Tawjeed Islamic and JHS', MT),
    ('M3 site name 3', 'Ayaarno M/A Primary and JHS', MT),
    ('M3 site name 4', 'Buokrom Block A M/A Primary and JHS', MT),
    ('Table 2b gender confound', '%.4f' % SS['categorical_confounding']['gender']['cramers_v'], MT),
    ('R6 gender confound', '%.4f' % SS['categorical_confounding']['gender']['cramers_v'], RT),
    ('R8 site recoverability acc', '%.4f' % SS['site_recoverability']['cv_accuracy'], RT),
    ('R8 site recoverability base', '%.4f' % SS['site_recoverability']['majority_baseline'], RT),
]
for f in COMPF:
    school_checks.append(('R8 LOSO %s AUC-PR' % f['held_out_site'],
                          '%.4f' % f['summary']['auc_pr']['mean'], RT))
    school_checks.append(('R8 LOSO %s test n' % f['held_out_site'], str(f['test_n']), RT))
for label, needle, hay in school_checks:
    expect(label, needle, hay, 'prose')
# The UNRECORDED declaration must be gone.
expect_absent('M3 UNRECORDED site count',
              'the exact N of sites is reported here as UNRECORDED', MT, 'Methods')
expect_absent('M18 school identifier absent claim',
              'which is absent from the collected data', MT, 'Methods')

for label, needle, where in fails:
    print('  FAIL  %-34s %r' % (label, needle))
print('  %d of %d numeric/name claims located in the revised prose'
      % (len(oks), len(oks) + len(fails)))
print('  (includes %d school-identifier claims and 2 absence checks)'
      % len(school_checks))

# ================================================ B. Methods <-> Results agree
print()
print('=' * 78)
print('B. METHODS <-> RESULTS AGREEMENT')
print('=' * 78)
b_fail = []

model_terms = ['proposed XGBoost', 'default XGBoost', 'Decision Tree', 'TabTransformer',
               'Attendance rule', 'Attendance ranker']
for t in model_terms:
    inm, inr = (t.lower() in MT.lower()), (t.lower() in RT.lower())
    status = 'both' if (inm and inr) else ('Methods only' if inm else
                                           ('Results only' if inr else 'NEITHER'))
    print('  %-22s %s' % (t, status))
    if not (inm and inr):
        b_fail.append(t)

for term, label in [('RPS', 'rank preservation'), ('DAS', 'directional agreement'),
                    ('AUC-PR', 'primary metric'), ('scale_pos_weight', 'Layer 2'),
                    ('sampling_strategy', 'Layer 1'), ('F2', 'threshold objective')]:
    inm, inr = term in MT, term in RT
    print('  %-22s Methods=%s Results=%s  (%s)' % (term, inm, inr, label))
    if not (inm and inr):
        b_fail.append(term)

# A method described in Methods must be reflected in Results.
for phrase, where in [('ten-seed', RT), ('minimum detectable effect', RT),
                      ('leave-one-out', RT)]:
    if phrase.lower() not in where.lower():
        b_fail.append('Methods declares %r but Results does not report it' % phrase)
        print('  FAIL: Methods declares %r but Results does not report it' % phrase)

# The old model/pipeline names must not survive anywhere.
for stale in ['EduTrace_Main_2.ipynb', 'final_summary_v3.json', 'synthetic_student_data_v3',
              'pipeline.py', '/content/drive']:
    if stale in BOTH:
        b_fail.append('stale reference: ' + stale)
        print('  FAIL: stale reference to %r survives in the manuscript' % stale)
if not b_fail:
    print('  no disagreement found')

# ============================================ C. forbidden language while open
print()
print('=' * 78)
print('C. LANGUAGE THE FORWARD-FIX BLOCKS FORBID WHILE Q21 / Q22 / Q2 STAND')
print('=' * 78)
c_fail = []
forbidden = [
    # Q22: the negative-result headline may not be written anywhere
    (r'outperform\w*\s+machine learning', 'Q22 — the attendance-rule-beats-ML sentence'),
    (r'machine learning\s+\w*\s*underperform', 'Q22 — the same claim, inverted'),
    # Q21: no validated-contribution claim
    (r'validated contribution', 'Q21 — contribution claimed as validated'),
    (r'demonstrat\w+ that SHAPtoSMS', 'Q21 — contribution claimed as demonstrated'),
    # general over-claim
    (r'\bproves\b', 'over-claim: "proves"'),
    (r'\bno effect\b', 'C3 — "no effect" is forbidden for an underpowered comparison'),
    (r'statistically equivalent', 'C3 — equivalence claim from an underpowered test'),
]
for pat, why in forbidden:
    hits = [m.group(0) for m in re.finditer(pat, BOTH, re.I)]
    if hits:
        c_fail.append(why)
        print('  FAIL: %s -> found %s' % (why, set(hits)))
    else:
        print('  clear: %s' % why)

# Q2: "held-out test performance" must never appear unqualified.
unqual = re.findall(r'held-out test performance(?!\s*(?:for students|,\s*because))', BOTH, re.I)
if unqual:
    c_fail.append('Q2 — unqualified "held-out test performance"')
    print('  FAIL: unqualified "held-out test performance" appears %d time(s)' % len(unqual))
else:
    print('  clear: Q2 — no unqualified "held-out test performance"')

# The required qualifier must be present.
if 'next-year performance for students with prior-year' in BOTH:
    print('  present: the Q2 qualifier sentence')
else:
    c_fail.append('Q2 qualifier sentence missing')
    print('  FAIL: the Q2 qualifier sentence is missing')

# ===================================================================== verdict
print()
print('=' * 78)
total_fail = len(fails) + len(b_fail) + len(c_fail)
print('CONSISTENCY PASS: %d failure(s)' % total_fail)
if total_fail:
    print('  A (numeric):', len(fails))
    print('  B (M<->R)  :', len(b_fail))
    print('  C (language):', len(c_fail))
    sys.exit(1)
print('  A numeric claims : all located and matching')
print('  B Methods<->Results: agree on model names, metric names and terminology')
print('  C language        : nothing the forward-fix blocks forbid')
sys.exit(0)
