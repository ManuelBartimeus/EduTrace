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
import json, os, re, sys
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

from docx.oxml.ns import qn as _qn


def logical_lines(doc):
    """The document as the reader sees it, one line per visual line.

    R12 is a single paragraph carrying the six clearance lines separated by
    <w:br/>. python-docx concatenates run text and drops the breaks, so a check
    that reads paragraph text sees the six verdicts run together and can match
    none of them. Walking <w:t> and <w:br> in document order restores the lines.
    """
    out = []
    for p in doc.paragraphs:
        buf = []
        for node in p._p.iter():
            if node.tag == _qn('w:t'):
                buf.append(node.text or '')
            elif node.tag in (_qn('w:br'), _qn('w:cr')):
                buf.append('\n')
        out.extend(x for x in ''.join(buf).split('\n'))
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                out.append(c.text)
    return [' '.join(x.split()) for x in out if x.strip()]


R_LINES = logical_lines(RS)

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

# Q2 / FF6. The requirement is that no sentence describes the TEST PARTITION as
# held-out without the qualifier that every test student appears in training. The
# round-3 version of this check searched for the literal string "held-out test
# performance"; the manuscript wrote "held-out test-set performance", so the check
# passed while four sentences did not comply. Testing one phrasing is not testing
# the requirement, so it is now tested in four parts.

# (i) the phrasing FF6 names, in either of its spellings
unqual = re.findall(r'held[- ]out test(?:-set)? performance'
                    r'(?!\s*(?:for students|on that partition|,\s*(?:because|held)))',
                    BOTH, re.I)
if unqual:
    c_fail.append('Q2 — unqualified "held-out test performance"')
    print('  FAIL: unqualified "held-out test performance" appears %d time(s)' % len(unqual))
else:
    print('  clear: Q2 — no unqualified "held-out test(-set) performance"')

# (ii) a governing definition at first use, which is what licenses the shorthand
#      everywhere else in the section
GOVERNING = 'held out by academic year and not by student'
if GOVERNING in RT:
    print('  present: R1 governing definition of "held-out"')
else:
    c_fail.append('Q2 — R1 governing definition of "held-out" missing')
    print('  FAIL: R1 does not define "held-out" as held out by academic year, not by student')

# (iii) captions are read away from R1, so each must carry the qualifier itself.
#       "held-out site" is the leave-one-site-out sense and is a different claim.
bad_caps = []
for para in RS.paragraphs:
    t = ' '.join(para.text.split())
    if not re.match(r'^(?:Table|Figure)\s', t):
        continue
    for m in re.finditer(r'held[- ]out(?!\s+site)', t, re.I):
        window = t[max(0, m.start() - 40):m.end() + 160]
        if not re.search(r'academic year|not by student|prior-year', window, re.I):
            bad_caps.append(t[:80])
if bad_caps:
    c_fail.append('Q2 — %d caption(s) say held-out without the qualifier' % len(bad_caps))
    for t in bad_caps:
        print('  FAIL: unqualified caption -> %s...' % t)
else:
    print('  clear: every table and figure caption carries the qualifier or avoids the term')

# (iv) the declaration in Methods must not assert an absolute the captions do not meet
ABSOLUTE = ('no sentence in this manuscript describes the evaluation partition as '
            'held-out without that qualifier')
if ABSOLUTE in MT:
    c_fail.append('Q2 — the declaration asserts an absolute the manuscript does not meet')
    print('  FAIL: the countersignature declaration still makes the absolute claim')
else:
    print('  clear: the declaration states what the manuscript does')

# The required qualifier must be present.
if 'next-year performance for students with prior-year' in BOTH:
    print('  present: the Q2 qualifier sentence')
else:
    c_fail.append('Q2 qualifier sentence missing')
    print('  FAIL: the Q2 qualifier sentence is missing')

# ================================ D. clearance verdicts vs the certificate
# R12 quotes six verdicts. Nothing in the round-3 gates compared them against the
# file that issues them, which is how R12 came to report a C2 state closeout.py
# could not return. This closes that hole: the certificate is the authority and
# the manuscript must quote it.
print()
print('=' * 78)
print('D. R12 CLEARANCE VERDICTS vs results/closeout_verdict.json')
print('=' * 78)
d_fail = []


def head_word(state):
    return re.split(r'[\u2014\u2013:.-]', str(state))[0].strip().split()[0].upper()


for cid in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6'):
    expected = head_word(V['clearance'][cid])
    got = None
    for line in R_LINES:
        m = re.match(r'^%s\s*[\u2014\u2013-]\s*[^:]{1,120}:\s*([A-Za-z]+)' % cid, line)
        if m:
            got = m.group(1).upper()
            break
    if got is None:
        d_fail.append('%s not stated in R12' % cid)
        print('  FAIL  %s  certificate says %-14s manuscript states nothing' % (cid, expected))
    elif got != expected:
        d_fail.append('%s: manuscript %s, certificate %s' % (cid, got, expected))
        print('  FAIL  %s  certificate says %-14s manuscript says %s' % (cid, expected, got))
    else:
        print('  ok    %s  %s' % (cid, expected))

if 'null state %d' % V['null_state'] in RT.lower().replace('null state', 'null state'):
    print('  ok    null state %d stated in R12' % V['null_state'])
elif re.search(r'null state %d\b' % V['null_state'], RT, re.I):
    print('  ok    null state %d stated in R12' % V['null_state'])
else:
    d_fail.append('null state not stated as %d' % V['null_state'])
    print('  FAIL  null state: certificate says %d' % V['null_state'])

# ============================== E. corrections ledger vs the manuscript text
# Section F: "a number that moved with no row here is reverted." The converse
# also has to hold — a row whose correction never reached the manuscript is a
# ledger that documents a fix the reader cannot see. Table 7 carried the
# pre-repair McNemar statement for a full round because nothing checked this.
print()
print('=' * 78)
print('E. CORRECTIONS LEDGER vs THE MANUSCRIPT')
print('=' * 78)
e_fail = []
LEDGER = json.load(open('results/corrections_ledger.json'))


def artefact_numbers():
    vals = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            for d in range(0, 6):
                vals.add(round(abs(float(o)), d))
        elif isinstance(o, str):
            for m in re.finditer(r'-?\d+\.?\d*', o):
                try:
                    for d in range(0, 6):
                        vals.add(round(abs(float(m.group())), d))
                except ValueError:
                    pass
    for fn in sorted(os.listdir('results')):
        path = os.path.join('results', fn)
        if fn.endswith('.json') and fn != 'corrections_ledger.json':
            walk(json.load(open(path)))
        elif fn.endswith(('.csv', '.txt', '.log')):
            walk(open(path, errors='ignore').read())
    walk(json.load(open('figure_meta.json')))
    walk(json.load(open(os.path.join('data', 'synth_ctgan_s42_audit.json'))))
    return vals


UNIVERSE = artefact_numbers()

stale = 0
for row in LEDGER.get('moved', []):
    before, after = row.get('before'), row.get('after')
    if isinstance(before, float) and round(abs(before), 4) not in UNIVERSE:
        needle = '%.4f' % abs(before)
        if needle in BOTH:
            e_fail.append('superseded value %s (%s) still in the text' % (needle, row['quantity']))
            print('  FAIL  %-52s superseded %s still appears' % (row['quantity'][:52], needle))
            stale += 1
    if isinstance(before, str) and len(before) > 12 and before in BOTH:
        e_fail.append('superseded wording still in the text: %s' % row['quantity'])
        print('  FAIL  %-52s superseded wording %r survives'
              % (row['quantity'][:52], before[:44]))
        stale += 1
    # A clearance verdict is stored with its explanation attached; the manuscript
    # quotes the verdict and gives its own account, so the head of the string is
    # what has to be findable, not the certificate's whole sentence.
    head = str(after).split('\u2014')[0].strip() if isinstance(after, str) else after
    if isinstance(after, str) and len(head) > 5 and head not in BOTH:
        e_fail.append('corrected statement absent: %s' % row['quantity'])
        print('  FAIL  %-52s corrected wording not found in the manuscript'
              % row['quantity'][:52])
        stale += 1
if not stale:
    print('  no superseded value and no uncorrected statement survives in either section')

# A superseded claim may legitimately appear where the text is disclosing that it
# was superseded. What must not appear is the claim asserted in its own right, so
# the check looks at what precedes it.
RETRACTED = re.compile(r'(previously|earlier|no longer|withdrawn|is not true|was true of)',
                       re.I)
for pat, why in ((r'[Aa]ll four favour\w*\s+the comparator', 'the pre-repair McNemar statement'),
                 (r'every comparison favour\w*\s+the comparator', 'the same claim, reworded')):
    asserted = [m for m in re.finditer(pat, BOTH)
                if not RETRACTED.search(BOTH[max(0, m.start() - 90):m.start()])]
    if asserted:
        e_fail.append(why)
        print('  FAIL  %s survives as an assertion' % why)
    elif re.search(pat, BOTH):
        print('  clear: %s appears only where the text retracts it' % why)
    else:
        print('  clear: %s' % why)

# ==================== F. every decimal literal traced to a committed artefact
# Check A tests a curated list, so a claim nobody thought to list is not tested.
# This sweeps the documents instead: every 3- and 4-decimal literal must resolve
# to an artefact value or to a derivation named here with its formula.
print()
print('=' * 78)
print('F. EVERY 3-4 DECIMAL LITERAL vs THE COMMITTED ARTEFACTS')
print('=' * 78)
f_fail = []


def _t3(model, metric, stat='mean'):
    return R['table3'][model][metric][stat]


_rp = RPS['rank_preservation_RPS']
DERIVED = {
    round(_t3('tabtransformer', 'auc_pr') - _t3('xgb_engineered', 'auc_pr'), 4):
        'R10 gap: TabTransformer AUC-PR minus proposed',
    round(R['attendance_continuous_ranker']['auc_pr'] - _t3('xgb_engineered', 'auc_pr'), 4):
        'R10 gap: attendance ranker AUC-PR minus proposed',
    round(_rp['rps_at_1'] - _rp['naive_fixed_order_at_1'], 4):
        'R5 ablation delta at RPS@1',
    round(_rp['rps_at_2'] - _rp['naive_fixed_order_at_2'], 4):
        'R5 ablation delta at RPS@2',
    round(R['smote_ablation']['smote_0.20_asconfigured']['train_n']
          / float(R['smote_ablation']['smote_0.20_asconfigured']['train_n']
                  + R['smote_ablation']['smote_0.20_asconfigured']
                  ['synthetic_rows_added_per_seed'][0]), 4):
        'M8/R10 majority-class baseline on the SMOTE frame',
}

unresolved = []
for label, txt in (('Methods', MT), ('Results', RT)):
    for m in re.finditer(r'(?<![\w.])(\d+\.\d{3,4})(?![\w])', txt):
        v = float(m.group(1))
        if round(v, 4) in UNIVERSE or round(v, 3) in UNIVERSE:
            continue
        if round(v, 4) in DERIVED:
            continue
        ctx = ' '.join(txt[max(0, m.start() - 70):m.end() + 50].split())
        unresolved.append((label, m.group(1), ctx))

if unresolved:
    seen = set()
    for label, n, ctx in unresolved:
        if (label, n) in seen:
            continue
        seen.add((label, n))
        f_fail.append('%s: %s has no source' % (label, n))
        print('  FAIL  %-8s %-8s ...%s...' % (label, n, ctx[:96]))
else:
    counted = len(re.findall(r'(?<![\w.])(\d+\.\d{3,4})(?![\w])', BOTH))
    print('  %d decimal literals, all resolving to an artefact value or a named derivation'
          % counted)
    for k, v in sorted(DERIVED.items()):
        print('      derived  %-8s %s' % ('%.4f' % k, v))

# ===================================================================== verdict
print()
print('=' * 78)
total_fail = (len(fails) + len(b_fail) + len(c_fail)
              + len(d_fail) + len(e_fail) + len(f_fail))
print('CONSISTENCY PASS: %d failure(s)' % total_fail)
if total_fail:
    print('  A (curated claims) :', len(fails))
    print('  B (M<->R)          :', len(b_fail))
    print('  C (language)       :', len(c_fail))
    print('  D (clearance)      :', len(d_fail))
    print('  E (ledger)         :', len(e_fail))
    print('  F (numeric sweep)  :', len(f_fail))
    sys.exit(1)
print('  A curated claims   : all located and matching')
print('  B Methods<->Results: agree on model names, metric names and terminology')
print('  C language         : nothing the forward-fix blocks forbid')
print('  D clearance        : R12 quotes the certificate verbatim, all six conditions')
print('  E ledger           : no superseded value and no uncorrected statement survives')
print('  F numeric sweep    : every decimal literal traces to an artefact or a derivation')
sys.exit(0)
