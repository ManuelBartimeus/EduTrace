#!/usr/bin/env python3
"""
Round-4 manuscript corrections, applied to docs_out/ rather than rebuilt from
docs_in/.

Why not re-run the chain from docs_in. The round-3 release edit (M21's tag and
the platform-boundedness sentence) was applied to docs_out directly, so the
edit_docs_round2 -> _school -> _tabtransformer chain no longer reproduces the
committed documents. Re-running it here would silently drop that edit. This
script therefore takes the committed docs_out as its input, and every
replacement asserts the exact text it expects to find, so a document in an
unexpected state fails loudly instead of being half-edited.

It is idempotent: if a correction is already present the script says so and
moves on, which is what makes it safe to re-run after a partial application.

Every figure is read from the artefacts at edit time. Nothing here is typed.

The seven corrections, each traceable to a finding in the post-implementation
audit of the round-3 push:

  R12  C2 verdict now reads the state the certificate actually issues
  R11  Table 7's McNemar directions row, which the round-3 correction missed
  R5   DAS@1 compared against the DAS naive baseline, not the RPS one
  R1   a governing definition of "held-out", and the captions that travel alone
  R3   "sign changes" renamed to what the statistic counts
  M18  the countersignature declaration made accurate
  M19  the notebook's location stated correctly
"""
import json
import os

from docx import Document
from docx.oxml.ns import qn

RES = os.path.join('results')
R = json.load(open(os.path.join(RES, 'results.json')))
RPS = json.load(open(os.path.join(RES, 'rps_results.json')))
V = json.load(open(os.path.join(RES, 'closeout_verdict.json')))

M_PATH = os.path.join('docs_out', 'Group 3_Methods Section.docx')
R_PATH = os.path.join('docs_out', 'Group 3_Results and Analysis Section.docx')

CHANGES = []
SKIPPED = []


def log(item, what):
    CHANGES.append((item, what))


def _texts(p):
    """Every <w:t> node under a paragraph, in document order."""
    return p._p.findall('.//' + qn('w:t'))


def para_text(p):
    return ''.join(n.text or '' for n in _texts(p))


def replace_in_paragraph(p, old, new):
    """Replace `old` with `new` inside a paragraph without disturbing anything else.

    The obvious implementation — write the whole new string into the first run and
    delete the rest — destroys any <w:br/> the paragraph carries, because the line
    breaks live in the runs being deleted. R12 is a single paragraph holding the
    six clearance lines separated by breaks, so that implementation silently
    collapses the certificate into one block of prose. This edits the <w:t> nodes
    the matched span actually covers and leaves every other node, break included,
    exactly where it was.
    """
    nodes = _texts(p)
    full = ''.join(n.text or '' for n in nodes)
    start = full.find(old)
    if start < 0:
        return False
    end = start + len(old)
    pos, written = 0, False
    for n in nodes:
        s = n.text or ''
        a, b = pos, pos + len(s)
        pos = b
        if b <= start or a >= end:
            continue
        head = s[:start - a] if a < start else ''
        tail = s[end - a:] if b > end else ''
        n.text = head + ('' if written else new) + tail
        n.set(qn('xml:space'), 'preserve')
        written = True
    return True


def find(doc, snippet):
    for i, p in enumerate(doc.paragraphs):
        if snippet in para_text(p):
            return i, p
    raise KeyError('not found: %r' % snippet[:80])


def swap(doc, locator, old, new, item, what):
    """Replace `old` with `new` in the paragraph located by `locator`.

    Idempotent: if `new` is already in place the edit is recorded as skipped.
    Anything else is an error — a document in an unexpected state must not be
    edited on a guess.
    """
    try:
        i, p = find(doc, locator)
    except KeyError:
        i, p = find(doc, new[:60])
    if new in para_text(p):
        SKIPPED.append((item, what))
        return p
    if not replace_in_paragraph(p, old, new):
        raise AssertionError('%s: expected text not found: %r' % (item, old[:110]))
    log(item, what)
    return p


def find_cell(doc, snippet):
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                if any(snippet in para_text(p) for p in c.paragraphs):
                    return c
    raise KeyError('cell not found: %r' % snippet[:80])


def swap_cell(doc, locator, old, new, item, what):
    # `locator` names the row for the reader; the edit is made in whichever cell of
    # that row actually holds the text, which is not the one carrying the label.
    try:
        c = find_cell(doc, old)
    except KeyError:
        c = find_cell(doc, new)
    if any(new in para_text(p) for p in c.paragraphs):
        SKIPPED.append((item, what))
        return c
    for p in c.paragraphs:
        if replace_in_paragraph(p, old, new):
            log(item, what)
            return c
    raise AssertionError('%s: expected cell text not found: %r' % (item, old[:110]))


# =========================================================================
# figures read from the artefacts
# =========================================================================
XM = RPS['cross_model_sensitivity']
DAS_NAIVE_1 = XM['das_naive_fixed_order_at_1']
RPS_NAIVE_1 = XM['rps_naive_fixed_order_at_1']

MCN = R['stats']['mcnemar']
FAV_PROP = [r for r in MCN if 'proposed' in str(r.get('direction', '')).lower()]
FAV_COMP = [r for r in MCN if r not in FAV_PROP]

C2_STATE = V['clearance']['C2']
C2_WORD = C2_STATE.split('—')[0].strip().split()[0]

N_POS = V['v1a_summary']['n_positive']
N_NEG = V['v1a_summary']['n_negative']
N_MINORITY = V['v1a_summary']['sign_flips']

LOO = {r['dropped']: r['auc_pr'] for r in V['v1b_leave_one_out']}
LOO_DROP_SYNTH = LOO.get('synthetic')
LOO_DROP_REAL = LOO.get('real')

assert C2_WORD == 'UNVERIFIABLE', (
    'closeout_verdict.json still reports C2 as %r. Run closeout.py under the '
    'round-4 rule before editing the manuscript — the document must never state '
    'a verdict the certificate does not issue.' % C2_STATE)

rd = Document(R_PATH)
md = Document(M_PATH)

# =========================================================================
# R12 — the C2 verdict
# =========================================================================
c2_old = ('C2 — survives leave-one-out: FAIL. Dropping the synthetic supplement raises '
          'the primary metric and dropping the real records collapses it (R3), so the '
          'result does not survive the leave-one-out check in the direction the design '
          'assumes.')
c2_new = (
    'C2 — survives leave-one-out: %s. The condition asks whether every leave-one-out '
    'fold preserves the sign of the effect, and C1 has established that the effect has '
    'no stable sign at this sample size, so there is no sign for a fold to preserve. '
    'The certificate withholds a verdict here rather than returning one: closeout.py '
    'returns this state whenever C1 fails, and results/closeout_verdict.json records it. '
    'The measurements are unchanged and are reported in full in R3 — dropping the '
    'synthetic supplement raises the primary metric to %.4f and dropping the real '
    'records collapses it to %.4f — so what is withheld is the verdict and not the '
    'evidence. An earlier version of the certificate returned PASS or FAIL here by '
    'comparing each fold against the sign of a mean delta that lies well inside its own '
    'spread, which made the verdict turn on floating-point distance from zero rather '
    'than on the data; the change of state is recorded in the corrections ledger.'
    % (C2_WORD, LOO_DROP_SYNTH, LOO_DROP_REAL))
swap(rd, 'C2 — survives leave-one-out', c2_old, c2_new, 'R12',
     'C2 reports the state the certificate issues (%s), with the leave-one-out '
     'measurements retained and the change ledgered' % C2_WORD)

# =========================================================================
# R12 / R3 — what the ten-seed sign statistic counts
# =========================================================================
c1_old = ('The proposed-versus-attendance-ranker delta changed sign %d times across the '
          'ten seeds (R3)' % N_MINORITY)
c1_new = ('The proposed-versus-attendance-ranker delta carried the minority sign at %d '
          'of the ten seeds (R3)' % N_MINORITY)
swap(rd, 'C1 — sign stability across ten seeds', c1_old, c1_new, 'R12',
     'C1 describes the statistic the certificate computes — min(positive, negative) — '
     'rather than a count of transitions through an unordered set')

r3_old = ('the sign was negative at %d of the ten seeds and positive at %d, giving %d '
          'sign changes across the set' % (N_NEG, N_POS, N_MINORITY))
r3_new = ('the sign was negative at %d of the ten seeds and positive at %d, so %d of the '
          'ten carry the minority sign (sign_flips = %d in '
          'results/closeout_verdict.json)' % (N_NEG, N_POS, N_MINORITY, N_MINORITY))
swap(rd, 'Ten-seed sign stability', r3_old, r3_new, 'R3',
     'the same correction in the narrative: the seeds are unordered, so the statistic '
     'is a count of minority-sign seeds')

# =========================================================================
# R11 / Table 7 — the McNemar directions row the round-3 correction missed
# =========================================================================
dir_old = 'All four favour the comparator (R4, R10)'
dir_new = ('%d favour the comparator, %d favours the proposed model (R4, R10)'
           % (len(FAV_COMP), len(FAV_PROP)))
swap_cell(rd, 'McNemar directions', dir_old, dir_new, 'R11 / Table 7',
          'reconciliation row brought into line with Table 4 and R4/R10, which the '
          'round-3 direction correction did not reach')

# =========================================================================
# R5 — DAS compared against the DAS naive baseline
# =========================================================================
das_old = 'DAS@1 %.4f versus %.4f)' % (XM['das_at_1'], RPS_NAIVE_1)
das_new = 'DAS@1 %.4f versus %.4f)' % (XM['das_at_1'], DAS_NAIVE_1)
swap(rd, 'On the earlier cross-model population', das_old, das_new, 'R5',
     'directional agreement is compared against the directional-agreement naive '
     'baseline (%.4f), not the rank-preservation one (%.4f)'
     % (DAS_NAIVE_1, RPS_NAIVE_1))

art_old = ('the naive fixed-order baseline, the DAS figures under their own name, and '
           'the cross-model sensitivity population')
art_new = ('the naive fixed-order baseline, the DAS figures under their own name with '
           'their own baseline, and the cross-model sensitivity population with a '
           'separately named naive baseline for each of the two quantities')
swap(rd, 'These quantities are committed as a standalone artefact', art_old, art_new,
     'R5', 'the artefact description matches what the file now carries, so every figure '
           'quoted in the subsection is readable off it')

# =========================================================================
# R1 — a governing definition of "held-out", and the captions that travel alone
# =========================================================================
r1_old = ('The held-out evaluation partition comprised 248 real student-year records, of '
          'which 241 were non-dropout and 7 were dropout, a positive rate of 2.82% '
          '(7/248).')
r1_new = ('The evaluation partition comprised 248 real student-year records, of which 241 '
          'were non-dropout and 7 were dropout, a positive rate of 2.82% (7/248). It is '
          'held out by academic year and not by student: every student in it also appears '
          'in the 2024 training partition (Methods M9; R8), so wherever this section calls '
          'the partition held-out that is the sense meant, and every figure reported on it '
          'is next-year performance for students with prior-year register history rather '
          'than performance on students unseen in training.')
swap(rd, 'evaluation partition comprised 248 real student-year records', r1_old, r1_new,
     'R1', 'governing definition of "held-out" placed at its first use, as Q2 forward-fix '
           'block FF6 requires')

swap(rd, 'the primary metric is AUC-PR, named as primary in Methods M15',
     'Table 3 reports held-out test-set performance as mean ± standard deviation across '
     'the five seeds',
     'Table 3 reports test-set performance on that partition — held out by academic year, '
     'not by student (R1) — as mean ± standard deviation across the five seeds',
     'R2', 'the qualifier carried at the point the headline table is introduced')

swap(rd, 'Table 3. Held-out test-set performance',
     'Table 3. Held-out test-set performance (248 real records, 7 dropout), mean ± SD '
     'across five seeds.',
     'Table 3. Test-set performance (248 real records, 7 dropout; the partition is held '
     'out by academic year and not by student — every test student also appears in '
     'training, Methods M9 and R8), mean ± SD across five seeds.',
     'Table 3', 'caption qualified, because a caption is read away from R1')

swap(rd, 'Figure 4. ROC curves', 'on the held-out test set (seed 42)',
     'on the 248-record test partition (held out by academic year, not by student; '
     'seed 42)',
     'Figure 4', 'caption qualified')

swap(rd, 'Figure 6. SHAP feature importance', 'computed on the full held-out test set',
     'computed on the full 248-record test partition',
     'Figure 6', 'caption qualified')

swap(rd, 'Figure S1. Signed SHAP attributions', 'proposed XGBoost, full held-out test set',
     'proposed XGBoost, full 248-record test partition',
     'Figure S1', 'caption qualified')

# =========================================================================
# Methods — the declaration, and the notebook's location
# =========================================================================
decl_old = ('Pending countersignature, no sentence in this manuscript describes the '
            'evaluation partition as held-out without that qualifier.')
decl_new = ('Every use of the term held-out in the Results section is governed by the '
            'definition given at its first use in R1 — held out by academic year and not '
            'by student — and the tables and figure captions that report on the partition '
            'carry that qualifier with them. Pending countersignature, no sentence in this '
            'manuscript describes the partition as held out from students unseen in '
            'training, and none reports performance on it without that qualifier in '
            'force.')
swap(md, 'The student-disjoint sensitivity re-score specified in M9 is declared',
     decl_old, decl_new, 'M18 / declaration',
     'the declaration now states what the manuscript does, rather than asserting an '
     'absolute the captions did not meet')

m19_old = ('both present at the repository root of the tagged release')
m19_new = ('both committed in the tagged release — the pipeline at the repository root '
           'and the notebook under notebooks/')
swap(md, 'The reproduction path is EduTrace_Revised_Pipeline.py', m19_old, m19_new,
     'M19', 'the notebook is located where its own path says it is')

rd.save(R_PATH)
md.save(M_PATH)

print('updated:\n  %s\n  %s\n' % (R_PATH, M_PATH))
for item, what in CHANGES:
    print('  [%-14s] %s' % (item, what))
for item, what in SKIPPED:
    print('  [%-14s] already applied — %s' % (item, what))

prev = []
p = os.path.join('results', 'manuscript_changes.json')
if os.path.exists(p):
    prev = json.load(open(p))
json.dump(prev + [{'document': 'Methods' if i.startswith('M') else 'Results',
                   'item': i, 'change': w, 'round': 4} for i, w in CHANGES],
          open(p, 'w'), indent=2)
print('\n%d round-4 edits applied, %d already in place (%d edits recorded in total)'
      % (len(CHANGES), len(SKIPPED), len(prev) + len(CHANGES)))
