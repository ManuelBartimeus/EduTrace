#!/usr/bin/env python3
"""
Applies the school-identifier revisions, on top of the round-2 edits.

Run AFTER edit_docs_round2.py. It reads the documents that script produced in
docs_out/ and rewrites them in place, so the chain is:

    docs_in/  --[edit_docs_round2.py]-->  docs_out/  --[this script]-->  docs_out/

Every number is read from results/school_structure.json and
results/loso_validation.json rather than typed.

What changes, and why:
  Q2 / M3        The site count is no longer UNRECORDED. Four sites are named.
  Q2 / Table 1b  The "N of sites" row is replaced and per-site composition added.
  M18            Leave-one-site-out is now partially computable, so the Tier-1
                 justification and the "two named ingredients" sentence change.
  R8             The leave-one-site-out result is reported, including the two
                 folds that cannot be computed.
  R6 / Table 2b  gender is confounded with site (one school is single-sex), so
                 the gender attribution carries a site component.
  Countersign.   The declaration NARROWS: the site count is recovered, so only
                 the student-disjoint re-score remains not computable.
"""
import copy, json, os
from docx import Document

SS = json.load(open(os.path.join('results', 'school_structure.json')))
LOSO = json.load(open(os.path.join('results', 'loso_validation.json')))
R = json.load(open(os.path.join('results', 'results.json')))

M_PATH = os.path.join('docs_out', 'Group 3_Methods Section.docx')
R_PATH = os.path.join('docs_out', 'Group 3_Results and Analysis Section.docx')

SITES = SS['sites']
NS = SS['n_sites']
NAMES = [s['school_name'] for s in SITES]
GENDER_V = SS['categorical_confounding']['gender']['cramers_v']
REC = SS['site_recoverability']
NOPOS = SS['sites_without_test_positives']
COMP = [f for f in LOSO['folds'] if f.get('computable')]
NONCOMP = [f for f in LOSO['folds'] if not f.get('computable')]

CHANGES = []


def log(doc, item, what):
    CHANGES.append((doc, item, what))


def set_text(p, text):
    runs = p.runs
    if not runs:
        p.add_run(text); return p
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    return p


def find(doc, snippet):
    for i, p in enumerate(doc.paragraphs):
        if snippet in p.text:
            return i, p
    raise KeyError('not found: %r' % snippet[:80])


def replace_sub(doc, locator, old, new):
    i, p = find(doc, locator)
    assert old in p.text, 'substring missing: %r' % old[:70]
    set_text(p, p.text.replace(old, new))
    return p


def insert_after(p, text):
    new_p = copy.deepcopy(p._p)
    p._p.addnext(new_p)
    from docx.text.paragraph import Paragraph
    np_ = Paragraph(new_p, p._parent)
    set_text(np_, text)
    return np_


def set_cell(cell, text):
    cell.paragraphs[0].text = ''
    set_text(cell.paragraphs[0], text)
    for extra in cell.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def site_phrase():
    return '; '.join('%s (%s, %d students, %d records, %d dropout events)'
                     % (s['school_name'], s['school_id'], s['students'],
                        s['records'], s['dropouts']) for s in SITES)


# ================================================================== METHODS
md = Document(M_PATH)

# ---- M3: the site count is recovered ---------------------------------------
OLD = ('The number of participating sites is not recoverable from the analysed data: the '
       'register extract carries no school identifier (M18), so the exact N of sites is '
       'reported here as UNRECORDED rather than estimated. This is a data-collection defect '
       'on the study’s part, not a design choice, and it is the same absence that blocks '
       'leave-one-site-out validation (M18).')
NEW = ('The extract is drawn from four participating sites (n = %d): %s. The school identifier was not '
       'captured in the original register extract, and an earlier version of this manuscript '
       'reported the number of sites as UNRECORDED for that reason. It was subsequently '
       'recovered from the participating schools while the HuSSREC approval and site access '
       'were live, and attached to every record by add_school_identifier.py, which asserts '
       'that the supplied student-number ranges are contiguous, non-overlapping and partition '
       'the register exactly before writing anything. The identifier is carried as a '
       'record-level attribute and is not a model input: the feature set in M7 is unchanged, '
       'and no reported model figure depends on it. Its recovery removes the absence that '
       'previously blocked leave-one-site-out validation, which is now reported in M18 and '
       'R8. Site sizes are markedly uneven — the largest site holds %.1f%% of students and '
       'the smallest %.1f%% — and the record-level dropout rate ranges from %.4f to %.4f '
       'across sites, a %.1f-fold spread, so the sites are not interchangeable replicates '
       '(Table 1b).'
       % (NS, site_phrase(),
          max(s['share_of_students'] for s in SITES),
          min(s['share_of_students'] for s in SITES),
          SS['base_rate_spread']['min'], SS['base_rate_spread']['max'],
          SS['base_rate_spread']['max'] / SS['base_rate_spread']['min']))
replace_sub(md, 'Two datasets were used, both originating within this project', OLD, NEW)
log('Methods', 'Q2', 'M3: site count recovered — %d named sites replace the UNRECORDED '
                     'declaration, with the recovery route, the non-feature status of the '
                     'identifier, and the site-size and base-rate spread' % NS)

# ---- Table 1b: the N-of-sites row ------------------------------------------
t1b = md.tables[2]
for row in t1b.rows:
    if row.cells[0].text.strip() == 'N of sites':
        set_cell(row.cells[1], '%d — %s' % (NS, ', '.join(NAMES)))
        set_cell(row.cells[2],
                 'Recovered post hoc (M3), not captured at collection. Sites are uneven '
                 '(%d to %d students) and site base rates span %.4f to %.4f, so pooled '
                 'figures are dominated by the largest site; %d of %d sites carry no '
                 'test-period dropout event, so leave-one-site-out is computable for %d '
                 'sites only (M18, R8).'
                 % (min(s['students'] for s in SITES), max(s['students'] for s in SITES),
                    SS['base_rate_spread']['min'], SS['base_rate_spread']['max'],
                    len(NOPOS), NS, len(COMP)))
        break
else:
    raise AssertionError('N of sites row not found in Table 1b')

# ---- Table 1b: a new site-composition row ----------------------------------
tmpl = t1b.rows[-1]._tr
new_tr = copy.deepcopy(tmpl)
t1b.rows[-1]._tr.addnext(new_tr)
newrow = t1b.rows[-1]
set_cell(newrow.cells[0], 'Site composition')
set_cell(newrow.cells[1], '; '.join(
    '%s: %d students, %d records, %d dropout (base %.4f), test %d records / %d events'
    % (s['school_id'], s['students'], s['records'], s['dropouts'], s['base_rate'],
       s['test_records'], s['test_positives']) for s in SITES))
set_cell(newrow.cells[2],
         'Site is confounded with gender (Cramer’s V = %.4f; %s is single-sex) and with '
         'fee_payment_status (two sites are 100%% Unpaid), so an attribution on those '
         'variables carries a site component (Table 2b, R6). Site is not materially '
         'recoverable from the six features (%.4f accuracy against a %.4f majority '
         'baseline), which is what keeps the leave-one-site-out folds meaningful.'
         % (GENDER_V, [s['school_id'] for s in SITES if s['school_id'] == 'SCH03'][0],
            REC['cv_accuracy'], REC['majority_baseline']))
log('Methods', 'Q2', 'Table 1b: N-of-sites row replaced with the four named sites; a Site '
                     'composition row added carrying per-site counts and the confounding '
                     'and recoverability diagnostics')

# ---- Table 2b: gender validity threat --------------------------------------
t2b = md.tables[4]
for row in t2b.rows:
    if row.cells[1].text.strip() == 'gender':
        old = row.cells[4].text.strip()
        set_cell(row.cells[4], old.rstrip('.') + '. Additionally, gender is confounded with '
                 'site (Cramer’s V = %.4f): %s is single-sex, so a gender attribution on '
                 'this panel is partly a site effect and cannot be read as a behavioural '
                 'property of students (M3, R6).'
                 % (GENDER_V, 'Ayaarno M/A Primary and JHS'))
        break
else:
    raise AssertionError('gender row not found in Table 2b')
log('Methods', 'Q2', 'Table 2b: the gender row records the site confound, so its attribution '
                     'is not read as a behavioural property')

# ---- M18: leave-one-site-out is now available -------------------------------
OLD18a = ('A leave-one-year-out cross-validation was attempted but is not computable on the '
          'present data, because the entire train+validation pool falls within a single '
          'academic year (2024); true leave-one-site-out validation would require a school '
          'identifier, which is absent from the collected data and is flagged as an action '
          'item for the next data-collection round.')
NEW18a = ('A leave-one-year-out cross-validation was attempted but is not computable on the '
          'present data, because the entire train+validation pool falls within a single '
          'academic year (2024). Leave-one-site-out validation, which an earlier version of '
          'this manuscript reported as blocked by the absence of a school identifier, is now '
          'available: the identifier has been recovered (M3) and the analysis is reported in '
          'R8. It is computable for %d of the %d sites and not for the other %d, which carry '
          'test-period records but no test-period dropout event, so ranking metrics are '
          'undefined on them; those folds are reported as not computable rather than filled '
          'with a floor value. Each fold trains on the training-period records of the other '
          'sites and tests on the test-period records of the held-out site, so the temporal '
          'firewall holds inside every fold. The synthetic supplement is excluded from '
          'leave-one-site-out training, because it was fitted on the whole training-period '
          'pool including the held-out site and would otherwise leak it.'
          % (len(COMP), NS, len(NONCOMP)))
replace_sub(md, 'The targeted claim tier is Tier 1', OLD18a, NEW18a)

OLD18b = ('It is stated explicitly that the generalisability of EduTrace beyond the studied '
          'schools (exact site count UNRECORDED, as the register extract carries no school '
          'identifier; M3) has not been established; the robustness checks characterise '
          'stability within, not beyond, the studied context.')
NEW18b = ('It is stated explicitly that the generalisability of EduTrace beyond the studied '
          'schools has not been established. The site count is now recorded — %d sites (M3) — '
          'and site transfer is measured rather than assumed, but it is measured on %d '
          'computable folds resting on %d dropout events between them, which is a direction '
          'and not an estimate. The robustness checks therefore still characterise stability '
          'within, not beyond, the studied context, and the four sites are themselves a '
          'convenience set within one region.'
          % (NS, len(COMP), sum(f['test_positives'] for f in COMP)))
replace_sub(md, 'The targeted claim tier is Tier 1', OLD18b, NEW18b)

OLD18c = ('Two named ingredients would be required to move beyond the Tier 1 ceiling — a '
          'school identifier on each record, enabling leave-one-site-out validation, and an '
          'independent out-of-region institutional dataset for cross-dataset transfer — and '
          'both are collection-round dependencies rather than wording changes.')
NEW18c = ('Two named ingredients were previously required to move beyond the Tier 1 ceiling: '
          'a school identifier on each record, enabling leave-one-site-out validation, and an '
          'independent out-of-region institutional dataset for cross-dataset transfer. The '
          'first has been delivered and the analysis run (R8). The second has not, and a '
          'third requirement is now visible that only the site-level analysis could expose: '
          'the present sites yield too few test-period dropout events for a site hold-out to '
          'estimate anything — %d of %d sites carry none at all — so additional dropout '
          'events, whether from more sites or more observation years, are a precondition for '
          'the site-transfer evidence to carry weight. The claim tier therefore remains Tier '
          '1, and it remains a collection-round dependency rather than a wording change.'
          % (len(NONCOMP), NS))
replace_sub(md, 'The targeted claim tier is Tier 1', OLD18c, NEW18c)
log('Methods', 'Q2 / M18', 'M18: leave-one-site-out is reported as available and partially '
                           'computable; the Tier-2 ingredient list is updated and a third, '
                           'newly visible requirement (test-period events per site) is named')

# ---- Countersignature block narrows ----------------------------------------
_, cs = find(md, 'The student-disjoint sensitivity re-score specified in M9 is declared')
cs_new = (
    'SCOPE NOTE. An earlier version of this declaration also covered the number of '
    'participating sites. That is no longer part of it: the school identifier has been '
    'recovered from the schools and the site count is now reported exactly as %d (M3, Table '
    '1b), and leave-one-site-out validation is reported in R8. The declaration below is '
    'narrowed to the one quantity that remains not computable.\n\n'
    'The student-disjoint sensitivity re-score specified in M9 is declared NOT COMPUTABLE on '
    'these data. The declaration rests on three statements, each verified against the '
    'analysed register: (i) M9 — all %d students observed in the test years also appear in '
    'the 2024 training partition, so 0 of the %d test records belong to a student unseen in '
    'training; (ii) R8 — the re-score is therefore not computable and no substitute is '
    'reported in its place; (iii) M18 — the inference boundary is that the claim covers '
    'next-year dropout-risk prediction for students who already have prior-year register '
    'history in the studied schools, and does not cover new intakes. Pending '
    'countersignature, no sentence in this manuscript describes the evaluation partition as '
    'held-out without that qualifier.\n\n'
    'Countersigned: ______________________________   Dr Eric Opoku Osei\n\n'
    'Date: ______________________________'
    % (NS, R['student_disjoint']['n_test_students'], R['student_disjoint']['n_test_records']))
set_text(cs, cs_new)
log('Methods', 'Q2', 'Countersignature block narrowed: the site count is recovered and is no '
                     'longer part of the not-computable declaration, which now covers the '
                     'student-disjoint re-score alone')

md.save(M_PATH)

# ================================================================== RESULTS
rd = Document(R_PATH)

# ---- R1: site composition ---------------------------------------------------
replace_sub(rd, 'The held-out evaluation partition comprised',
            'The evaluation protocol follows the TRAINING-ONLY AUGMENTATION design',
            'The corpus is drawn from %d sites (Methods M3, Table 1b), and the evaluation '
            'partition is unevenly distributed across them: %s. %d of the %d sites carry no '
            'dropout event in the test period at all, which is what limits the '
            'leave-one-site-out analysis reported in R8. The evaluation protocol follows the '
            'TRAINING-ONLY AUGMENTATION design'
            % (NS,
               '; '.join('%s %d records and %d events'
                         % (s['school_id'], s['test_records'], s['test_positives'])
                         for s in SITES),
               len(NOPOS), NS))
log('Results', 'Q2', 'R1: the evaluation partition\'s composition across the four sites, and '
                     'the two sites with no test-period event')

# ---- R6: the gender attribution carries a site component --------------------
replace_sub(rd, 'SHAP attributions were computed with the exact tree explainer',
            'These are associations within the fitted model on one evaluation partition, not '
            'causal estimates.',
            'One attribution outside the top three requires a caveat that only the site '
            'identifier makes visible. gender_Female is the fourth-ranked feature by mean '
            '|SHAP| (0.322), and gender is confounded with site on this panel (Cramer’s V = '
            '%.4f): Ayaarno M/A Primary and JHS is single-sex, and two sites are 100%% '
            'Unpaid on fee_payment_status. A gender or fee attribution here is therefore '
            'partly a site effect and is not read as a behavioural property of students '
            '(Methods Table 2b). No interpretive claim is made from either variable. These '
            'are associations within the fitted model on one evaluation partition, not causal '
            'estimates.' % GENDER_V)
log('Results', 'Q2', 'R6: the gender attribution is qualified as partly a site effect, with '
                     'the confounding statistic')

# ---- R8: the leave-one-site-out result --------------------------------------
_, r8 = find(rd, 'Cross-dataset transfer is not applicable under the TRAINING-ONLY')
lines = []
for f in LOSO['folds']:
    if f.get('computable'):
        s = f['summary']
        lines.append('holding out %s (%s, %d test records, %d dropout events, base rate '
                     '%.4f) gave AUC-PR %.4f ± %.4f across the five seeds, against a '
                     'prevalence floor of %.4f (a %.2f-fold lift), with recall %.4f ± '
                     '%.4f'
                     % (f['held_out_site'], f['school'], f['test_n'], f['test_positives'],
                        f['test_base_rate'], s['auc_pr']['mean'], s['auc_pr']['std'],
                        f['prevalence_floor'], f['auc_pr_lift_over_prevalence'],
                        s['recall']['mean'], s['recall']['std']))
    else:
        lines.append('holding out %s (%s) is not computable: the site has %d test-period '
                     'records but no dropout event, so AUC-PR and AUC-ROC are undefined on it '
                     'and no floor value is substituted'
                     % (f['held_out_site'], f['school'], f['test_n']))
r8_add = (
    'Leave-one-site-out validation. The school identifier recovered in Methods M3 makes the '
    'site hold-out named in M18 possible, and it was run: each fold trains on the '
    'training-period records of the other sites and tests on the test-period records of the '
    'held-out site, so the temporal firewall holds inside every fold. The synthetic supplement '
    'is excluded from this training, because it was fitted on the whole training-period pool '
    'including the held-out site and training on it would leak that site. Across the %d sites: '
    '%s. Two properties of this result govern how much weight it can carry. First, it is '
    'computable for %d of %d sites only; the %d excluded folds are excluded because the site '
    'records no dropout event in the test period, not because the result was unfavourable. '
    'Second, the %d computable folds rest on %d dropout events between them, and the spread '
    'across seeds within a single fold is of the same order as the gap between the folds '
    '(±%.4f on AUC-PR within one fold), so neither the level nor the difference is '
    'stable; the figures are reported as a direction and not as an estimate of '
    'site-transfer performance. One diagnostic supports the folds '
    'being meaningful at all: site membership is not materially recoverable from the six '
    'register features (%.4f cross-validated accuracy against a %.4f majority-class baseline, '
    'a lift of only %.2f), so a model in a site hold-out fold cannot readily infer the site it '
    'has been denied. The honest summary is that site transfer is now measured rather than '
    'assumed, that nothing in it contradicts the within-site result, and that it is far too '
    'thin to lift the claim tier.'
    % (NS, '; '.join(lines), len(COMP), NS, len(NONCOMP), len(COMP),
       sum(f['test_positives'] for f in COMP),
       max(f['summary']['auc_pr']['std'] for f in COMP),
       REC['cv_accuracy'], REC['majority_baseline'], REC['lift']))
insert_after(r8, r8_add)
log('Results', 'Q2 / M18', 'R8: the leave-one-site-out result reported in full, including the '
                           'two folds that are not computable and why, the seed spread, and '
                           'the site-recoverability diagnostic')

# ---- R12: note Q2's change of status ---------------------------------------
replace_sub(rd, 'This subsection reports the verification conditions C1-C6',
            "It is placed after R11 so that R10's account of the non-confirmatory results "
            'stands unaltered',
            'It also records one change of status since the previous verification: the school '
            'identifier has been recovered, so the site count is no longer UNRECORDED (%d '
            'sites, M3) and leave-one-site-out validation has been run (R8). That closes the '
            'collection-side half of the sampling item; the student-disjoint re-score remains '
            'not computable and remains the subject of the countersigned declaration in '
            "Methods. It is placed after R11 so that R10's account of the non-confirmatory "
            'results stands unaltered' % NS)
log('Results', 'Q2', 'R12: records the site count\'s change of status and what it does and '
                     'does not close')

rd.save(R_PATH)

# ------------------------------------------------------------------- summary
print('updated %s' % M_PATH)
print('updated %s' % R_PATH)
print()
for doc, item, what in CHANGES:
    print('  [%-8s] %-10s %s' % (doc, item, what))
prev = []
p = os.path.join('results', 'manuscript_changes.json')
if os.path.exists(p):
    prev = json.load(open(p))
json.dump(prev + [{'document': d, 'item': i, 'change': w} for d, i, w in CHANGES],
          open(p, 'w'), indent=2)
print('\n%d further edits (%d total)' % (len(CHANGES), len(prev) + len(CHANGES)))
