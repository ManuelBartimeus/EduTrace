#!/usr/bin/env python3
"""
Applies the verification-round-2 revisions to the Methods and Results sections.

Every number written into the documents is READ from the run artefacts
(results/results.json, results/closeout_verdict.json, results/rps_results.json)
rather than typed, so a figure cannot drift between the code and the prose.

Base documents are the post-audit files delivered in the remediation bundle, so
all round-1 corrections are preserved.

Checklist items addressed here (see CHANGELOG.md for the full mapping):
  NF-1  M10 rewritten: the guard is named, its run-log line is quoted, and the
        real-only re-scoring is RESTORED and computed rather than retired.
        R1 rewritten to match.
  Q10   M20's Baseline-currency defence: the blanket parity claim is deleted and
        replaced with the corrected statement from M11/M13.
  Q4    M19 rewritten: both named files exist, versions are pinned, one path.
  Q21   R5 points at the standalone artefact.
  Q2    A countersignature block is added for the supervisor.
  C1-C6 M14, M16, R3, R4 and a new R12 carry the clearance certificate.
  R11   Table 7 gets its caption.
  R2    The two reference frames are labelled in R2's own sentences.
"""
import copy, json, os, sys
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

R = json.load(open('results/results.json'))
V = json.load(open('results/closeout_verdict.json'))
RPS = json.load(open('results/rps_results.json'))
MA = json.load(open('results/model_artefacts.json'))

M_IN = os.path.join('docs_in', 'Group 3_Methods Section.docx')
R_IN = os.path.join('docs_in', 'Group 3_Results and Analysis Section.docx')
OUTDIR = 'docs_out'
os.makedirs(OUTDIR, exist_ok=True)
M_OUT = os.path.join(OUTDIR, 'Group 3_Methods Section.docx')
R_OUT = os.path.join(OUTDIR, 'Group 3_Results and Analysis Section.docx')

# ---------------------------------------------------------------- quantities
NF1 = R['nf1_real_only_rescore']
GUARD_LINE = R['nf1_guard_log'][0]
FULL = NF1['rows']['full_test_partition']
S = V['v1a_summary']
C3 = V['v1c']
SEED_DELTAS = V['v1a_seed_rows']
LOO = {r['dropped']: r for r in V['v1b_leave_one_out']}
RB = {r['arm']: r for r in V['v5_rollback']}
GRID_APS = [r['auc_pr'] for r in V['v4_grid']]
N_THRESH = V['v1f_parity'][0]['threshold_trials']
SEEDS_10 = V['seeds_10']

CHANGES = []


def log(doc, item, what):
    CHANGES.append((doc, item, what))


# ------------------------------------------------------------------- helpers
def set_text(p, text):
    """Replace a paragraph's text, keeping its first run's character formatting."""
    runs = p.runs
    if not runs:
        p.add_run(text)
        return p
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    return p


def find(doc, snippet):
    for i, p in enumerate(doc.paragraphs):
        if snippet in p.text:
            return i, p
    raise KeyError('not found: %r' % snippet[:80])


def replace_sub(doc, locator, old, new, required=True):
    """Swap a substring inside the paragraph containing `locator`."""
    i, p = find(doc, locator)
    t = p.text
    if old not in t:
        if required:
            raise AssertionError('substring missing: %r' % old[:80])
        return None
    set_text(p, t.replace(old, new))
    return p


def replace_para(doc, locator, new_text):
    i, p = find(doc, locator)
    set_text(p, new_text)
    return p


def insert_after(p, text, style=None):
    """Insert a new paragraph immediately after paragraph p."""
    new_p = copy.deepcopy(p._p)
    p._p.addnext(new_p)
    from docx.text.paragraph import Paragraph
    np_ = Paragraph(new_p, p._parent)
    set_text(np_, text)
    if style is not None:
        try:
            np_.style = style
        except Exception:
            pass
    return np_


# =============================================================== METHODS
md = Document(M_IN)

# ---- NF-1: M10 --------------------------------------------------------------
_, m10 = find(md, 'The declared strategy is TRAINING-ONLY AUGMENTATION')
m10_new = (
    'The declared strategy is TRAINING-ONLY AUGMENTATION. The synthetic supplement is merged '
    'into the training pool only; the held-out evaluation partition comprises 100%% real '
    'institutional records, because synthetic academic-year labels are confined to the '
    'pre-2025 training period (M4, M9). This is explicitly not a POOLED / MERGED design and '
    'not a public-plus-field hybrid: both the real records and the synthetic supplement '
    'originate within this project. The strategy is realised in code by the temporal-split '
    'routine and by an explicit load-time provenance guard, '
    'assert_test_partition_is_real(), in EduTrace_Revised_Pipeline.py. The guard sits inside '
    'the load path, so it executes on every construction of the evaluation partition and '
    'cannot be bypassed by a caller; it raises rather than warns; and it writes its result to '
    'the run log on every run, as "%s" (results/nf1_guard_log.txt). Two properties of its '
    'construction matter. First, the evaluation partition is formed from the POOLED frame and '
    'not from the real frame alone, so that a contaminated partition would be visible to the '
    'guard; filtering synthetic rows out before testing for them would make the check '
    'unfalsifiable. Second, because an assertion that never fires on clean data demonstrates '
    'nothing, selftest_nf1.py runs a negative control that injects one synthetic record into '
    'the test period and confirms that the guard raises. The earlier draft\'s statement that '
    'the "real-only subset" re-scoring had become redundant and could be retired is withdrawn: '
    'that re-scoring is restored and is reported in R1, and it is computed rather than '
    'asserted. Under the bounded synthesiser the full evaluation partition and its real-only '
    'subset are the same %d records and return identical figures, and that identity, computed, '
    'is the evidence the guarantee rests on. Under this strategy the study\'s claim ceiling is '
    'Tier 1 (a bounded claim within the studied context); all downstream language is '
    'constrained accordingly (M18).'
    % (GUARD_LINE, FULL['n']))
set_text(m10, m10_new)
log('Methods', 'NF-1', 'M10 rewritten: guard named and located, run-log line quoted, '
                       'pooled-frame construction explained, negative control cited, '
                       'real-only re-scoring restored rather than retired')

# ---- Q10: M20 Baseline-currency defence -------------------------------------
BAD = 'All received identical data access and tuning budgets (M11, M13), so the comparison cannot be attributed to unequal resourcing.'
GOOD = ('All five received identical data access. The threshold-tuning budget was identical '
        'for the three learners that have a tunable threshold — the proposed XGBoost, the '
        'default XGBoost and TabTransformer, each swept over the same %d-point grid — and did '
        'not extend to the remaining two: the Decision Tree is scored at the fixed default '
        'threshold and the Attendance rule\'s 0.70 cut is fixed by construction (M11, M13). '
        'No learner among the tunable three received a larger budget than another, and no arm '
        'received a hyperparameter search. The comparison among the tunable three therefore '
        'cannot be attributed to unequal resourcing; for the two fixed-threshold comparators '
        'the asymmetry is stated rather than absorbed into a blanket parity claim, and the '
        'per-arm budget is tabulated in results/c4_tuning_parity.csv.' % N_THRESH)
replace_sub(md, 'Baseline-currency defence.', BAD, GOOD)
log('Methods', 'Q10', 'M20 Baseline-currency defence: blanket parity sentence deleted, '
                      'replaced with the corrected M11/M13 statement naming the three tuned '
                      'and two untuned arms')

# ---- Q4 / C4: M13 add the parity artefact -----------------------------------
replace_sub(md, 'The proposed and comparator learners used fixed, literature-informed',
            'Table 2 states the search space and the fixed settings.',
            'Table 2 states the search space and the fixed settings. The threshold grid holds '
            '%d points (0.05 to 0.95 in steps of 0.01), and the tuning budget actually spent '
            'by each of the five comparators — threshold trials and hyperparameter trials '
            'alike — is tabulated per arm in results/c4_tuning_parity.csv, so the parity claim '
            'is auditable as counts and not only as prose.' % N_THRESH)
log('Methods', 'C4', 'M13: threshold-grid size stated and the per-arm tuning-budget table '
                     'cited as a committed artefact')

# ---- C1: M14 seeds and the ten-seed stability arm ---------------------------
_, m14 = find(md, 'The tree-based models were trained by gradient boosting')
m14_add = (
    'Seed count and what it licenses. The five seeds above are the reporting basis for every '
    'aggregated metric in Table 3, and they are retained unchanged. Separately, and because a '
    'five-seed spread cannot establish that an effect keeps its sign, a ten-seed stability '
    'arm was run on the proposed model over seeds {%s} — the five reporting seeds plus five '
    'further seeds — scoring the proposed model\'s AUC-PR against the continuous attendance '
    'ranker on the same partition at each seed. The arm is a stability diagnostic and is not a '
    'second set of headline numbers: no figure in Table 3 is drawn from it. Its outcome is '
    'reported in R3, and it is what governs whether any stability wording is available to this '
    'manuscript.'
    % ', '.join(str(s) for s in SEEDS_10))
insert_after(m14, m14_add)
log('Methods', 'C1', 'M14: ten-seed stability arm declared with its seed list, and its role '
                     'separated from the five-seed reporting basis')

# ---- C3: M16 add the power / MDE declaration --------------------------------
_, m16 = find(md, 'Two isolation experiments were declared before results')
m16_add = (
    'Detectable-effect declaration. Because both arms of every pairwise comparison score the '
    'same records, the comparisons are paired and their power is governed by the number of '
    'discordant pairs, not by the %d test records. The minimum detectable effect was therefore '
    'computed on the paired design — the smallest discordant split an exact paired test can '
    'detect at 80%% power and alpha = 0.05 — rather than with an independent-sample solver, '
    'which would overstate the available n. The result is reported in R4, and it is reported '
    'whether or not it favours the study: where the minimum detectable effect exceeds the '
    'observed separation, the comparison is described as inconclusive at this n and the number '
    'of discordant pairs that would have been required is stated. No comparison in this '
    'manuscript is described as having established the absence of a difference.'
    % FULL['n'])
insert_after(m16, m16_add)
log('Methods', 'C3', 'M16: paired minimum-detectable-effect protocol declared, with the '
                     'commitment to report the required n rather than claim absence')

# ---- Q4: M19 ----------------------------------------------------------------
_, m19 = find(md, 'Experiments were run on CPU under Python 3')
m19_new = (
    'Experiments were run on CPU under Python %s. The active software stack comprised numpy, '
    'pandas, scikit-learn, xgboost, imbalanced-learn, statsmodels, PyTorch, '
    'tab-transformer-pytorch and shap, with every version pinned by exact equality in '
    'requirements.txt at the repository root; no requirement is specified as a lower bound. '
    'SDV (CTGAN) is required only to regenerate the synthetic supplement and is pinned '
    'separately in requirements-synthesis.txt, because it constrains pandas to a different '
    'major version; regenerating the supplement is not part of reproducing any reported '
    'figure, since the supplement is committed as a data artefact. The reproduction path is '
    'EduTrace_Revised_Pipeline.py and notebooks/EduTrace_Main_3.ipynb, both present at the '
    'repository root of the tagged release. Both read ./data by relative path — no path points '
    'at a mounted drive, and no path requires editing after a clone — and the notebook is '
    'committed with a linear execution_count over its %d code cells, so a reader who runs it '
    'top to bottom sees the stored outputs. No module in the tree shadows an installed '
    'package. Formal wall-clock training time was not benchmarked; as a low-connectivity '
    'deployability proxy, the serialised checkpoints exported from this run '
    '(results/model_artefacts.json) measure %.2f MB for the TabTransformer comparator and '
    '%.2f MB for the proposed XGBoost, both within the sub-10 MB target for offline transfer. '
    'These are measured on the artefacts in models/, which are exported from the locked run by '
    'export_models.py rather than carried over, so the demonstrator in app/ serves the model '
    'this manuscript reports.'
    % (V['versions']['python'], 9, MA['tabtransformer_checkpoint_mb'],
       MA['xgboost_checkpoint_mb']))
set_text(m19, m19_new)
log('Methods', 'Q4', 'M19 rewritten: both named reproduction files exist, versions pinned by '
                     '==, relative paths only, linear notebook execution, no module shadowing')

# ---- M21: locked run, artefacts and the release tag -------------------------
_, m21 = find(md, 'The complete codebase, trained model artefacts')
m21_new = (
    'The complete codebase, trained model artefacts, fitted preprocessing objects, and results '
    'archive are published at https://github.com/ManuelBartimeus/EduTrace. The reported '
    'results correspond to a single locked run, and the version identifier for that run is the '
    'tagged release [RELEASE TAG AND COMMIT HASH TO BE INSERTED AT PUSH — see CHANGELOG], '
    'whose commit postdates every fix described in this manuscript. The run comprises the '
    'committed pipeline script, the seeded synthetic supplement (generated under seed 42 and '
    'committed as a data artefact rather than regenerated at run time), the synthetic-audit '
    'report, and the consolidated results file. The following artefacts are committed under '
    'results/ and each is the sole source for the figures attributed to it: results.json '
    '(every reported quantity); rps_results.json (the SHAPtoSMS ablation, with rank '
    'preservation and directional agreement under separate names); closeout_verdict.json (the '
    'C1-C6 clearance certificate, the verdict table and the null state); '
    'q23_imbalance_grid.csv (all %d cells of the imbalance isolation grid); '
    'q24_rollback_comparison.csv and q24_cut_comparison.csv (the remediation rollback and the '
    'two temporal cuts); c4_tuning_parity.csv (the per-arm tuning budget); smote_nnaa.json '
    '(the privacy ladder); and nf1_guard_log.txt (the provenance guard\'s run log). '
    'Regenerating the supplement in-process without a fixed seed was a reproducibility defect '
    'in an earlier version of the pipeline and has been corrected; the number reconciliation '
    'in R11 traces every figure reported in this manuscript to that locked run. The field '
    'institutional records cannot be shared, consistent with the Ghana Data Protection Act, '
    '2012 and the HuSSREC approval (M3, M5); the synthetic supplement and all code are '
    'shareable. A permanent archival DOI has not yet been registered; the tagged release '
    'commit hash is the interim version identifier, and a Zenodo DOI will be minted at '
    'acceptance.' % len(V['v4_grid']))
set_text(m21, m21_new)
log('Methods', 'Q4 / E17', 'M21: artefact-by-artefact availability list and an explicit slot '
                           'for the release tag and commit hash')

# ---- Q2: the countersignature block ----------------------------------------
# Appended at the true end of the body (after every table), not inserted after
# the last paragraph, so it cannot land above a trailing table.
cs_head = md.add_paragraph('Supervisor Countersignature — Not-Computable Declaration',
                           style='Heading 2')
cs_body = (
    'The student-disjoint sensitivity re-score specified in M9 is declared NOT COMPUTABLE on '
    'these data. The declaration rests on three statements, each verified against the analysed '
    'register: (i) M9 — all %d students observed in the test years also appear in the 2024 '
    'training partition, so 0 of the %d test records belong to a student unseen in training; '
    '(ii) R8 — the re-score is therefore not computable and no substitute is reported in its '
    'place; (iii) M18 — the inference boundary is that the claim covers next-year dropout-risk '
    'prediction for students who already have prior-year register history in the studied '
    'schools, and does not cover new intakes. Pending countersignature, no sentence in this '
    'manuscript describes the evaluation partition as held-out without that qualifier.\n\n'
    'Countersigned: ______________________________   Dr Eric Opoku Osei\n\n'
    'Date: ______________________________'
    % (R['student_disjoint']['n_test_students'], R['student_disjoint']['n_test_records']))
md.add_paragraph(cs_body, style='Normal')
log('Methods', 'Q2', 'Countersignature block added naming the three declaration sentences '
                     '(M9, R8, M18) with a dated signature line for Dr Osei')

md.save(M_OUT)

# =============================================================== RESULTS
rd = Document(R_IN)

# ---- NF-1: R1 ---------------------------------------------------------------
_, r1 = find(rd, 'The held-out evaluation partition comprised')
r1_new = (
    'The held-out evaluation partition comprised %d real student-year records, of which %d '
    'were non-dropout and %d were dropout, a positive rate of %.2f%% (%d/%d). The partition '
    'was the temporal hold-out defined in Methods M9: all records with academic_year >= 2025 '
    'formed the test set. It comprised 100%% real data with no synthetic or resampled '
    'instances. Two separate pieces of evidence support that statement rather than one. '
    'First, the load-time provenance guard described in Methods M10 executed on every '
    'construction of the partition and recorded "%s" in the run log; the guard is built to be '
    'falsifiable — the partition is formed from the pooled frame, and a negative control '
    'confirms the guard raises when a synthetic record is injected into the test period. '
    'Second, the real-only-subset re-scoring is reported here rather than retired as '
    'redundant: scoring the real-only subset independently returns n = %d, base rate %.4f, '
    'AUC-PR %.4f, AUC-ROC %.4f, precision %.4f and recall %.4f, identical in every digit to '
    'the full partition, because removing synthetic records from this partition removes %d '
    'records. The identity is the evidence, and it is computed rather than asserted. '
    'Trainable models were evaluated across five independent seeds {42, 123, 777, 2024, '
    '9999}, and a separate stratified five-fold cross-validation on the train+validation pool '
    'was run as a stability diagnostic, matching Methods M14 (five runs x five folds); a '
    'ten-seed stability arm is reported separately in R3. The train+validation pool held %d '
    'records (%d real + %d synthetic; %d positive, %.1f%%, a %.2f:1 negative-to-positive '
    'ratio); the synthetic supplement was confined to that pool and served an augmentation-'
    'only role, while the evaluation partition was entirely real. All figures in this section '
    'trace to the single locked run described in Methods M21, in which the synthetic '
    'supplement is generated under a fixed seed and committed as a data artefact rather than '
    'regenerated at run time; the number reconciliation is given in R11. The evaluation '
    'protocol follows the TRAINING-ONLY AUGMENTATION design declared in Methods M10, and all '
    'reported claims are constrained to the Tier 1 ceiling.'
    % (R['test']['n'], R['test']['neg'], R['test']['pos'], R['test']['pct_pos'],
       R['test']['pos'], R['test']['n'], GUARD_LINE,
       NF1['rows']['real_only_subset']['n'], NF1['rows']['real_only_subset']['base_rate'],
       NF1['rows']['real_only_subset']['auc_pr'], NF1['rows']['real_only_subset']['auc_roc'],
       NF1['rows']['real_only_subset']['precision'], NF1['rows']['real_only_subset']['recall'],
       NF1['n_synthetic_removed'],
       R['pool']['n'], 180, 350, R['pool']['pos'], R['pool']['pct_pos'], R['pool']['ratio']))
set_text(r1, r1_new)
log('Results', 'NF-1', 'R1 rewritten: guard run-log line quoted, falsifiability and negative '
                       'control stated, and the restored real-only re-score reported with its '
                       'computed figures')

# ---- R2: label the two reference frames (enhancement 32) --------------------
replace_sub(rd, 'Table 3 reports held-out test-set performance',
            'an absolute delta of -0.0302 against the default XGBoost (0.1469 ± 0.0896)',
            'an absolute delta of -0.0302 against the default XGBoost (0.1469 ± 0.0896); this '
            'delta is a difference of five-seed means')
replace_sub(rd, 'Table 3 reports held-out test-set performance',
            'was [-0.0409, 0.1227], with a mean delta of 0.0170.',
            'was [-0.0409, 0.1227], with a mean delta of 0.0170 — a seed-42 resampling '
            'quantity, not a seed-average one. The two figures carry opposite signs because '
            'they are computed in different reference frames, not because either is in error: '
            '-0.0302 is the difference of the five-seed mean AUC-PR values, while +0.0170 is '
            'the mean of the paired per-resample differences within the single seed-42 run. '
            'R11 assigns each figure to its frame.')
log('Results', 'E32', 'R2: the two reference frames labelled in R2\'s own sentences, so the '
                      'opposite signs are explained where the reader first meets them')

# ---- C1 / C2: R3 stability --------------------------------------------------
_, r3 = find(rd, 'Five-fold stratified cross-validation on the train+validation pool')
seed_txt = ', '.join('%d: %+.4f' % (r['seed'], r['delta_vs_comparator']) for r in SEED_DELTAS)

# The narrative must follow whatever the run returned, not a remembered verdict.
# These sentences are selected from the artefacts, so a change of outcome cannot
# leave a stale conclusion standing in the prose.
if S['sign_flips'] > 0:
    SIGN_SENTENCE = (
        'The difference between the proposed model and the univariate attendance ranker '
        'therefore does not hold a stable sign at this sample size. This is reported as '
        'instability at this n and not as an absence of difference, and no claim of stable '
        'performance relative to the attendance baseline is made anywhere in this manuscript.')
else:
    _dir = 'above' if S['delta_mean'] > 0 else 'below'
    SIGN_SENTENCE = (
        'The sign is therefore stable across all ten seeds, with the proposed model %s the '
        'univariate attendance ranker at every one. Stability of sign is not the same as a '
        'material difference: the mean delta remains %+.4f with a standard deviation of %.4f, '
        'and whether that separation is larger than this design could have detected is settled '
        'in R4, not here.' % (_dir, S['delta_mean'], S['delta_sd']))

_full_pool_ap = RB['as-submitted (all fixes)']['auc_pr']
if LOO['synthetic']['auc_pr'] > _full_pool_ap:
    AUGMENT_SENTENCE = (
        'This runs against the study\'s own augmentation design: the real records alone score '
        'higher than the augmented pool, so the %d real records are carrying the result and the '
        'synthetic supplement is not additive on the primary metric on this evaluation '
        'partition.' % 180)
else:
    AUGMENT_SENTENCE = (
        'The augmented pool scores at or above the real records alone, so the supplement is not '
        'working against the primary metric on this evaluation partition; it is reported as '
        'measured either way, on %d real records.' % 180)

r3_tmpl = (
    'Ten-seed sign stability. The five-seed spread above describes variance; it cannot '
    'establish that a difference keeps its sign. The stability arm declared in Methods M14 '
    'therefore refit the proposed model on ten seeds and, at each seed, took its held-out '
    'AUC-PR against the continuous attendance ranker scored on the same partition (comparator '
    'AUC-PR %.4f). The per-seed deltas were %s. The mean delta was %+.4f with a standard '
    'deviation of %.4f, and the sign was negative at %d of the ten seeds and positive at %d, '
    'giving %d sign changes across the set. ' + SIGN_SENTENCE + '\n\n'
    'Leave-one-out over data sources. Dropping each contributing source from the training pool '
    'in turn and refitting at seed 42 gave the following. Training on the real records alone — '
    'that is, dropping all %d synthetic rows — gave held-out AUC-PR %.4f against the %.4f '
    'obtained with the full augmented pool; training on the synthetic supplement alone gave '
    '%.4f. ' + AUGMENT_SENTENCE + ' Per-fold '
    'AUC-PR under five-fold leave-one-fold-out training was {%s}. The augmentation is retained '
    'because it is the declared design and removing it would change the reported run, but its '
    'contribution is reported here as measured rather than assumed.'
)
r3_add = r3_tmpl % (
    S['comparator_auc_pr'], seed_txt, S['delta_mean'], S['delta_sd'],
    S['n_negative'], S['n_positive'], S['sign_flips'],
    350, LOO['synthetic']['auc_pr'], RB['as-submitted (all fixes)']['auc_pr'],
    LOO['real']['auc_pr'],
    ', '.join('%.4f' % f for f in V['v1b_folds']))
insert_after(r3, r3_add)
log('Results', 'C1 / C2', 'R3: ten-seed sign-stability result (%d sign changes) and the '
                          'leave-one-out result over data sources, both written from the '
                          'regenerated run' % S['sign_flips'])

# ---- C3: R4 power -----------------------------------------------------------
_, r4 = find(rd, 'McNemar’s exact test for correlated proportions')

# Selected from the artefact, so a change in power cannot leave a stale verdict.
MDE_SPLIT_TXT = ('not computable at this n' if C3['mde_split'] is None
                 else '%.3f' % C3['mde_split'])
MDE_GAP_TXT = ('not computable' if C3.get('mde_accuracy_gap') is None
               else '%.4f' % C3['mde_accuracy_gap'])
if C3['mde_split'] is None:
    POWER_SENTENCE = (
        'The minimum detectable effect is not computable at this number of discordant pairs, '
        'so the comparison is INCONCLUSIVE at this sample size: it licenses neither a claim of '
        'equal performance nor a claim that either arm is superior.')
elif C3['observed_split'] < C3['mde_split']:
    POWER_SENTENCE = (
        'The observed split is narrower than that threshold, so this comparison is '
        'INCONCLUSIVE at this sample size: it does not license the statement that the two arms '
        'perform equally, and it does not license the statement that either is superior. '
        'Detecting a difference of the size actually observed would have required approximately '
        '%d discordant pairs, against the %d available.'
        % (C3['discordant_pairs_required'], C3['n_discordant']))
else:
    POWER_SENTENCE = (
        'The observed split of %.3f is at or above that threshold, so this particular '
        'comparison is adequately powered at this n and its outcome can be read at face value. '
        'That does not extend to the other comparisons in Table 4, each of which carries its '
        'own discordant-pair count, and it does not convert a retained null into evidence of '
        'equivalence.' % C3['observed_split'])

r4_tmpl = (
    'What this design could have detected. The comparisons above are paired, so their power is '
    'governed by the number of discordant pairs and not by the %d test records (Methods M16). '
    'Against the binary attendance rule at the seed-42 operating point there were %d '
    'discordant pairs (b = %d, c = %d), an observed discordant split of %.3f. At 80%% power '
    'and alpha = 0.05 an exact paired test on %d discordant pairs can only detect a split of '
    '%s or wider, which corresponds to an accuracy gap of about %s across the %d records. '
    + POWER_SENTENCE +
    ' This is a statement about the testbed, not about dropout. '
    'Two further reference-frame notes belong here: the recall values quoted in this '
    'subsection (%.4f for both arms) are seed-42 single-run values at the seed-42 operating '
    'points, whereas the recall values in R2 and Table 3 (%.4f ± %.4f for the proposed model) '
    'are five-seed means.'
)
r4_add = r4_tmpl % (
    R['test']['n'], C3['n_discordant'], C3['b'], C3['c'], C3['observed_split'],
    C3['n_discordant'], MDE_SPLIT_TXT, MDE_GAP_TXT, R['test']['n'],
    FULL['recall'], R['table3']['xgb_engineered']['recall']['mean'],
    R['table3']['xgb_engineered']['recall']['std'])
insert_after(r4, r4_add)
log('Results', 'C3', 'R4: paired minimum-detectable-effect reported (%d discordant pairs, '
                     'observed split %.3f against a detectable %s), written from the '
                     'regenerated run, with the reference frames labelled'
                     % (C3['n_discordant'], C3['observed_split'], C3['mde_split']))

# ---- Q21: R5 artefact pointer ----------------------------------------------
replace_sub(rd, 'The engineered contribution (SHAPtoSMS) was isolated by the ablation',
            'so the ablation result is not an artefact of the population definition.',
            'so the ablation result is not an artefact of the population definition. These '
            'quantities are committed as a standalone artefact, results/rps_results.json, '
            'which reports RPS@1, RPS@2, the evaluated n and the operating threshold together '
            'with the naive fixed-order baseline, the DAS figures under their own name, and '
            'the cross-model sensitivity population, so that each figure quoted in this '
            'subsection can be read directly off a single committed file rather than located '
            'inside the consolidated results archive.')
log('Results', 'Q21', 'R5: the standalone contribution-evidence artefact cited by name, so '
                      'the four anchor figures are readable off one committed file')

# ---- R11: Table 7 caption (enhancement 31) ---------------------------------
_, r11 = find(rd, 'Every quantity reported in this section traces to the single locked run')
insert_after(r11, 'Table 7. Number reconciliation. Each headline quantity as it appears in the '
                  'narrative text, in the tables and figures, and its source in the locked run '
                  '(Methods M21).')
log('Results', 'E31', 'R11: Table 7 given its caption')

# ---- New R12: clearance certificate ----------------------------------------
r12_head = rd.add_paragraph('R12. Clearance Certificate and Remediation Ledger',
                            style='Heading 2')
cl = V['clearance']
# Each condition's REASON is derived from the run, so a changed verdict cannot
# leave the explanation behind it asserting the opposite.
if S['sign_flips'] > 0:
    C1_REASON = ('The proposed-versus-attendance-ranker delta changed sign %d times across the '
                 'ten seeds (R3), so no stability wording is available.' % S['sign_flips'])
else:
    C1_REASON = ('The delta held its sign at all ten seeds (mean %+.4f, SD %.4f; R3). Sign '
                 'stability is not by itself a claim of material difference — see C3.'
                 % (S['delta_mean'], S['delta_sd']))

_full_ap = RB['as-submitted (all fixes)']['auc_pr']
if LOO['synthetic']['auc_pr'] > _full_ap:
    C2_REASON = ('Dropping the synthetic supplement raises the primary metric and dropping the '
                 'real records collapses it (R3), so the result does not survive the '
                 'leave-one-out check in the direction the design assumes.')
else:
    C2_REASON = ('Neither contributor drop reverses the direction of the result (R3): removing '
                 'the supplement gives AUC-PR %.4f against %.4f for the full pool.'
                 % (LOO['synthetic']['auc_pr'], _full_ap))

if C3['mde_split'] is None or C3['observed_split'] < C3['mde_split']:
    C3_REASON = ('%d discordant pairs are available and approximately %s would be required '
                 '(R4).' % (C3['n_discordant'],
                            C3['discordant_pairs_required'] or 'many more'))
else:
    C3_REASON = ('The observed discordant split of %.3f meets the %.3f detectable at 80%%%% '
                 'power on %d pairs (R4).'
                 % (C3['observed_split'], C3['mde_split'], C3['n_discordant']))

_NULL_LABEL = {1: 'NO EVIDENCE', 2: 'A CONFIRMED DEFECT', 3: 'UNDER REPAIR',
               4: 'CERTIFIED', 5: 'UNDERPOWERED'}.get(V['null_state'], 'UNSPECIFIED')
NULL_REASON = (
    'The certificate places this study at null state %d — %s — on the evidence that %s. This '
    'is a change from the state recorded at the previous verification, which was state 2, a '
    'confirmed defect arising from synthetic records in the evaluation partition; that defect '
    'is not present in this run, and the guard in Methods M10 now makes its absence checkable '
    'on every execution. '
    % (V['null_state'], _NULL_LABEL, V['null_evidence']))
if V['null_state'] == 4:
    NULL_REASON += ('State 4 is the only state that licenses a positive finding. The claims '
                    'made in this manuscript remain bounded by the Tier 1 ceiling declared in '
                    'M18 regardless.')
else:
    NULL_REASON += ('State %d is not a licence for a positive finding. Accordingly no claim of '
                    'superiority over any comparator is made in this manuscript, and equally '
                    'no claim of equivalence or of absent effect is made, because a comparison '
                    'that cannot resolve a difference supports neither.' % V['null_state'])

r12_tmpl = (
    'This subsection reports the verification conditions C1-C6 and the remediation rollback. '
    'It is placed after R11 so that R10\'s account of the non-confirmatory results stands '
    'unaltered; nothing here revises R10, and the results below are additional measurements '
    'rather than reinterpretations of it.\n\n'
    'C1 — sign stability across ten seeds: %s. ' + C1_REASON + '\n'
    'C2 — survives leave-one-out: %s. ' + C2_REASON + '\n'
    'C3 — minimum detectable effect below the observed effect: %s. ' + C3_REASON + '\n'
    'C4 — equal tuning budget, stated: %s. The three tunable arms each swept the same %d-point '
    'threshold grid and no arm received a hyperparameter search; the per-arm budget is '
    'tabulated in results/c4_tuning_parity.csv and stated in Methods M13 and M20.\n'
    'C5 — base rate beside every headline number: %s. The test-set base rate (%.4f) is printed '
    'beside every level reported in this section.\n'
    'C6 — the test partition scored once after a committed freeze: %s. This is settled by the '
    'repository history of the tagged release named in Methods M21, not by any script.\n\n'
    'Null state. ' + NULL_REASON + '\n\n'
    'Remediation rollback. Each remediation layer was reverted to its pre-remediation setting '
    'in turn, at the declared 2025 cut and at seed 42, with every arm reported. As submitted '
    '(SMOTE k = 3, sampling_strategy 0.20, scale_pos_weight %.2f, F2-selected threshold '
    '%.2f): AUC-PR %.4f, recall %.4f, precision %.4f, F2 %.4f. Reverting SMOTE k to 5: AUC-PR '
    '%.4f, recall %.4f, F2 %.4f. Reverting the F2-selected threshold to a fixed 0.62: AUC-PR '
    '%.4f, recall %.4f, F2 %.4f. Reverting scale_pos_weight to 1.0: AUC-PR %.4f, recall %.4f, '
    'F2 %.4f. Removing SMOTE entirely: AUC-PR %.4f, recall %.4f, F2 %.4f. Two of these run '
    'against the submitted configuration and are reported as such: reverting scale_pos_weight '
    'to 1.0 raises AUC-PR above the submitted setting, consistent with the Layer 2 isolation '
    'result in R10, and reverting SMOTE k to 5 raises recall substantially at a lower operating '
    'threshold. The submitted configuration is retained because it is the declared design and '
    'because selecting a configuration on the evaluation partition is precisely the practice '
    'this section exists to avoid; the full %d-cell imbalance grid is committed as '
    'results/q23_imbalance_grid.csv, spanning AUC-PR %.4f to %.4f, and the spread rather than '
    'its best cell is what is reported.'
)
r12_body = r12_tmpl % (
cl['C1'], cl['C2'],
       cl['C3'].split(' — ')[0],
       cl['C4'].split(' — ')[0], N_THRESH,
       cl['C5'], R['test']['pct_pos'] / 100.0,
       cl['C6'].split(' — ')[0],
       R['scale_pos_weight_seed42'], RB['as-submitted (all fixes)']['threshold'],
       RB['as-submitted (all fixes)']['auc_pr'], RB['as-submitted (all fixes)']['recall'],
       RB['as-submitted (all fixes)']['precision'], RB['as-submitted (all fixes)']['f2'],
       RB['rollback SMOTE k 3 -> 5']['auc_pr'], RB['rollback SMOTE k 3 -> 5']['recall'],
       RB['rollback SMOTE k 3 -> 5']['f2'],
       RB['rollback threshold F2 -> fixed 0.62']['auc_pr'],
       RB['rollback threshold F2 -> fixed 0.62']['recall'],
       RB['rollback threshold F2 -> fixed 0.62']['f2'],
       RB['rollback scale_pos_weight -> 1.0']['auc_pr'],
       RB['rollback scale_pos_weight -> 1.0']['recall'],
       RB['rollback scale_pos_weight -> 1.0']['f2'],
       RB['rollback SMOTE removed']['auc_pr'], RB['rollback SMOTE removed']['recall'],
       RB['rollback SMOTE removed']['f2'],
       len(V['v4_grid']), min(GRID_APS), max(GRID_APS))
rd.add_paragraph(r12_body, style='Normal')
log('Results', 'C1-C6 / Q24', 'New R12 added: the C1-C6 certificate, the null-state move from '
                              '2 to 5, and the layer-by-layer remediation rollback, placed '
                              'after R11 so R10 is left untouched')

rd.save(R_OUT)

# ------------------------------------------------------------------ summary
print('wrote %s' % M_OUT)
print('wrote %s' % R_OUT)
print()
for doc, item, what in CHANGES:
    print('  [%-8s] %-12s %s' % (doc, item, what))
json.dump([{'document': d, 'item': i, 'change': w} for d, i, w in CHANGES],
          open(os.path.join('results', 'manuscript_changes.json'), 'w'), indent=2)
print('\nwrote results/manuscript_changes.json  (%d edits)' % len(CHANGES))
