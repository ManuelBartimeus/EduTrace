#!/usr/bin/env python3
"""
Corrections ledger (close-out Section F), generated rather than hand-written.

Section F requires one row per number that moves, with its cause, and warns that
"a number that moved with no row here is reverted". The determinism repair moves
every model-derived figure, so the ledger has to be produced from the artefacts
themselves — a hand-typed table of that size would be wrong within a day.

The ledger now covers TWO rounds of correction, each against its own baseline and
each with its own cause. Collapsing them into one diff would attribute a round-4
change to the round-3 repair, which is precisely the misattribution Section F
exists to prevent.

  round 3  results_pre_determinism/  ->  results_pre_round4/   thread pinning
  round 4  results_pre_round4/       ->  results/              C2 rule; DAS baseline

Both baselines are committed, so every row in the table below can be recomputed
from the repository. (The round-3 baseline directory was named `results_prev`
when the round-3 ledger was first produced and was renamed before the push, which
left this script unable to find it; the directory names are now the committed
ones and the script is re-runnable from a clean clone.)

It emits two things:

  1. results/corrections_ledger.json — the complete leaf-level diff, every
     changed value, for audit.
  2. results/corrections_ledger.md — the Section F table, restricted to the
     quantities that actually appear in the Methods and Results sections, because
     those are the numbers a reader can check against the manuscript.

The distinction matters and is stated in the output: the full diff is the
evidence, the table is the readable summary of it. Neither is a sample.
"""
import json
import os

# (label, baseline directory, regenerated directory)
ROUND3 = ('Round 3 — determinism repair', 'results_pre_determinism', 'results_pre_round4')
ROUND4 = ('Round 4 — post-audit corrections', 'results_pre_round4', 'results')

CAUSE3 = ('Estimator thread count was not pinned. XGBoost and torch accumulate '
          'floating-point sums in thread-completion order, so the same seed on a '
          'machine with a different core count produced different trees, weights and '
          'metrics. Confirmed by changing only OMP_NUM_THREADS: 329 of 858 leaves '
          'moved and two clearance conditions changed verdict.')
FIX3 = ('n_jobs=1 on every XGBoost and scikit-learn estimator; torch pinned to one '
        'thread with deterministic algorithms enabled and a seeded DataLoader '
        'generator; the run regenerated once under the pinned settings. '
        'selftest_determinism.py verifies thread-count independence on any machine.')
CONF3 = ('OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped')

CAUSE4 = ('Two defects found by an independent audit of the round-3 push. (a) C2 was '
          'decided by comparing each leave-one-out fold against the sign of a mean '
          'delta that C1 had already shown to be unstable, so its PASS/FAIL turned on '
          'which side of zero a quantity indistinguishable from zero fell. (b) The RPS '
          'artefact wrote only the rank-preservation naive baseline for the cross-model '
          'population, under an unqualified name, so R5 quoted that baseline beside a '
          'directional-agreement figure.')
FIX4 = ('(a) closeout.py returns UNVERIFIABLE for C2 when C1 has failed, leaving '
        'C2_PASS and therefore the null state and the certificate count untouched. '
        '(b) rps_results.json carries both naive baselines for the cross-model '
        'population, each named for the quantity it belongs to, and the generator that '
        'writes the R5 sentence reads each figure from its own key. No model was '
        'refitted and no estimator setting changed.')
CONF4 = ('Clean-clone re-run under the round-3 pins: 859/859 leaves of results.json '
         'bit-identical, so nothing model-derived moved')


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def leaf_diff(a, b, path='', out=None, tol=1e-9):
    if out is None:
        out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append((path + '/' + str(k), None, b[k], 'added'))
            elif k not in b:
                out.append((path + '/' + str(k), a[k], None, 'removed'))
            else:
                leaf_diff(a[k], b[k], path + '/' + str(k), out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, 'len %d' % len(a), 'len %d' % len(b), 'length'))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                leaf_diff(x, y, path + '[%d]' % i, out, tol)
    else:
        if isinstance(a, float) and isinstance(b, float):
            if abs(a - b) > tol:
                out.append((path, a, b, 'changed'))
        elif a != b:
            out.append((path, a, b, 'changed'))
    return out


def g(d, *keys, default=None):
    """Safe nested get."""
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return default if cur is None else cur


def build(PREV, NEW, manual_rows=()):
    R0, R1 = load(PREV, 'results.json'), load(NEW, 'results.json')
    V0, V1 = load(PREV, 'closeout_verdict.json'), load(NEW, 'closeout_verdict.json')
    P0, P1 = load(PREV, 'rps_results.json'), load(NEW, 'rps_results.json')
    L0, L1 = load(PREV, 'loso_validation.json'), load(NEW, 'loso_validation.json')
    N0, N1 = load(PREV, 'smote_nnaa.json'), load(NEW, 'smote_nnaa.json')
    S0, S1 = load(PREV, 'school_structure.json'), load(NEW, 'school_structure.json')

    diffs = {}
    for label, a, b in (('results.json', R0, R1), ('closeout_verdict.json', V0, V1),
                        ('rps_results.json', P0, P1), ('loso_validation.json', L0, L1),
                        ('smote_nnaa.json', N0, N1), ('school_structure.json', S0, S1)):
        if a is None or b is None:
            continue
        diffs[label] = leaf_diff(a, b)

    total = sum(len(v) for v in diffs.values())

    # ---- the quantities that appear in the manuscript ----------------------
    def t3(R, model, metric, stat='mean'):
        return g(R, 'table3', model, metric, stat)

    def _mc(R, comparison, field):
        for row in (R or {}).get('stats', {}).get('mcnemar', []):
            if row.get('comparison') == comparison:
                v = row.get(field)
                return round(v, 4) if isinstance(v, float) else v
        return None

    _MA1 = load(NEW, 'model_artefacts.json') or {}
    _MA0 = load(PREV, 'model_artefacts.json')

    def _ma(k, fallback):
        """Checkpoint sizes, from the baseline's own artefact where it has one.

        The round-3 baseline predates export_models.py, so there is no
        model_artefacts.json under it and the round-2 figures carried in the
        manuscript are the honest 'before'. Every later baseline has the file,
        and reading it is what keeps a size that did not move out of the moved
        column.
        """
        return (_MA0 or {}).get(k, fallback) if _MA0 is not None else fallback

    def _ma_after(k):
        return _MA1.get(k)

    rows = [
        # (manuscript location, quantity, before, after)
        ('R2 / Table 3', 'Proposed XGBoost AUC-PR (5-seed mean)',
         t3(R0, 'xgb_engineered', 'auc_pr'), t3(R1, 'xgb_engineered', 'auc_pr')),
        ('R2 / Table 3', 'Proposed XGBoost AUC-PR (SD)',
         t3(R0, 'xgb_engineered', 'auc_pr', 'std'), t3(R1, 'xgb_engineered', 'auc_pr', 'std')),
        ('R2 / Table 3', 'Default XGBoost AUC-PR',
         t3(R0, 'xgb_default', 'auc_pr'), t3(R1, 'xgb_default', 'auc_pr')),
        ('R2 / Table 3', 'TabTransformer AUC-PR',
         t3(R0, 'tabtransformer', 'auc_pr'), t3(R1, 'tabtransformer', 'auc_pr')),
        ('R2 / Table 3', 'Decision Tree AUC-PR',
         t3(R0, 'decision_tree', 'auc_pr'), t3(R1, 'decision_tree', 'auc_pr')),
        ('R2 / Table 3', 'Proposed XGBoost recall',
         t3(R0, 'xgb_engineered', 'recall'), t3(R1, 'xgb_engineered', 'recall')),
        ('R2 / Table 3', 'TabTransformer recall',
         t3(R0, 'tabtransformer', 'recall'), t3(R1, 'tabtransformer', 'recall')),
        ('R2 / R4 / R10', 'Proposed seed-42 AUC-PR',
         g(R0, 'table3_seed42', 'xgb_engineered', 'auc_pr'),
         g(R1, 'table3_seed42', 'xgb_engineered', 'auc_pr')),
        ('R2 / R10', 'Proposed seed-42 AUC-ROC',
         g(R0, 'table3_seed42', 'xgb_engineered', 'auc_roc'),
         g(R1, 'table3_seed42', 'xgb_engineered', 'auc_roc')),
        ('R5 / R7 / Table 6', 'Seed-42 operating threshold (proposed)',
         g(R0, 'seed42_thresholds', 'xgb_engineered'),
         g(R1, 'seed42_thresholds', 'xgb_engineered')),
        ('R5 / Table 5', 'Records flagged at the seed-42 threshold',
         g(R0, 'seed42_flags', 'xgb_engineered'), g(R1, 'seed42_flags', 'xgb_engineered')),
        ('M8 / M12', 'scale_pos_weight at seed 42',
         g(R0, 'scale_pos_weight_seed42'), g(R1, 'scale_pos_weight_seed42')),
        ('R5 / Figure 8', 'RPS@1 (rank-preserving)',
         g(P0, 'rank_preservation_RPS', 'rps_at_1'), g(P1, 'rank_preservation_RPS', 'rps_at_1')),
        ('R5 / Figure 8', 'RPS@2 (rank-preserving)',
         g(P0, 'rank_preservation_RPS', 'rps_at_2'), g(P1, 'rank_preservation_RPS', 'rps_at_2')),
        ('R5 / Figure 8', 'RPS@1 (naive fixed order)',
         g(P0, 'rank_preservation_RPS', 'naive_fixed_order_at_1'),
         g(P1, 'rank_preservation_RPS', 'naive_fixed_order_at_1')),
        ('R5 / Figure 8', 'DAS@1 (rank-preserving)',
         g(P0, 'directional_agreement_DAS', 'das_at_1'),
         g(P1, 'directional_agreement_DAS', 'das_at_1')),
        ('R5 / Table 5', 'Ablation population n (proposed own threshold)',
         g(P0, 'n_evaluated'), g(P1, 'n_evaluated')),
        ('R5', 'Cross-model sensitivity population n',
         g(P0, 'cross_model_sensitivity', 'n_evaluated'),
         g(P1, 'cross_model_sensitivity', 'n_evaluated')),
        ('R10', 'SMOTE 0.50 arm AUC-PR',
         g(R0, 'smote_ablation', 'smote_0.50', 'summary', 'auc_pr', 'mean'),
         g(R1, 'smote_ablation', 'smote_0.50', 'summary', 'auc_pr', 'mean')),
        ('R10', 'Layer 2 spw=1 arm AUC-PR',
         g(R0, 'layer2_ablation', 'spw_1__f2_threshold', 'summary', 'auc_pr', 'mean'),
         g(R1, 'layer2_ablation', 'spw_1__f2_threshold', 'summary', 'auc_pr', 'mean')),
        ('R10', 'SMOTE k=3 NNAA at seed 42',
         g(N0, 'summary', 'SMOTE k=3', 'seed42'), g(N1, 'summary', 'SMOTE k=3', 'seed42')),
        ('R3', 'Ten-seed delta vs attendance ranker (mean)',
         g(V0, 'v1a_summary', 'delta_mean'), g(V1, 'v1a_summary', 'delta_mean')),
        # Named for what the statistic is — min(n_positive, n_negative), the number
        # of seeds carrying the minority sign — not for "sign changes", which would
        # be an order-dependent count of a set that has no order.
        ('R3 / R12', 'Ten-seed delta, seeds carrying the minority sign',
         g(V0, 'v1a_summary', 'sign_flips'), g(V1, 'v1a_summary', 'sign_flips')),
        ('R5', 'Cross-model DAS@1 naive baseline (artefact key)',
         g(P0, 'cross_model_sensitivity', 'das_naive_fixed_order_at_1'),
         g(P1, 'cross_model_sensitivity', 'das_naive_fixed_order_at_1')),
        ('R3', 'Leave-one-out, synthetic dropped (AUC-PR)',
         (V0.get('v1b_leave_one_out') or [{}, {}])[1].get('auc_pr') if V0 else None,
         (V1.get('v1b_leave_one_out') or [{}, {}])[1].get('auc_pr') if V1 else None),
        ('R4', 'Discordant pairs vs the attendance rule',
         g(V0, 'v1c', 'n_discordant'), g(V1, 'v1c', 'n_discordant')),
        ('R4', 'Discordant pairs required at 80% power',
         g(V0, 'v1c', 'discordant_pairs_required'), g(V1, 'v1c', 'discordant_pairs_required')),
        ('R8', 'LOSO SCH01 AUC-PR',
         next((f.get('summary', {}).get('auc_pr', {}).get('mean')
               for f in (L0 or {}).get('folds', []) if f.get('held_out_site') == 'SCH01'), None),
         next((f.get('summary', {}).get('auc_pr', {}).get('mean')
               for f in (L1 or {}).get('folds', []) if f.get('held_out_site') == 'SCH01'), None)),
        ('R8', 'LOSO SCH02 AUC-PR',
         next((f.get('summary', {}).get('auc_pr', {}).get('mean')
               for f in (L0 or {}).get('folds', []) if f.get('held_out_site') == 'SCH02'), None),
         next((f.get('summary', {}).get('auc_pr', {}).get('mean')
               for f in (L1 or {}).get('folds', []) if f.get('held_out_site') == 'SCH02'), None)),
        ('R4 / Table 4', 'McNemar vs TabTransformer: discordant b',
         _mc(R0, 'TabTransformer', 'b'), _mc(R1, 'TabTransformer', 'b')),
        ('R4 / Table 4', 'McNemar vs TabTransformer: discordant c',
         _mc(R0, 'TabTransformer', 'c'), _mc(R1, 'TabTransformer', 'c')),
        ('R4 / Table 4', 'McNemar vs TabTransformer: BH-adjusted p',
         _mc(R0, 'TabTransformer', 'p_bh'), _mc(R1, 'TabTransformer', 'p_bh')),
        ('R4 / Table 4 / R10', 'McNemar vs TabTransformer: DIRECTION',
         _mc(R0, 'TabTransformer', 'direction'), _mc(R1, 'TabTransformer', 'direction')),
        ('R4 / Table 4', 'McNemar vs Default XGBoost: discordant b/c',
         '%s/%s' % (_mc(R0, 'Default XGBoost', 'b'), _mc(R0, 'Default XGBoost', 'c')),
         '%s/%s' % (_mc(R1, 'Default XGBoost', 'b'), _mc(R1, 'Default XGBoost', 'c'))),
        ('R4 / Table 4', 'McNemar vs Attendance rule: discordant b/c',
         '%s/%s' % (_mc(R0, 'Attendance rule (binary)', 'b'),
                    _mc(R0, 'Attendance rule (binary)', 'c')),
         '%s/%s' % (_mc(R1, 'Attendance rule (binary)', 'b'),
                    _mc(R1, 'Attendance rule (binary)', 'c'))),
        ('Table 3', 'TabTransformer AUC-ROC',
         t3(R0, 'tabtransformer', 'auc_roc'), t3(R1, 'tabtransformer', 'auc_roc')),
        ('Table 3', 'TabTransformer F2',
         t3(R0, 'tabtransformer', 'f2'), t3(R1, 'tabtransformer', 'f2')),
        ('Table 3', 'TabTransformer precision',
         t3(R0, 'tabtransformer', 'precision'), t3(R1, 'tabtransformer', 'precision')),
        ('R7 / Figure 7', 'TabTransformer seed-42 threshold',
         g(R0, 'seed42_thresholds', 'tabtransformer'),
         g(R1, 'seed42_thresholds', 'tabtransformer')),
        ('R5', 'TabTransformer seed-42 flagged count',
         g(R0, 'seed42_flags', 'tabtransformer'), g(R1, 'seed42_flags', 'tabtransformer')),
        ('M19', 'TabTransformer checkpoint size (MB)',
         _ma('tabtransformer_checkpoint_mb', 1.25), _ma_after('tabtransformer_checkpoint_mb')),
        ('M19', 'Proposed XGBoost checkpoint size (MB)',
         _ma('xgboost_checkpoint_mb', 0.29), _ma_after('xgboost_checkpoint_mb')),
        ('R12', 'C1 verdict', g(V0, 'clearance', 'C1'), g(V1, 'clearance', 'C1')),
        ('R12', 'C2 verdict', g(V0, 'clearance', 'C2'), g(V1, 'clearance', 'C2')),
        ('R12', 'C3 verdict', g(V0, 'clearance', 'C3'), g(V1, 'clearance', 'C3')),
        ('R12', 'Null state', g(V0, 'null_state'), g(V1, 'null_state')),
    ]

    rows = list(rows) + [tuple(r) for r in manual_rows]
    moved = [r for r in rows if r[2] != r[3]]
    held = [r for r in rows if r[2] == r[3]]
    return dict(rows=rows, moved=moved, held=held, diffs=diffs, total=total)


def section(label, prev, new, cause, fix, confirmed, res):
    """One round's Section F table plus the scope statement that frames it."""
    md = ['| WHERE | QUANTITY | CAUSE | CONFIRMED BY | FIX APPLIED | BEFORE | AFTER |',
          '|---|---|---|---|---|---|---|']
    for where, what, before, after in res['moved']:
        md.append('| %s | %s | %s | %s | %s | `%s` | `%s` |'
                  % (where, what, cause, confirmed, fix, before, after))
    if len(md) == 2:
        md.append('| — | no manuscript-facing quantity moved | — | — | — | — | — |')
    b = []
    b.append('## %s\n' % label)
    b.append('Baseline `%s/` against `%s/`. Leaf-level differences across all '
             'regenerated artefacts: **%d**.\n' % (prev, new, res['total']))
    for k, v in sorted(res['diffs'].items()):
        b.append('- `%s`: %d changed leaves' % (k, len(v)))
    b.append('')
    b.append('**%d of %d manuscript-facing quantities moved; %d held.**\n'
             % (len(res['moved']), len(res['rows']), len(res['held'])))
    b.append('\n'.join(md))
    b.append('')
    return '\n'.join(b)


def main():
    # Round 4 carries two rows that are not artefact-to-artefact movements: a
    # figure the manuscript quoted from the wrong key, and a statement in Table 7
    # that the round-3 ledger corrected in R4 and R10 but did not reach. Section F
    # asks for one row per number that moves as the reader sees it, so they are
    # declared here rather than left out because no JSON leaf holds the old value.
    _p4 = load(ROUND4[2], 'rps_results.json') or {}
    _xm = _p4.get('cross_model_sensitivity', {})
    _r4 = load(ROUND4[2], 'results.json') or {}
    _mcn = _r4.get('stats', {}).get('mcnemar', [])
    _prop = [r for r in _mcn if 'proposed' in str(r.get('direction', '')).lower()]
    _comp = [r for r in _mcn if r not in _prop]
    manual4 = [
        ('R5', 'Cross-model DAS@1 comparison baseline AS QUOTED IN R5',
         _xm.get('rps_naive_fixed_order_at_1'), _xm.get('das_naive_fixed_order_at_1')),
        ('R11 / Table 7', 'McNemar directions row',
         'All four favour the comparator',
         '%d favour the comparator, %d favours the proposed model'
         % (len(_comp), len(_prop))),
    ]

    l3, p3, n3 = ROUND3
    l4, p4, n4 = ROUND4
    r3 = build(p3, n3)
    r4 = build(p4, n4, manual_rows=manual4)

    short_cause3 = 'Unpinned estimator thread count (non-deterministic reduction order)'
    short_fix3 = ('n_jobs=1 everywhere, torch single-thread + deterministic, seeded '
                  'DataLoader; run regenerated once')
    short_cause4 = ('C2 decided on the sign of an unstable mean delta; RPS artefact wrote '
                    'one naive baseline under an unqualified name')
    short_fix4 = ('C2 returns UNVERIFIABLE when C1 fails; both naive baselines written and '
                  'read from their own keys; no refit')

    body = []
    body.append('# Corrections ledger (close-out Section F)\n')
    body.append('Two rounds of correction, each against its own committed baseline and each '
                'with its own cause. The complete leaf-by-leaf diff for both is committed as '
                '`results/corrections_ledger.json`. The tables below are not samples of it: '
                'each is every quantity that appears in the Methods or Results sections, '
                'which is the set a reader can check against the manuscript.\n')
    body.append(section(l3, p3, n3, short_cause3, CONF3, short_fix3, r3))
    body.append('### Cause (round 3)\n')
    body.append(CAUSE3)
    body.append('\n### Fix (round 3)\n')
    body.append(FIX3)
    body.append('')
    body.append(section(l4, p4, n4, short_cause4, CONF4, short_fix4, r4))
    body.append('### Cause (round 4)\n')
    body.append(CAUSE4)
    body.append('\n### Fix (round 4)\n')
    body.append(FIX4)
    body.append('')
    if r3['held']:
        body.append('\n### Quantities that did NOT move across either round\n')
        stayed = {(w, q) for w, q, b_, a_ in r3['held']} & {(w, q) for w, q, b_, a_ in r4['held']}
        for where, what, before, _ in r3['held']:
            if (where, what) in stayed:
                body.append('- %s — %s: `%s`' % (where, what, before))

    merged_moved = r3['moved'] + r4['moved']
    os.makedirs('results', exist_ok=True)
    json.dump(dict(
        total_changed_leaves=r3['total'] + r4['total'],
        by_artefact={k: len(v) for k, v in r4['diffs'].items()},
        cause=CAUSE4, fix=FIX4,
        rounds=[dict(label=l3, baseline=p3, regenerated=n3, cause=CAUSE3, fix=FIX3,
                     confirmed_by=CONF3, changed_leaves=r3['total'],
                     by_artefact={k: len(v) for k, v in r3['diffs'].items()},
                     moved=[dict(where=w, quantity=q, before=b_, after=a_)
                            for w, q, b_, a_ in r3['moved']],
                     full_diff={k: [dict(path=p_, before=x, after=y, kind=z)
                                    for p_, x, y, z in v] for k, v in r3['diffs'].items()}),
                dict(label=l4, baseline=p4, regenerated=n4, cause=CAUSE4, fix=FIX4,
                     confirmed_by=CONF4, changed_leaves=r4['total'],
                     by_artefact={k: len(v) for k, v in r4['diffs'].items()},
                     moved=[dict(where=w, quantity=q, before=b_, after=a_)
                            for w, q, b_, a_ in r4['moved']],
                     full_diff={k: [dict(path=p_, before=x, after=y, kind=z)
                                    for p_, x, y, z in v] for k, v in r4['diffs'].items()})],
        manuscript_quantities=[dict(where=w, quantity=q, before=b_, after=a_)
                               for w, q, b_, a_ in r4['rows']],
        moved=[dict(where=w, quantity=q, before=b_, after=a_)
               for w, q, b_, a_ in merged_moved],
        full_diff={k: [dict(path=p_, before=x, after=y, kind=z)
                       for p_, x, y, z in v] for k, v in r4['diffs'].items()}),
        open('results/corrections_ledger.json', 'w'), indent=2, default=str)
    open('results/corrections_ledger.md', 'w').write('\n'.join(body) + '\n')

    for label, res in ((l3, r3), (l4, r4)):
        print('%s: %d changed leaves, %d of %d manuscript quantities moved'
              % (label, res['total'], len(res['moved']), len(res['rows'])))
        for where, what, before, after in res['moved']:
            print('  %-16s %-52s %s -> %s' % (where, what[:52], before, after))
    print('\nwritten: results/corrections_ledger.json and .md')


if __name__ == '__main__':
    main()
