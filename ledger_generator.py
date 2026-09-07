#!/usr/bin/env python3
"""
Corrections ledger (close-out Section F), generated rather than hand-written.

Section F requires one row per number that moves, with its cause, and warns that
"a number that moved with no row here is reverted". The determinism repair moves
every model-derived figure, so the ledger has to be produced from the artefacts
themselves — a hand-typed table of that size would be wrong within a day.

This script compares the artefacts BEFORE the determinism repair (results_prev/)
against the regenerated ones (results/) and emits two things:

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

PREV, NEW = 'results_prev', 'results'


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


def main():
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

    CAUSE = ('Estimator thread count was not pinned. XGBoost and torch accumulate '
             'floating-point sums in thread-completion order, so the same seed on a '
             'machine with a different core count produced different trees, weights and '
             'metrics. Confirmed by changing only OMP_NUM_THREADS: 329 of 858 leaves '
             'moved and two clearance conditions changed verdict.')
    FIX = ('n_jobs=1 on every XGBoost and scikit-learn estimator; torch pinned to one '
           'thread with deterministic algorithms enabled and a seeded DataLoader '
           'generator; the run regenerated once under the pinned settings. '
           'selftest_determinism.py verifies thread-count independence on any machine.')

    # ---- the quantities that appear in the manuscript ----------------------
    def t3(R, model, metric, stat='mean'):
        return g(R, 'table3', model, metric, stat)

    def _mc(R, comparison, field):
        for row in (R or {}).get('stats', {}).get('mcnemar', []):
            if row.get('comparison') == comparison:
                v = row.get(field)
                return round(v, 4) if isinstance(v, float) else v
        return None

    _MA = load(NEW, 'model_artefacts.json') or {}

    def _ma(k):
        return _MA.get(k)

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
        ('R3', 'Ten-seed delta sign changes',
         g(V0, 'v1a_summary', 'sign_flips'), g(V1, 'v1a_summary', 'sign_flips')),
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
        ('M19', 'TabTransformer checkpoint size (MB)', 1.25, _ma('tabtransformer_checkpoint_mb')),
        ('M19', 'Proposed XGBoost checkpoint size (MB)', 0.29, _ma('xgboost_checkpoint_mb')),
        ('R12', 'C1 verdict', g(V0, 'clearance', 'C1'), g(V1, 'clearance', 'C1')),
        ('R12', 'C2 verdict', g(V0, 'clearance', 'C2'), g(V1, 'clearance', 'C2')),
        ('R12', 'C3 verdict', g(V0, 'clearance', 'C3'), g(V1, 'clearance', 'C3')),
        ('R12', 'Null state', g(V0, 'null_state'), g(V1, 'null_state')),
    ]

    moved = [r for r in rows if r[2] != r[3]]
    held = [r for r in rows if r[2] == r[3]]

    # ---- markdown table ----------------------------------------------------
    md = []
    md.append('| WHERE | QUANTITY | CAUSE | CONFIRMED BY | FIX APPLIED | BEFORE | AFTER |')
    md.append('|---|---|---|---|---|---|---|')
    short_cause = 'Unpinned estimator thread count (non-deterministic reduction order)'
    short_conf = 'OMP_NUM_THREADS experiment: 329/858 leaves moved, 2 clearance conditions flipped'
    short_fix = 'n_jobs=1 everywhere, torch single-thread + deterministic, seeded DataLoader; run regenerated once'
    for where, what, before, after in moved:
        md.append('| %s | %s | %s | %s | %s | `%s` | `%s` |'
                  % (where, what, short_cause, short_conf, short_fix, before, after))
    table = '\n'.join(md)

    body = []
    body.append('### Ledger scope\n')
    body.append('Leaf-level differences across all regenerated artefacts: **%d**.\n' % total)
    for k, v in sorted(diffs.items()):
        body.append('- `%s`: %d changed leaves' % (k, len(v)))
    body.append('')
    body.append('The complete leaf-by-leaf diff is committed as '
                '`results/corrections_ledger.json`. The table below is not a sample of it: '
                'it is every quantity that appears in the Methods or Results sections, '
                'which is the set a reader can check against the manuscript.\n')
    body.append('**%d of %d manuscript-facing quantities moved; %d held.**\n'
                % (len(moved), len(rows), len(held)))
    body.append(table)
    body.append('')
    if held:
        body.append('\n### Quantities that did NOT move\n')
        for where, what, before, _ in held:
            body.append('- %s — %s: `%s`' % (where, what, before))
    body.append('\n### Cause\n')
    body.append(CAUSE)
    body.append('\n### Fix\n')
    body.append(FIX)

    os.makedirs('results', exist_ok=True)
    json.dump(dict(total_changed_leaves=total,
                   by_artefact={k: len(v) for k, v in diffs.items()},
                   cause=CAUSE, fix=FIX,
                   manuscript_quantities=[dict(where=w, quantity=q, before=b, after=a)
                                          for w, q, b, a in rows],
                   moved=[dict(where=w, quantity=q, before=b, after=a)
                          for w, q, b, a in moved],
                   full_diff={k: [dict(path=p, before=x, after=y, kind=z)
                                  for p, x, y, z in v] for k, v in diffs.items()}),
              open('results/corrections_ledger.json', 'w'), indent=2, default=str)
    open('results/corrections_ledger.md', 'w').write('\n'.join(body) + '\n')

    print('total changed leaves: %d' % total)
    for k, v in sorted(diffs.items()):
        print('  %-28s %d' % (k, len(v)))
    print('\nmanuscript-facing quantities: %d moved, %d held' % (len(moved), len(held)))
    for where, what, before, after in moved:
        print('  %-16s %-46s %s -> %s' % (where, what[:46], before, after))
    print('\nwritten: results/corrections_ledger.json and .md')


if __name__ == '__main__':
    main()
