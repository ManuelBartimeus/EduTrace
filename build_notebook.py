#!/usr/bin/env python3
"""
Builds notebooks/EduTrace_Main_3.ipynb — the second file Methods M19 names as
the reproduction path — and executes it top to bottom so the stored
execution_count sequence is linear.

Q4 named four defects in the previous notebook: a PROJECT_DIR pointing at a
mounted Drive rather than the repository's own data/, a non-monotonic
execution_count with two cells never run, a cell that mutated a results store in
place before a later cell wrote it out, and a local shap.py shadowing the
installed package. This notebook has none of them: it reads ./data and
./results by relative path, it holds no mutable global state, and it is executed
once, in order, by this script.
"""
import os, subprocess, sys
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'notebooks', 'EduTrace_Main_3.ipynb')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

cells = [
    new_markdown_cell(
        "# EduTrace — reproduction notebook\n"
        "\n"
        "This notebook is the second of the two files Methods M19 names as the reproduction "
        "path. The first is `EduTrace_Revised_Pipeline.py`.\n"
        "\n"
        "It runs top to bottom from a clean clone with no path editing. Every path below is "
        "relative to the repository root, so nothing reads a mounted Drive.\n"
        "\n"
        "**Two modes.** By default the notebook *reads the locked artefacts* in `results/` and "
        "displays every reported quantity, which takes seconds. Setting `REGENERATE = True` "
        "re-runs the full pipeline from `data/` instead (about 25 minutes on CPU) and then "
        "displays the same quantities from the regenerated file. Both paths read the same "
        "committed inputs and land on the same numbers; the default exists so a reviewer can "
        "check the reconciliation without waiting for TabTransformer to train."),

    new_code_cell(
        "import json, os, subprocess, sys\n"
        "\n"
        "# Repository root, whether the notebook is opened from notebooks/ or from the root.\n"
        "ROOT = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.')\n"
        "os.chdir(ROOT)\n"
        "sys.path.insert(0, ROOT)\n"
        "\n"
        "REAL    = os.path.join('data', 'real_student_data_CLEANED_school.csv')\n"
        "SYNTH   = os.path.join('data', 'synth_ctgan_s42.csv')\n"
        "RESULTS = os.path.join('results', 'results.json')\n"
        "\n"
        "REGENERATE = False   # True -> re-run the pipeline instead of reading the locked run\n"
        "\n"
        "print('repository root :', ROOT)\n"
        "for p in (REAL, SYNTH):\n"
        "    print('%-42s exists=%s' % (p, os.path.exists(p)))"),

    new_markdown_cell(
        "## 1. The provenance guard (NF-1, Q9)\n"
        "\n"
        "`assert_test_partition_is_real()` lives in the load path of "
        "`EduTrace_Revised_Pipeline.py`. It raises on any synthetic row in the evaluation "
        "partition. The self-test below also runs the negative control — it injects a "
        "synthetic row into the test period and confirms the guard fires — because an "
        "assertion that never fires on clean data proves nothing."),

    new_code_cell(
        "print(subprocess.run([sys.executable, 'selftest_nf1.py'],\n"
        "                     capture_output=True, text=True).stdout)"),

    new_markdown_cell(
        "## 2. The locked run\n"
        "\n"
        "Reads `results/results.json`, or regenerates it when `REGENERATE` is set."),

    new_code_cell(
        "if REGENERATE:\n"
        "    subprocess.run([sys.executable, 'EduTrace_Revised_Pipeline.py',\n"
        "                    '--real', REAL, '--synth', SYNTH, '--out', RESULTS], check=True)\n"
        "\n"
        "R = json.load(open(RESULTS))\n"
        "print('keys in the results archive:', len(R))\n"
        "print()\n"
        "print('train+validation pool :', R['pool'])\n"
        "print('evaluation partition  :', R['test'])"),

    new_markdown_cell(
        "## 3. Table 3 — held-out performance, mean ± SD across five seeds\n"
        "\n"
        "The base rate is printed beside the metrics (clearance condition C5): AUC-PR is "
        "bounded below by class prevalence, so a level is not interpretable without it."),

    new_code_cell(
        "base = R['test']['pct_pos'] / 100.0\n"
        "print('test-set base rate: %.4f  (%d positives of %d)\\n'\n"
        "      % (base, R['test']['pos'], R['test']['n']))\n"
        "print('%-24s %-18s %-18s %-18s' % ('model', 'AUC-PR', 'AUC-ROC', 'recall'))\n"
        "for k, v in R['table3'].items():\n"
        "    print('%-24s %.4f +- %.4f   %.4f +- %.4f   %.4f +- %.4f'\n"
        "          % (k, v['auc_pr']['mean'], v['auc_pr']['std'],\n"
        "             v['auc_roc']['mean'], v['auc_roc']['std'],\n"
        "             v['recall']['mean'], v['recall']['std']))\n"
        "print()\n"
        "print('attendance rule (binary)   single-point AUC-ROC %.4f | recall %.4f | precision %.4f'\n"
        "      % (R['attendance_binary_rule']['single_point_auc_roc'],\n"
        "         R['attendance_binary_rule']['recall'],\n"
        "         R['attendance_binary_rule']['precision']))\n"
        "print('attendance ranker (cont.)  AUC-ROC %.4f | AUC-PR %.4f'\n"
        "      % (R['attendance_continuous_ranker']['auc_roc'],\n"
        "         R['attendance_continuous_ranker']['auc_pr']))\n"
        "print()\n"
        "print('seed-42 thresholds:', R['seed42_thresholds'])\n"
        "print('seed-42 scale_pos_weight:', R['scale_pos_weight_seed42'])"),

    new_markdown_cell(
        "## 4. NF-1 — the restored real-only re-scoring\n"
        "\n"
        "An earlier draft of M10 retired this analysis as redundant. It is computed here "
        "rather than asserted. Under a correctly bounded synthesiser the two rows are "
        "identical, because the real-only subset *is* the evaluation partition — and that "
        "identity, computed, is the evidence."),

    new_code_cell(
        "nf1 = R['nf1_real_only_rescore']\n"
        "for label, row in nf1['rows'].items():\n"
        "    print('%-22s n=%3d  positives=%d  base=%.4f  AUC-PR=%.4f  recall=%.4f  precision=%.4f'\n"
        "          % (label, row['n'], row['positives'], row['base_rate'],\n"
        "             row['auc_pr'], row['recall'], row['precision']))\n"
        "print()\n"
        "print('subsets identical      :', nf1['subsets_identical'])\n"
        "print('synthetic rows removed :', nf1['n_synthetic_removed'])\n"
        "print()\n"
        "for line in R['nf1_guard_log']:\n"
        "    print(line)"),

    new_markdown_cell(
        "## 5. Q21 — the contribution-evidence anchor\n"
        "\n"
        "RPS and DAS are reported under separate names (M17). RPS asks whether the packed "
        "alert names the model's own top-k SHAP features in order; DAS asks whether a top-k "
        "attribution carries a sign agreeing with the observed outcome. The original "
        "implementation computed the second and called it the first."),

    new_code_cell(
        "rps = json.load(open(os.path.join('results', 'rps_results.json')))\n"
        "print('population        :', rps['population'])\n"
        "print('threshold_used    :', rps['threshold_used'])\n"
        "print('n_evaluated       :', rps['n_evaluated'])\n"
        "print('true dropouts in that population:', rps['true_dropouts_in_population'])\n"
        "print()\n"
        "for block in ('rank_preservation_RPS', 'directional_agreement_DAS'):\n"
        "    print(block)\n"
        "    for k, v in rps[block].items():\n"
        "        if k != 'definition':\n"
        "            print('   %-26s %s' % (k, v))\n"
        "print()\n"
        "print('cross-model sensitivity:', {k: v for k, v in rps['cross_model_sensitivity'].items()\n"
        "                                   if k != 'population'})\n"
        "print()\n"
        "print('Table 5 character compliance:', rps['table5_character_compliance'])\n"
        "print()\n"
        "for m in rps['example_alerts'][:2]:\n"
        "    print('  (%d chars) %s' % (len(m), m))"),

    new_markdown_cell(
        "## 6. Q23 — the imbalance isolation grid, every cell\n"
        "\n"
        "Sixteen cells: four SMOTE levels × two `scale_pos_weight` settings × two threshold "
        "arms. The spread is what is reported. No cell is selected."),

    new_code_cell(
        "import csv\n"
        "with open(os.path.join('results', 'q23_imbalance_grid.csv')) as fh:\n"
        "    grid = list(csv.DictReader(fh))\n"
        "print('%9s %6s %13s %9s %8s %8s %8s' % ('smote', 'spw', 'thresh', 'AUC-PR',\n"
        "                                        'recall', 'prec', 'F2'))\n"
        "for r in grid:\n"
        "    print('%9s %6s %13s %9s %8s %8s %8s'\n"
        "          % (r['smote'], r['spw'], r['threshold_arm'], r['auc_pr'],\n"
        "             r['recall'], r['precision'], r['f2']))\n"
        "aps = [float(r['auc_pr']) for r in grid]\n"
        "print('\\nAUC-PR across the grid: min=%.4f max=%.4f range=%.4f'\n"
        "      % (min(aps), max(aps), max(aps) - min(aps)))"),

    new_markdown_cell(
        "## 7. Q24 — rollback and temporal integrity\n"
        "\n"
        "Each remediation layer reverted to its pre-remediation setting, one at a time, at the "
        "cut the Method declares (2025). Results that run against the study's own design "
        "choices are printed here rather than summarised away."),

    new_code_cell(
        "for name in ('q24_rollback_comparison.csv', 'q24_cut_comparison.csv'):\n"
        "    print('---', name)\n"
        "    with open(os.path.join('results', name)) as fh:\n"
        "        for row in csv.DictReader(fh):\n"
        "            print('   ', {k: row[k] for k in list(row)[:8]})\n"
        "    print()"),

    new_markdown_cell(
        "## 8. The clearance certificate and the null state\n"
        "\n"
        "C1–C6 and the null state, from `closeout.py`. The null state governs what may be "
        "written: only state 4 licenses a positive finding."),

    new_code_cell(
        "V = json.load(open(os.path.join('results', 'closeout_verdict.json')))\n"
        "print('%-6s %-44s %s' % ('ITEM', 'CHECK', 'VERDICT'))\n"
        "for r in V['rows']:\n"
        "    print('%-6s %-44s %s' % (r['item'], r['check'], r['verdict']))\n"
        "print()\n"
        "for c, state in V['clearance'].items():\n"
        "    print('  %s  %s' % (c, state))\n"
        "print()\n"
        "print('NULL STATE : %d' % V['null_state'])\n"
        "print('EVIDENCE   : %s' % V['null_evidence'])\n"
        "print()\n"
        "s = V['v1a_summary']\n"
        "print('ten-seed delta vs the attendance ranker: %+.4f +- %.4f  '\n"
        "      '(%d positive / %d negative, %d sign flips)'\n"
        "      % (s['delta_mean'], s['delta_sd'], s['n_positive'], s['n_negative'],\n"
        "         s['sign_flips']))\n"
        "c3 = V['v1c']\n"
        "print('paired MDE: %d discordant pairs observed, split %.3f; %d would be required'\n"
        "      % (c3['n_discordant'], c3['observed_split'], c3['discordant_pairs_required']))"),

    new_markdown_cell(
        "## 9. Reconciliation\n"
        "\n"
        "The quantities printed above are the ones Results R11 / Table 7 reconciles. Every "
        "figure in the manuscript traces to `results/results.json` from this single locked "
        "run."),
]

nb = new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': sys.version.split()[0]}})

nbf.write(nb, OUT)
print('wrote (unexecuted):', OUT)

# Execute in place so execution_count is linear and the stored outputs are real.
res = subprocess.run(
    [sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute',
     '--inplace', '--ExecutePreprocessor.timeout=2400',
     '--ExecutePreprocessor.kernel_name=python3', OUT],
    capture_output=True, text=True, cwd=HERE)
print(res.stdout[-2000:]); print(res.stderr[-2000:])

nb2 = nbf.read(OUT, as_version=4)
counts = [c.get('execution_count') for c in nb2.cells if c.cell_type == 'code']
print('execution_count sequence:', counts)
print('monotonic and complete   :', counts == sorted(counts) and None not in counts)
