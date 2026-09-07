#!/usr/bin/env python3
"""
Sampling-frame and site-structure diagnostics (Q2 / M3 / Table 1b / M18).

Now that the school identifier exists, three things can be measured that the
manuscript previously had to leave unstated:

  1. The sampling frame per site — students, records, dropout events, base rate
     and partition sizes. This is the Table 1b "N of sites" row and the M3
     provenance declaration.

  2. Whether site membership is CONFOUNDED with a register variable. If a school
     is (say) single-sex, then a coefficient or attribution on gender is partly a
     school effect, and the manuscript should say so rather than let a reader
     infer a behavioural reading.

  3. Whether site membership is RECOVERABLE from the six register features. This
     is the sharper question for M18: if a model can identify the school from the
     features alone, then a leave-one-site-out fold is weaker evidence of site
     transfer than it appears, because the model can condition on site
     implicitly even when the identifier is withheld.

Every number here is descriptive. Nothing in this file trains or alters the
reported model.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import EduTrace_Revised_Pipeline as P

SRC = os.path.join('data', 'real_student_data_CLEANED_school.csv')
OUT = os.path.join('results', 'school_structure.json')


def cramers_v(table):
    chi2 = sps.chi2_contingency(table)[0]
    n = table.values.sum()
    r, k = table.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float('nan')


def main():
    df = pd.read_csv(SRC)
    df['_sid'] = df['student_id'].str.split('_').str[0]
    names = dict(df[['school_id', 'school_name']].drop_duplicates().values)
    sites = sorted(df['school_id'].unique())

    print('=' * 96)
    print('1. SAMPLING FRAME BY SITE   (M3, Table 1b)')
    print('=' * 96)
    print('%-7s %-36s %8s %8s %9s %10s %9s %8s %9s %8s'
          % ('CODE', 'SCHOOL', 'STUDENTS', 'RECORDS', 'DROPOUTS', 'BASE RATE',
             'TRAIN_N', 'TR_POS', 'TEST_N', 'TE_POS'))
    frame = []
    for s in sites:
        d = df[df['school_id'] == s]
        tr = d[d[P.YEAR] < P.SPLIT_YEAR]; te = d[d[P.YEAR] >= P.SPLIT_YEAR]
        row = dict(school_id=s, school_name=names[s],
                   students=int(d['_sid'].nunique()), records=int(len(d)),
                   dropouts=int(d[P.TARGET].sum()),
                   base_rate=round(float(d[P.TARGET].mean()), 4),
                   train_records=int(len(tr)), train_positives=int(tr[P.TARGET].sum()),
                   test_records=int(len(te)), test_positives=int(te[P.TARGET].sum()),
                   share_of_students=round(100.0 * d['_sid'].nunique() / df['_sid'].nunique(), 1),
                   share_of_records=round(100.0 * len(d) / len(df), 1))
        frame.append(row)
        print('%-7s %-36s %8d %8d %9d %10.4f %9d %8d %9d %8d'
              % (s, names[s][:36], row['students'], row['records'], row['dropouts'],
                 row['base_rate'], row['train_records'], row['train_positives'],
                 row['test_records'], row['test_positives']))
    print('%-7s %-36s %8d %8d %9d %10.4f %9d %8d %9d %8d'
          % ('TOTAL', '4 sites', df['_sid'].nunique(), len(df), int(df[P.TARGET].sum()),
             float(df[P.TARGET].mean()),
             int((df[P.YEAR] < P.SPLIT_YEAR).sum()),
             int(df[df[P.YEAR] < P.SPLIT_YEAR][P.TARGET].sum()),
             int((df[P.YEAR] >= P.SPLIT_YEAR).sum()),
             int(df[df[P.YEAR] >= P.SPLIT_YEAR][P.TARGET].sum())))

    br = [r['base_rate'] for r in frame]
    print('\n  site base rates span %.4f to %.4f (a %.1fx spread across sites)'
          % (min(br), max(br), max(br) / max(min(br), 1e-9)))
    print('  largest site holds %.1f%% of students; smallest holds %.1f%%'
          % (max(r['share_of_students'] for r in frame),
             min(r['share_of_students'] for r in frame)))
    zero_pos = [r['school_id'] for r in frame if r['test_positives'] == 0]
    print('  sites with NO test-period dropout event: %d (%s)'
          % (len(zero_pos), ', '.join(zero_pos) or '—'))

    print()
    print('=' * 96)
    print('2. IS SITE CONFOUNDED WITH A REGISTER VARIABLE?')
    print('=' * 96)
    conf = {}
    for col in P.CAT:
        tab = pd.crosstab(df['school_id'], df[col])
        chi2, p, dof, _ = sps.chi2_contingency(tab)
        v = cramers_v(tab)
        conf[col] = dict(cramers_v=round(v, 4), chi2=round(float(chi2), 2),
                         p_value=float(p), table=tab.to_dict())
        flag = ('SEVERE' if v >= 0.7 else 'strong' if v >= 0.5 else
                'moderate' if v >= 0.3 else 'weak')
        print("\n  %s  — Cramer's V = %.4f (%s), chi2 = %.2f, p = %.3g"
              % (col, v, flag, chi2, p))
        print(tab.to_string().replace('\n', '\n    '))
        # name the extreme cells
        pct = (tab.T / tab.sum(axis=1)).T * 100
        for s in tab.index:
            top = pct.loc[s].idxmax()
            if pct.loc[s, top] >= 95:
                print('      NOTE: %s is %.1f%% "%s" — %s is effectively constant within '
                      'this site' % (s, pct.loc[s, top], top, col))

    print('\n  distance_km by site (continuous):')
    dist = {}
    for s in sites:
        d = df[df['school_id'] == s]['distance_km']
        dist[s] = dict(min=float(d.min()), max=float(d.max()),
                       mean=round(float(d.mean()), 3),
                       values=sorted(float(x) for x in d.unique()))
        print('    %-7s min=%.1f max=%.1f mean=%.3f values=%s'
              % (s, d.min(), d.max(), d.mean(), dist[s]['values']))
    kw = sps.kruskal(*[df[df['school_id'] == s]['distance_km'].dropna() for s in sites])
    print('    Kruskal-Wallis across sites: H = %.2f, p = %.3g' % (kw.statistic, kw.pvalue))

    print()
    print('=' * 96)
    print('3. IS SITE RECOVERABLE FROM THE SIX REGISTER FEATURES?')
    print('=' * 96)
    print('  A classifier is asked to predict school_id from the feature set the model')
    print('  actually sees. High accuracy means a leave-one-site-out fold is weaker')
    print('  evidence than it looks, because site can be inferred rather than withheld.')
    X = df[P.CAT + P.CONT].copy()
    X[P.CONT] = SimpleImputer(strategy='median').fit_transform(X[P.CONT])
    X[P.CAT] = SimpleImputer(strategy='most_frequent').fit_transform(X[P.CAT])
    X = pd.get_dummies(X, columns=P.CAT)
    X[P.CONT] = MinMaxScaler().fit_transform(X[P.CONT])
    y = df['school_id'].values
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X.values, y, cv=cv, scoring='accuracy', n_jobs=1)
    majority = float(pd.Series(y).value_counts(normalize=True).max())
    print('    5-fold accuracy predicting school_id: %.4f ± %.4f' % (acc.mean(), acc.std()))
    print('    majority-class baseline             : %.4f' % majority)
    print('    lift over majority                  : %.2fx' % (acc.mean() / majority))
    recoverable = bool(acc.mean() > majority + 0.15)
    print('    site is materially recoverable from the features: %s' % recoverable)

    payload = dict(
        n_sites=len(sites), sites=frame,
        site_names=names,
        base_rate_spread=dict(min=min(br), max=max(br)),
        sites_without_test_positives=zero_pos,
        categorical_confounding=conf,
        distance_km_by_site=dist,
        distance_kruskal=dict(H=round(float(kw.statistic), 2), p=float(kw.pvalue)),
        site_recoverability=dict(
            cv_accuracy=round(float(acc.mean()), 4), cv_std=round(float(acc.std()), 4),
            majority_baseline=round(majority, 4),
            lift=round(float(acc.mean() / majority), 2),
            materially_recoverable=recoverable))
    os.makedirs('results', exist_ok=True)
    json.dump(payload, open(OUT, 'w'), indent=2, default=str)
    print('\nwritten: %s' % OUT)


if __name__ == '__main__':
    main()
