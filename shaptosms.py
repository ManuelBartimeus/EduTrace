"""
SHAPtoSMS — implementation of Algorithm 1 exactly as specified in Methods M12.

Manuscript spec (M12, Algorithm 1):
  INPUT : SHAP attribution vector phi for student s; raw feature values;
          lexicon L; character limit C = 160
  OUTPUT: natural-language SMS alert msg, with len(msg) <= C
  1: Candidate set <- features with phi_i > 0; if empty, use all features
  2: Rank candidates by |phi_i| descending -> F = [f_1, ..., f_k]
  3: header <- 'ALERT:{student_id} risk. Factors:'
  4: msg <- header; included <- []
  5: for each f_i in F (rank order):
  6:     if f_i categorical: label <- L[f_i][raw_value]
  7:     else: severity by |phi_i| band; label <- L[f_i][severity]
  8:     fragment <- ' ' + label + ','
  9:     if len(msg)+len(fragment)+len(footer)+1 <= C: msg += fragment; included.append(f_i)
  10:    else: break
  11: msg <- strip_trailing_comma(msg) + '. Contact guardian.EduTrace.'
  12: return msg, included

Two fidelity metrics are computed here, kept deliberately separate because the
supplied codebase conflated them (see RPS_NOTE below).

RPS_NOTE
--------
The original pipeline's `rps_at_k` checked whether the top-k feature by |SHAP|
carried a SHAP sign agreeing with the record's true label. That is a
*directional-agreement* statistic, not a rank-preservation statistic: it never
compares two orderings. Methods M17 describes RPS as measuring "whether the
packing and lexicalisation stages faithfully preserve the model's attribution
ordering", which is a different quantity. Both are computed below under
distinct names so the manuscript can report what it actually measures.
"""

import numpy as np

C_LIMIT = 160
HEADER_TMPL = "ALERT:{sid} risk. Factors:"
FOOTER = ". Contact guardian.EduTrace."

# Lexicon L. Categorical features map raw value -> phrase.
# Continuous features map severity band -> phrase.
LEXICON = {
    "attendance_rate": {"high": "very low attendance",
                        "med":  "low attendance",
                        "low":  "attendance dipping"},
    "exam_score":      {"high": "very weak exam score",
                        "med":  "weak exam score",
                        "low":  "exam score slipping"},
    "distance_km":     {"high": "lives very far",
                        "med":  "lives far",
                        "low":  "long trip to school"},
    "att_delta":       {"high": "attendance falling fast",
                        "med":  "attendance falling",
                        "low":  "attendance drifting down"},
    "exam_delta":      {"high": "scores falling fast",
                        "med":  "scores falling",
                        "low":  "scores drifting down"},
    "prior_risk_flag": {"high": "flagged last year",
                        "med":  "flagged last year",
                        "low":  "flagged last year"},
    "terms_observed":  {"high": "short school history",
                        "med":  "short school history",
                        "low":  "short school history"},
    "fee_payment_status": {"Unpaid": "fees unpaid",
                           "Partially Paid": "fees part-paid",
                           "Fully Paid": "fees paid"},
    "grade_level": {"Grade 7": "in Grade 7",
                    "Grade 8": "in Grade 8",
                    "Grade 9": "in Grade 9"},
    "gender": {"Male": "male student", "Female": "female student"},
}

CATEGORICAL_BASES = {"fee_payment_status", "grade_level", "gender"}


def _severity(abs_phi, bands):
    """Severity band for a continuous feature, from |phi| against global terciles."""
    lo, hi = bands
    if abs_phi >= hi:
        return "high"
    if abs_phi >= lo:
        return "med"
    return "low"


def _base_and_value(feature_name, raw_row):
    """Split a (possibly one-hot) column name into its base feature and raw value.

    Algorithm 1 step 6 lexicalises a categorical feature as L[f_i][raw_value] —
    the STUDENT'S OWN value, not the identity of whichever one-hot column
    carried the attribution. So `gender_Male` on a female student's record
    renders "female student". Callers additionally de-duplicate by base feature
    so one register variable is named at most once per alert.
    """
    for base in CATEGORICAL_BASES:
        if feature_name == base or feature_name.startswith(base + "_"):
            return base, raw_row.get(base)
    return feature_name, raw_row.get(feature_name)


def label_for(feature_name, raw_row, abs_phi, bands):
    """Lexicalisation stage: map a feature to its natural-language phrase."""
    base, value = _base_and_value(feature_name, raw_row)
    entry = LEXICON.get(base)
    if entry is None:
        return None
    if base in CATEGORICAL_BASES:
        return entry.get(str(value))
    return entry.get(_severity(abs_phi, bands))


def build_alert(sid, phi, feature_names, raw_row, bands, order="rank"):
    """
    Generate one SMS alert.

    order = "rank"      -> Algorithm 1 as specified (strict descending |phi|)
    order = "canonical" -> ablation baseline (fixed canonical order, same
                           candidate set, same lexicon, same budget)

    Returns (msg, included_feature_names, candidate_order).
    """
    phi = np.asarray(phi, dtype=float)

    # Step 1 — candidate set: features with phi_i > 0; if empty, use all.
    cand = [i for i in range(len(phi)) if phi[i] > 0]
    if not cand:
        cand = list(range(len(phi)))

    # Step 2 — rank candidates.
    if order == "rank":
        cand_sorted = sorted(cand, key=lambda i: -abs(phi[i]))
    else:
        canon = ["fee_payment_status", "grade_level", "gender",
                 "attendance_rate", "exam_score", "distance_km",
                 "att_delta", "exam_delta", "prior_risk_flag", "terms_observed"]
        def canon_key(i):
            base, _ = _base_and_value(feature_names[i], raw_row)
            return canon.index(base) if base in canon else len(canon)
        cand_sorted = sorted(cand, key=canon_key)

    # De-duplicate by base register variable, keeping each variable's
    # highest-ranked one-hot column. One variable is named at most once.
    seen, deduped = set(), []
    for i in cand_sorted:
        base, _ = _base_and_value(feature_names[i], raw_row)
        if base in seen:
            continue
        seen.add(base)
        deduped.append(i)

    # Steps 3-4.
    msg = HEADER_TMPL.format(sid=sid)
    included = []

    # Steps 5-10 — greedy packing under the character budget.
    for i in deduped:
        lab = label_for(feature_names[i], raw_row, abs(phi[i]), bands)
        if lab is None:
            continue
        frag = " " + lab + ","
        if len(msg) + len(frag) + len(FOOTER) + 1 <= C_LIMIT:
            msg += frag
            included.append(feature_names[i])
        else:
            break

    # Step 11.
    msg = msg.rstrip(",") + FOOTER
    return msg, included, [feature_names[i] for i in deduped]


# ----------------------------------------------------------------------------
# Fidelity metrics
# ----------------------------------------------------------------------------

def rank_preservation_at_k(included, gold_order, k):
    """
    TRUE rank preservation (what Methods M17 describes).

    Does the packed alert's first k named factors match the model's own
    top-k SHAP ranking, in order? Compared against the same gold ranking
    derived from the same model whose attributions are packed.
    """
    if len(gold_order) < k or len(included) < k:
        return 0.0
    return float(all(included[j] == gold_order[j] for j in range(k)))


def directional_agreement_at_k(y_true, phi_row, k):
    """
    The statistic the ORIGINAL pipeline computed and labelled RPS.

    Does any of the top-k features by |phi| carry a SHAP sign that agrees with
    the record's true label (positive attribution for a dropout, negative for a
    non-dropout)? This measures attribution/outcome direction agreement, not
    ordering fidelity. Reported under its own name to avoid the M17 conflation.
    """
    phi_row = np.asarray(phi_row, dtype=float)
    top = np.argsort(np.abs(phi_row))[::-1][:k]
    for fi in top:
        if y_true == 1 and phi_row[fi] > 0:
            return 1.0
        if y_true == 0 and phi_row[fi] < 0:
            return 1.0
    return 0.0


def directional_agreement_at_k_fixed(y_true, phi_row, fixed_order_idx, k):
    """Naive fixed-order counterpart of directional_agreement_at_k."""
    phi_row = np.asarray(phi_row, dtype=float)
    for fi in fixed_order_idx[:k]:
        if y_true == 1 and phi_row[fi] > 0:
            return 1.0
        if y_true == 0 and phi_row[fi] < 0:
            return 1.0
    return 0.0


def severity_bands(shap_matrix, continuous_idx):
    """Global terciles of |phi| over continuous features, for severity banding."""
    vals = np.abs(shap_matrix[:, continuous_idx]).ravel()
    vals = vals[vals > 0]
    if vals.size == 0:
        return (0.0, 0.0)
    return (float(np.percentile(vals, 33)), float(np.percentile(vals, 66)))
