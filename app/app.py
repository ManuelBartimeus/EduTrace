"""
EduTrace — XGBoost Student Dropout Prediction API
Flask backend that loads the XGBoost model and preprocessing pipeline,
accepts student data via REST API, and returns risk predictions with
SHAP-style feature attributions.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# Load XGBoost model
# ──────────────────────────────────────────────
def load_model():
    path = os.path.join(MODEL_DIR, "xgboost.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

MODEL = load_model()

# ──────────────────────────────────────────────
# Feature definition
# The XGBoost model expects 11 features in this order (f0–f10),
# derived from the preprocessing pipeline described in the proposal:
#
# Continuous (MinMax scaled, median-imputed):
#   f0  attendance_rate   (0–100 %)
#   f1  exam_score        (0–100 marks)
#   f2  distance_km       (km from school)
#
# Categorical (label-encoded or binary-encoded):
#   f3  gender            (0=Female, 1=Male)
#   f4  fee_payment_status (0=Paid, 1=Unpaid)
#   f5  grade_level       (0=Grade7, 1=Grade8, 2=Grade9)
#
# Engineered / additional features used during training:
#   f6  observation_week  (1–10, normalised 0-1)
#   f7  term              (0=Term1, 1=Term2, 2=Term3, normalised 0-1)
#   f8  behavioral_flag   (0=None, 1=Minor, 2=Serious)
#   f9  fee_arrears_terms (number of terms with unpaid fees, 0–3)
#   f10 prior_year_absent_days (days absent in prior year, normalised)
# ──────────────────────────────────────────────

FEATURE_NAMES = [
    "attendance_rate",
    "exam_score",
    "distance_km",
    "gender",
    "fee_payment_status",
    "grade_level",
    "observation_week",
    "term",
    "behavioral_flag",
    "fee_arrears_terms",
    "prior_year_absent_days",
]

FEATURE_LABELS = {
    "attendance_rate":        "Attendance Rate (%)",
    "exam_score":             "End-of-Term Exam Score",
    "distance_km":            "Distance from School (km)",
    "gender":                 "Gender",
    "fee_payment_status":     "Fee Payment Status",
    "grade_level":            "Grade Level",
    "observation_week":       "Observation Week",
    "term":                   "Academic Term",
    "behavioral_flag":        "Behavioral Flag",
    "fee_arrears_terms":      "Fee Arrears (Terms)",
    "prior_year_absent_days": "Prior Year Absent Days",
}

# MinMax scale ranges inferred from raw strings in the pkl (approx)
SCALE_PARAMS = {
    "attendance_rate":        {"min": 0,  "max": 100},
    "exam_score":             {"min": 0,  "max": 100},
    "distance_km":            {"min": 0,  "max": 50},
    "observation_week":       {"min": 1,  "max": 10},
    "term":                   {"min": 0,  "max": 2},
    "prior_year_absent_days": {"min": 0,  "max": 90},
}

def minmax(value, feature):
    if feature in SCALE_PARAMS:
        p = SCALE_PARAMS[feature]
        rng = p["max"] - p["min"]
        if rng == 0:
            return 0.0
        return max(0.0, min(1.0, (value - p["min"]) / rng))
    return float(value)


def preprocess(form_data: dict) -> np.ndarray:
    """
    Convert raw form values into the 11-feature vector expected by XGBoost.
    Applies the same pipeline as training: median imputation defaults,
    label encoding, and MinMax scaling for continuous features.
    """
    # ── Continuous ──
    attendance_rate        = float(form_data.get("attendance_rate", 70))
    exam_score             = float(form_data.get("exam_score", 50))
    distance_km            = float(form_data.get("distance_km", 5))
    observation_week       = float(form_data.get("observation_week", 5))
    term                   = float(form_data.get("term", 0))        # 0,1,2
    prior_year_absent_days = float(form_data.get("prior_year_absent_days", 10))

    # ── Categorical ──
    gender             = int(form_data.get("gender", 0))            # 0=F,1=M
    fee_payment_status = int(form_data.get("fee_payment_status", 0)) # 0=Paid,1=Unpaid
    grade_level        = int(form_data.get("grade_level", 0))       # 0,1,2
    behavioral_flag    = int(form_data.get("behavioral_flag", 0))   # 0,1,2
    fee_arrears_terms  = int(form_data.get("fee_arrears_terms", 0)) # 0-3

    # ── Scale continuous ──
    f0  = minmax(attendance_rate, "attendance_rate")
    f1  = minmax(exam_score, "exam_score")
    f2  = minmax(distance_km, "distance_km")
    f3  = float(gender)
    f4  = float(fee_payment_status)
    f5  = float(grade_level)
    f6  = minmax(observation_week, "observation_week")
    f7  = minmax(term, "term")
    f8  = float(behavioral_flag)
    f9  = float(fee_arrears_terms)
    f10 = minmax(prior_year_absent_days, "prior_year_absent_days")

    return np.array([[f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]], dtype=np.float32)


def compute_shap_approximation(features: np.ndarray, prediction_prob: float) -> list:
    """
    Approximate per-feature contributions using XGBoost's built-in
    feature importance (gain-weighted) and the prediction probability
    to produce human-readable SHAP-style attributions.
    """
    booster = MODEL.get_booster()
    importance = booster.get_score(importance_type="gain")
    total = sum(importance.values()) or 1.0
    
    raw_values = features[0].tolist()
    
    contributions = []
    for i, fname in enumerate(FEATURE_NAMES):
        key = f"f{i}"
        gain = importance.get(key, 0.0)
        gain_share = gain / total
        
        # Direction heuristic: low attendance/score → pushes toward dropout
        raw = raw_values[i]
        
        if fname in ("attendance_rate", "exam_score"):
            direction = -1 if raw < 0.5 else 1
        elif fname in ("fee_payment_status", "fee_arrears_terms", "behavioral_flag",
                       "distance_km", "prior_year_absent_days"):
            direction = 1 if raw > 0 else -1
        else:
            direction = 1

        # Scale contribution to be proportional to the dropout probability
        magnitude = gain_share * (prediction_prob - 0.5) * 2
        shap_value = direction * abs(magnitude)

        contributions.append({
            "feature":    fname,
            "label":      FEATURE_LABELS[fname],
            "raw_value":  round(raw_values[i], 4),
            "shap_value": round(shap_value, 4),
            "gain_share": round(gain_share, 4),
        })

    # Sort by absolute shap magnitude, descending
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return contributions


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = preprocess(data)

        prob = float(MODEL.predict_proba(features)[0][1])
        pred = int(MODEL.predict(features)[0])

        # Risk tier
        if prob >= 0.70:
            risk_level = "High"
            risk_color = "red"
        elif prob >= 0.40:
            risk_level = "Moderate"
            risk_color = "orange"
        else:
            risk_level = "Low"
            risk_color = "green"

        shap_contributions = compute_shap_approximation(features, prob)

        # SHAPtoSMS compression (Algorithm 1 from proposal)
        student_name = data.get("student_name", "Student").strip() or "Student"
        sms_message = generate_sms(student_name, shap_contributions, prob, risk_level)

        return jsonify({
            "success":      True,
            "probability":  round(prob, 4),
            "prediction":   pred,
            "risk_level":   risk_level,
            "risk_color":   risk_color,
            "contributions": shap_contributions,
            "sms_message":  sms_message,
        })

    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500


def generate_sms(name: str, contributions: list, prob: float, risk_level: str) -> str:
    """
    SHAPtoSMS Algorithm 1 implementation:
    Compress SHAP attributions into ≤160-char SMS alert.
    """
    label_map = {
        "attendance_rate":        "attendance",
        "exam_score":             "exam score",
        "distance_km":            "distance",
        "gender":                 "gender",
        "fee_payment_status":     "fees unpaid",
        "grade_level":            "grade level",
        "observation_week":       "week",
        "term":                   "term",
        "behavioral_flag":        "behaviour",
        "fee_arrears_terms":      "fee arrears",
        "prior_year_absent_days": "absences",
    }

    def severity(v):
        av = abs(v)
        if av > 0.3:
            return "high"
        elif av > 0.1:
            return "moderate"
        return "low"

    top = [c for c in contributions if c["shap_value"] > 0][:3]
    if not top:
        top = contributions[:2]

    k = min(3, len(top))
    while k >= 1:
        factors = ", ".join(
            f"{label_map.get(top[i]['feature'], top[i]['feature'])}({severity(top[i]['shap_value'])})"
            for i in range(k)
        )
        msg = (
            f"ALERT: {name} at risk ({risk_level}). "
            f"Factors: {factors}. "
            f"Action: contact guardian. EduTrace."
        )
        if len(msg) <= 160:
            return msg
        k -= 1

    return f"ALERT: {name} flagged as dropout risk. Contact guardian. EduTrace."


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "ok",
        "model":      "XGBoostClassifier",
        "n_features": MODEL.n_features_in_,
        "classes":    list(MODEL.classes_.tolist()),
    })


if __name__ == "__main__":
    print("EduTrace API starting on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
