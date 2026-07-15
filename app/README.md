# 🎓 EduTrace — Student Dropout Risk Predictor

An explainable AI early warning system that predicts student dropout risk in rural Ghanaian schools using **XGBoost** and **SHAP feature attributions**, with a built-in **SHAPtoSMS** alert generator.

---

## 📁 Project Structure

```
Edutrace/
├── app.py                  # Flask backend API
├── index.html              # Frontend UI
├── requirements.txt        # Python dependencies
├── xgboost.pkl             # Trained XGBoost model
├── cat_imputer.pkl         # Categorical feature imputer
├── cont_imputer.pkl        # Continuous feature imputer
├── label_encoders.pkl      # Label encoders
└── minmax_scaler.pkl       # MinMax scaler
```

---

## ⚙️ Requirements

- **Python 3.10+**
- **pip** (comes with Python)
- A modern web browser (Chrome, Firefox, Edge)

---

## 🚀 Setup & Installation

### Step 1 — Install Python
If you don't have Python installed, download it from:
👉 https://www.python.org/downloads/

> During installation, make sure to check **"Add Python to PATH"**

---

### Step 2 — Open a Terminal in the Project Folder

**Windows:**
1. Open File Explorer and navigate to the `Edutrace` folder
2. Click the address bar, type `cmd`, and press Enter

**Or using PowerShell:**
```powershell
cd path\to\Edutrace
```

---

### Step 3 — Install Dependencies

Run this command in the terminal:

```bash
pip install -r requirements.txt
```

This installs:
- `flask` — web server
- `flask-cors` — cross-origin support
- `xgboost` — the ML model library
- `scikit-learn` — preprocessing utilities
- `numpy` — numerical computation

> ⏳ This may take 1–2 minutes on first install.

---

### Step 4 — Start the Application

```bash
python app.py
```

You should see output like:
```
EduTrace API starting on http://localhost:5000
 * Running on http://127.0.0.1:5000
 * Debugger is active!
```

---

### Step 5 — Open the App

Open your browser and go to:

```
http://localhost:5000
```

The EduTrace interface will load automatically. ✅

---

## 🖥️ How to Use

1. **Enter student details** in the form on the left:
   - Attendance rate, exam score, grade level, term, week
   - Fee payment status and arrears
   - Distance from school, gender, behavioral flag, prior absences

2. **Click "Run Risk Assessment"**

3. **View the results** on the right:
   - 🎯 **Risk gauge** — dropout probability (0–100%)
   - 📊 **SHAP feature bars** — which factors drove the prediction
   - 📱 **SMS preview** — the alert a teacher would receive

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend UI |
| `GET` | `/api/health` | Check model status |
| `POST` | `/api/predict` | Run a prediction |

### Example API call (`/api/predict`):

```json
POST http://localhost:5000/api/predict
Content-Type: application/json

{
  "student_name": "Kwame Asante",
  "attendance_rate": 45,
  "exam_score": 38,
  "distance_km": 12,
  "gender": 1,
  "fee_payment_status": 1,
  "grade_level": 0,
  "observation_week": 6,
  "term": 1,
  "behavioral_flag": 1,
  "fee_arrears_terms": 2,
  "prior_year_absent_days": 30
}
```

### Response:
```json
{
  "success": true,
  "probability": 0.748,
  "risk_level": "High",
  "prediction": 1,
  "contributions": [...],
  "sms_message": "ALERT: Kwame Asante at risk (High). Factors: attendance(high), fees unpaid(moderate). Action: contact guardian. EduTrace."
}
```

---

## 🛑 Stopping the Server

Press `Ctrl + C` in the terminal to stop the Flask server.

---

## 🔁 Restarting After Closing

Every time you want to use the app again, just run:

```bash
python app.py
```

Then visit `http://localhost:5000` in your browser.

---

## 🧠 Model Features

The XGBoost model uses these 11 features:

| # | Feature | Type | Description |
|---|---------|------|-------------|
| f0 | `attendance_rate` | Continuous | % of days attended (0–100) |
| f1 | `exam_score` | Continuous | End-of-term exam marks (0–100) |
| f2 | `distance_km` | Continuous | Distance from school in km |
| f3 | `gender` | Categorical | 0 = Female, 1 = Male |
| f4 | `fee_payment_status` | Categorical | 0 = Paid, 1 = Unpaid |
| f5 | `grade_level` | Categorical | 0 = Grade 7, 1 = Grade 8, 2 = Grade 9 |
| f6 | `observation_week` | Continuous | Week of term (1–10) |
| f7 | `term` | Categorical | 0 = Term 1, 1 = Term 2, 2 = Term 3 |
| f8 | `behavioral_flag` | Categorical | 0 = None, 1 = Minor, 2 = Serious |
| f9 | `fee_arrears_terms` | Ordinal | Number of terms with unpaid fees (0–3) |
| f10 | `prior_year_absent_days` | Continuous | Absence days in prior year (0–90) |

---

## 📱 SHAPtoSMS Protocol

Based on **Algorithm 1** from the research proposal. Compresses SHAP attributions into a ≤160-character SMS alert:

```
ALERT: [Name] at risk ([Level]). Factors: [F1](severity), [F2](severity), [F3](severity). Action: contact guardian. EduTrace.
```

Severity levels: `high` (|φ| > 0.3) · `moderate` (0.1–0.3) · `low` (< 0.1)

---

## 👥 Research Team

| Name | Student ID | Role |
|------|-----------|------|
| Bartimeus Manuel Nii Osabu | 3382422 | Lead Researcher |
| Karikari David Kweku | 3394522 | Co-Researcher |
| Manu Richeal Pokuah | 3398922 | Co-Researcher |
| Antwi Jeffrey Twum | 3372922 | Co-Researcher |
| Oti Wisdom Boakye | 3419922 | Co-Researcher |

**Supervisor:** Dr. Eric Opoku Osei  
**Institution:** KNUST, Ghana  
**Programme:** KNUST Supervised Research Programme 2025–2026

---

## 🔗 Repository

GitHub: https://github.com/AJ-280/SHAPtoSMS-EduTrace
