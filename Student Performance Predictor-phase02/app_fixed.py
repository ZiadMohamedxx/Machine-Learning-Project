from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, feature columns, and scaler
model    = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
scaler   = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Must exactly match the numeric columns the scaler was fit on during training
NUMERIC_COLS = [
    "study_hours", "class_attendance", "sleep_hours",
    "facility_rating", "exam_difficulty", "study_efficiency"
    # BUG FIX: original app.py was missing "study_efficiency" here,
    # so that column was never scaled — causing a silent mismatch.
]

FACILITY_MAP   = {"low": 1, "medium": 2, "high": 3}
DIFFICULTY_MAP = {"easy": 1, "moderate": 2, "hard": 3}

# Valid values for categorical fields (lowercase — we normalise before checking)
VALID_FACILITY   = {"low", "medium", "high"}
VALID_DIFFICULTY = {"easy", "moderate", "hard"}
VALID_INTERNET   = {"yes", "no"}
VALID_SLEEP_Q    = {"poor", "average", "good"}
VALID_STUDY_M    = {"coaching", "online videos", "mixed", "self-study", "group study"}

# ─────────────────────────────────────────────────────────────────
# IQR caps computed from the training dataset (Exam_Score_Prediction.csv).
# The notebook's Data Cleaning step Winsorises these three columns before
# fitting the scaler. We MUST apply the identical caps here, otherwise
# out-of-range inputs (e.g. sleep_hours=24) produce z-scores far outside
# the model's training distribution, leading to nonsensical predictions.
# ─────────────────────────────────────────────────────────────────
IQR_CAPS = {
    # column: (lower_cap, upper_cap)  — derived from Q1/Q3 ± 1.5×IQR
    "study_hours":      (0.0,  12.0),   # Q1=2, Q3=6, IQR=4  → cap=[−4,12] → clamp≥0
    "class_attendance": (10.25, 100.0), # Q1=55.1, Q3=85.0, IQR=29.9 → clamp≤100
    "sleep_hours":      (1.0,  13.0),   # Q1=5.5, Q3=8.5, IQR=3.0
}


# =========================
# Input Validation
# =========================
def validate_input(data):
    errors = []

    # ── Numeric fields ──
    try:
        study_hours = float(data.get("study_hours", ""))
        if not (0 <= study_hours <= 24):
            errors.append("study_hours must be between 0 and 24.")
    except (TypeError, ValueError):
        errors.append("study_hours is required and must be a number.")

    try:
        attendance = float(data.get("class_attendance", ""))
        if not (0 <= attendance <= 100):
            errors.append("class_attendance must be between 0 and 100.")
    except (TypeError, ValueError):
        errors.append("class_attendance is required and must be a number.")

    try:
        sleep_hours = float(data.get("sleep_hours", ""))
        if not (0 <= sleep_hours <= 24):
            errors.append("sleep_hours must be between 0 and 24.")
    except (TypeError, ValueError):
        errors.append("sleep_hours is required and must be a number.")

    # ── Categorical fields ──
    facility = str(data.get("facility_rating", "")).strip().lower()
    if facility not in VALID_FACILITY:
        errors.append(f"facility_rating must be one of: {', '.join(sorted(VALID_FACILITY))}.")

    difficulty = str(data.get("exam_difficulty", "")).strip().lower()
    if difficulty not in VALID_DIFFICULTY:
        errors.append(f"exam_difficulty must be one of: {', '.join(sorted(VALID_DIFFICULTY))}.")

    internet = str(data.get("internet_access", "")).strip().lower()
    if internet not in VALID_INTERNET:
        errors.append(f"internet_access must be one of: {', '.join(sorted(VALID_INTERNET))}.")

    sleep_q = str(data.get("sleep_quality", "")).strip().lower()
    if sleep_q not in VALID_SLEEP_Q:
        errors.append(f"sleep_quality must be one of: {', '.join(sorted(VALID_SLEEP_Q))}.")

    study_m = str(data.get("study_method", "")).strip().lower()
    if study_m not in VALID_STUDY_M:
        errors.append(f"study_method must be one of: {', '.join(sorted(VALID_STUDY_M))}.")

    return errors


# =========================
# Explanation logic
# =========================
def explain_prediction(data, prediction):
    reasons = []

    study_hours      = float(data.get("study_hours", 0))
    class_attendance = float(data.get("class_attendance", 0))
    sleep_hours      = float(data.get("sleep_hours", 0))
    sleep_q          = str(data.get("sleep_quality", "")).lower()
    study_m          = str(data.get("study_method", "")).lower()
    internet         = str(data.get("internet_access", "")).lower()
    difficulty       = str(data.get("exam_difficulty", "")).lower()

    if study_hours < 2:
        reasons.append("Low study hours (less than 2 hours/day)")
    if class_attendance < 60:
        reasons.append("Low class attendance (below 60%)")
    if sleep_hours < 5:
        reasons.append("Insufficient sleep (less than 5 hours)")
    if sleep_q == "poor":
        reasons.append("Poor sleep quality")
    if study_m == "self-study":
        reasons.append("Studying alone without structured guidance")
    if internet == "no":
        reasons.append("No internet access for study resources")
    if difficulty == "hard":
        reasons.append("Exam difficulty was high")

    # BUG FIX: "Strong academic habits" message was shown for ANY prediction,
    # including Fail/Pass, because it only checked prediction == "Distinction"
    # but that check was always at the end regardless of what was predicted.
    # It should ONLY appear when the prediction is actually Distinction.
    if prediction == "Distinction":
        reasons.append("Strong academic habits and effective study strategy")

    return reasons


# =========================
# HTML Frontend
# =========================
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Student Performance Predictor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #f4f6fb;
  padding: 40px 20px;
  min-height: 100vh;
}

.container {
  max-width: 820px;
  margin: auto;
  background: white;
  padding: 36px;
  border-radius: 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}

h1 {
  text-align: center;
  font-size: 1.7rem;
  margin-bottom: 8px;
  color: #1e293b;
}

.subtitle {
  text-align: center;
  color: #64748b;
  margin-bottom: 28px;
  font-size: 0.9rem;
}

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0 24px;
}

@media(max-width: 580px){ .grid { grid-template-columns: 1fr; } }

.field { margin-bottom: 18px; }

label {
  display: block;
  font-size: 0.85rem;
  font-weight: 600;
  color: #374151;
  margin-bottom: 5px;
}

label .hint {
  font-weight: 400;
  color: #9ca3af;
  font-size: 0.78rem;
  margin-left: 4px;
}

input, select {
  width: 100%;
  padding: 9px 12px;
  border-radius: 8px;
  border: 1.5px solid #d1d5db;
  font-size: 0.95rem;
  transition: border-color .2s;
  background: #f9fafb;
}

input:focus, select:focus {
  outline: none;
  border-color: #2563eb;
  background: white;
}

input.invalid, select.invalid {
  border-color: #ef4444;
  background: #fff5f5;
}

.field-error {
  color: #dc2626;
  font-size: 0.78rem;
  margin-top: 4px;
  display: none;
}

.field-error.visible { display: block; }

button {
  margin-top: 10px;
  padding: 13px;
  width: 100%;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 10px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background .2s;
}

button:hover { background: #1d4ed8; }
button:disabled { background: #93c5fd; cursor: not-allowed; }

#result {
  margin-top: 24px;
  padding: 20px 24px;
  border-radius: 12px;
  display: none;
  font-size: 0.97rem;
}

.result-header {
  font-size: 1.15rem;
  font-weight: 700;
  margin-bottom: 6px;
}

.confidence-bar-wrap { margin: 10px 0; }

.confidence-bar-bg {
  height: 10px;
  border-radius: 5px;
  background: rgba(0,0,0,0.1);
  overflow: hidden;
  margin-top: 4px;
}

.confidence-bar-fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.6s ease;
}

.reasons-list {
  margin-top: 12px;
  padding-left: 18px;
}

.reasons-list li {
  margin-bottom: 4px;
  font-size: 0.9rem;
}

.fail-box  { background:#fee2e2; border:1.5px solid #ef4444; color:#991b1b; }
.pass-box  { background:#fef3c7; border:1.5px solid #f59e0b; color:#92400e; }
.dist-box  { background:#dcfce7; border:1.5px solid #22c55e; color:#166534; }
.err-box   { background:#f1f5f9; border:1.5px solid #94a3b8; color:#1e293b; }

.fail-fill { background:#ef4444; }
.pass-fill { background:#f59e0b; }
.dist-fill { background:#22c55e; }

.probabilities {
  display: flex;
  gap: 10px;
  margin-top: 12px;
  flex-wrap: wrap;
}

.prob-chip {
  background: rgba(0,0,0,0.07);
  border-radius: 6px;
  padding: 4px 10px;
  font-size: 0.82rem;
  font-weight: 600;
}

.spinner {
  display: none;
  text-align: center;
  padding: 12px;
  color: #6b7280;
  font-size: 0.9rem;
}

.warning-note {
  margin-top: 10px;
  padding: 8px 12px;
  background: #fff7ed;
  border: 1px solid #fed7aa;
  border-radius: 6px;
  font-size: 0.82rem;
  color: #92400e;
}
</style>
</head>
<body>

<div class="container">
  <h1>🎓 Student Performance Predictor</h1>
  <p class="subtitle">Enter student information to predict academic performance.</p>

  <form id="form" novalidate>
    <div class="grid">

      <div class="field">
        <label>Study Hours <span class="hint">(0–24 hrs/day)</span></label>
        <input id="study_hours" name="study_hours" type="number" step="0.1" min="0" max="24"
               placeholder="e.g. 3.5" required>
        <div class="field-error" id="err_study_hours">Please enter a value between 0 and 24.</div>
      </div>

      <div class="field">
        <label>Class Attendance <span class="hint">(%)</span></label>
        <input id="class_attendance" name="class_attendance" type="number" step="0.1" min="0" max="100"
               placeholder="e.g. 75" required>
        <div class="field-error" id="err_class_attendance">Please enter a value between 0 and 100.</div>
      </div>

      <div class="field">
        <label>Sleep Hours <span class="hint">(0–24 hrs/day)</span></label>
        <input id="sleep_hours" name="sleep_hours" type="number" step="0.1" min="0" max="24"
               placeholder="e.g. 7" required>
        <div class="field-error" id="err_sleep_hours">Please enter a value between 0 and 24.</div>
      </div>

      <div class="field">
        <label>Sleep Quality</label>
        <select id="sleep_quality" name="sleep_quality" required>
          <option value="">-- Select --</option>
          <option value="poor">Poor</option>
          <option value="average">Average</option>
          <option value="good">Good</option>
        </select>
        <div class="field-error" id="err_sleep_quality">Please select sleep quality.</div>
      </div>

      <div class="field">
        <label>Facility Rating</label>
        <select id="facility_rating" name="facility_rating" required>
          <option value="">-- Select --</option>
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
        </select>
        <div class="field-error" id="err_facility_rating">Please select facility rating.</div>
      </div>

      <div class="field">
        <label>Exam Difficulty</label>
        <select id="exam_difficulty" name="exam_difficulty" required>
          <option value="">-- Select --</option>
          <option value="easy">Easy</option>
          <option value="moderate">Moderate</option>
          <option value="hard">Hard</option>
        </select>
        <div class="field-error" id="err_exam_difficulty">Please select exam difficulty.</div>
      </div>

      <div class="field">
        <label>Internet Access</label>
        <select id="internet_access" name="internet_access" required>
          <option value="">-- Select --</option>
          <option value="yes">Yes</option>
          <option value="no">No</option>
        </select>
        <div class="field-error" id="err_internet_access">Please select internet access.</div>
      </div>

      <div class="field">
        <label>Study Method</label>
        <select id="study_method" name="study_method" required>
          <option value="">-- Select --</option>
          <option value="coaching">Coaching</option>
          <option value="online videos">Online Videos</option>
          <option value="mixed">Mixed</option>
          <option value="self-study">Self Study</option>
          <option value="group study">Group Study</option>
        </select>
        <div class="field-error" id="err_study_method">Please select a study method.</div>
      </div>

    </div><!-- /.grid -->

    <button type="submit" id="submitBtn">Predict Performance</button>
  </form>

  <div class="spinner" id="spinner">⏳ Running prediction...</div>

  <div id="result"></div>
</div>

<script>
// ── Client-side validation ──
const NUMERIC_FIELDS = [
  { id: "study_hours",      min: 0, max: 24  },
  { id: "class_attendance", min: 0, max: 100 },
  { id: "sleep_hours",      min: 0, max: 24  }
];

const SELECT_FIELDS = [
  "sleep_quality", "facility_rating", "exam_difficulty",
  "internet_access", "study_method"
];

function clearErrors() {
  document.querySelectorAll(".field-error").forEach(el => el.classList.remove("visible"));
  document.querySelectorAll("input, select").forEach(el => el.classList.remove("invalid"));
}

function showError(fieldId, msg) {
  const el = document.getElementById("err_" + fieldId);
  const input = document.getElementById(fieldId);
  if (el) { el.textContent = msg; el.classList.add("visible"); }
  if (input) input.classList.add("invalid");
}

function validateForm(data) {
  let valid = true;

  for (const { id, min, max } of NUMERIC_FIELDS) {
    const raw = data[id];
    const val = parseFloat(raw);
    if (raw === "" || raw === null || raw === undefined || isNaN(val)) {
      showError(id, "This field is required.");
      valid = false;
    } else if (val < min || val > max) {
      showError(id, `Value must be between ${min} and ${max}.`);
      valid = false;
    }
  }

  for (const id of SELECT_FIELDS) {
    if (!data[id]) {
      showError(id, "Please make a selection.");
      valid = false;
    }
  }

  return valid;
}

// ── Remove invalid styling on change ──
document.querySelectorAll("input, select").forEach(el => {
  el.addEventListener("input", () => {
    el.classList.remove("invalid");
    const errEl = document.getElementById("err_" + el.id);
    if (errEl) errEl.classList.remove("visible");
  });
});

// ── Form submit ──
document.getElementById("form").addEventListener("submit", async function(e) {
  e.preventDefault();
  clearErrors();

  const formData = new FormData(e.target);
  const data = {};
  formData.forEach((v, k) => { data[k] = v; });

  // Convert numeric
  ["study_hours", "class_attendance", "sleep_hours"].forEach(k => {
    data[k] = data[k] !== "" ? Number(data[k]) : "";
  });

  if (!validateForm(data)) return;

  const btn = document.getElementById("submitBtn");
  const spinner = document.getElementById("spinner");
  btn.disabled = true;
  spinner.style.display = "block";
  document.getElementById("result").style.display = "none";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    const result = await response.json();
    renderResult(result);

  } catch (err) {
    renderError("Network error — could not reach the server.");
  } finally {
    btn.disabled = false;
    spinner.style.display = "none";
  }
});

function renderResult(result) {
  const box = document.getElementById("result");
  box.style.display = "block";
  box.className = "";

  if (result.error) {
    renderError(result.error);
    return;
  }

  const pred = result.prediction;
  const confPct = result.confidence != null ? (result.confidence * 100).toFixed(1) : null;

  let boxClass = "err-box", fillClass = "";
  if (pred === "Fail")             { boxClass = "fail-box"; fillClass = "fail-fill"; }
  else if (pred === "Pass")        { boxClass = "pass-box"; fillClass = "pass-fill"; }
  else if (pred === "Distinction") { boxClass = "dist-box"; fillClass = "dist-fill"; }

  box.classList.add(boxClass);

  const emoji = pred === "Fail" ? "❌" : pred === "Pass" ? "⚠️" : "✅";

  let html = `<div class="result-header">${emoji} Prediction: ${pred}</div>`;

  if (confPct !== null) {
    html += `
      <div class="confidence-bar-wrap">
        <span style="font-size:0.88rem">Confidence: <strong>${confPct}%</strong></span>
        <div class="confidence-bar-bg">
          <div class="confidence-bar-fill ${fillClass}" style="width:${confPct}%"></div>
        </div>
      </div>`;
  }

  if (result.probabilities) {
    html += `<div class="probabilities">`;
    for (const [cls, prob] of Object.entries(result.probabilities)) {
      html += `<span class="prob-chip">${cls}: ${(prob * 100).toFixed(1)}%</span>`;
    }
    html += `</div>`;
  }

  if (result.reasons && result.reasons.length > 0) {
    html += `<ul class="reasons-list">`;
    result.reasons.forEach(r => { html += `<li>${r}</li>`; });
    html += `</ul>`;
  }

  if (result.clamped && result.clamped.length > 0) {
    html += `<div class="warning-note">⚠️ Note: the following inputs were outside the model's training range and were automatically clamped: ${result.clamped.join(", ")}.</div>`;
  }

  box.innerHTML = html;
}

function renderError(msg) {
  const box = document.getElementById("result");
  box.style.display = "block";
  box.className = "err-box";
  box.innerHTML = `<div class="result-header">⚠️ Error</div><p style="margin-top:8px">${msg}</p>`;
}
</script>

</body>
</html>
"""


# =========================
# Routes
# =========================

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON body."}), 400

    # ── Server-side validation ──
    validation_errors = validate_input(data)
    if validation_errors:
        return jsonify({"error": " | ".join(validation_errors)}), 422

    try:
        # Normalise string casing (must match training preprocessing)
        for key in ["facility_rating", "exam_difficulty", "internet_access",
                    "sleep_quality", "study_method"]:
            if key in data:
                data[key] = str(data[key]).strip().lower()

        input_df = pd.DataFrame([data])

        # ── Ordinal encode ordered categoricals ──
        input_df["facility_rating"] = input_df["facility_rating"].map(FACILITY_MAP)
        input_df["exam_difficulty"] = input_df["exam_difficulty"].map(DIFFICULTY_MAP)

        # ── Numeric cast ──
        for col in ["study_hours", "class_attendance", "sleep_hours",
                    "facility_rating", "exam_difficulty"]:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(float)

        # ── BUG FIX: Apply the same IQR-based Winsorisation that was applied
        # during training data cleaning. Without this step, extreme values
        # (e.g. sleep_hours=24) produce z-scores ~10 SD above the training
        # distribution, completely breaking the scaler and producing nonsense
        # predictions (e.g. "sleeping 24h → Pass").
        # The caps are derived from the training data's Q1/Q3 ± 1.5×IQR.
        # ──
        clamped_fields = []
        for col, (lo, hi) in IQR_CAPS.items():
            original = float(input_df[col].iloc[0])
            clamped  = float(np.clip(original, lo, hi))
            if clamped != original:
                clamped_fields.append(f"{col} ({original} → {clamped})")
            input_df[col] = clamped

        # ── Feature engineering (must match training) ──
        input_df["study_efficiency"] = (
            input_df["study_hours"] * input_df["class_attendance"] / 100
        )

        # ── One-hot encode nominal categoricals ──
        nominal_cols = ["internet_access", "sleep_quality", "study_method"]
        input_df = pd.get_dummies(input_df, columns=nominal_cols)

        # ── Align to training feature set ──
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[features]

        # ── Scale numeric columns using saved scaler ──
        # BUG FIX: NUMERIC_COLS now includes "study_efficiency" — it was
        # missing in the original app.py, so study_efficiency was passed
        # unscaled to the model, which had been trained on scaled values.
        input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])

        x = input_df.values

        # ── Predict ──
        pred = model.predict(x)[0]

        best_prob = None
        probs = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x)[0]
            probs = {cls: round(float(p[i]), 4)
                     for i, cls in enumerate(model.classes_)}
            best_prob = probs.get(pred)

        reasons = explain_prediction(data, pred)

        return jsonify({
            "prediction":    pred,
            "confidence":    best_prob,
            "probabilities": probs,
            "reasons":       reasons,
            "clamped":       clamped_fields,   # tells the UI which inputs were clamped
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(debug=True)
