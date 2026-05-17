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

NUMERIC_COLS = [
    "study_hours", "class_attendance", "sleep_hours",
    "facility_rating", "exam_difficulty", "study_efficiency"
]

FACILITY_MAP   = {"low": 1, "medium": 2, "high": 3}
DIFFICULTY_MAP = {"easy": 1, "moderate": 2, "hard": 3}

# =========================
# Explanation logic
# =========================
def explain_prediction(data, prediction):
    reasons = []

    study_hours     = float(data.get("study_hours", 0))
    class_attendance = float(data.get("class_attendance", 0))
    sleep_hours     = float(data.get("sleep_hours", 0))

    if study_hours < 2:
        reasons.append("Low study hours (less than 2 hours/day)")
    if class_attendance < 60:
        reasons.append("Low class attendance (below 60%)")
    if sleep_hours < 5:
        reasons.append("Insufficient sleep (less than 5 hours)")
    if data.get("sleep_quality") == "poor":
        reasons.append("Poor sleep quality")
    if data.get("study_method") == "self-study":
        reasons.append("Studying alone without structured guidance")
    if data.get("internet_access") == "no":
        reasons.append("No internet access for study resources")
    if data.get("exam_difficulty") == "hard":
        reasons.append("Exam difficulty was high")
    if prediction == "Distinction":
        reasons.append("Strong academic habits and effective study strategy")

    return reasons


# =========================
# HTML Frontend
# =========================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Student Performance Predictor</title>
<style>
body{
font-family: Arial;
background:#f4f6fb;
padding:40px;
}

.container{
max-width:800px;
margin:auto;
background:white;
padding:30px;
border-radius:10px;
box-shadow:0 10px 25px rgba(0,0,0,0.1);
}

h1{
text-align:center;
margin-bottom:25px;
}

label{
display:block;
margin-top:15px;
font-weight:bold;
}

input,select{
width:100%;
padding:10px;
margin-top:5px;
border-radius:6px;
border:1px solid #ccc;
box-sizing:border-box;
}

button{
margin-top:20px;
padding:12px;
width:100%;
background:#2563eb;
color:white;
border:none;
border-radius:8px;
font-size:16px;
cursor:pointer;
}

button:hover{
background:#1d4ed8;
}

#result{
margin-top:25px;
padding:18px;
border-radius:10px;
display:none;
white-space:pre-line;
font-size:15px;
font-weight:bold;
}

.fail-box{
background:#fee2e2;
border:1px solid #ef4444;
color:#991b1b;
}

.pass-box{
background:#fef3c7;
border:1px solid #f59e0b;
color:#92400e;
}

.dist-box{
background:#dcfce7;
border:1px solid #22c55e;
color:#166534;
}
</style>
</head>
<body>

<div class="container">
<h1>Student Performance Predictor</h1>

<form id="form">
<label>Study Hours</label>
<input name="study_hours" type="number" step="0.1" required>

<label>Class Attendance (%)</label>
<input name="class_attendance" type="number" step="0.1" required>

<label>Sleep Hours</label>
<input name="sleep_hours" type="number" step="0.1" required>

<label>Facility Rating</label>
<select name="facility_rating">
<option value="low">Low</option>
<option value="medium">Medium</option>
<option value="high">High</option>
</select>

<label>Exam Difficulty</label>
<select name="exam_difficulty">
<option value="easy">Easy</option>
<option value="moderate">Moderate</option>
<option value="hard">Hard</option>
</select>

<label>Internet Access</label>
<select name="internet_access">
<option value="yes">Yes</option>
<option value="no">No</option>
</select>

<label>Sleep Quality</label>
<select name="sleep_quality">
<option value="poor">Poor</option>
<option value="average">Average</option>
<option value="good">Good</option>
</select>

<label>Study Method</label>
<select name="study_method">
<option value="coaching">Coaching</option>
<option value="online videos">Online Videos</option>
<option value="mixed">Mixed</option>
<option value="self-study">Self Study</option>
<option value="group study">Group Study</option>
</select>

<button type="submit">Predict</button>
</form>

<div id="result"></div>
</div>

<script>
document.getElementById("form").addEventListener("submit", async function(e){
    e.preventDefault();

    const formData = new FormData(e.target);
    let data = {};
    formData.forEach((v,k) => { data[k] = v; });

    // Convert numeric fields
    ["study_hours","class_attendance","sleep_hours"].forEach(k => {
        data[k] = Number(data[k]);
    });

    let response = await fetch("/predict",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify(data)
    });

    let result = await response.json();

    let box = document.getElementById("result");
    box.style.display = "block";
    box.className = "";

    if(result.prediction === "Fail"){
        box.classList.add("fail-box");
    } else if(result.prediction === "Pass"){
        box.classList.add("pass-box");
    } else if(result.prediction === "Distinction"){
        box.classList.add("dist-box");
    }

    let output = "Prediction: " + result.prediction + "\\n";

    if(result.confidence !== null && result.confidence !== undefined){
        output += "Confidence: " + (result.confidence * 100).toFixed(1) + "%\\n";
    }

    if(result.reasons && result.reasons.length > 0){
        output += "\\nReasons:\\n";
        result.reasons.forEach(r => { output += "- " + r + "\\n"; });
    }

    box.innerText = output;
});
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
    data = request.get_json(silent=True) or {}

    try:
        input_df = pd.DataFrame([data])

        # ── Ordinal encode facility_rating and exam_difficulty ──
        input_df["facility_rating"] = input_df["facility_rating"].map(FACILITY_MAP)
        input_df["exam_difficulty"]  = input_df["exam_difficulty"].map(DIFFICULTY_MAP)

        # ── Numeric cast ──
        for col in ["study_hours", "class_attendance", "sleep_hours",
                    "facility_rating", "exam_difficulty"]:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(float)

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
        input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])

        x = input_df.values

        # ── Predict ──
        pred = model.predict(x)[0]

        best_prob = None
        probs = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x)[0]
            probs = {cls: round(float(p[i]), 6) for i, cls in enumerate(model.classes_)}
            best_prob = probs.get(pred)

        reasons = explain_prediction(data, pred)

        return jsonify({
            "prediction":   pred,
            "confidence":   best_prob,
            "probabilities": probs,
            "reasons":      reasons
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(debug=True)
