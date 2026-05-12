from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and feature columns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

# =========================
# Explanation logic
# =========================
def explain_prediction(data, prediction):

    reasons = []

    if float(data["study_hours"]) < 2:
        reasons.append("Low study hours")

    if float(data["class_attendance"]) < 60:
        reasons.append("Low class attendance")

    if float(data["sleep_hours"]) < 5:
        reasons.append("Insufficient sleep")

    if data["sleep_quality"] == "poor":
        reasons.append("Poor sleep quality")

    if data["study_method"] == "self-study":
        reasons.append("Studying alone without structured guidance")

    if data["internet_access"] == "no":
        reasons.append("No internet access for study resources")

    if float(data["exam_difficulty"]) >= 4:
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

<label>Facility Rating (1-5)</label>
<input name="facility_rating" type="number" min="1" max="5" required>

<label>Exam Difficulty (1-5)</label>
<input name="exam_difficulty" type="number" min="1" max="5" required>

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

    formData.forEach((v,k)=>{
        if(["study_hours","class_attendance","sleep_hours","facility_rating","exam_difficulty"].includes(k)){
            data[k] = Number(v);
        } else {
            data[k] = v;
        }
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
    }
    else if(result.prediction === "Pass"){
        box.classList.add("pass-box");
    }
    else if(result.prediction === "Distinction"){
        box.classList.add("dist-box");
    }

    let output = "Prediction: " + result.prediction + "\\n";

    if(result.confidence !== null && result.confidence !== undefined){
        output += "Confidence: " + (result.confidence * 100).toFixed(1) + "%\\n";
    }

    if(result.reasons && result.reasons.length > 0){
        output += "\\nReasons:\\n";
        result.reasons.forEach(r=>{
            output += "- " + r + "\\n";
        });
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
        # Convert incoming form data to DataFrame
        input_df = pd.DataFrame([data])

        # Numeric columns used by the model
        numeric_cols = [
            "age",
            "study_hours",
            "class_attendance",
            "sleep_hours",
            "facility_rating",
            "exam_difficulty"
        ]

        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(float)

        # Apply same one-hot encoding style used during training
        input_df = pd.get_dummies(input_df)

        # Add any missing training columns
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        # Keep columns in exact same order as training
        input_df = input_df[features]

        x = input_df.values

        # Predict class
        pred = model.predict(x)[0]

        # Predict probabilities if available
        probs = None
        best_prob = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x)[0]
            probs = {cls: round(float(p[i]), 6) for i, cls in enumerate(model.classes_)}
            best_prob = probs.get(pred)

        # Explanation
        reasons = explain_prediction(data, pred)

        response = {
            "prediction": pred,
            "confidence": best_prob,
            "probabilities": probs,
            "reasons": reasons
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 400
    
if __name__ == "__main__":
    app.run(debug=True)