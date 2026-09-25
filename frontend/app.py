"""
CardioScan AI — Cardiovascular Disease Risk Intelligence
=========================================================
Production-grade Flask Web Application & REST API
Backend: Flask, Scikit-Learn, XGBoost, NumPy, Pandas
Frontend: HTML5, CSS3 (Modern Bento/Glassmorphism), Vanilla ES6 JavaScript

Reuses pre-trained model: cardiovascular_model.pkl
Applies exact preprocessing: MinMaxScaler + BMI feature engineering (from cleanData.ipynb)
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ----------------------------------------------------------------------------
# 1. Model Loading & Preprocessing Pipeline
# ----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cardiovascular_model.pkl")

# Exact MinMaxScaler parameters computed during training in cleanData.ipynb
# Applied to num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
SCALER_BOUNDS = {
    "age": (29.0, 64.0),
    "height": (120.0, 207.0),
    "weight": (40.0, 107.0),
    "ap_hi": (70.0, 240.0),
    "ap_lo": (40.0, 180.0),
    "bmi": (13.520822, 62.5),
}

EXPECTED_FEATURES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "bmi"
]

print(f"Loading trained XGBoost model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Failed to load model: {e}")
    model = None

# Evaluation metrics and feature importances extracted from training notebooks
MODEL_METRICS = {
    "model_name": "Extreme Gradient Boosting (XGBoost)",
    "accuracy": 0.7373,
    "precision": 0.7560,
    "recall": 0.6820,
    "f1_score": 0.72,
    "cv_score": 0.7391,
    "total_samples": 70000,
    "cleaned_samples": 60195,
    "test_samples": 12039,
    "confusion_matrix": {
        "true_negative": 4873,
        "false_positive": 1294,
        "false_negative": 1869,
        "true_positive": 4003
    },
    "hyperparameters": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss"
    },
    "feature_importances": [
        {"feature": "Systolic BP (ap_hi)", "code": "ap_hi", "importance": 41.93},
        {"feature": "Diastolic BP (ap_lo)", "code": "ap_lo", "importance": 19.23},
        {"feature": "Serum Cholesterol", "code": "cholesterol", "importance": 14.07},
        {"feature": "Patient Age", "code": "age", "importance": 8.93},
        {"feature": "Physical Activity", "code": "active", "importance": 3.19},
        {"feature": "Tobacco Smoking", "code": "smoke", "importance": 2.45},
        {"feature": "Fasting Glucose", "code": "gluc", "importance": 2.38},
        {"feature": "Alcohol Consumption", "code": "alco", "importance": 2.13},
        {"feature": "Body Mass Index (BMI)", "code": "bmi", "importance": 1.73},
        {"feature": "Weight", "code": "weight", "importance": 1.51},
        {"feature": "Biological Sex", "code": "gender", "importance": 1.30},
        {"feature": "Height", "code": "height", "importance": 1.15}
    ]
}


def preprocess_patient_data(raw_dict: dict) -> pd.DataFrame:
    """
    Applies the exact MinMaxScaler preprocessing from cleanData.ipynb:
    1. Height in m, weight in kg -> BMI.
    2. Continuous features scaled with clipping to [0.0, 1.0].
    3. Categorical variables retained as integers.
    """
    df = pd.DataFrame([raw_dict])[EXPECTED_FEATURES].copy()
    
    for col, (col_min, col_max) in SCALER_BOUNDS.items():
        clipped = np.clip(df[col], col_min, col_max)
        df[col] = (clipped - col_min) / (col_max - col_min)
        
    return df


# ----------------------------------------------------------------------------
# 2. Flask Web Routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    """Serves the single-page HTML5/CSS3/JS application."""
    return render_template("index.html")


@app.route("/api/info", methods=["GET"])
def get_info():
    """Returns model metrics, architecture details, and feature importances."""
    return jsonify({
        "status": "success",
        "data": MODEL_METRICS
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Inference endpoint:
    Accepts patient vitals & lifestyle JSON.
    Returns calibrated probability, risk classification, and contributing risk factors.
    """
    if model is None:
        return jsonify({"status": "error", "message": "Model is not loaded on server."}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "Empty request body"}), 400

        # Extract & validate fields
        age = float(data.get("age", 45))
        gender = int(data.get("gender", 1))  # 1: Female, 2: Male
        height = float(data.get("height", 168))
        weight = float(data.get("weight", 70))
        ap_hi = float(data.get("ap_hi", 120))
        ap_lo = float(data.get("ap_lo", 80))
        cholesterol = int(data.get("cholesterol", 1))  # 1, 2, 3
        gluc = int(data.get("gluc", 1))                # 1, 2, 3
        smoke = int(data.get("smoke", 0))              # 0, 1
        alco = int(data.get("alco", 0))                # 0, 1
        active = int(data.get("active", 1))            # 0, 1

        # Physiological validation
        if ap_lo >= ap_hi:
            return jsonify({
                "status": "error",
                "message": "Diastolic BP (ap_lo) cannot be greater than or equal to Systolic BP (ap_hi)."
            }), 400

        if height <= 0:
            return jsonify({"status": "error", "message": "Height must be greater than zero."}), 400

        # Calculate BMI
        height_m = height / 100.0
        bmi = round(weight / (height_m ** 2), 2)

        # Build raw dict
        raw_patient = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active,
            "bmi": bmi
        }

        # Preprocess with MinMaxScaler
        processed_df = preprocess_patient_data(raw_patient)

        # Run inference
        prediction_class = int(model.predict(processed_df)[0])
        probability = float(model.predict_proba(processed_df)[0][1])

        # Clinical Risk Stratification
        if probability < 0.35:
            risk_tier = "Low Risk"
            risk_badge = "low"
            risk_color = "#10b981"
            recommendation = (
                "Patient profile indicates low 10-year cardiovascular risk. "
                "Maintain healthy dietary patterns, regular aerobic exercise, and routine preventive checkups."
            )
        elif probability < 0.65:
            risk_tier = "Moderate Risk"
            risk_badge = "moderate"
            risk_color = "#f59e0b"
            recommendation = (
                "Borderline or elevated markers identified. Recommend lipid profile monitoring, "
                "blood pressure management, dietary adjustments, and active cardiovascular screening."
            )
        else:
            risk_tier = "High Risk"
            risk_badge = "high"
            risk_color = "#ef4444"
            recommendation = (
                "Significant cardiovascular risk factors detected. Formal medical consultation, "
                "comprehensive diagnostic cardiac evaluation, and targeted clinical intervention are strongly advised."
            )

        # Primary risk contributors for this specific patient
        contributing_drivers = []
        if ap_hi >= 140:
            contributing_drivers.append(f"Systolic Hypertension ({int(ap_hi)} mmHg)")
        elif ap_hi >= 130:
            contributing_drivers.append(f"Pre-hypertensive Systolic BP ({int(ap_hi)} mmHg)")

        if ap_lo >= 90:
            contributing_drivers.append(f"Diastolic Hypertension ({int(ap_lo)} mmHg)")

        if cholesterol == 3:
            contributing_drivers.append("High Serum Cholesterol (Well Above Normal)")
        elif cholesterol == 2:
            contributing_drivers.append("Elevated Serum Cholesterol (Above Normal)")

        if age >= 55:
            contributing_drivers.append(f"Advanced Age Factor ({int(age)} years)")

        if smoke == 1:
            contributing_drivers.append("Active Tobacco Smoking Habit")

        if bmi >= 30.0:
            contributing_drivers.append(f"Clinical Obesity (BMI: {bmi})")
        elif bmi >= 25.0:
            contributing_drivers.append(f"Overweight Category (BMI: {bmi})")

        if gluc == 3:
            contributing_drivers.append("High Fasting Glucose (Marked Hyperglycemia)")
        elif gluc == 2:
            contributing_drivers.append("Elevated Blood Glucose")

        if active == 0:
            contributing_drivers.append("Sedentary Lifestyle (Low Physical Activity)")

        if alco == 1:
            contributing_drivers.append("Alcohol Consumption")

        # Blood pressure clinical staging
        if ap_hi < 120 and ap_lo < 80:
            bp_stage = "Normal Blood Pressure"
        elif 120 <= ap_hi < 130 and ap_lo < 80:
            bp_stage = "Elevated Blood Pressure"
        elif (130 <= ap_hi < 140) or (80 <= ap_lo < 90):
            bp_stage = "Hypertension Stage 1"
        else:
            bp_stage = "Hypertension Stage 2"

        return jsonify({
            "status": "success",
            "prediction": prediction_class,
            "probability": round(probability, 4),
            "probability_percent": round(probability * 100, 1),
            "confidence": round(max(probability, 1 - probability) * 100, 1),
            "risk_tier": risk_tier,
            "risk_badge": risk_badge,
            "risk_color": risk_color,
            "recommendation": recommendation,
            "contributing_drivers": contributing_drivers,
            "patient_metrics": {
                "bmi": bmi,
                "bp_stage": bp_stage,
                "pulse_pressure": int(ap_hi - ap_lo)
            },
            "model_features": EXPECTED_FEATURES,
            "normalized_vector": processed_df.iloc[0].to_dict()
        })

    except Exception as exc:
        return jsonify({"status": "error", "message": f"Inference error: {str(exc)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f">> CardioScan AI Flask Server starting at http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
