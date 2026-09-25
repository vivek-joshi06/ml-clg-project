"""
CardioScan AI — FastAPI Machine Learning Backend
=================================================
Production-ready REST API for Cardiovascular Disease Risk Prediction.
Designed for deployment on Render, Railway, Fly.io, or AWS.

Features:
- Preserves trained XGBoost model (cardiovascular_model.pkl)
- Exact MinMaxScaler bounds and BMI feature engineering from cleanData.ipynb
- Robust Pydantic request validation and physiological checks
- Full CORS configuration for cross-origin frontend connectivity
- OpenAPI interactive documentation at /docs
"""

import os
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# 1. FastAPI Application & CORS Setup
# ----------------------------------------------------------------------------
app = FastAPI(
    title="CardioScan AI — Cardiovascular Disease Risk Predictor API",
    description=(
        "High-performance machine learning backend for cardiovascular risk stratification. "
        "Powered by Extreme Gradient Boosting (XGBoost) trained on 70,000 clinical records."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for all origins so any frontend (Render, Vercel, Netlify, localhost) can consume the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# 2. Model Loading & Exact Preprocessing Pipeline
# ----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cardiovascular_model.pkl")

# Exact MinMaxScaler parameters computed during training in cleanData.ipynb
# Applied to continuous columns: ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
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

print(f">> Loading trained model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
    print(">> XGBoost model loaded successfully!")
except Exception as e:
    print(f"!! Warning: Could not load model file '{MODEL_PATH}': {e}")
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
        "true_positive": 4003,
    },
    "hyperparameters": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
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
        {"feature": "Height", "code": "height", "importance": 1.15},
    ],
}


def preprocess_patient_data(raw_dict: dict) -> pd.DataFrame:
    """
    Applies the exact MinMaxScaler preprocessing from cleanData.ipynb:
    1. Continuous features scaled with clipping to [0.0, 1.0].
    2. Categorical features retained as integers.
    3. Guarantees column ordering matches the trained XGBoost estimator.
    """
    df = pd.DataFrame([raw_dict])[EXPECTED_FEATURES].copy()

    for col, (col_min, col_max) in SCALER_BOUNDS.items():
        clipped = np.clip(df[col], col_min, col_max)
        df[col] = (clipped - col_min) / (col_max - col_min)

    return df


# ----------------------------------------------------------------------------
# 3. Request / Response Pydantic Schemas
# ----------------------------------------------------------------------------
class PatientInput(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Patient age in years (e.g., 48)")
    gender: int = Field(..., ge=1, le=2, description="Biological sex (1: Female, 2: Male)")
    height: float = Field(..., ge=50, le=250, description="Height in centimeters (e.g., 170.0)")
    weight: float = Field(..., ge=20, le=300, description="Weight in kilograms (e.g., 72.0)")
    ap_hi: float = Field(..., ge=40, le=300, description="Systolic blood pressure in mmHg (e.g., 122)")
    ap_lo: float = Field(..., ge=20, le=250, description="Diastolic blood pressure in mmHg (e.g., 82)")
    cholesterol: int = Field(..., ge=1, le=3, description="Serum cholesterol (1: Normal, 2: Above Normal, 3: Well Above)")
    gluc: int = Field(..., ge=1, le=3, description="Fasting glucose (1: Normal, 2: Above Normal, 3: Well Above)")
    smoke: int = Field(..., ge=0, le=1, description="Active tobacco smoker (0: No, 1: Yes)")
    alco: int = Field(..., ge=0, le=1, description="Alcohol consumption (0: No, 1: Yes)")
    active: int = Field(..., ge=0, le=1, description="Regular physical activity (0: No, 1: Yes)")
    bmi: Optional[float] = Field(None, description="Optional pre-calculated BMI; computed automatically if omitted")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 48.0,
                "gender": 2,
                "height": 172.0,
                "weight": 76.0,
                "ap_hi": 125.0,
                "ap_lo": 82.0,
                "cholesterol": 1,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1,
            }
        }
    }


class PatientMetrics(BaseModel):
    bmi: float
    bp_stage: str
    pulse_pressure: int


class PredictionResponse(BaseModel):
    status: str
    prediction: int = Field(..., description="0 = No CVD Indicated, 1 = CVD Indicated")
    probability: float = Field(..., description="Calibrated CVD probability between 0.0 and 1.0")
    probability_percent: float = Field(..., description="Probability expressed as a percentage")
    confidence: float = Field(..., description="Model confidence score in the predicted class (%)")
    risk_tier: str = Field(..., description="Low Risk | Moderate Risk | High Risk")
    risk_badge: str = Field(..., description="low | moderate | high")
    risk_color: str = Field(..., description="Hex color for UI badge rendering")
    recommendation: str = Field(..., description="Clinical decision support guidance")
    contributing_drivers: List[str] = Field(..., description="Patient-specific elevated risk factors")
    patient_metrics: PatientMetrics
    model_features: List[str]
    normalized_vector: Dict[str, float]


# ----------------------------------------------------------------------------
# 4. API Endpoints
# ----------------------------------------------------------------------------
@app.get("/", tags=["Health & Status"])
def root():
    """Service status and API landing endpoint."""
    return {
        "status": "online",
        "service": "CardioScan AI — FastAPI Machine Learning Backend",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health", tags=["Health & Status"])
def health():
    """Health check endpoint for Render, container orchestrators, and uptime monitors."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model file not loaded.",
        )
    return {"status": "healthy", "model": "loaded"}


@app.get("/info", tags=["Model Analytics"])
@app.get("/api/info", tags=["Model Analytics"], include_in_schema=False)
def get_info():
    """Returns model architecture parameters, performance metrics, and feature importances."""
    return {
        "status": "success",
        "data": MODEL_METRICS,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    include_in_schema=False,
)
def predict(patient: PatientInput):
    """
    Cardiovascular Risk Prediction Endpoint:
    Accepts patient demographics, hemodynamics, biomarkers, and lifestyle.
    Applies exact MinMaxScaler bounds and returns calibrated risk probability,
    confidence score, clinical risk tier, and primary risk contributors.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained ML model is not loaded on this server.",
        )

    # Physiological validation
    if patient.ap_lo >= patient.ap_hi:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Diastolic BP (ap_lo) cannot be equal to or greater than Systolic BP (ap_hi).",
        )

    # Calculate BMI if not provided
    if patient.bmi is not None and patient.bmi > 0:
        bmi = round(float(patient.bmi), 2)
    else:
        height_m = patient.height / 100.0
        bmi = round(patient.weight / (height_m ** 2), 2)

    raw_payload = {
        "age": float(patient.age),
        "gender": int(patient.gender),
        "height": float(patient.height),
        "weight": float(patient.weight),
        "ap_hi": float(patient.ap_hi),
        "ap_lo": float(patient.ap_lo),
        "cholesterol": int(patient.cholesterol),
        "gluc": int(patient.gluc),
        "smoke": int(patient.smoke),
        "alco": int(patient.alco),
        "active": int(patient.active),
        "bmi": float(bmi),
    }

    # Preprocess with MinMaxScaler
    processed_df = preprocess_patient_data(raw_payload)

    # Run inference
    try:
        prediction_class = int(model.predict(processed_df)[0])
        probability = float(model.predict_proba(processed_df)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )

    # Risk Tier Stratification
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

    # Primary contributing risk factors
    contributing_drivers = []
    if patient.ap_hi >= 140:
        contributing_drivers.append(f"Systolic Hypertension ({int(patient.ap_hi)} mmHg)")
    elif patient.ap_hi >= 130:
        contributing_drivers.append(f"Pre-hypertensive Systolic BP ({int(patient.ap_hi)} mmHg)")

    if patient.ap_lo >= 90:
        contributing_drivers.append(f"Diastolic Hypertension ({int(patient.ap_lo)} mmHg)")

    if patient.cholesterol == 3:
        contributing_drivers.append("High Serum Cholesterol (Well Above Normal)")
    elif patient.cholesterol == 2:
        contributing_drivers.append("Elevated Serum Cholesterol (Above Normal)")

    if patient.age >= 55:
        contributing_drivers.append(f"Advanced Age Factor ({int(patient.age)} years)")

    if patient.smoke == 1:
        contributing_drivers.append("Active Tobacco Smoking Habit")

    if bmi >= 30.0:
        contributing_drivers.append(f"Clinical Obesity (BMI: {bmi})")
    elif bmi >= 25.0:
        contributing_drivers.append(f"Overweight Category (BMI: {bmi})")

    if patient.gluc == 3:
        contributing_drivers.append("High Fasting Glucose (Marked Hyperglycemia)")
    elif patient.gluc == 2:
        contributing_drivers.append("Elevated Blood Glucose")

    if patient.active == 0:
        contributing_drivers.append("Sedentary Lifestyle (Low Physical Activity)")

    if patient.alco == 1:
        contributing_drivers.append("Alcohol Consumption")

    # Blood pressure clinical staging
    if patient.ap_hi < 120 and patient.ap_lo < 80:
        bp_stage = "Normal Blood Pressure"
    elif 120 <= patient.ap_hi < 130 and patient.ap_lo < 80:
        bp_stage = "Elevated Blood Pressure"
    elif (130 <= patient.ap_hi < 140) or (80 <= patient.ap_lo < 90):
        bp_stage = "Hypertension Stage 1"
    else:
        bp_stage = "Hypertension Stage 2"

    return {
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
            "pulse_pressure": int(patient.ap_hi - patient.ap_lo),
        },
        "model_features": EXPECTED_FEATURES,
        "normalized_vector": processed_df.iloc[0].to_dict(),
    }


# ----------------------------------------------------------------------------
# 5. Local Development Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n=======================================================")
    print(f">> Starting CardioScan AI FastAPI on port {port}")
    print(f">> Interactive docs available at http://127.0.0.1:{port}/docs")
    print(f"=======================================================\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
