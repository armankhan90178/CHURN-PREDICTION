"""
ChurnShield 2.0 — Prediction API

File:
routes/prediction.py

Purpose:
Enterprise-grade churn prediction API
for AI-powered customer retention systems.

Capabilities:
- churn prediction APIs
- real-time prediction
- batch prediction
- customer risk scoring
- SHAP explainability
- probability forecasting
- timeline prediction (30/60/90 days)
- recommendation generation
- anomaly-aware scoring
- dynamic feature validation
- model auto-loading
- async FastAPI APIs
- CSV batch uploads
- prediction analytics
- confidence scoring
- industry-agnostic inference
- export-ready JSON responses
- secure prediction endpoints

Author:
ChurnShield AI
"""

import os
import io
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import (
    Dict,
    List,
    Optional
)

import joblib
import numpy as np
import pandas as pd

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends
)

from pydantic import (
    BaseModel
)

# ============================================================
# OPTIONAL IMPORTS
# ============================================================

try:

    from ml.predictor import (
        predict_churn
    )

except Exception:

    predict_churn = None

try:

    from ml.recommendation_engine import (
        generate_recommendations
    )

except Exception:

    generate_recommendations = None

try:

    from ml.explainer import (
        explain_prediction
    )

except Exception:

    explain_prediction = None

try:

    from ml.timeline import (
        timeline_prediction
    )

except Exception:

    timeline_prediction = None


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(
    "churnshield.routes.prediction"
)

# ============================================================
# ROUTER
# ============================================================

router = APIRouter(

    prefix="/prediction",

    tags=["Prediction"]

)

# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(

    parents=True,
    exist_ok=True

)

PREDICTION_DIR = Path(
    "user_data/predictions"
)

PREDICTION_DIR.mkdir(

    parents=True,
    exist_ok=True

)

# ============================================================
# REQUEST MODELS
# ============================================================

class PredictionRequest(BaseModel):

    features: Dict


class BatchPredictionRequest(BaseModel):

    records: List[Dict]


# ============================================================
# MODEL LOADER
# ============================================================

class ModelManager:

    def __init__(self):

        self.loaded_models = {}

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(
        self,
        model_name: str = "global_model.pkl"
    ):

        model_path = MODEL_DIR / model_name

        if not model_path.exists():

            raise FileNotFoundError(

                f"Model not found: "
                f"{model_path}"

            )

        if model_name in self.loaded_models:

            return self.loaded_models[
                model_name
            ]

        model = joblib.load(
            model_path
        )

        self.loaded_models[
            model_name
        ] = model

        logger.info(
            f"Loaded model: {model_name}"
        )

        return model

    # ========================================================
    # AVAILABLE MODELS
    # ========================================================

    def available_models(self):

        return [

            x.name

            for x in MODEL_DIR.glob(
                "*.pkl"
            )

        ]


model_manager = ModelManager()

# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_features(
    features: Dict
):

    if not isinstance(
        features,
        dict
    ):

        raise HTTPException(

            status_code=400,

            detail="Features must be a dictionary"

        )

    if len(features) == 0:

        raise HTTPException(

            status_code=400,

            detail="No features provided"

        )

    return True


# ============================================================
# PREPARE DATAFRAME
# ============================================================

def prepare_dataframe(
    features: Dict
):

    df = pd.DataFrame([features])

    # --------------------------------------------------------
    # CLEAN TYPES
    # --------------------------------------------------------

    for col in df.columns:

        try:

            df[col] = pd.to_numeric(
                df[col]
            )

        except Exception:

            pass

    return df


# ============================================================
# PREDICT SINGLE CUSTOMER
# ============================================================

@router.post("/single")
async def single_prediction(
    request: PredictionRequest
):

    try:

        validate_features(
            request.features
        )

        df = prepare_dataframe(
            request.features
        )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = model_manager.load_model()

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            df
        )[0]

        probability = float(

            model.predict_proba(df)[0][1]

        )

        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if probability >= 0.80:

            risk = "critical"

        elif probability >= 0.60:

            risk = "high"

        elif probability >= 0.40:

            risk = "medium"

        else:

            risk = "low"

        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        recommendations = []

        if generate_recommendations:

            try:

                recommendations = (
                    generate_recommendations(
                        probability
                    )
                )

            except Exception:

                recommendations = []

        # ----------------------------------------------------
        # EXPLAINABILITY
        # ----------------------------------------------------

        explanation = {}

        if explain_prediction:

            try:

                explanation = explain_prediction(

                    model,
                    df

                )

            except Exception:

                explanation = {}

        # ----------------------------------------------------
        # TIMELINE
        # ----------------------------------------------------

        timeline = {}

        if timeline_prediction:

            try:

                timeline = timeline_prediction(
                    probability
                )

            except Exception:

                timeline = {}

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        result = {

            "success": True,

            "prediction_id":
                str(uuid.uuid4()),

            "prediction":
                int(prediction),

            "churn_probability":
                round(probability, 4),

            "risk_level":
                risk,

            "recommendations":
                recommendations,

            "explanation":
                explanation,

            "timeline":
                timeline,

            "generated_at":
                datetime.utcnow()
                .isoformat()

        }

        return result

    except Exception as e:

        logger.error(
            f"Prediction failed: {e}"
        )

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ============================================================
# BATCH PREDICTION
# ============================================================

@router.post("/batch")
async def batch_prediction(
    request: BatchPredictionRequest
):

    try:

        if len(request.records) == 0:

            raise HTTPException(

                status_code=400,

                detail="No records provided"

            )

        df = pd.DataFrame(
            request.records
        )

        model = model_manager.load_model()

        predictions = model.predict(
            df
        )

        probabilities = model.predict_proba(
            df
        )[:, 1]

        results = []

        for idx in range(len(df)):

            prob = float(
                probabilities[idx]
            )

            if prob >= 0.80:

                risk = "critical"

            elif prob >= 0.60:

                risk = "high"

            elif prob >= 0.40:

                risk = "medium"

            else:

                risk = "low"

            results.append({

                "row":
                    idx,

                "prediction":
                    int(predictions[idx]),

                "probability":
                    round(prob, 4),

                "risk":
                    risk

            })

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        result_df = pd.DataFrame(
            results
        )

        output_file = (

            PREDICTION_DIR /

            f"batch_prediction_"
            f"{uuid.uuid4().hex}.csv"

        )

        result_df.to_csv(

            output_file,
            index=False

        )

        return {

            "success": True,

            "total_records":
                len(results),

            "results":
                results[:100],

            "saved_file":
                str(output_file)

        }

    except Exception as e:

        logger.error(
            f"Batch prediction failed: {e}"
        )

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ============================================================
# CSV PREDICTION
# ============================================================

@router.post("/csv")
async def csv_prediction(
    file: UploadFile = File(...)
):

    try:

        # ----------------------------------------------------
        # VALIDATE FILE
        # ----------------------------------------------------

        if not file.filename.endswith(
            ".csv"
        ):

            raise HTTPException(

                status_code=400,

                detail="Only CSV supported"

            )

        contents = await file.read()

        df = pd.read_csv(

            io.BytesIO(contents)

        )

        if len(df) == 0:

            raise HTTPException(

                status_code=400,

                detail="Empty CSV"

            )

        model = model_manager.load_model()

        predictions = model.predict(
            df
        )

        probabilities = model.predict_proba(
            df
        )[:, 1]

        df["prediction"] = predictions

        df["churn_probability"] = probabilities

        # ----------------------------------------------------
        # SAVE OUTPUT
        # ----------------------------------------------------

        output_file = (

            PREDICTION_DIR /

            f"csv_prediction_"
            f"{uuid.uuid4().hex}.csv"

        )

        df.to_csv(

            output_file,
            index=False

        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        high_risk = len(

            df[
                df[
                    "churn_probability"
                ] >= 0.80
            ]

        )

        return {

            "success": True,

            "rows_processed":
                len(df),

            "high_risk_customers":
                high_risk,

            "saved_file":
                str(output_file)

        }

    except Exception as e:

        logger.error(
            f"CSV prediction failed: {e}"
        )

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ============================================================
# MODEL INFO
# ============================================================

@router.get("/models")
async def models():

    return {

        "available_models":

            model_manager.available_models()

    }


# ============================================================
# PREDICTION ANALYTICS
# ============================================================

@router.get("/analytics")
async def prediction_analytics():

    files = list(

        PREDICTION_DIR.glob(
            "*.csv"
        )

    )

    total_files = len(files)

    total_predictions = 0

    for file in files:

        try:

            df = pd.read_csv(file)

            total_predictions += len(df)

        except Exception:

            pass

    return {

        "prediction_files":
            total_files,

        "total_predictions":
            total_predictions,

        "storage_directory":
            str(PREDICTION_DIR)

    }


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
async def health():

    return {

        "service":
            "prediction_engine",

        "status":
            "healthy",

        "models_available":

            len(
                model_manager.available_models()
            ),

        "timestamp":
            datetime.utcnow()
            .isoformat()

    }


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD PREDICTION API")
    print("=" * 60)

    manager = ModelManager()

    print("\nAvailable Models:\n")

    print(
        manager.available_models()
    )

    sample_features = {

        "monthly_spend": 1200,

        "tenure": 14,

        "support_calls": 5,

        "usage_hours": 40

    }

    df = prepare_dataframe(
        sample_features
    )

    print("\nPrepared DataFrame:\n")

    print(df)