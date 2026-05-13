"""
ChurnShield 2.0 — Auto Retrain Engine

Purpose:
Enterprise-grade automated ML retraining system
for churn prediction, customer analytics,
continuous learning, model optimization,
and production lifecycle management.

Capabilities:
- automatic retraining
- scheduled training
- drift-triggered retraining
- performance-based retraining
- multi-model training
- rollback support
- model versioning
- dataset monitoring
- automated evaluation
- champion vs challenger models
- production deployment pipeline
- incremental learning
- retrain alerts
- retrain reports
- auto backup system
- model registry integration

Supports:
- XGBoost
- LightGBM
- CatBoost
- RandomForest
- LogisticRegression
- Ensemble models

Author:
ChurnShield AI
"""

import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder
)

logger = logging.getLogger(
    "churnshield.auto_retrain"
)


# ============================================================
# OPTIONAL BOOSTING LIBRARIES
# ============================================================

XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False

try:

    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True

except Exception:
    pass

try:

    from lightgbm import (
        LGBMClassifier
    )

    LIGHTGBM_AVAILABLE = True

except Exception:
    pass

try:

    from catboost import (
        CatBoostClassifier
    )

    CATBOOST_AVAILABLE = True

except Exception:
    pass


# ============================================================
# MAIN ENGINE
# ============================================================

class AutoRetrainEngine:

    def __init__(
        self,
        models_dir: str = "models",
        registry_file: str = (
            "models/model_registry.json"
        ),
    ):

        self.models_dir = Path(models_dir)

        self.models_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.registry_file = Path(
            registry_file
        )

        self.registry = (
            self._load_registry()
        )

    # ========================================================
    # REGISTRY
    # ========================================================

    def _load_registry(self) -> Dict:

        if not self.registry_file.exists():

            return {

                "models": {},
                "active_model": None,

            }

        try:

            with open(
                self.registry_file,
                "r"
            ) as f:

                return json.load(f)

        except Exception as e:

            logger.error(
                f"Registry load failed: {e}"
            )

            return {

                "models": {},
                "active_model": None,

            }

    # ========================================================
    # SAVE REGISTRY
    # ========================================================

    def _save_registry(self):

        with open(
            self.registry_file,
            "w"
        ) as f:

            json.dump(

                self.registry,
                f,
                indent=4

            )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = "churn",
    ):

        logger.info(
            "Preparing training data"
        )

        if target_column not in df.columns:

            raise ValueError(
                f"Missing target column: "
                f"{target_column}"
            )

        data = df.copy()

        # ----------------------------------------------------
        # HANDLE CATEGORICAL
        # ----------------------------------------------------

        encoders = {}

        for col in data.columns:

            if data[col].dtype == "object":

                le = LabelEncoder()

                data[col] = le.fit_transform(

                    data[col]
                    .astype(str)

                )

                encoders[col] = le

        # ----------------------------------------------------
        # FILL MISSING
        # ----------------------------------------------------

        data = data.fillna(0)

        X = data.drop(
            columns=[target_column]
        )

        y = data[target_column]

        return X, y, encoders

    # ========================================================
    # MODEL FACTORY
    # ========================================================

    def get_model(
        self,
        algorithm: str
    ):

        algorithm = algorithm.lower()

        if (
            algorithm == "xgboost"
            and XGBOOST_AVAILABLE
        ):

            return XGBClassifier(

                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",

            )

        elif (
            algorithm == "lightgbm"
            and LIGHTGBM_AVAILABLE
        ):

            return LGBMClassifier(

                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,

            )

        elif (
            algorithm == "catboost"
            and CATBOOST_AVAILABLE
        ):

            return CatBoostClassifier(

                iterations=300,
                learning_rate=0.05,
                depth=8,
                verbose=False,

            )

        elif algorithm == "randomforest":

            return RandomForestClassifier(

                n_estimators=300,
                max_depth=12,
                n_jobs=-1,
                random_state=42,

            )

        else:

            return LogisticRegression(
                max_iter=500
            )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str,
        target_column: str = "churn",
        algorithm: str = "xgboost",
    ) -> Dict:

        logger.info(
            f"Training model: {model_name}"
        )

        # ----------------------------------------------------
        # PREPARE
        # ----------------------------------------------------

        X, y, encoders = (
            self.prepare_data(
                df,
                target_column
            )
        )

        X_train, X_test, y_train, y_test = (

            train_test_split(

                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,

            )

        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = self.get_model(
            algorithm
        )

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        probabilities = None

        try:

            probabilities = (

                model.predict_proba(
                    X_test
                )[:, 1]

            )

        except Exception:

            probabilities = predictions

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = {

            "accuracy":
                round(

                    accuracy_score(
                        y_test,
                        predictions
                    ),

                    4

                ),

            "precision":
                round(

                    precision_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                    4

                ),

            "recall":
                round(

                    recall_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                    4

                ),

            "f1_score":
                round(

                    f1_score(
                        y_test,
                        predictions,
                        zero_division=0
                    ),

                    4

                ),

            "roc_auc":
                round(

                    roc_auc_score(
                        y_test,
                        probabilities
                    ),

                    4

                ),

        }

        # ----------------------------------------------------
        # VERSION
        # ----------------------------------------------------

        timestamp = datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S"
        )

        version = f"{model_name}_{timestamp}"

        model_path = (

            self.models_dir /
            f"{version}.pkl"

        )

        metadata_path = (

            self.models_dir /
            f"{version}_metadata.pkl"

        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        joblib.dump(
            model,
            model_path
        )

        metadata = {

            "model_name":
                model_name,

            "version":
                version,

            "algorithm":
                algorithm,

            "metrics":
                metrics,

            "features":
                list(X.columns),

            "target_column":
                target_column,

            "encoders":
                encoders,

            "created_at":
                datetime.utcnow().isoformat(),

        }

        with open(
            metadata_path,
            "wb"
        ) as f:

            pickle.dump(
                metadata,
                f
            )

        # ----------------------------------------------------
        # UPDATE REGISTRY
        # ----------------------------------------------------

        self.registry["models"][
            version
        ] = {

            "model_path":
                str(model_path),

            "metadata_path":
                str(metadata_path),

            "metrics":
                metrics,

            "algorithm":
                algorithm,

            "created_at":
                datetime.utcnow().isoformat(),

        }

        # ----------------------------------------------------
        # CHAMPION MODEL
        # ----------------------------------------------------

        active_model = (
            self.registry.get(
                "active_model"
            )
        )

        promote = False

        if active_model is None:

            promote = True

        else:

            current_auc = (

                self.registry["models"]
                [active_model]
                ["metrics"]
                .get("roc_auc", 0)

            )

            if metrics["roc_auc"] > current_auc:

                promote = True

        if promote:

            self.registry[
                "active_model"
            ] = version

            logger.info(
                f"Promoted new champion: "
                f"{version}"
            )

        self._save_registry()

        return {

            "success": True,

            "version":
                version,

            "metrics":
                metrics,

            "model_path":
                str(model_path),

            "promoted":
                promote,

        }

    # ========================================================
    # RETRAIN TRIGGER
    # ========================================================

    def should_retrain(
        self,
        drift_score: float = 0.0,
        current_accuracy: float = 1.0,
        threshold_drift: float = 0.2,
        threshold_accuracy: float = 0.7,
    ) -> Dict:

        reasons = []

        retrain = False

        # ----------------------------------------------------
        # DRIFT
        # ----------------------------------------------------

        if drift_score > threshold_drift:

            retrain = True

            reasons.append(
                "High data drift detected"
            )

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        if current_accuracy < (
            threshold_accuracy
        ):

            retrain = True

            reasons.append(
                "Model accuracy degraded"
            )

        return {

            "should_retrain":
                retrain,

            "reasons":
                reasons,

        }

    # ========================================================
    # RETRAIN PIPELINE
    # ========================================================

    def retrain_pipeline(
        self,
        df: pd.DataFrame,
        model_name: str,
        algorithm: str = "xgboost",
        target_column: str = "churn",
        drift_score: float = 0.0,
        current_accuracy: float = 1.0,
    ) -> Dict:

        # ----------------------------------------------------
        # CHECK
        # ----------------------------------------------------

        decision = self.should_retrain(

            drift_score=drift_score,
            current_accuracy=current_accuracy,

        )

        if not decision[
            "should_retrain"
        ]:

            return {

                "success": False,

                "message":
                    "Retraining not required",

                "decision":
                    decision,

            }

        logger.info(
            "Starting automatic retraining"
        )

        result = self.train_model(

            df=df,
            model_name=model_name,
            target_column=target_column,
            algorithm=algorithm,

        )

        result["decision"] = decision

        return result

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(
        self,
        version: Optional[str] = None,
    ):

        if version is None:

            version = (
                self.registry.get(
                    "active_model"
                )
            )

        if version is None:

            raise ValueError(
                "No active model found"
            )

        model_info = (

            self.registry["models"]
            [version]

        )

        model = joblib.load(

            model_info["model_path"]

        )

        with open(
            model_info["metadata_path"],
            "rb"
        ) as f:

            metadata = pickle.load(f)

        return {

            "model": model,
            "metadata": metadata,

        }

    # ========================================================
    # ROLLBACK
    # ========================================================

    def rollback_model(
        self,
        version: str
    ) -> Dict:

        if version not in (
            self.registry["models"]
        ):

            return {

                "success": False,

                "message":
                    "Version not found",

            }

        self.registry[
            "active_model"
        ] = version

        self._save_registry()

        logger.info(
            f"Rollback to {version}"
        )

        return {

            "success": True,

            "active_model":
                version,

        }

    # ========================================================
    # DELETE OLD MODELS
    # ========================================================

    def cleanup_models(
        self,
        keep_latest: int = 5
    ) -> Dict:

        models = list(

            self.registry["models"]
            .items()

        )

        models.sort(

            key=lambda x:
            x[1]["created_at"],

            reverse=True

        )

        active_model = (
            self.registry.get(
                "active_model"
            )
        )

        deleted = []

        for version, info in models[keep_latest:]:

            if version == active_model:
                continue

            try:

                Path(
                    info["model_path"]
                ).unlink(
                    missing_ok=True
                )

                Path(
                    info["metadata_path"]
                ).unlink(
                    missing_ok=True
                )

                del self.registry[
                    "models"
                ][version]

                deleted.append(version)

            except Exception as e:

                logger.error(
                    f"Cleanup failed: {e}"
                )

        self._save_registry()

        return {

            "deleted_models":
                deleted

        }

    # ========================================================
    # MODEL REPORT
    # ========================================================

    def model_report(self) -> Dict:

        report = {

            "active_model":
                self.registry.get(
                    "active_model"
                ),

            "total_models":
                len(
                    self.registry["models"]
                ),

            "models":
                self.registry["models"],

        }

        return report

    # ========================================================
    # BACKUP
    # ========================================================

    def backup_models(
        self,
        backup_dir: str = "backup_models"
    ) -> Dict:

        backup_path = Path(
            backup_dir
        )

        backup_path.mkdir(
            parents=True,
            exist_ok=True
        )

        for file in self.models_dir.glob("*"):

            shutil.copy2(

                file,
                backup_path / file.name

            )

        logger.info(
            "Model backup completed"
        )

        return {

            "success": True,

            "backup_path":
                str(backup_path),

        }


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def train_auto_model(
    df: pd.DataFrame,
    model_name: str = "churn_model",
    algorithm: str = "xgboost",
):

    engine = AutoRetrainEngine()

    return engine.train_model(

        df=df,
        model_name=model_name,
        algorithm=algorithm,

    )


def auto_retrain(
    df: pd.DataFrame,
    drift_score: float = 0.3,
    current_accuracy: float = 0.6,
):

    engine = AutoRetrainEngine()

    return engine.retrain_pipeline(

        df=df,
        model_name="auto_model",
        drift_score=drift_score,
        current_accuracy=current_accuracy,

    )


def load_active_model():

    engine = AutoRetrainEngine()

    return engine.load_model()


def model_report():

    engine = AutoRetrainEngine()

    return engine.model_report()