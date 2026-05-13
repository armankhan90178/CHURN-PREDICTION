"""
ChurnShield 2.0 — Ensemble Learning Engine

Purpose:
Enterprise-grade ensemble modeling framework
for churn prediction, customer intelligence,
retention analytics, and high-accuracy ML.

Capabilities:
- stacking ensemble
- voting ensemble
- blending ensemble
- weighted ensemble
- multi-model orchestration
- auto-weight optimization
- probability averaging
- soft voting
- hard voting
- meta-model stacking
- dynamic ensemble scoring
- confidence estimation
- uncertainty detection
- model diversity analysis
- ensemble explainability
- production inference pipeline

Supports:
- XGBoost
- LightGBM
- CatBoost
- RandomForest
- LogisticRegression
- GradientBoosting
- Any sklearn-compatible model

Author:
ChurnShield AI
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
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
    StratifiedKFold,
    cross_val_predict,
)

logger = logging.getLogger(
    "churnshield.ensemble"
)


# ============================================================
# OPTIONAL MODELS
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

class EnsembleEngine:

    def __init__(
        self,
        models_dir: str = "models/ensemble",
    ):

        self.models_dir = Path(
            models_dir
        )

        self.models_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.base_models = {}

        self.meta_model = None

        self.ensemble_model = None

    # ========================================================
    # BASE MODELS
    # ========================================================

    def create_base_models(self):

        models = {

            "random_forest":
                RandomForestClassifier(

                    n_estimators=300,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,

                ),

            "gradient_boosting":
                GradientBoostingClassifier(

                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,

                ),

            "logistic_regression":
                LogisticRegression(

                    max_iter=500

                ),

        }

        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        if XGBOOST_AVAILABLE:

            models["xgboost"] = (

                XGBClassifier(

                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",

                )

            )

        # ----------------------------------------------------
        # LIGHTGBM
        # ----------------------------------------------------

        if LIGHTGBM_AVAILABLE:

            models["lightgbm"] = (

                LGBMClassifier(

                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,

                )

            )

        # ----------------------------------------------------
        # CATBOOST
        # ----------------------------------------------------

        if CATBOOST_AVAILABLE:

            models["catboost"] = (

                CatBoostClassifier(

                    iterations=300,
                    learning_rate=0.05,
                    depth=6,
                    verbose=False,

                )

            )

        self.base_models = models

        return models

    # ========================================================
    # SOFT VOTING
    # ========================================================

    def train_soft_voting(
        self,
        X,
        y
    ) -> Dict:

        logger.info(
            "Training soft voting ensemble"
        )

        if not self.base_models:

            self.create_base_models()

        estimators = [

            (name, model)
            for name, model
            in self.base_models.items()

        ]

        ensemble = VotingClassifier(

            estimators=estimators,
            voting="soft",
            n_jobs=-1,

        )

        ensemble.fit(X, y)

        self.ensemble_model = ensemble

        predictions = ensemble.predict(X)

        probabilities = (

            ensemble.predict_proba(X)[:, 1]

        )

        metrics = self.calculate_metrics(

            y,
            predictions,
            probabilities

        )

        return {

            "ensemble_type":
                "soft_voting",

            "metrics":
                metrics,

            "models":
                list(
                    self.base_models.keys()
                ),

        }

    # ========================================================
    # HARD VOTING
    # ========================================================

    def train_hard_voting(
        self,
        X,
        y
    ) -> Dict:

        logger.info(
            "Training hard voting ensemble"
        )

        if not self.base_models:

            self.create_base_models()

        estimators = [

            (name, model)
            for name, model
            in self.base_models.items()

        ]

        ensemble = VotingClassifier(

            estimators=estimators,
            voting="hard",
            n_jobs=-1,

        )

        ensemble.fit(X, y)

        self.ensemble_model = ensemble

        predictions = ensemble.predict(X)

        metrics = self.calculate_metrics(

            y,
            predictions,
            predictions

        )

        return {

            "ensemble_type":
                "hard_voting",

            "metrics":
                metrics,

        }

    # ========================================================
    # STACKING
    # ========================================================

    def train_stacking(
        self,
        X,
        y,
        folds: int = 5,
    ) -> Dict:

        logger.info(
            "Training stacking ensemble"
        )

        if not self.base_models:

            self.create_base_models()

        skf = StratifiedKFold(

            n_splits=folds,
            shuffle=True,
            random_state=42,

        )

        meta_features = np.zeros(

            (len(X), len(self.base_models))

        )

        trained_models = {}

        # ----------------------------------------------------
        # LEVEL 1
        # ----------------------------------------------------

        for idx, (name, model) in enumerate(

            self.base_models.items()

        ):

            logger.info(
                f"Training base model: {name}"
            )

            oof_preds = cross_val_predict(

                clone(model),
                X,
                y,
                cv=skf,
                method="predict_proba",

            )[:, 1]

            meta_features[:, idx] = (
                oof_preds
            )

            model.fit(X, y)

            trained_models[name] = model

        # ----------------------------------------------------
        # META MODEL
        # ----------------------------------------------------

        meta_model = LogisticRegression(
            max_iter=500
        )

        meta_model.fit(
            meta_features,
            y
        )

        meta_predictions = meta_model.predict(
            meta_features
        )

        meta_probabilities = (

            meta_model.predict_proba(
                meta_features
            )[:, 1]

        )

        metrics = self.calculate_metrics(

            y,
            meta_predictions,
            meta_probabilities

        )

        self.base_models = trained_models

        self.meta_model = meta_model

        return {

            "ensemble_type":
                "stacking",

            "metrics":
                metrics,

            "base_models":
                list(
                    trained_models.keys()
                ),

        }

    # ========================================================
    # WEIGHTED ENSEMBLE
    # ========================================================

    def train_weighted_ensemble(
        self,
        X,
        y
    ) -> Dict:

        logger.info(
            "Training weighted ensemble"
        )

        if not self.base_models:

            self.create_base_models()

        model_predictions = {}

        model_scores = {}

        # ----------------------------------------------------
        # TRAIN INDIVIDUALS
        # ----------------------------------------------------

        for name, model in (

            self.base_models.items()

        ):

            model.fit(X, y)

            probabilities = (

                model.predict_proba(X)[:, 1]

            )

            predictions = (
                probabilities > 0.5
            ).astype(int)

            auc = roc_auc_score(
                y,
                probabilities
            )

            model_predictions[name] = (
                probabilities
            )

            model_scores[name] = auc

        # ----------------------------------------------------
        # NORMALIZE WEIGHTS
        # ----------------------------------------------------

        total = sum(
            model_scores.values()
        )

        weights = {

            k: v / total
            for k, v
            in model_scores.items()

        }

        # ----------------------------------------------------
        # WEIGHTED PREDICTION
        # ----------------------------------------------------

        final_probs = np.zeros(len(X))

        for name, probs in (

            model_predictions.items()

        ):

            final_probs += (
                probs * weights[name]
            )

        final_preds = (
            final_probs > 0.5
        ).astype(int)

        metrics = self.calculate_metrics(

            y,
            final_preds,
            final_probs

        )

        self.ensemble_weights = weights

        return {

            "ensemble_type":
                "weighted",

            "metrics":
                metrics,

            "weights":
                weights,

        }

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        X,
        method: str = "stacking",
    ) -> Dict:

        if method == "stacking":

            if self.meta_model is None:

                raise ValueError(
                    "Stacking model not trained"
                )

            meta_features = []

            for model in (
                self.base_models.values()
            ):

                probs = (

                    model.predict_proba(X)[:, 1]

                )

                meta_features.append(probs)

            meta_features = np.column_stack(
                meta_features
            )

            probabilities = (

                self.meta_model.predict_proba(
                    meta_features
                )[:, 1]

            )

        else:

            if self.ensemble_model is None:

                raise ValueError(
                    "Ensemble model not trained"
                )

            probabilities = (

                self.ensemble_model
                .predict_proba(X)[:, 1]

            )

        predictions = (
            probabilities > 0.5
        ).astype(int)

        confidence = self.confidence_score(
            probabilities
        )

        return {

            "predictions":
                predictions,

            "probabilities":
                probabilities,

            "confidence":
                confidence,

        }

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def confidence_score(
        self,
        probabilities
    ):

        confidence = np.abs(
            probabilities - 0.5
        ) * 2

        return confidence

    # ========================================================
    # DIVERSITY
    # ========================================================

    def model_diversity(
        self,
        X
    ) -> Dict:

        predictions = {}

        for name, model in (

            self.base_models.items()

        ):

            predictions[name] = (

                model.predict(X)

            )

        diversity_scores = {}

        names = list(
            predictions.keys()
        )

        for i in range(len(names)):

            for j in range(i + 1, len(names)):

                model_a = names[i]
                model_b = names[j]

                disagreement = np.mean(

                    predictions[model_a]
                    !=
                    predictions[model_b]

                )

                diversity_scores[
                    f"{model_a}_vs_{model_b}"
                ] = round(
                    float(disagreement),
                    4
                )

        return diversity_scores

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def feature_importance(
        self,
        feature_names: List[str]
    ) -> Dict:

        importance_data = {}

        for name, model in (

            self.base_models.items()

        ):

            if hasattr(
                model,
                "feature_importances_"
            ):

                importances = (
                    model.feature_importances_
                )

                importance_data[name] = {

                    feature_names[i]:
                    round(
                        float(importances[i]),
                        4
                    )

                    for i in range(
                        len(feature_names)
                    )

                }

        return importance_data

    # ========================================================
    # METRICS
    # ========================================================

    def calculate_metrics(
        self,
        y_true,
        predictions,
        probabilities,
    ) -> Dict:

        return {

            "accuracy":
                round(

                    accuracy_score(
                        y_true,
                        predictions
                    ),

                    4

                ),

            "precision":
                round(

                    precision_score(
                        y_true,
                        predictions,
                        zero_division=0
                    ),

                    4

                ),

            "recall":
                round(

                    recall_score(
                        y_true,
                        predictions,
                        zero_division=0
                    ),

                    4

                ),

            "f1_score":
                round(

                    f1_score(
                        y_true,
                        predictions,
                        zero_division=0
                    ),

                    4

                ),

            "roc_auc":
                round(

                    roc_auc_score(
                        y_true,
                        probabilities
                    ),

                    4

                ),

        }

    # ========================================================
    # SAVE
    # ========================================================

    def save_ensemble(
        self,
        name: str = "ensemble_model"
    ) -> Dict:

        timestamp = datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S"
        )

        save_path = (

            self.models_dir /
            f"{name}_{timestamp}.pkl"

        )

        data = {

            "base_models":
                self.base_models,

            "meta_model":
                self.meta_model,

            "ensemble_model":
                self.ensemble_model,

            "created_at":
                datetime.utcnow().isoformat(),

        }

        with open(
            save_path,
            "wb"
        ) as f:

            pickle.dump(
                data,
                f
            )

        logger.info(
            f"Ensemble saved: {save_path}"
        )

        return {

            "success": True,

            "path":
                str(save_path),

        }

    # ========================================================
    # LOAD
    # ========================================================

    def load_ensemble(
        self,
        path: str
    ) -> Dict:

        with open(
            path,
            "rb"
        ) as f:

            data = pickle.load(f)

        self.base_models = data[
            "base_models"
        ]

        self.meta_model = data[
            "meta_model"
        ]

        self.ensemble_model = data[
            "ensemble_model"
        ]

        logger.info(
            "Ensemble loaded"
        )

        return {

            "success": True,

            "loaded_path":
                path,

        }

    # ========================================================
    # BENCHMARK
    # ========================================================

    def benchmark_models(
        self,
        X,
        y
    ) -> Dict:

        benchmark_results = {}

        if not self.base_models:

            self.create_base_models()

        for name, model in (

            self.base_models.items()

        ):

            model.fit(X, y)

            preds = model.predict(X)

            probs = (
                model.predict_proba(X)[:, 1]
            )

            benchmark_results[name] = (

                self.calculate_metrics(
                    y,
                    preds,
                    probs
                )

            )

        return benchmark_results


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def train_soft_ensemble(
    X,
    y
):

    engine = EnsembleEngine()

    return engine.train_soft_voting(
        X,
        y
    )


def train_stacking_ensemble(
    X,
    y
):

    engine = EnsembleEngine()

    return engine.train_stacking(
        X,
        y
    )


def train_weighted_ensemble(
    X,
    y
):

    engine = EnsembleEngine()

    return engine.train_weighted_ensemble(
        X,
        y
    )


def benchmark_models(
    X,
    y
):

    engine = EnsembleEngine()

    return engine.benchmark_models(
        X,
        y
    )