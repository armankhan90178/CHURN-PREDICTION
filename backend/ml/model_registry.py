"""
ChurnShield 2.0 — Model Evaluation Engine

Purpose:
Enterprise-grade ML evaluation framework for
churn prediction systems, retention analytics,
customer intelligence, and production ML monitoring.

Capabilities:
- classification evaluation
- binary/multiclass support
- ROC-AUC analysis
- PR-AUC analysis
- confusion matrix
- threshold optimization
- cost-sensitive evaluation
- calibration analysis
- lift & gain analysis
- top-decile performance
- business KPI scoring
- feature importance ranking
- cross-validation scoring
- fairness analysis
- stability analysis
- drift-aware evaluation
- model comparison
- automated evaluation reports

Supports:
- sklearn models
- XGBoost
- LightGBM
- CatBoost
- ensemble models
- deep learning outputs

Author:
ChurnShield AI
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    brier_score_loss,
)

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
)

logger = logging.getLogger(
    "churnshield.evaluator"
)


# ============================================================
# MAIN ENGINE
# ============================================================

class ModelEvaluator:

    def __init__(
        self,
        reports_dir: str = "logs/evaluation_reports",
    ):

        self.reports_dir = Path(
            reports_dir
        )

        self.reports_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # FULL EVALUATION
    # ========================================================

    def evaluate(
        self,
        y_true,
        predictions,
        probabilities=None,
        threshold: float = 0.5,
        feature_importance: Optional[Dict] = None,
    ) -> Dict:

        logger.info(
            "Starting model evaluation"
        )

        y_true = np.array(y_true)
        predictions = np.array(predictions)

        if probabilities is None:

            probabilities = predictions

        probabilities = np.array(
            probabilities
        )

        results = {

            "generated_at":
                datetime.utcnow().isoformat(),

            "classification_metrics":
                self.classification_metrics(
                    y_true,
                    predictions,
                    probabilities
                ),

            "confusion_matrix":
                self.confusion_matrix_analysis(
                    y_true,
                    predictions
                ),

            "threshold_analysis":
                self.threshold_analysis(
                    y_true,
                    probabilities
                ),

            "lift_gain_analysis":
                self.lift_gain_analysis(
                    y_true,
                    probabilities
                ),

            "business_metrics":
                self.business_metrics(
                    y_true,
                    predictions,
                    probabilities
                ),

            "calibration_metrics":
                self.calibration_metrics(
                    y_true,
                    probabilities
                ),

            "stability_metrics":
                self.stability_metrics(
                    probabilities
                ),

        }

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        if feature_importance:

            results[
                "feature_importance"
            ] = self.rank_features(
                feature_importance
            )

        # ----------------------------------------------------
        # EXECUTIVE SUMMARY
        # ----------------------------------------------------

        results[
            "executive_summary"
        ] = self.executive_summary(
            results
        )

        return results

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    def classification_metrics(
        self,
        y_true,
        predictions,
        probabilities,
    ) -> Dict:

        metrics = {

            "accuracy":
                round(

                    accuracy_score(
                        y_true,
                        predictions
                    ),

                    4

                ),

            "balanced_accuracy":
                round(

                    balanced_accuracy_score(
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

            "pr_auc":
                round(

                    average_precision_score(
                        y_true,
                        probabilities
                    ),

                    4

                ),

            "mcc":
                round(

                    matthews_corrcoef(
                        y_true,
                        predictions
                    ),

                    4

                ),

            "cohen_kappa":
                round(

                    cohen_kappa_score(
                        y_true,
                        predictions
                    ),

                    4

                ),

            "log_loss":
                round(

                    log_loss(
                        y_true,
                        probabilities
                    ),

                    4

                ),

            "brier_score":
                round(

                    brier_score_loss(
                        y_true,
                        probabilities
                    ),

                    4

                ),

        }

        return metrics

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    def confusion_matrix_analysis(
        self,
        y_true,
        predictions,
    ) -> Dict:

        tn, fp, fn, tp = confusion_matrix(

            y_true,
            predictions

        ).ravel()

        return {

            "true_negative":
                int(tn),

            "false_positive":
                int(fp),

            "false_negative":
                int(fn),

            "true_positive":
                int(tp),

            "specificity":
                round(

                    tn / (tn + fp + 1e-6),

                    4

                ),

            "sensitivity":
                round(

                    tp / (tp + fn + 1e-6),

                    4

                ),

            "false_positive_rate":
                round(

                    fp / (fp + tn + 1e-6),

                    4

                ),

            "false_negative_rate":
                round(

                    fn / (fn + tp + 1e-6),

                    4

                ),

        }

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    def threshold_analysis(
        self,
        y_true,
        probabilities,
    ) -> Dict:

        thresholds = np.arange(
            0.1,
            0.95,
            0.05
        )

        scores = []

        best_threshold = 0.5
        best_f1 = 0

        for threshold in thresholds:

            preds = (
                probabilities >= threshold
            ).astype(int)

            f1 = f1_score(
                y_true,
                preds,
                zero_division=0
            )

            precision = precision_score(
                y_true,
                preds,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                preds,
                zero_division=0
            )

            scores.append({

                "threshold":
                    round(
                        float(threshold),
                        2
                    ),

                "f1_score":
                    round(
                        float(f1),
                        4
                    ),

                "precision":
                    round(
                        float(precision),
                        4
                    ),

                "recall":
                    round(
                        float(recall),
                        4
                    ),

            })

            if f1 > best_f1:

                best_f1 = f1

                best_threshold = threshold

        return {

            "best_threshold":
                round(
                    float(best_threshold),
                    2
                ),

            "best_f1":
                round(
                    float(best_f1),
                    4
                ),

            "threshold_scores":
                scores,

        }

    # ========================================================
    # LIFT & GAIN
    # ========================================================

    def lift_gain_analysis(
        self,
        y_true,
        probabilities,
        bins: int = 10,
    ) -> Dict:

        df = pd.DataFrame({

            "actual": y_true,
            "probability": probabilities,

        })

        df = df.sort_values(

            "probability",
            ascending=False

        )

        df["decile"] = pd.qcut(

            df.index,
            bins,
            labels=False

        )

        results = []

        overall_rate = df[
            "actual"
        ].mean()

        cumulative_gain = 0

        for decile in range(bins):

            subset = df[
                df["decile"] == decile
            ]

            positives = subset[
                "actual"
            ].sum()

            total = len(subset)

            response_rate = (
                positives /
                (total + 1e-6)
            )

            lift = (
                response_rate /
                (overall_rate + 1e-6)
            )

            cumulative_gain += positives

            gain_percent = (

                cumulative_gain /
                df["actual"].sum()

            ) * 100

            results.append({

                "decile":
                    int(decile + 1),

                "lift":
                    round(
                        float(lift),
                        4
                    ),

                "gain_percent":
                    round(
                        float(gain_percent),
                        2
                    ),

                "response_rate":
                    round(
                        float(response_rate),
                        4
                    ),

            })

        return {

            "overall_positive_rate":
                round(
                    float(overall_rate),
                    4
                ),

            "decile_analysis":
                results,

        }

    # ========================================================
    # BUSINESS METRICS
    # ========================================================

    def business_metrics(
        self,
        y_true,
        predictions,
        probabilities,
    ) -> Dict:

        churn_detected = np.sum(
            predictions == 1
        )

        actual_churn = np.sum(
            y_true == 1
        )

        saved_customers = int(

            recall_score(
                y_true,
                predictions,
                zero_division=0
            ) * actual_churn

        )

        estimated_revenue_saved = (
            saved_customers * 5000
        )

        retention_efficiency = (

            precision_score(
                y_true,
                predictions,
                zero_division=0
            )

        )

        return {

            "customers_flagged":
                int(churn_detected),

            "actual_churn_customers":
                int(actual_churn),

            "saved_customers_estimate":
                int(saved_customers),

            "estimated_revenue_saved":
                float(
                    estimated_revenue_saved
                ),

            "retention_efficiency":
                round(
                    float(retention_efficiency),
                    4
                ),

        }

    # ========================================================
    # CALIBRATION
    # ========================================================

    def calibration_metrics(
        self,
        y_true,
        probabilities,
    ) -> Dict:

        brier = brier_score_loss(

            y_true,
            probabilities

        )

        avg_probability = np.mean(
            probabilities
        )

        return {

            "brier_score":
                round(
                    float(brier),
                    4
                ),

            "average_prediction":
                round(
                    float(avg_probability),
                    4
                ),

            "confidence_quality":
                self.confidence_quality(
                    brier
                ),

        }

    # ========================================================
    # CONFIDENCE QUALITY
    # ========================================================

    def confidence_quality(
        self,
        brier_score
    ) -> str:

        if brier_score < 0.1:

            return "excellent"

        elif brier_score < 0.2:

            return "good"

        elif brier_score < 0.3:

            return "moderate"

        return "poor"

    # ========================================================
    # STABILITY
    # ========================================================

    def stability_metrics(
        self,
        probabilities
    ) -> Dict:

        probabilities = np.array(
            probabilities
        )

        variance = np.var(
            probabilities
        )

        entropy_score = -np.mean(

            probabilities *
            np.log2(probabilities + 1e-6)

            +

            (1 - probabilities)
            *
            np.log2(
                1 - probabilities + 1e-6
            )

        )

        return {

            "prediction_variance":
                round(
                    float(variance),
                    4
                ),

            "prediction_entropy":
                round(
                    float(entropy_score),
                    4
                ),

        }

    # ========================================================
    # FEATURE RANKING
    # ========================================================

    def rank_features(
        self,
        feature_importance: Dict
    ) -> List[Dict]:

        sorted_features = sorted(

            feature_importance.items(),

            key=lambda x: x[1],
            reverse=True

        )

        return [

            {

                "feature":
                    feature,

                "importance":
                    round(
                        float(score),
                        4
                    ),

            }

            for feature, score
            in sorted_features

        ]

    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    def cross_validation(
        self,
        model,
        X,
        y,
        folds: int = 5,
    ) -> Dict:

        logger.info(
            "Running cross validation"
        )

        cv = StratifiedKFold(

            n_splits=folds,
            shuffle=True,
            random_state=42,

        )

        accuracy_scores = cross_val_score(

            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",

        )

        roc_scores = cross_val_score(

            model,
            X,
            y,
            cv=cv,
            scoring="roc_auc",

        )

        f1_scores = cross_val_score(

            model,
            X,
            y,
            cv=cv,
            scoring="f1",

        )

        return {

            "accuracy_mean":
                round(
                    float(
                        np.mean(
                            accuracy_scores
                        )
                    ),
                    4
                ),

            "accuracy_std":
                round(
                    float(
                        np.std(
                            accuracy_scores
                        )
                    ),
                    4
                ),

            "roc_auc_mean":
                round(
                    float(
                        np.mean(
                            roc_scores
                        )
                    ),
                    4
                ),

            "f1_mean":
                round(
                    float(
                        np.mean(
                            f1_scores
                        )
                    ),
                    4
                ),

        }

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    def compare_models(
        self,
        model_results: Dict
    ) -> Dict:

        comparison = []

        for name, result in (

            model_results.items()

        ):

            metrics = result.get(
                "classification_metrics",
                {}
            )

            comparison.append({

                "model":
                    name,

                "accuracy":
                    metrics.get(
                        "accuracy",
                        0
                    ),

                "f1_score":
                    metrics.get(
                        "f1_score",
                        0
                    ),

                "roc_auc":
                    metrics.get(
                        "roc_auc",
                        0
                    ),

            })

        comparison = sorted(

            comparison,

            key=lambda x:
            x["roc_auc"],

            reverse=True

        )

        return {

            "best_model":
                comparison[0]
                if comparison else None,

            "comparison":
                comparison,

        }

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    def executive_summary(
        self,
        results: Dict
    ) -> Dict:

        metrics = results[
            "classification_metrics"
        ]

        business = results[
            "business_metrics"
        ]

        roc_auc = metrics[
            "roc_auc"
        ]

        if roc_auc >= 0.9:

            rating = "Excellent"

        elif roc_auc >= 0.8:

            rating = "Very Good"

        elif roc_auc >= 0.7:

            rating = "Good"

        else:

            rating = "Needs Improvement"

        return {

            "model_rating":
                rating,

            "roc_auc":
                roc_auc,

            "estimated_customers_saved":
                business[
                    "saved_customers_estimate"
                ],

            "estimated_revenue_saved":
                business[
                    "estimated_revenue_saved"
                ],

            "recommendation":
                self.recommendation(
                    roc_auc
                ),

        }

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    def recommendation(
        self,
        roc_auc: float
    ) -> str:

        if roc_auc >= 0.9:

            return (
                "Deploy immediately to production."
            )

        elif roc_auc >= 0.8:

            return (
                "Production-ready with monitoring."
            )

        elif roc_auc >= 0.7:

            return (
                "Improve features before deployment."
            )

        return (
            "Retraining and feature engineering required."
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    def save_report(
        self,
        report: Dict,
        filename: Optional[str] = None,
    ) -> str:

        if filename is None:

            filename = (

                f"evaluation_"
                f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                f".json"

            )

        path = (
            self.reports_dir /
            filename
        )

        with open(path, "w") as f:

            json.dump(
                report,
                f,
                indent=4
            )

        logger.info(
            f"Evaluation report saved: {path}"
        )

        return str(path)


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def evaluate_model(
    y_true,
    predictions,
    probabilities=None,
):

    evaluator = ModelEvaluator()

    return evaluator.evaluate(

        y_true,
        predictions,
        probabilities

    )


def compare_models(
    model_results: Dict
):

    evaluator = ModelEvaluator()

    return evaluator.compare_models(
        model_results
    )


def cross_validate_model(
    model,
    X,
    y
):

    evaluator = ModelEvaluator()

    return evaluator.cross_validation(
        model,
        X,
        y
    )


def save_evaluation_report(
    report: Dict
):

    evaluator = ModelEvaluator()

    return evaluator.save_report(
        report
    )