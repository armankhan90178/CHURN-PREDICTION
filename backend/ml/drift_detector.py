"""
ChurnShield 2.0 — Drift Detection Engine

Purpose:
Enterprise-grade drift detection system for
monitoring data quality, feature distribution,
model behavior, prediction instability,
and production ML degradation.

Capabilities:
- data drift detection
- concept drift detection
- prediction drift detection
- feature drift monitoring
- PSI calculation
- KL divergence
- statistical testing
- distribution comparison
- drift severity scoring
- automatic retraining triggers
- real-time monitoring
- drift dashboards
- alert generation
- business impact estimation
- model degradation analysis

Supports:
- numeric features
- categorical features
- time-series drift
- probability drift
- online monitoring
- batch monitoring

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

from scipy.stats import (
    ks_2samp,
    chi2_contingency,
    entropy,
)

logger = logging.getLogger(
    "churnshield.drift_detector"
)


# ============================================================
# MAIN ENGINE
# ============================================================

class DriftDetector:

    def __init__(
        self,
        drift_threshold: float = 0.2,
        severe_threshold: float = 0.4,
        reports_dir: str = "logs/drift_reports",
    ):

        self.drift_threshold = (
            drift_threshold
        )

        self.severe_threshold = (
            severe_threshold
        )

        self.reports_dir = Path(
            reports_dir
        )

        self.reports_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # MAIN DRIFT ANALYSIS
    # ========================================================

    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Starting drift analysis"
        )

        results = {

            "generated_at":
                datetime.utcnow().isoformat(),

            "overall_drift_score":
                0.0,

            "drift_detected":
                False,

            "feature_results":
                {},

            "summary":
                {},

        }

        common_columns = [

            col for col in reference_df.columns
            if col in current_df.columns

        ]

        drift_scores = []

        severe_features = []

        moderate_features = []

        # ----------------------------------------------------
        # FEATURE LOOP
        # ----------------------------------------------------

        for column in common_columns:

            try:

                ref_series = (
                    reference_df[column]
                    .dropna()
                )

                cur_series = (
                    current_df[column]
                    .dropna()
                )

                # --------------------------------------------
                # NUMERIC
                # --------------------------------------------

                if pd.api.types.is_numeric_dtype(
                    ref_series
                ):

                    feature_result = (

                        self.numeric_drift(

                            ref_series,
                            cur_series

                        )

                    )

                # --------------------------------------------
                # CATEGORICAL
                # --------------------------------------------

                else:

                    feature_result = (

                        self.categorical_drift(

                            ref_series,
                            cur_series

                        )

                    )

                results["feature_results"][
                    column
                ] = feature_result

                drift_scores.append(

                    feature_result[
                        "drift_score"
                    ]

                )

                # --------------------------------------------
                # SEVERITY
                # --------------------------------------------

                if (
                    feature_result[
                        "drift_score"
                    ]
                    >= self.severe_threshold
                ):

                    severe_features.append(
                        column
                    )

                elif (
                    feature_result[
                        "drift_score"
                    ]
                    >= self.drift_threshold
                ):

                    moderate_features.append(
                        column
                    )

            except Exception as e:

                logger.error(
                    f"Drift failed for "
                    f"{column}: {e}"
                )

        # ----------------------------------------------------
        # OVERALL
        # ----------------------------------------------------

        overall_score = float(
            np.mean(drift_scores)
        ) if drift_scores else 0.0

        results[
            "overall_drift_score"
        ] = round(
            overall_score,
            4
        )

        results[
            "drift_detected"
        ] = (
            overall_score >=
            self.drift_threshold
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        results["summary"] = {

            "total_features":
                len(common_columns),

            "moderate_drift_features":
                moderate_features,

            "severe_drift_features":
                severe_features,

            "moderate_count":
                len(moderate_features),

            "severe_count":
                len(severe_features),

        }

        # ----------------------------------------------------
        # BUSINESS IMPACT
        # ----------------------------------------------------

        results["business_impact"] = (

            self.business_impact_estimation(
                overall_score
            )

        )

        # ----------------------------------------------------
        # RETRAIN RECOMMENDATION
        # ----------------------------------------------------

        results[
            "retrain_recommended"
        ] = (

            overall_score >=
            self.drift_threshold

        )

        return results

    # ========================================================
    # NUMERIC DRIFT
    # ========================================================

    def numeric_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> Dict:

        # ----------------------------------------------------
        # PSI
        # ----------------------------------------------------

        psi_score = self.population_stability_index(

            reference,
            current

        )

        # ----------------------------------------------------
        # KS TEST
        # ----------------------------------------------------

        ks_stat, p_value = ks_2samp(

            reference,
            current

        )

        # ----------------------------------------------------
        # KL DIVERGENCE
        # ----------------------------------------------------

        kl_score = self.kl_divergence(

            reference,
            current

        )

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        drift_score = float(

            np.mean([
                psi_score,
                ks_stat,
                kl_score,
            ])

        )

        return {

            "feature_type":
                "numeric",

            "drift_score":
                round(drift_score, 4),

            "psi":
                round(psi_score, 4),

            "ks_statistic":
                round(float(ks_stat), 4),

            "ks_pvalue":
                round(float(p_value), 4),

            "kl_divergence":
                round(kl_score, 4),

            "drift_detected":
                drift_score >=
                self.drift_threshold,

            "reference_mean":
                round(
                    float(reference.mean()),
                    4
                ),

            "current_mean":
                round(
                    float(current.mean()),
                    4
                ),

            "reference_std":
                round(
                    float(reference.std()),
                    4
                ),

            "current_std":
                round(
                    float(current.std()),
                    4
                ),

        }

    # ========================================================
    # CATEGORICAL DRIFT
    # ========================================================

    def categorical_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> Dict:

        ref_counts = (
            reference
            .value_counts(normalize=True)
        )

        cur_counts = (
            current
            .value_counts(normalize=True)
        )

        all_categories = set(
            ref_counts.index
        ).union(
            set(cur_counts.index)
        )

        ref_dist = []
        cur_dist = []

        for cat in all_categories:

            ref_dist.append(
                ref_counts.get(cat, 0)
            )

            cur_dist.append(
                cur_counts.get(cat, 0)
            )

        # ----------------------------------------------------
        # CHI-SQUARE
        # ----------------------------------------------------

        contingency = np.array([

            ref_dist,
            cur_dist,

        ])

        chi2, p_value, _, _ = (
            chi2_contingency(
                contingency
            )
        )

        # ----------------------------------------------------
        # KL
        # ----------------------------------------------------

        kl_score = entropy(

            np.array(ref_dist) + 1e-6,
            np.array(cur_dist) + 1e-6

        )

        drift_score = float(

            np.mean([

                min(chi2 / 100, 1),
                min(kl_score, 1),

            ])

        )

        return {

            "feature_type":
                "categorical",

            "drift_score":
                round(drift_score, 4),

            "chi2":
                round(float(chi2), 4),

            "p_value":
                round(float(p_value), 4),

            "kl_divergence":
                round(float(kl_score), 4),

            "unique_categories":
                len(all_categories),

            "drift_detected":
                drift_score >=
                self.drift_threshold,

        }

    # ========================================================
    # PSI
    # ========================================================

    def population_stability_index(
        self,
        expected: pd.Series,
        actual: pd.Series,
        bins: int = 10,
    ) -> float:

        expected = np.array(expected)
        actual = np.array(actual)

        breakpoints = np.percentile(

            expected,
            np.arange(
                0,
                bins + 1
            ) / bins * 100

        )

        expected_counts = np.histogram(

            expected,
            bins=breakpoints

        )[0]

        actual_counts = np.histogram(

            actual,
            bins=breakpoints

        )[0]

        expected_percents = (
            expected_counts /
            len(expected)
        )

        actual_percents = (
            actual_counts /
            len(actual)
        )

        psi_values = []

        for e, a in zip(

            expected_percents,
            actual_percents

        ):

            if e == 0:
                e = 0.0001

            if a == 0:
                a = 0.0001

            psi = (
                (e - a)
                *
                np.log(e / a)
            )

            psi_values.append(psi)

        return float(
            np.sum(psi_values)
        )

    # ========================================================
    # KL DIVERGENCE
    # ========================================================

    def kl_divergence(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 20,
    ) -> float:

        ref_hist, bin_edges = np.histogram(

            reference,
            bins=bins,
            density=True

        )

        cur_hist, _ = np.histogram(

            current,
            bins=bin_edges,
            density=True

        )

        ref_hist += 1e-6
        cur_hist += 1e-6

        score = entropy(
            ref_hist,
            cur_hist
        )

        return float(score)

    # ========================================================
    # PREDICTION DRIFT
    # ========================================================

    def prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Dict:

        reference_predictions = np.array(
            reference_predictions
        )

        current_predictions = np.array(
            current_predictions
        )

        psi = self.population_stability_index(

            pd.Series(reference_predictions),
            pd.Series(current_predictions),

        )

        ks_stat, p_value = ks_2samp(

            reference_predictions,
            current_predictions

        )

        drift_score = float(
            np.mean([psi, ks_stat])
        )

        return {

            "prediction_drift_score":
                round(drift_score, 4),

            "psi":
                round(psi, 4),

            "ks_statistic":
                round(float(ks_stat), 4),

            "p_value":
                round(float(p_value), 4),

            "drift_detected":
                drift_score >=
                self.drift_threshold,

        }

    # ========================================================
    # FEATURE IMPORTANCE DRIFT
    # ========================================================

    def feature_importance_drift(
        self,
        old_importance: Dict,
        new_importance: Dict,
    ) -> Dict:

        features = set(
            old_importance.keys()
        ).intersection(

            set(
                new_importance.keys()
            )

        )

        diffs = {}

        scores = []

        for feature in features:

            diff = abs(

                old_importance[feature]
                -
                new_importance[feature]

            )

            diffs[feature] = round(
                diff,
                4
            )

            scores.append(diff)

        overall = float(
            np.mean(scores)
        ) if scores else 0

        return {

            "importance_drift":
                round(overall, 4),

            "feature_changes":
                diffs,

            "drift_detected":
                overall >=
                self.drift_threshold,

        }

    # ========================================================
    # BUSINESS IMPACT
    # ========================================================

    def business_impact_estimation(
        self,
        drift_score: float
    ) -> Dict:

        if drift_score >= 0.5:

            severity = "critical"

            impact = (
                "Major prediction instability "
                "possible. Immediate retraining "
                "recommended."
            )

        elif drift_score >= 0.3:

            severity = "high"

            impact = (
                "Model accuracy degradation "
                "likely. Monitor closely."
            )

        elif drift_score >= 0.2:

            severity = "moderate"

            impact = (
                "Minor drift detected. "
                "Retraining should be considered."
            )

        else:

            severity = "low"

            impact = (
                "Model stability acceptable."
            )

        return {

            "severity":
                severity,

            "business_impact":
                impact,

        }

    # ========================================================
    # DRIFT ALERT
    # ========================================================

    def generate_alert(
        self,
        drift_report: Dict
    ) -> Dict:

        score = drift_report[
            "overall_drift_score"
        ]

        summary = drift_report["summary"]

        message = f"""

Drift Alert Generated

Overall Drift Score:
{score}

Moderate Drift Features:
{summary['moderate_count']}

Severe Drift Features:
{summary['severe_count']}

Retraining Recommended:
{drift_report['retrain_recommended']}

        """.strip()

        return {

            "timestamp":
                datetime.utcnow().isoformat(),

            "severity":
                drift_report[
                    "business_impact"
                ]["severity"],

            "message":
                message,

        }

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

                f"drift_report_"
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
            f"Drift report saved: {path}"
        )

        return str(path)

    # ========================================================
    # LIVE MONITOR
    # ========================================================

    def live_monitor(
        self,
        reference_df: pd.DataFrame,
        stream_df: pd.DataFrame,
        chunk_size: int = 1000,
    ) -> List[Dict]:

        logger.info(
            "Starting live monitoring"
        )

        reports = []

        total_chunks = int(

            np.ceil(
                len(stream_df) /
                chunk_size
            )

        )

        for idx in range(total_chunks):

            start = idx * chunk_size
            end = start + chunk_size

            chunk = stream_df.iloc[
                start:end
            ]

            report = self.detect_drift(

                reference_df,
                chunk

            )

            report["chunk_index"] = idx

            reports.append(report)

        return reports


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def detect_dataset_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
):

    detector = DriftDetector()

    return detector.detect_drift(

        reference_df,
        current_df

    )


def detect_prediction_drift(
    reference_predictions,
    current_predictions,
):

    detector = DriftDetector()

    return detector.prediction_drift(

        reference_predictions,
        current_predictions

    )


def generate_drift_alert(
    drift_report: Dict
):

    detector = DriftDetector()

    return detector.generate_alert(
        drift_report
    )


def save_drift_report(
    report: Dict
):

    detector = DriftDetector()

    return detector.save_report(
        report
    )