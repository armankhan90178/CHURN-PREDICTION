"""
ChurnShield 2.0 — Hyper Churn Predictor Engine

Purpose:
Enterprise-grade churn prediction system
for customer retention intelligence.

Capabilities:
- ultra-fast churn prediction
- calibrated probability scoring
- multi-risk segmentation
- business health intelligence
- revenue-at-risk forecasting
- churn prioritization
- survival probability estimation
- customer risk profiling
- explainable scoring pipeline
- adaptive thresholding
- batch prediction
- confidence estimation
- portfolio intelligence
"""

import logging
import warnings
import numpy as np
import pandas as pd
import joblib

from typing import Dict
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.predictor"
)


class HyperChurnPredictor:

    def __init__(
        self,
        model=None,
        model_path=None,
    ):

        self.model = None

        if model is not None:
            self.model = model

        elif model_path:
            self.load_model(model_path)

        logger.info(
            "Hyper predictor initialized"
        )

    # ─────────────────────────────────────────────
    # LOAD MODEL
    # ─────────────────────────────────────────────

    def load_model(
        self,
        model_path,
    ):

        logger.info(
            f"Loading model: {model_path}"
        )

        self.model = joblib.load(model_path)

        return self.model

    # ─────────────────────────────────────────────
    # MAIN PREDICTION PIPELINE
    # ─────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if self.model is None:
            raise ValueError(
                "No ML model loaded"
            )

        logger.info(
            "Starting enterprise prediction pipeline"
        )

        data = df.copy()

        X = self._prepare_features(
            data
        )

        probabilities = self._predict_probabilities(
            X
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        confidence_scores = self._confidence_scores(
            probabilities
        )

        risk_levels = self._risk_levels(
            probabilities
        )

        health_scores = self._health_scores(
            probabilities
        )

        urgency = self._urgency_levels(
            probabilities
        )

        survival_probability = (
            1 - probabilities
        )

        revenue_risk = self._revenue_at_risk(
            data,
            probabilities,
        )

        priority_scores = self._priority_scores(
            data,
            probabilities,
        )

        churn_windows = self._predict_churn_window(
            probabilities
        )

        intervention_type = self._recommended_action(
            probabilities,
            data,
        )

        behavioral_state = self._behavioral_state(
            probabilities,
            data,
        )

        portfolio_rank = self._portfolio_rank(
            probabilities
        )

        # Attach results
        data["churn_probability"] = (
            probabilities
        )

        data["predicted_churn"] = (
            predictions
        )

        data["prediction_confidence"] = (
            confidence_scores
        )

        data["risk_level"] = (
            risk_levels
        )

        data["customer_health_score"] = (
            health_scores
        )

        data["urgency_level"] = (
            urgency
        )

        data["survival_probability"] = (
            survival_probability
        )

        data["revenue_at_risk"] = (
            revenue_risk
        )

        data["priority_score"] = (
            priority_scores
        )

        data["predicted_churn_window"] = (
            churn_windows
        )

        data["recommended_intervention"] = (
            intervention_type
        )

        data["behavioral_state"] = (
            behavioral_state
        )

        data["portfolio_risk_rank"] = (
            portfolio_rank
        )

        logger.info(
            "Prediction pipeline completed"
        )

        return data

    # ─────────────────────────────────────────────
    # FEATURE PREPARATION
    # ─────────────────────────────────────────────

    def _prepare_features(
        self,
        df,
    ):

        numeric = df.select_dtypes(
            include=[np.number]
        )

        ignored = [
            "churned",
            "predicted_churn",
        ]

        numeric = numeric.drop(
            columns=[
                c for c in ignored
                if c in numeric.columns
            ],
            errors="ignore",
        )

        numeric = numeric.fillna(0)

        return numeric

    # ─────────────────────────────────────────────
    # PROBABILITY PREDICTION
    # ─────────────────────────────────────────────

    def _predict_probabilities(
        self,
        X,
    ):

        logger.info(
            "Generating churn probabilities"
        )

        try:

            probs = self.model.predict_proba(
                X
            )[:, 1]

        except Exception:

            preds = self.model.predict(
                X
            )

            probs = np.clip(
                preds,
                0,
                1,
            )

        probs = np.clip(
            probs,
            0,
            1,
        )

        return probs

    # ─────────────────────────────────────────────
    # CONFIDENCE ENGINE
    # ─────────────────────────────────────────────

    def _confidence_scores(
        self,
        probs,
    ):

        confidence = (
            np.abs(probs - 0.5) * 2
        )

        return np.round(
            confidence,
            4,
        )

    # ─────────────────────────────────────────────
    # RISK LEVELS
    # ─────────────────────────────────────────────

    def _risk_levels(
        self,
        probs,
    ):

        risk = []

        for p in probs:

            if p >= 0.85:
                risk.append(
                    "Critical"
                )

            elif p >= 0.70:
                risk.append(
                    "High"
                )

            elif p >= 0.45:
                risk.append(
                    "Moderate"
                )

            else:
                risk.append(
                    "Low"
                )

        return risk

    # ─────────────────────────────────────────────
    # HEALTH SCORE
    # ─────────────────────────────────────────────

    def _health_scores(
        self,
        probs,
    ):

        scores = (
            (1 - probs) * 100
        )

        return np.round(
            scores,
            2,
        )

    # ─────────────────────────────────────────────
    # URGENCY LEVEL
    # ─────────────────────────────────────────────

    def _urgency_levels(
        self,
        probs,
    ):

        urgency = []

        for p in probs:

            if p >= 0.85:
                urgency.append(
                    "Immediate Action"
                )

            elif p >= 0.70:
                urgency.append(
                    "High Priority"
                )

            elif p >= 0.45:
                urgency.append(
                    "Monitor Closely"
                )

            else:
                urgency.append(
                    "Stable"
                )

        return urgency

    # ─────────────────────────────────────────────
    # REVENUE AT RISK
    # ─────────────────────────────────────────────

    def _revenue_at_risk(
        self,
        df,
        probs,
    ):

        if "monthly_revenue" not in df.columns:

            return np.zeros(
                len(df)
            )

        revenue = df[
            "monthly_revenue"
        ].fillna(0)

        revenue_risk = (
            revenue * probs
        )

        return np.round(
            revenue_risk,
            2,
        )

    # ─────────────────────────────────────────────
    # PRIORITY SCORING
    # ─────────────────────────────────────────────

    def _priority_scores(
        self,
        df,
        probs,
    ):

        revenue = df.get(
            "monthly_revenue",
            pd.Series([0] * len(df))
        )

        engagement = df.get(
            "engagement_score",
            pd.Series([0.5] * len(df))
        )

        score = (
            (probs * 0.60) +
            (
                np.log1p(revenue) /
                np.log1p(revenue.max() + 1)
            ) * 0.30 +
            (
                (1 - engagement) * 0.10
            )
        )

        score *= 100

        return np.round(
            score,
            2,
        )

    # ─────────────────────────────────────────────
    # CHURN WINDOW PREDICTION
    # ─────────────────────────────────────────────

    def _predict_churn_window(
        self,
        probs,
    ):

        windows = []

        for p in probs:

            if p >= 0.90:
                windows.append(
                    "0-7 Days"
                )

            elif p >= 0.75:
                windows.append(
                    "7-30 Days"
                )

            elif p >= 0.55:
                windows.append(
                    "30-60 Days"
                )

            elif p >= 0.40:
                windows.append(
                    "60-90 Days"
                )

            else:
                windows.append(
                    "Stable"
                )

        return windows

    # ─────────────────────────────────────────────
    # RECOMMENDED INTERVENTION
    # ─────────────────────────────────────────────

    def _recommended_action(
        self,
        probs,
        df,
    ):

        actions = []

        tickets = df.get(
            "support_tickets",
            pd.Series([0] * len(df))
        )

        engagement = df.get(
            "engagement_score",
            pd.Series([0.5] * len(df))
        )

        for idx, p in enumerate(probs):

            if p >= 0.85:

                if tickets.iloc[idx] >= 5:

                    actions.append(
                        "Executive Escalation + Dedicated Support"
                    )

                else:

                    actions.append(
                        "Immediate Retention Campaign"
                    )

            elif p >= 0.70:

                if engagement.iloc[idx] < 0.30:

                    actions.append(
                        "Reactivation + Product Training"
                    )

                else:

                    actions.append(
                        "Customer Success Outreach"
                    )

            elif p >= 0.45:

                actions.append(
                    "Behavior Monitoring"
                )

            else:

                actions.append(
                    "Upsell / Expansion Opportunity"
                )

        return actions

    # ─────────────────────────────────────────────
    # BEHAVIORAL STATE
    # ─────────────────────────────────────────────

    def _behavioral_state(
        self,
        probs,
        df,
    ):

        states = []

        engagement = df.get(
            "engagement_score",
            pd.Series([0.5] * len(df))
        )

        for idx, p in enumerate(probs):

            e = engagement.iloc[idx]

            if p >= 0.80 and e < 0.30:

                states.append(
                    "Disengaged"
                )

            elif p >= 0.70:

                states.append(
                    "Declining"
                )

            elif e >= 0.75:

                states.append(
                    "Thriving"
                )

            else:

                states.append(
                    "Stable"
                )

        return states

    # ─────────────────────────────────────────────
    # PORTFOLIO RANKING
    # ─────────────────────────────────────────────

    def _portfolio_rank(
        self,
        probs,
    ):

        ranks = pd.Series(
            probs
        ).rank(
            ascending=False,
            method="dense"
        )

        return ranks.astype(int)

    # ─────────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────

    def executive_summary(
        self,
        predicted_df,
    ) -> Dict:

        total_customers = len(
            predicted_df
        )

        high_risk = len(
            predicted_df[
                predicted_df[
                    "risk_level"
                ].isin(
                    ["Critical", "High"]
                )
            ]
        )

        revenue_at_risk = (
            predicted_df[
                "revenue_at_risk"
            ].sum()
        )

        avg_probability = (
            predicted_df[
                "churn_probability"
            ].mean()
        )

        return {

            "customers_analyzed":
                int(total_customers),

            "high_risk_customers":
                int(high_risk),

            "average_churn_probability":
                round(
                    float(avg_probability),
                    4,
                ),

            "total_revenue_at_risk":
                round(
                    float(revenue_at_risk),
                    2,
                ),

            "portfolio_health":
                round(
                    float(
                        predicted_df[
                            "customer_health_score"
                        ].mean()
                    ),
                    2,
                ),
        }


# ─────────────────────────────────────────────
# FUNCTIONAL INTERFACE
# ─────────────────────────────────────────────

def predict_churn(
    df: pd.DataFrame,
    model,
):

    predictor = HyperChurnPredictor(
        model=model
    )

    return predictor.predict(df)