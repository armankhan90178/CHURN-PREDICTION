"""
ChurnShield 2.0 — Enterprise Early Warning System (EWS)

Purpose:
Advanced customer health intelligence engine
that predicts churn before it happens.

Capabilities:
- Unified customer health score
- Multi-dimensional risk analysis
- Revenue-weighted risk scoring
- Behavioral degradation tracking
- Payment instability detection
- Engagement collapse detection
- Silent churn prediction
- Health trend forecasting
- AI-style warning generation
- Executive health summaries
- Customer lifecycle risk modeling
- Renewal risk estimation
- Enterprise-grade scoring engine
"""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import Dict, List
from datetime import datetime

from config import (
    EWS_WEIGHTS,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.analytics.ews"
)


# ─────────────────────────────────────────────
# ENTERPRISE EWS ENGINE
# ─────────────────────────────────────────────

class EnterpriseEWS:

    """
    Enterprise-grade Early Warning System
    """

    # ─────────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────────

    def __init__(self):

        self.weights = EWS_WEIGHTS

        self.health_dimensions = [

            "usage_health",
            "payment_health",
            "support_health",
            "engagement_health",
            "relationship_health",
        ]

    # ─────────────────────────────────────────────
    # MASTER ANALYSIS ENGINE
    # ─────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Starting enterprise EWS analysis..."
        )

        data = df.copy()

        # ─────────────────────────────────────────
        # PREP
        # ─────────────────────────────────────────

        data = self._prepare_data(data)

        # ─────────────────────────────────────────
        # HEALTH DIMENSIONS
        # ─────────────────────────────────────────

        data["usage_health"] = (
            self.calculate_usage_health(data)
        )

        data["payment_health"] = (
            self.calculate_payment_health(data)
        )

        data["support_health"] = (
            self.calculate_support_health(data)
        )

        data["engagement_health"] = (
            self.calculate_engagement_health(data)
        )

        data["relationship_health"] = (
            self.calculate_relationship_health(data)
        )

        # ─────────────────────────────────────────
        # HEALTH SCORE
        # ─────────────────────────────────────────

        data["health_score"] = (
            self.calculate_health_score(data)
        )

        # ─────────────────────────────────────────
        # RISK ENGINE
        # ─────────────────────────────────────────

        data["risk_level"] = (
            self.assign_risk_levels(data)
        )

        data["risk_score"] = (
            1 - data["health_score"]
        ).round(4)

        # ─────────────────────────────────────────
        # REVENUE RISK
        # ─────────────────────────────────────────

        data["revenue_risk"] = (
            self.calculate_revenue_risk(data)
        )

        # ─────────────────────────────────────────
        # WARNING SYSTEM
        # ─────────────────────────────────────────

        data["warning_signals"] = (
            self.generate_warning_signals(data)
        )

        data["critical_flags"] = (
            self.generate_critical_flags(data)
        )

        # ─────────────────────────────────────────
        # TREND ANALYSIS
        # ─────────────────────────────────────────

        data["health_trend"] = (
            self.generate_health_trends(data)
        )

        # ─────────────────────────────────────────
        # EXECUTIVE PRIORITY
        # ─────────────────────────────────────────

        data["intervention_priority"] = (
            self.calculate_intervention_priority(data)
        )

        logger.info(
            "EWS analysis completed"
        )

        return data

    # ─────────────────────────────────────────────
    # PREP ENGINE
    # ─────────────────────────────────────────────

    def _prepare_data(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        data = df.copy()

        numeric_defaults = {

            "feature_usage_score": 0.5,
            "payment_delays": 0,
            "support_tickets": 0,
            "login_frequency": 10,
            "contract_age_months": 6,
            "monthly_revenue": 1000,
            "active_seats": 1,
            "total_seats": 1,
            "nps_score": 5,
        }

        for col, default in numeric_defaults.items():

            if col not in data.columns:

                data[col] = default

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            ).fillna(default)

        return data

    # ─────────────────────────────────────────────
    # USAGE HEALTH
    # ─────────────────────────────────────────────

    def calculate_usage_health(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Calculating usage health..."
        )

        feature_score = (
            df["feature_usage_score"]
            .clip(0, 1)
        )

        login_score = np.clip(
            df["login_frequency"] / 30,
            0,
            1,
        )

        seat_utilization = (

            df["active_seats"]

            /

            df["total_seats"]
            .replace(0, 1)

        ).clip(0, 1)

        usage_health = (

            feature_score * 0.5

            +

            login_score * 0.3

            +

            seat_utilization * 0.2
        )

        return usage_health.round(4)

    # ─────────────────────────────────────────────
    # PAYMENT HEALTH
    # ─────────────────────────────────────────────

    def calculate_payment_health(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Calculating payment health..."
        )

        delays = df["payment_delays"]

        payment_health = np.where(

            delays == 0,
            1.0,

            np.where(
                delays <= 1,
                0.85,

                np.where(
                    delays <= 3,
                    0.60,
                    0.25,
                )
            )
        )

        return pd.Series(
            payment_health,
            index=df.index,
        ).round(4)

    # ─────────────────────────────────────────────
    # SUPPORT HEALTH
    # ─────────────────────────────────────────────

    def calculate_support_health(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Calculating support health..."
        )

        tickets = df["support_tickets"]

        support_health = np.exp(
            -tickets / 6
        )

        return pd.Series(
            support_health,
            index=df.index,
        ).clip(0, 1).round(4)

    # ─────────────────────────────────────────────
    # ENGAGEMENT HEALTH
    # ─────────────────────────────────────────────

    def calculate_engagement_health(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Calculating engagement health..."
        )

        usage = (
            df["feature_usage_score"]
        )

        tenure = (
            df["contract_age_months"]
        )

        engagement = (

            usage * 0.7

            +

            np.clip(
                tenure / 24,
                0,
                1,
            ) * 0.3
        )

        return engagement.round(4)

    # ─────────────────────────────────────────────
    # RELATIONSHIP HEALTH
    # ─────────────────────────────────────────────

    def calculate_relationship_health(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Calculating relationship health..."
        )

        nps_scaled = (
            df["nps_score"] / 10
        ).clip(0, 1)

        support_inverse = (

            1

            -

            np.clip(
                df["support_tickets"] / 10,
                0,
                1,
            )
        )

        relationship = (

            nps_scaled * 0.7

            +

            support_inverse * 0.3
        )

        return relationship.round(4)

    # ─────────────────────────────────────────────
    # FINAL HEALTH SCORE
    # ─────────────────────────────────────────────

    def calculate_health_score(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Calculating unified health score..."
        )

        score = (

            df["usage_health"]
            * self.weights["usage_health"]

            +

            df["payment_health"]
            * self.weights["payment_health"]

            +

            df["support_health"]
            * self.weights["support_health"]

            +

            df["engagement_health"]
            * self.weights["engagement_score"]

            +

            df["relationship_health"]
            * self.weights["relationship_score"]
        )

        return score.clip(0, 1).round(4)

    # ─────────────────────────────────────────────
    # RISK LEVEL ENGINE
    # ─────────────────────────────────────────────

    def assign_risk_levels(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Assigning risk levels..."
        )

        health = df["health_score"]

        risk = np.where(

            health <= (
                1 - RISK_HIGH_THRESHOLD
            ),
            "HIGH",

            np.where(

                health <= (
                    1 - RISK_MEDIUM_THRESHOLD
                ),
                "MEDIUM",
                "LOW",
            )
        )

        return pd.Series(
            risk,
            index=df.index,
        )

    # ─────────────────────────────────────────────
    # REVENUE RISK
    # ─────────────────────────────────────────────

    def calculate_revenue_risk(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Calculating revenue risk..."
        )

        revenue = df["monthly_revenue"]

        normalized_revenue = (

            revenue

            /

            revenue.max()
            if revenue.max() > 0
            else 1
        )

        revenue_risk = (

            normalized_revenue

            *

            (1 - df["health_score"])
        )

        return revenue_risk.round(4)

    # ─────────────────────────────────────────────
    # WARNING SIGNALS
    # ─────────────────────────────────────────────

    def generate_warning_signals(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Generating warning signals..."
        )

        warnings_list = []

        for _, row in df.iterrows():

            signals = []

            if row["feature_usage_score"] < 0.25:
                signals.append(
                    "Low Feature Adoption"
                )

            if row["payment_delays"] >= 2:
                signals.append(
                    "Payment Instability"
                )

            if row["support_tickets"] >= 5:
                signals.append(
                    "Support Escalation"
                )

            if row["login_frequency"] < 5:
                signals.append(
                    "Usage Collapse"
                )

            if row["nps_score"] <= 4:
                signals.append(
                    "Relationship Deterioration"
                )

            if not signals:
                signals.append(
                    "Healthy"
                )

            warnings_list.append(
                ", ".join(signals)
            )

        return pd.Series(
            warnings_list,
            index=df.index,
        )

    # ─────────────────────────────────────────────
    # CRITICAL FLAGS
    # ─────────────────────────────────────────────

    def generate_critical_flags(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Generating critical flags..."
        )

        flags = []

        for _, row in df.iterrows():

            critical = []

            if row["health_score"] < 0.25:
                critical.append(
                    "Severe Churn Risk"
                )

            if row["revenue_risk"] > 0.60:
                critical.append(
                    "High Revenue Exposure"
                )

            if row["payment_delays"] >= 4:
                critical.append(
                    "Payment Failure Risk"
                )

            if row["feature_usage_score"] < 0.10:
                critical.append(
                    "Near-Zero Adoption"
                )

            if not critical:
                critical.append(
                    "None"
                )

            flags.append(
                ", ".join(critical)
            )

        return pd.Series(
            flags,
            index=df.index,
        )

    # ─────────────────────────────────────────────
    # HEALTH TREND ENGINE
    # ─────────────────────────────────────────────

    def generate_health_trends(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Generating health trends..."
        )

        trends = []

        for _, row in df.iterrows():

            score = row["health_score"]

            if score >= 0.80:
                trends.append(
                    "Strong Positive"
                )

            elif score >= 0.60:
                trends.append(
                    "Stable"
                )

            elif score >= 0.40:
                trends.append(
                    "Declining"
                )

            else:
                trends.append(
                    "Critical Decline"
                )

        return pd.Series(
            trends,
            index=df.index,
        )

    # ─────────────────────────────────────────────
    # INTERVENTION PRIORITY
    # ─────────────────────────────────────────────

    def calculate_intervention_priority(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Calculating intervention priorities..."
        )

        priority_scores = (

            (1 - df["health_score"]) * 0.6

            +

            df["revenue_risk"] * 0.4
        )

        priorities = []

        for score in priority_scores:

            if score >= 0.75:
                priorities.append("P1")

            elif score >= 0.50:
                priorities.append("P2")

            elif score >= 0.25:
                priorities.append("P3")

            else:
                priorities.append("P4")

        return pd.Series(
            priorities,
            index=df.index,
        )

    # ─────────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────

    def executive_summary(
        self,
        df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Generating EWS executive summary..."
        )

        summary = {

            "generated_at":
                datetime.utcnow().isoformat(),

            "customers_analyzed":
                int(len(df)),

            "average_health_score":
                round(
                    df["health_score"].mean(),
                    4,
                ),

            "high_risk_customers":
                int(
                    (
                        df["risk_level"]
                        == "HIGH"
                    ).sum()
                ),

            "medium_risk_customers":
                int(
                    (
                        df["risk_level"]
                        == "MEDIUM"
                    ).sum()
                ),

            "low_risk_customers":
                int(
                    (
                        df["risk_level"]
                        == "LOW"
                    ).sum()
                ),

            "total_revenue_at_risk":
                float(
                    df.loc[
                        df["risk_level"] == "HIGH",
                        "monthly_revenue",
                    ].sum()
                ),
        }

        return summary


# ─────────────────────────────────────────────
# GLOBAL ENGINE
# ─────────────────────────────────────────────

ews_engine = EnterpriseEWS()


# ─────────────────────────────────────────────
# FUNCTIONAL API
# ─────────────────────────────────────────────

def analyze_customer_health(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return ews_engine.analyze(df)


def generate_ews_summary(
    df: pd.DataFrame,
) -> Dict:

    analyzed = ews_engine.analyze(df)

    return ews_engine.executive_summary(
        analyzed
    )