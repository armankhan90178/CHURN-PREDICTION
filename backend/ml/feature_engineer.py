"""
ChurnShield 2.0 — Hyper Feature Engineering Engine

Enterprise-grade ML feature generation system.

Purpose:
Transform raw customer behavioral data
into high-signal ML intelligence features.

Capabilities:
- behavioral intelligence
- churn signal amplification
- financial engineering
- engagement scoring
- lifecycle intelligence
- customer health scoring
- anomaly feature generation
- interaction synthesis
- trend intelligence
- velocity detection
- seat analytics
- support burden analytics
- revenue quality scoring
- feature normalization
- ML-ready feature matrix generation
"""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import List
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.feature_engineer"
)


class HyperFeatureEngineer:

    def __init__(self):

        self.minmax_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()

    # ─────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────

    def engineer(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Starting hyper feature engineering pipeline"
        )

        data = df.copy()

        # Core stages
        data = self._financial_features(data)
        data = self._engagement_features(data)
        data = self._support_features(data)
        data = self._seat_features(data)
        data = self._contract_features(data)
        data = self._behavioral_features(data)
        data = self._risk_features(data)
        data = self._trend_features(data)
        data = self._velocity_features(data)
        data = self._segment_features(data)
        data = self._interaction_features(data)
        data = self._health_scores(data)
        data = self._anomaly_features(data)
        data = self._temporal_features(data)
        data = self._ai_meta_features(data)
        data = self._normalize_numeric_features(data)
        data = self._memory_optimize(data)

        logger.info(
            f"Feature engineering completed | Shape: {data.shape}"
        )

        return data

    # ─────────────────────────────────────────────
    # FINANCIAL FEATURES
    # ─────────────────────────────────────────────

    def _financial_features(self, df):

        if "monthly_revenue" not in df.columns:
            return df

        revenue = df["monthly_revenue"].fillna(0)

        df["annual_revenue"] = revenue * 12

        df["quarterly_revenue"] = revenue * 3

        df["log_revenue"] = np.log1p(revenue)

        df["revenue_percentile"] = (
            revenue.rank(pct=True)
        )

        df["high_value_customer"] = (
            revenue >= revenue.quantile(0.85)
        ).astype(int)

        df["low_value_customer"] = (
            revenue <= revenue.quantile(0.20)
        ).astype(int)

        df["revenue_stability_score"] = (
            revenue /
            max(revenue.std(), 1)
        )

        df["revenue_segment"] = pd.cut(
            revenue,
            bins=[
                -1,
                999,
                4999,
                14999,
                49999,
                np.inf,
            ],
            labels=[
                "Low",
                "Medium",
                "High",
                "Premium",
                "Enterprise",
            ]
        )

        return df

    # ─────────────────────────────────────────────
    # ENGAGEMENT FEATURES
    # ─────────────────────────────────────────────

    def _engagement_features(self, df):

        if "login_frequency" in df.columns:

            login = df["login_frequency"].fillna(0)

            df["daily_login_average"] = (
                login / 30
            )

            df["weekly_login_average"] = (
                login / 4
            )

            df["login_percentile"] = (
                login.rank(pct=True)
            )

            df["inactive_customer"] = (
                login <= 2
            ).astype(int)

            df["power_user"] = (
                login >= login.quantile(0.90)
            ).astype(int)

            df["login_health_score"] = np.select(
                [
                    login <= 2,
                    login <= 8,
                    login <= 20,
                    login > 20,
                ],
                [
                    0.1,
                    0.4,
                    0.7,
                    1.0,
                ],
                default=0.5
            )

        if "feature_usage_score" in df.columns:

            usage = df["feature_usage_score"].fillna(0)

            df["adoption_percentile"] = (
                usage.rank(pct=True)
            )

            df["low_adoption_flag"] = (
                usage < 0.20
            ).astype(int)

            df["feature_health_score"] = np.select(
                [
                    usage < 0.20,
                    usage < 0.50,
                    usage < 0.80,
                    usage >= 0.80,
                ],
                [
                    0.1,
                    0.4,
                    0.7,
                    1.0,
                ]
            )

        if {
            "login_health_score",
            "feature_health_score",
        }.issubset(df.columns):

            df["engagement_score"] = (
                (
                    df["login_health_score"] * 0.5
                ) +
                (
                    df["feature_health_score"] * 0.5
                )
            )

        return df

    # ─────────────────────────────────────────────
    # SUPPORT FEATURES
    # ─────────────────────────────────────────────

    def _support_features(self, df):

        if "support_tickets" not in df.columns:
            return df

        tickets = df["support_tickets"].fillna(0)

        df["support_ticket_log"] = (
            np.log1p(tickets)
        )

        df["high_support_flag"] = (
            tickets >= tickets.quantile(0.80)
        ).astype(int)

        df["support_percentile"] = (
            tickets.rank(pct=True)
        )

        df["support_severity"] = np.select(
            [
                tickets <= 1,
                tickets <= 3,
                tickets <= 7,
                tickets > 7,
            ],
            [
                "Low",
                "Moderate",
                "High",
                "Critical",
            ]
        )

        df["support_health_score"] = (
            1 /
            (1 + np.log1p(tickets))
        )

        return df

    # ─────────────────────────────────────────────
    # SEAT FEATURES
    # ─────────────────────────────────────────────

    def _seat_features(self, df):

        required = {
            "active_seats",
            "total_seats",
        }

        if not required.issubset(df.columns):
            return df

        active = df["active_seats"].fillna(0)
        total = df["total_seats"].replace(0, 1)

        utilization = active / total

        df["seat_utilization"] = utilization

        df["unused_seats"] = (
            total - active
        )

        df["seat_efficiency"] = (
            utilization *
            np.log1p(total)
        )

        df["underutilized_account"] = (
            utilization < 0.40
        ).astype(int)

        return df

    # ─────────────────────────────────────────────
    # CONTRACT FEATURES
    # ─────────────────────────────────────────────

    def _contract_features(self, df):

        if "contract_age_months" not in df.columns:
            return df

        age = df["contract_age_months"].fillna(0)

        df["customer_age_days"] = age * 30

        df["customer_maturity_score"] = np.select(
            [
                age <= 3,
                age <= 12,
                age <= 24,
                age > 24,
            ],
            [
                0.2,
                0.5,
                0.8,
                1.0,
            ]
        )

        df["new_customer"] = (
            age <= 3
        ).astype(int)

        df["loyal_customer"] = (
            age >= 24
        ).astype(int)

        return df

    # ─────────────────────────────────────────────
    # BEHAVIORAL FEATURES
    # ─────────────────────────────────────────────

    def _behavioral_features(self, df):

        if {
            "engagement_score",
            "support_health_score",
        }.issubset(df.columns):

            df["behavioral_health"] = (
                (
                    df["engagement_score"] * 0.7
                ) +
                (
                    df["support_health_score"] * 0.3
                )
            )

        if {
            "monthly_revenue",
            "engagement_score",
        }.issubset(df.columns):

            df["customer_value_index"] = (
                np.log1p(df["monthly_revenue"]) *
                df["engagement_score"]
            )

        return df

    # ─────────────────────────────────────────────
    # RISK FEATURES
    # ─────────────────────────────────────────────

    def _risk_features(self, df):

        risk = np.zeros(len(df))

        if "payment_delays" in df.columns:
            risk += (
                df["payment_delays"] * 0.30
            )

        if "support_ticket_log" in df.columns:
            risk += (
                df["support_ticket_log"] * 0.20
            )

        if "engagement_score" in df.columns:
            risk += (
                (1 - df["engagement_score"]) * 0.50
            )

        risk = np.clip(risk, 0, 1)

        df["churn_risk_signal"] = risk

        df["risk_segment"] = pd.cut(
            risk,
            bins=[
                -1,
                0.25,
                0.50,
                0.75,
                np.inf,
            ],
            labels=[
                "Low",
                "Moderate",
                "High",
                "Critical",
            ]
        )

        return df

    # ─────────────────────────────────────────────
    # TREND FEATURES
    # ─────────────────────────────────────────────

    def _trend_features(self, df):

        if {
            "monthly_revenue",
            "contract_age_months",
        }.issubset(df.columns):

            df["revenue_per_month_age"] = (
                df["monthly_revenue"] /
                df["contract_age_months"].replace(0, 1)
            )

        return df

    # ─────────────────────────────────────────────
    # VELOCITY FEATURES
    # ─────────────────────────────────────────────

    def _velocity_features(self, df):

        if {
            "login_frequency",
            "contract_age_months",
        }.issubset(df.columns):

            df["usage_velocity"] = (
                df["login_frequency"] /
                df["contract_age_months"].replace(0, 1)
            )

        return df

    # ─────────────────────────────────────────────
    # SEGMENT FEATURES
    # ─────────────────────────────────────────────

    def _segment_features(self, df):

        if "monthly_revenue" in df.columns:

            revenue = df["monthly_revenue"]

            df["top_10_percent_customer"] = (
                revenue >= revenue.quantile(0.90)
            ).astype(int)

        return df

    # ─────────────────────────────────────────────
    # INTERACTION FEATURES
    # ─────────────────────────────────────────────

    def _interaction_features(self, df):

        if {
            "engagement_score",
            "monthly_revenue",
        }.issubset(df.columns):

            df["revenue_engagement_interaction"] = (
                df["engagement_score"] *
                np.log1p(df["monthly_revenue"])
            )

        if {
            "support_ticket_log",
            "engagement_score",
        }.issubset(df.columns):

            df["friction_index"] = (
                df["support_ticket_log"] *
                (1 - df["engagement_score"])
            )

        return df

    # ─────────────────────────────────────────────
    # HEALTH SCORES
    # ─────────────────────────────────────────────

    def _health_scores(self, df):

        components = []

        possible = [
            "engagement_score",
            "support_health_score",
            "customer_maturity_score",
        ]

        for col in possible:

            if col in df.columns:
                components.append(df[col])

        if len(components) > 0:

            df["overall_customer_health"] = (
                np.mean(components, axis=0)
            )

        return df

    # ─────────────────────────────────────────────
    # ANOMALY FEATURES
    # ─────────────────────────────────────────────

    def _anomaly_features(self, df):

        numeric = df.select_dtypes(
            include=[np.number]
        )

        for col in numeric.columns:

            mean = numeric[col].mean()
            std = numeric[col].std()

            if std == 0:
                continue

            z_score = (
                (numeric[col] - mean) / std
            )

            df[f"{col}_anomaly"] = (
                np.abs(z_score) > 3
            ).astype(int)

        return df

    # ─────────────────────────────────────────────
    # TEMPORAL FEATURES
    # ─────────────────────────────────────────────

    def _temporal_features(self, df):

        date_cols = [
            c for c in df.columns
            if "date" in c
        ]

        for col in date_cols:

            try:

                parsed = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                df[f"{col}_year"] = parsed.dt.year
                df[f"{col}_month"] = parsed.dt.month
                df[f"{col}_weekday"] = parsed.dt.weekday

            except Exception:
                continue

        return df

    # ─────────────────────────────────────────────
    # AI META FEATURES
    # ─────────────────────────────────────────────

    def _ai_meta_features(self, df):

        if {
            "overall_customer_health",
            "monthly_revenue",
        }.issubset(df.columns):

            df["predicted_ltv_proxy"] = (
                df["overall_customer_health"] *
                df["monthly_revenue"] * 12
            )

        if {
            "churn_risk_signal",
            "monthly_revenue",
        }.issubset(df.columns):

            df["revenue_at_risk"] = (
                df["churn_risk_signal"] *
                df["monthly_revenue"]
            )

        return df

    # ─────────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────────

    def _normalize_numeric_features(self, df):

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        exclude = [
            "churned",
        ]

        scale_cols = [
            c for c in numeric_cols
            if c not in exclude
        ]

        try:

            df[scale_cols] = (
                self.minmax_scaler.fit_transform(
                    df[scale_cols]
                )
            )

        except Exception as e:

            logger.warning(
                f"Normalization failed: {e}"
            )

        return df

    # ─────────────────────────────────────────────
    # MEMORY OPTIMIZATION
    # ─────────────────────────────────────────────

    def _memory_optimize(self, df):

        for col in df.select_dtypes(
            include=["float64"]
        ).columns:

            df[col] = pd.to_numeric(
                df[col],
                downcast="float"
            )

        for col in df.select_dtypes(
            include=["int64"]
        ).columns:

            df[col] = pd.to_numeric(
                df[col],
                downcast="integer"
            )

        return df


# ─────────────────────────────────────────────
# FUNCTIONAL INTERFACE
# ─────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
):

    engineer = HyperFeatureEngineer()

    return engineer.engineer(df)