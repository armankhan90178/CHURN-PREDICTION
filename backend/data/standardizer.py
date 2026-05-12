"""
ChurnShield 2.0 — Hyper Intelligent Universal Standardizer

Enterprise-grade schema intelligence engine.

Purpose:
Convert ANY messy business dataset into
a fully standardized ChurnShield ML-ready dataset.

Core Features:
- AI-style semantic schema mapping
- Fuzzy intelligent column matching
- Automatic datatype correction
- Industry-aware transformations
- Smart feature reconstruction
- Universal churn normalization
- Revenue harmonization
- Temporal feature extraction
- Behavioral feature synthesis
- Data quality enrichment
- ML-ready output generation
- Cross-industry compatibility

Designed for:
- SaaS
- OTT
- Banking
- Telecom
- Healthcare
- Fitness
- Retail
- Gaming
- Insurance
- Education
- Any future industry
"""

import re
import logging
import warnings
import pandas as pd
import numpy as np

from typing import Dict
from difflib import SequenceMatcher
from datetime import datetime

from config import STANDARD_SCHEMA

warnings.filterwarnings("ignore")

logger = logging.getLogger("churnshield.standardizer")


class HyperUniversalStandardizer:

    def __init__(self):

        # Canonical schema
        self.standard_schema = STANDARD_SCHEMA

        # Semantic aliases
        self.aliases = {
            "customer_id": [
                "custid",
                "customerid",
                "userid",
                "subscriber_id",
                "account_id",
                "client_id",
                "member_id",
            ],

            "customer_name": [
                "name",
                "full_name",
                "client_name",
                "subscriber_name",
                "company_name",
            ],

            "monthly_revenue": [
                "mrr",
                "revenue",
                "monthly_spend",
                "monthly_value",
                "subscription_amount",
                "income",
                "sales",
            ],

            "support_tickets": [
                "tickets",
                "complaints",
                "issues",
                "queries",
                "cases",
            ],

            "login_frequency": [
                "logins",
                "sessions",
                "usage",
                "usage_count",
                "activity_count",
                "visit_frequency",
            ],

            "feature_usage_score": [
                "usage_score",
                "engagement_score",
                "feature_adoption",
                "adoption_score",
            ],

            "contract_age_months": [
                "tenure",
                "subscription_age",
                "customer_age",
                "months_active",
            ],

            "payment_delays": [
                "late_payments",
                "failed_payments",
                "payment_issues",
            ],

            "active_seats": [
                "active_users",
                "used_seats",
                "active_accounts",
            ],

            "total_seats": [
                "licensed_users",
                "total_users",
                "purchased_seats",
            ],

            "churned": [
                "churn",
                "cancelled",
                "left",
                "inactive",
                "is_churned",
                "target",
                "label",
                "status",
            ],
        }

        # Binary churn normalization map
        self.churn_map = {
            "yes": 1,
            "true": 1,
            "1": 1,
            "churned": 1,
            "cancelled": 1,
            "inactive": 1,
            "left": 1,
            "closed": 1,

            "no": 0,
            "false": 0,
            "0": 0,
            "active": 0,
            "retained": 0,
            "existing": 0,
            "live": 0,
        }

    # ─────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Starting hyper-standardization pipeline")

        data = df.copy()

        # Step 1
        data = self._normalize_columns(data)

        # Step 2
        mapping = self._generate_schema_mapping(data)

        logger.info(f"Schema mapping generated: {mapping}")

        data = data.rename(columns=mapping)

        # Step 3
        data = self._correct_datatypes(data)

        # Step 4
        data = self._generate_missing_standard_columns(data)

        # Step 5
        data = self._normalize_churn_column(data)

        # Step 6
        data = self._repair_business_logic(data)

        # Step 7
        data = self._extract_temporal_features(data)

        # Step 8
        data = self._generate_behavioral_features(data)

        # Step 9
        data = self._generate_financial_features(data)

        # Step 10
        data = self._generate_engagement_features(data)

        # Step 11
        data = self._generate_risk_indicators(data)

        # Step 12
        data = self._generate_ai_features(data)

        # Step 13
        data = self._finalize_schema(data)

        logger.info(
            f"Standardization completed successfully | Shape: {data.shape}"
        )

        return data

    # ─────────────────────────────────────────────
    # COLUMN NORMALIZATION
    # ─────────────────────────────────────────────

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        normalized = []

        for col in df.columns:

            clean = (
                str(col)
                .lower()
                .strip()
            )

            clean = re.sub(r"[^a-zA-Z0-9]+", "_", clean)
            clean = re.sub(r"_+", "_", clean)
            clean = clean.strip("_")

            normalized.append(clean)

        df.columns = normalized

        return df

    # ─────────────────────────────────────────────
    # AI-STYLE MAPPING ENGINE
    # ─────────────────────────────────────────────

    def _generate_schema_mapping(self, df: pd.DataFrame) -> Dict[str, str]:

        mapping = {}

        for column in df.columns:

            best_match = None
            best_score = 0

            # Direct exact match
            if column in self.standard_schema:
                mapping[column] = column
                continue

            # Alias match
            for target, aliases in self.aliases.items():

                if column in aliases:
                    mapping[column] = target
                    best_match = target
                    break

            if best_match:
                continue

            # Semantic fuzzy matching
            for target in self.standard_schema:

                score = SequenceMatcher(
                    None,
                    column,
                    target,
                ).ratio()

                if score > best_score:
                    best_score = score
                    best_match = target

            if best_score > 0.72:
                mapping[column] = best_match

        return mapping

    # ─────────────────────────────────────────────
    # DATATYPE CORRECTION
    # ─────────────────────────────────────────────

    def _correct_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in df.columns:

            try:

                # Numeric inference
                if df[col].dtype == "object":

                    numeric_candidate = pd.to_numeric(
                        df[col],
                        errors="coerce"
                    )

                    success_rate = numeric_candidate.notnull().mean()

                    if success_rate > 0.80:
                        df[col] = numeric_candidate
                        continue

                # Date inference
                if "date" in col or "time" in col:

                    parsed = pd.to_datetime(
                        df[col],
                        errors="coerce"
                    )

                    if parsed.notnull().mean() > 0.60:
                        df[col] = parsed

            except Exception:
                continue

        return df

    # ─────────────────────────────────────────────
    # GENERATE REQUIRED COLUMNS
    # ─────────────────────────────────────────────

    def _generate_missing_standard_columns(self, df: pd.DataFrame):

        defaults = {
            "customer_id": "UNKNOWN",
            "customer_name": "Unknown Customer",
            "plan_type": "Basic",
            "monthly_revenue": 0,
            "contract_age_months": 1,
            "login_frequency": 0,
            "feature_usage_score": 0.50,
            "support_tickets": 0,
            "payment_delays": 0,
            "active_seats": 1,
            "total_seats": 1,
            "nps_score": 5,
        }

        for col, default in defaults.items():

            if col not in df.columns:
                logger.info(f"Creating missing standard column: {col}")
                df[col] = default

        return df

    # ─────────────────────────────────────────────
    # CHURN NORMALIZATION
    # ─────────────────────────────────────────────

    def _normalize_churn_column(self, df: pd.DataFrame):

        if "churned" not in df.columns:
            logger.warning("No churn column found")
            return df

        df["churned"] = (
            df["churned"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(self.churn_map)
            .fillna(0)
            .astype(int)
        )

        return df

    # ─────────────────────────────────────────────
    # BUSINESS LOGIC REPAIR
    # ─────────────────────────────────────────────

    def _repair_business_logic(self, df: pd.DataFrame):

        # Ensure positive revenue
        if "monthly_revenue" in df.columns:
            df["monthly_revenue"] = df["monthly_revenue"].clip(lower=0)

        # Ensure valid seat relationship
        if {
            "active_seats",
            "total_seats",
        }.issubset(df.columns):

            df["total_seats"] = np.maximum(
                df["total_seats"],
                df["active_seats"]
            )

        # NPS range enforcement
        if "nps_score" in df.columns:
            df["nps_score"] = df["nps_score"].clip(-100, 100)

        return df

    # ─────────────────────────────────────────────
    # TEMPORAL FEATURES
    # ─────────────────────────────────────────────

    def _extract_temporal_features(self, df: pd.DataFrame):

        date_columns = [
            c for c in df.columns
            if "date" in c or "time" in c
        ]

        for col in date_columns:

            try:
                parsed = pd.to_datetime(df[col], errors="coerce")

                if parsed.notnull().mean() < 0.50:
                    continue

                df[col] = parsed

                df[f"{col}_year"] = parsed.dt.year
                df[f"{col}_month"] = parsed.dt.month
                df[f"{col}_day"] = parsed.dt.day
                df[f"{col}_weekday"] = parsed.dt.weekday
                df[f"{col}_quarter"] = parsed.dt.quarter

                current_date = pd.Timestamp.now()

                df[f"days_since_{col}"] = (
                    current_date - parsed
                ).dt.days

            except Exception:
                continue

        return df

    # ─────────────────────────────────────────────
    # BEHAVIORAL FEATURES
    # ─────────────────────────────────────────────

    def _generate_behavioral_features(self, df: pd.DataFrame):

        if "login_frequency" in df.columns:

            df["login_health"] = np.select(
                [
                    df["login_frequency"] <= 2,
                    df["login_frequency"] <= 10,
                    df["login_frequency"] > 10,
                ],
                [0.1, 0.5, 1.0],
                default=0.3,
            )

        if "feature_usage_score" in df.columns:

            df["adoption_segment"] = pd.cut(
                df["feature_usage_score"],
                bins=[-1, 0.2, 0.5, 0.8, np.inf],
                labels=[
                    "Very Low",
                    "Low",
                    "Healthy",
                    "Power User",
                ]
            )

        return df

    # ─────────────────────────────────────────────
    # FINANCIAL FEATURES
    # ─────────────────────────────────────────────

    def _generate_financial_features(self, df: pd.DataFrame):

        if "monthly_revenue" in df.columns:

            revenue = df["monthly_revenue"]

            df["annual_revenue"] = revenue * 12

            df["revenue_tier"] = pd.cut(
                revenue,
                bins=[-1, 999, 4999, 14999, 49999, np.inf],
                labels=[
                    "Low",
                    "Medium",
                    "High",
                    "Premium",
                    "Enterprise",
                ]
            )

            median_revenue = max(revenue.median(), 1)

            df["relative_revenue_score"] = (
                revenue / median_revenue
            )

        return df

    # ─────────────────────────────────────────────
    # ENGAGEMENT FEATURES
    # ─────────────────────────────────────────────

    def _generate_engagement_features(self, df: pd.DataFrame):

        components = []

        if "feature_usage_score" in df.columns:
            components.append(df["feature_usage_score"])

        if "login_health" in df.columns:
            components.append(df["login_health"])

        if {
            "active_seats",
            "total_seats",
        }.issubset(df.columns):

            utilization = (
                df["active_seats"] /
                df["total_seats"].replace(0, 1)
            )

            df["seat_utilization"] = utilization
            components.append(utilization)

        if len(components) > 0:

            df["engagement_score"] = np.mean(
                components,
                axis=0,
            )

        return df

    # ─────────────────────────────────────────────
    # RISK INDICATORS
    # ─────────────────────────────────────────────

    def _generate_risk_indicators(self, df: pd.DataFrame):

        risk = np.zeros(len(df))

        if "payment_delays" in df.columns:
            risk += df["payment_delays"] * 0.15

        if "support_tickets" in df.columns:
            risk += np.log1p(df["support_tickets"]) * 0.20

        if "feature_usage_score" in df.columns:
            risk += (1 - df["feature_usage_score"]) * 0.40

        if "login_health" in df.columns:
            risk += (1 - df["login_health"]) * 0.25

        df["churn_risk_signal"] = risk

        df["risk_segment"] = pd.cut(
            risk,
            bins=[-1, 0.3, 0.6, 1.0, np.inf],
            labels=[
                "Low Risk",
                "Medium Risk",
                "High Risk",
                "Critical",
            ]
        )

        return df

    # ─────────────────────────────────────────────
    # AI FEATURES
    # ─────────────────────────────────────────────

    def _generate_ai_features(self, df: pd.DataFrame):

        if "monthly_revenue" in df.columns and "engagement_score" in df.columns:

            df["customer_lifetime_proxy"] = (
                df["monthly_revenue"] *
                (1 + df["engagement_score"])
            )

        if "support_tickets" in df.columns:

            df["support_burden"] = np.log1p(
                df["support_tickets"]
            )

        if "contract_age_months" in df.columns:

            df["customer_maturity"] = pd.cut(
                df["contract_age_months"],
                bins=[-1, 3, 12, 24, np.inf],
                labels=[
                    "New",
                    "Growing",
                    "Established",
                    "Loyal",
                ]
            )

        return df

    # ─────────────────────────────────────────────
    # FINAL SCHEMA OPTIMIZATION
    # ─────────────────────────────────────────────

    def _finalize_schema(self, df: pd.DataFrame):

        # Ensure all canonical columns exist
        for col in self.standard_schema:

            if col not in df.columns:
                df[col] = np.nan

        # Remove duplicated columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Optimize memory
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Sort important columns first
        ordered = [
            c for c in self.standard_schema
            if c in df.columns
        ]

        remaining = [
            c for c in df.columns
            if c not in ordered
        ]

        df = df[ordered + remaining]

        return df


# ─────────────────────────────────────────────
# FUNCTIONAL INTERFACE
# ─────────────────────────────────────────────


def standardize_dataset(df: pd.DataFrame):
    """
    Functional wrapper for FastAPI routes.
    """

    standardizer = HyperUniversalStandardizer()
    return standardizer.standardize(df)