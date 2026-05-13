"""
ChurnShield 2.0 — Intelligent Schema Detector

Purpose:
Automatically understand ANY dataset structure.

Capabilities:
- schema inference
- semantic column detection
- AI-style business understanding
- churn target discovery
- entity detection
- feature categorization
- time-series detection
- identifier detection
- business domain inference
- ML readiness mapping
- relationship discovery
- confidence scoring

Works with:
- CSV
- Excel
- JSON
- ERP exports
- CRM exports
- Banking data
- Telecom data
- SaaS analytics
- E-commerce datasets
- Healthcare systems
- Any tabular business data

Author: ChurnShield AI
"""

import re
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

logger = logging.getLogger("churnshield.schema_detector")


# ============================================================
# MAIN ENGINE
# ============================================================

class IntelligentSchemaDetector:

    def __init__(self):

        # ----------------------------------------------------
        # CANONICAL BUSINESS FIELDS
        # ----------------------------------------------------

        self.canonical_schema = {

            "customer_id": [
                "customer_id",
                "cust_id",
                "userid",
                "user_id",
                "account_id",
                "subscriber_id",
                "member_id",
                "client_id",
                "id",
            ],

            "customer_name": [
                "name",
                "customer_name",
                "client_name",
                "full_name",
                "company_name",
                "business_name",
            ],

            "monthly_revenue": [
                "revenue",
                "mrr",
                "monthly_revenue",
                "sales",
                "amount",
                "billing",
                "invoice_amount",
                "subscription_amount",
                "payment_amount",
            ],

            "contract_age_months": [
                "tenure",
                "subscription_age",
                "months_active",
                "contract_months",
                "customer_age",
                "membership_duration",
            ],

            "support_tickets": [
                "tickets",
                "complaints",
                "issues",
                "support_requests",
                "cases",
                "queries",
            ],

            "payment_delays": [
                "late_payments",
                "payment_delay",
                "overdue_count",
                "failed_payments",
                "payment_failures",
            ],

            "login_frequency": [
                "logins",
                "sessions",
                "usage",
                "activity_count",
                "visit_count",
                "engagement",
            ],

            "feature_usage_score": [
                "feature_usage",
                "usage_score",
                "adoption_score",
                "engagement_score",
            ],

            "active_seats": [
                "active_users",
                "used_seats",
                "occupied_seats",
            ],

            "total_seats": [
                "licensed_seats",
                "purchased_seats",
                "team_size",
            ],

            "nps_score": [
                "nps",
                "satisfaction",
                "rating",
                "feedback_score",
                "csat",
            ],

            "churned": [
                "churn",
                "cancelled",
                "left",
                "inactive",
                "terminated",
                "closed",
                "status",
            ],
        }

        # ----------------------------------------------------
        # INDUSTRY KEYWORDS
        # ----------------------------------------------------

        self.industry_keywords = {

            "Telecom": [
                "call",
                "recharge",
                "data_usage",
                "sim",
                "airtel",
                "jio",
            ],

            "Banking": [
                "loan",
                "credit",
                "debit",
                "balance",
                "transaction",
            ],

            "SaaS": [
                "subscription",
                "seat",
                "api",
                "workspace",
                "tenant",
            ],

            "E-Commerce": [
                "order",
                "cart",
                "purchase",
                "product",
                "delivery",
            ],

            "Healthcare": [
                "patient",
                "doctor",
                "appointment",
                "medical",
            ],

            "Education": [
                "student",
                "course",
                "lesson",
                "quiz",
                "assignment",
            ],

            "Streaming": [
                "watch",
                "stream",
                "content",
                "viewer",
                "episode",
            ],
        }

    # ========================================================
    # MAIN ENTRY
    # ========================================================

    def detect(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        logger.info("Starting intelligent schema detection")

        data = df.copy()

        # Normalize columns
        original_columns = list(data.columns)

        normalized_columns = [
            self._normalize_column(col)
            for col in data.columns
        ]

        data.columns = normalized_columns

        results = {

            "original_columns":
                original_columns,

            "normalized_columns":
                normalized_columns,

            "schema_mapping":
                self._detect_schema_mapping(data),

            "column_types":
                self._detect_column_types(data),

            "business_entities":
                self._detect_business_entities(data),

            "target_column":
                self._detect_target_column(data),

            "identifier_columns":
                self._detect_identifiers(data),

            "datetime_columns":
                self._detect_datetime_columns(data),

            "industry_prediction":
                self._detect_industry(data),

            "feature_categories":
                self._categorize_features(data),

            "relationship_analysis":
                self._relationship_analysis(data),

            "data_density":
                self._data_density(data),

            "ml_capability":
                self._ml_capability(data),

            "confidence_scores":
                {},

            "recommendations":
                [],
        }

        # Generate confidence
        results["confidence_scores"] = (
            self._generate_confidence_scores(results)
        )

        # Recommendations
        results["recommendations"] = (
            self._generate_recommendations(results)
        )

        logger.info("Schema detection completed")

        return results

    # ========================================================
    # SCHEMA MAPPING
    # ========================================================

    def _detect_schema_mapping(
        self,
        df
    ):

        mapping = {}

        for column in df.columns:

            best_match = None
            best_score = 0

            for canonical, aliases in self.canonical_schema.items():

                for alias in aliases:

                    similarity = self._similarity(
                        column,
                        alias
                    )

                    if similarity > best_score:

                        best_score = similarity
                        best_match = canonical

            if best_score >= 0.70:

                mapping[column] = {

                    "mapped_to": best_match,
                    "confidence": round(best_score, 4),
                }

        return mapping

    # ========================================================
    # COLUMN TYPE DETECTION
    # ========================================================

    def _detect_column_types(
        self,
        df
    ):

        results = {}

        for col in df.columns:

            dtype = str(df[col].dtype)

            col_type = "unknown"

            if pd.api.types.is_numeric_dtype(df[col]):

                if df[col].nunique() <= 10:
                    col_type = "categorical_numeric"

                else:
                    col_type = "continuous_numeric"

            elif pd.api.types.is_datetime64_any_dtype(df[col]):

                col_type = "datetime"

            else:

                avg_length = (
                    df[col]
                    .astype(str)
                    .str.len()
                    .mean()
                )

                if avg_length > 50:
                    col_type = "text"

                else:
                    col_type = "categorical_text"

            results[col] = {

                "detected_type": col_type,
                "dtype": dtype,
            }

        return results

    # ========================================================
    # BUSINESS ENTITY DETECTION
    # ========================================================

    def _detect_business_entities(
        self,
        df
    ):

        entities = {

            "customer_columns": [],
            "financial_columns": [],
            "behavior_columns": [],
            "engagement_columns": [],
            "support_columns": [],
        }

        for col in df.columns:

            lower = col.lower()

            # Customer
            if any(
                k in lower
                for k in ["customer", "client", "user", "member"]
            ):

                entities["customer_columns"].append(col)

            # Financial
            if any(
                k in lower
                for k in [
                    "revenue",
                    "amount",
                    "billing",
                    "sales",
                    "payment",
                ]
            ):

                entities["financial_columns"].append(col)

            # Behavior
            if any(
                k in lower
                for k in [
                    "usage",
                    "login",
                    "activity",
                    "session",
                ]
            ):

                entities["behavior_columns"].append(col)

            # Engagement
            if any(
                k in lower
                for k in [
                    "feature",
                    "engagement",
                    "adoption",
                ]
            ):

                entities["engagement_columns"].append(col)

            # Support
            if any(
                k in lower
                for k in [
                    "ticket",
                    "complaint",
                    "issue",
                    "support",
                ]
            ):

                entities["support_columns"].append(col)

        return entities

    # ========================================================
    # TARGET DETECTION
    # ========================================================

    def _detect_target_column(
        self,
        df
    ):

        candidates = []

        for col in df.columns:

            lower = col.lower()

            score = 0

            if any(
                k in lower
                for k in [
                    "churn",
                    "cancel",
                    "leave",
                    "inactive",
                    "closed",
                ]
            ):
                score += 50

            unique_values = df[col].nunique()

            if unique_values <= 5:
                score += 20

            if set(
                df[col]
                .dropna()
                .astype(str)
                .str.lower()
                .unique()
            ).issubset(
                {"0", "1", "yes", "no", "true", "false"}
            ):
                score += 30

            if score > 0:

                candidates.append({

                    "column": col,
                    "score": score,
                })

        if not candidates:
            return None

        candidates = sorted(
            candidates,
            key=lambda x: x["score"],
            reverse=True
        )

        return candidates[0]

    # ========================================================
    # IDENTIFIER DETECTION
    # ========================================================

    def _detect_identifiers(
        self,
        df
    ):

        identifiers = []

        for col in df.columns:

            uniqueness_ratio = (
                df[col].nunique() / len(df)
            )

            if uniqueness_ratio > 0.90:

                identifiers.append({

                    "column": col,
                    "uniqueness_ratio":
                        round(uniqueness_ratio, 4),
                })

        return identifiers

    # ========================================================
    # DATETIME DETECTION
    # ========================================================

    def _detect_datetime_columns(
        self,
        df
    ):

        datetime_cols = []

        for col in df.columns:

            lower = col.lower()

            if any(
                k in lower
                for k in [
                    "date",
                    "time",
                    "timestamp",
                    "created",
                    "updated",
                ]
            ):

                datetime_cols.append(col)

        return datetime_cols

    # ========================================================
    # INDUSTRY DETECTION
    # ========================================================

    def _detect_industry(
        self,
        df
    ):

        scores = {}

        all_text = " ".join(df.columns).lower()

        for industry, keywords in self.industry_keywords.items():

            score = 0

            for keyword in keywords:

                if keyword.lower() in all_text:
                    score += 1

            scores[industry] = score

        best_industry = max(
            scores,
            key=scores.get
        )

        return {

            "industry":
                best_industry,

            "confidence":
                round(
                    scores[best_industry]
                    / max(1, len(self.industry_keywords[
                        best_industry
                    ])),
                    4,
                ),

            "all_scores":
                scores,
        }

    # ========================================================
    # FEATURE CATEGORIES
    # ========================================================

    def _categorize_features(
        self,
        df
    ):

        categories = {

            "numerical": [],
            "categorical": [],
            "datetime": [],
            "binary": [],
            "text": [],
        }

        for col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]):

                if df[col].nunique() <= 2:
                    categories["binary"].append(col)

                else:
                    categories["numerical"].append(col)

            else:

                avg_len = (
                    df[col]
                    .astype(str)
                    .str.len()
                    .mean()
                )

                if avg_len > 40:
                    categories["text"].append(col)

                else:
                    categories["categorical"].append(col)

        return categories

    # ========================================================
    # RELATIONSHIP ANALYSIS
    # ========================================================

    def _relationship_analysis(
        self,
        df
    ):

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        strong_relationships = []

        if len(numeric_cols) >= 2:

            corr = df[numeric_cols].corr()

            for i in corr.columns:

                for j in corr.columns:

                    if i == j:
                        continue

                    val = corr.loc[i, j]

                    if abs(val) >= 0.70:

                        strong_relationships.append({

                            "feature_1": i,
                            "feature_2": j,
                            "correlation":
                                round(float(val), 4),
                        })

        return {

            "strong_relationships":
                strong_relationships[:20]
        }

    # ========================================================
    # DATA DENSITY
    # ========================================================

    def _data_density(
        self,
        df
    ):

        total_cells = df.shape[0] * df.shape[1]

        filled_cells = total_cells - df.isna().sum().sum()

        density = filled_cells / total_cells

        return {

            "density":
                round(density, 4),

            "missing_ratio":
                round(1 - density, 4),
        }

    # ========================================================
    # ML CAPABILITY
    # ========================================================

    def _ml_capability(
        self,
        df
    ):

        numeric_features = len(
            df.select_dtypes(include=np.number).columns
        )

        has_target = (
            self._detect_target_column(df)
            is not None
        )

        score = 0

        if numeric_features >= 5:
            score += 40

        if has_target:
            score += 40

        if len(df) >= 500:
            score += 20

        return {

            "ml_ready":
                score >= 70,

            "score":
                score,

            "numeric_features":
                numeric_features,

            "has_target":
                has_target,
        }

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def _generate_confidence_scores(
        self,
        results
    ):

        scores = {

            "schema_mapping_confidence": 0,
            "industry_detection_confidence": 0,
            "target_detection_confidence": 0,
        }

        mappings = results["schema_mapping"]

        if mappings:

            avg_conf = np.mean([
                x["confidence"]
                for x in mappings.values()
            ])

            scores["schema_mapping_confidence"] = (
                round(float(avg_conf), 4)
            )

        industry_conf = results[
            "industry_prediction"
        ]["confidence"]

        scores[
            "industry_detection_confidence"
        ] = industry_conf

        if results["target_column"]:

            scores[
                "target_detection_confidence"
            ] = round(
                results["target_column"]["score"] / 100,
                4,
            )

        return scores

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def _generate_recommendations(
        self,
        results
    ):

        recommendations = []

        if not results["target_column"]:

            recommendations.append(
                "No churn target column detected. "
                "Use label generator."
            )

        if results["ml_capability"]["score"] < 60:

            recommendations.append(
                "Dataset needs richer numerical features."
            )

        if results["data_density"]["missing_ratio"] > 0.20:

            recommendations.append(
                "High missing data detected."
            )

        if not recommendations:

            recommendations.append(
                "Dataset structure is production-ready."
            )

        return recommendations

    # ========================================================
    # HELPERS
    # ========================================================

    def _normalize_column(
        self,
        col
    ):

        col = str(col).lower().strip()

        col = re.sub(r"[^a-z0-9]+", "_", col)

        col = re.sub(r"_+", "_", col)

        return col.strip("_")

    def _similarity(
        self,
        a,
        b
    ):

        return SequenceMatcher(
            None,
            a,
            b
        ).ratio()


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def detect_schema(
    df: pd.DataFrame
):

    detector = IntelligentSchemaDetector()

    return detector.detect(df)