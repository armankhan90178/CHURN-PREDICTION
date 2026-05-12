"""
ChurnShield 2.0 — Hyper Explainable AI Engine

Enterprise-grade churn explanation system.

Purpose:
Explain WHY customers churn using:
- SHAP explainability
- risk decomposition
- behavioral reasoning
- business-friendly narratives
- feature contribution analysis
- customer-level intelligence
- portfolio insights
- actionable recommendations

Capabilities:
- local explanations
- global explanations
- top churn drivers
- customer risk narratives
- AI-generated business reasoning
- feature importance ranking
- segment-level analysis
- executive insights
- churn factor scoring
- recommendation engine
"""

import logging
import warnings
import numpy as np
import pandas as pd
import shap

from typing import Dict, List

warnings.filterwarnings("ignore")

logger = logging.getLogger("churnshield.explainer")


class HyperChurnExplainer:

    def __init__(self, model):

        self.model = model

        logger.info("Initializing SHAP explainer")

        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            self.explainer = shap.Explainer(model)

    # ─────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────

    def explain(
        self,
        df: pd.DataFrame,
        top_n: int = 10,
    ) -> Dict:

        logger.info("Starting enterprise explainability pipeline")

        data = df.copy()

        numeric_data = self._prepare_features(data)

        # SHAP VALUES
        shap_values = self._generate_shap_values(numeric_data)

        # GLOBAL IMPORTANCE
        global_importance = self._generate_global_importance(
            numeric_data,
            shap_values,
            top_n,
        )

        # CUSTOMER EXPLANATIONS
        customer_explanations = self._generate_customer_explanations(
            data,
            numeric_data,
            shap_values,
        )

        # SEGMENT INSIGHTS
        segment_insights = self._generate_segment_insights(
            data,
            shap_values,
        )

        # EXECUTIVE SUMMARY
        executive_summary = self._generate_executive_summary(
            global_importance,
            customer_explanations,
        )

        logger.info("Explainability pipeline completed")

        return {
            "global_feature_importance": global_importance,
            "customer_explanations": customer_explanations,
            "segment_insights": segment_insights,
            "executive_summary": executive_summary,
        }

    # ─────────────────────────────────────────────
    # FEATURE PREPARATION
    # ─────────────────────────────────────────────

    def _prepare_features(self, df):

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
    # SHAP GENERATION
    # ─────────────────────────────────────────────

    def _generate_shap_values(self, X):

        logger.info("Generating SHAP values")

        shap_values = self.explainer.shap_values(X)

        # Binary classification fix
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values

    # ─────────────────────────────────────────────
    # GLOBAL FEATURE IMPORTANCE
    # ─────────────────────────────────────────────

    def _generate_global_importance(
        self,
        X,
        shap_values,
        top_n=10,
    ):

        logger.info("Calculating global importance")

        importance = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": importance,
        })

        importance_df = (
            importance_df
            .sort_values(
                by="importance",
                ascending=False
            )
        )

        top_features = []

        for _, row in importance_df.head(top_n).iterrows():

            top_features.append({
                "feature": row["feature"],
                "importance": round(
                    float(row["importance"]),
                    5
                ),
                "business_impact": self._feature_business_impact(
                    row["feature"]
                ),
            })

        return top_features

    # ─────────────────────────────────────────────
    # CUSTOMER-LEVEL EXPLANATIONS
    # ─────────────────────────────────────────────

    def _generate_customer_explanations(
        self,
        original_df,
        X,
        shap_values,
    ):

        logger.info("Generating customer explanations")

        explanations = []

        for idx in range(len(X)):

            row = original_df.iloc[idx]

            customer_id = row.get(
                "customer_id",
                f"CUST_{idx}"
            )

            customer_name = row.get(
                "customer_name",
                "Unknown"
            )

            customer_shap = shap_values[idx]

            feature_contributions = []

            for feature, value in zip(
                X.columns,
                customer_shap,
            ):

                feature_contributions.append({
                    "feature": feature,
                    "impact": float(value),
                    "direction": (
                        "increases_churn"
                        if value > 0
                        else "reduces_churn"
                    ),
                })

            feature_contributions = sorted(
                feature_contributions,
                key=lambda x: abs(x["impact"]),
                reverse=True,
            )

            top_drivers = feature_contributions[:5]

            narrative = self._build_customer_narrative(
                row,
                top_drivers,
            )

            recommendations = self._generate_recommendations(
                row,
                top_drivers,
            )

            explanations.append({
                "customer_id": customer_id,
                "customer_name": customer_name,
                "top_drivers": top_drivers,
                "risk_narrative": narrative,
                "recommendations": recommendations,
            })

        return explanations

    # ─────────────────────────────────────────────
    # SEGMENT INSIGHTS
    # ─────────────────────────────────────────────

    def _generate_segment_insights(
        self,
        df,
        shap_values,
    ):

        logger.info("Generating segment insights")

        insights = {}

        if "persona" in df.columns:

            for persona in df["persona"].dropna().unique():

                subset = df[
                    df["persona"] == persona
                ]

                insights[persona] = {
                    "customers": int(len(subset)),
                    "avg_revenue": round(
                        float(
                            subset.get(
                                "monthly_revenue",
                                pd.Series([0])
                            ).mean()
                        ),
                        2,
                    ),
                    "avg_churn_probability": round(
                        float(
                            subset.get(
                                "churn_probability",
                                pd.Series([0])
                            ).mean()
                        ),
                        4,
                    ),
                }

        return insights

    # ─────────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────

    def _generate_executive_summary(
        self,
        global_importance,
        customer_explanations,
    ):

        logger.info("Generating executive summary")

        total_customers = len(customer_explanations)

        high_risk = 0

        for item in customer_explanations:

            if any(
                d["impact"] > 0.5
                for d in item["top_drivers"]
            ):
                high_risk += 1

        top_global_driver = (
            global_importance[0]["feature"]
            if global_importance
            else "Unknown"
        )

        return {
            "total_customers_analyzed": total_customers,
            "high_risk_customers": high_risk,
            "top_global_churn_driver": top_global_driver,
            "key_business_message": (
                f"Primary churn risk across portfolio "
                f"is driven by '{top_global_driver}'."
            ),
        }

    # ─────────────────────────────────────────────
    # BUSINESS IMPACT LABELING
    # ─────────────────────────────────────────────

    def _feature_business_impact(self, feature):

        mapping = {

            "feature_usage_score":
                "Product adoption strongly impacts retention",

            "payment_delays":
                "Late payments correlate with churn risk",

            "support_tickets":
                "Support friction drives dissatisfaction",

            "monthly_revenue":
                "Revenue tier impacts customer loyalty",

            "login_frequency":
                "Low engagement indicates churn tendency",

            "seat_utilization":
                "Unused seats reduce perceived value",

            "nps_score":
                "Customer satisfaction impacts retention",
        }

        return mapping.get(
            feature,
            "Behavioral influence on churn",
        )

    # ─────────────────────────────────────────────
    # CUSTOMER NARRATIVE
    # ─────────────────────────────────────────────

    def _build_customer_narrative(
        self,
        row,
        top_drivers,
    ):

        positive = []
        negative = []

        for driver in top_drivers:

            if driver["impact"] > 0:
                negative.append(driver["feature"])
            else:
                positive.append(driver["feature"])

        narrative = []

        if negative:
            narrative.append(
                f"Customer shows churn signals due to "
                f"{', '.join(negative[:3])}."
            )

        if positive:
            narrative.append(
                f"Retention strengths include "
                f"{', '.join(positive[:2])}."
            )

        revenue = row.get("monthly_revenue", 0)

        if revenue > 20000:
            narrative.append(
                "This is a high-value account."
            )

        return " ".join(narrative)

    # ─────────────────────────────────────────────
    # RECOMMENDATION ENGINE
    # ─────────────────────────────────────────────

    def _generate_recommendations(
        self,
        row,
        top_drivers,
    ):

        recommendations = []

        features = [
            d["feature"]
            for d in top_drivers
        ]

        if "feature_usage_score" in features:
            recommendations.append(
                "Increase onboarding and feature adoption campaigns"
            )

        if "support_tickets" in features:
            recommendations.append(
                "Assign customer success manager immediately"
            )

        if "payment_delays" in features:
            recommendations.append(
                "Offer billing assistance or flexible payment options"
            )

        if "login_frequency" in features:
            recommendations.append(
                "Launch engagement reactivation workflow"
            )

        if "seat_utilization" in features:
            recommendations.append(
                "Educate customer on unused product capabilities"
            )

        if len(recommendations) == 0:
            recommendations.append(
                "Monitor customer engagement continuously"
            )

        return recommendations


# ─────────────────────────────────────────────
# FUNCTIONAL INTERFACE
# ─────────────────────────────────────────────

def explain_predictions(
    df: pd.DataFrame,
    model,
):

    explainer = HyperChurnExplainer(model)

    return explainer.explain(df)