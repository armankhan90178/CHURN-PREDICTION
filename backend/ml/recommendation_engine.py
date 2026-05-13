"""
ChurnShield 2.0 — Recommendation Engine

Purpose:
Enterprise-grade AI recommendation system for
customer retention, churn prevention,
revenue protection, and engagement optimization.

Capabilities:
- retention recommendations
- churn intervention planning
- customer-level recommendations
- segment-level recommendations
- revenue-saving strategies
- personalized offers
- dynamic risk scoring
- next best action
- AI retention playbooks
- upsell/cross-sell opportunities
- pricing optimization suggestions
- loyalty strategies
- behavioral intervention
- escalation intelligence
- communication timing optimization
- recommendation prioritization
- confidence scoring
- ROI estimation

Supports:
- B2B SaaS
- Telecom
- OTT
- Banking
- Insurance
- Ecommerce
- Healthcare
- EdTech
- Subscription businesses
- Any churn dataset

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

logger = logging.getLogger(
    "churnshield.recommendation_engine"
)


# ============================================================
# MAIN ENGINE
# ============================================================

class RecommendationEngine:

    def __init__(
        self,
        config: Optional[Dict] = None,
        save_dir: str = (
            "user_data/recommendations"
        ),
    ):

        self.config = config or {}

        self.save_dir = Path(
            save_dir
        )

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.recommendation_library = (
            self.load_recommendation_library()
        )

    # ========================================================
    # RECOMMENDATION LIBRARY
    # ========================================================

    def load_recommendation_library(
        self
    ) -> Dict:

        return {

            "high_risk": [

                "Immediate retention call",
                "Offer premium discount",
                "Assign dedicated success manager",
                "Trigger escalation workflow",
                "Priority support access",
                "Personalized engagement campaign",
                "Loyalty bonus activation",
                "Executive intervention",

            ],

            "medium_risk": [

                "Send targeted email campaign",
                "Offer product tutorial",
                "Recommend upgrade benefits",
                "Customer satisfaction survey",
                "Usage optimization tips",
                "Feature awareness campaign",
                "Limited-time retention offer",

            ],

            "low_risk": [

                "Upsell premium plans",
                "Referral incentives",
                "Community engagement",
                "Reward loyalty points",
                "Early renewal offer",

            ],

            "inactive_users": [

                "Reactivation email",
                "Win-back discount",
                "Feature announcement",
                "Limited access offer",

            ],

            "payment_issues": [

                "Flexible payment options",
                "Payment reminder automation",
                "Billing support assistance",

            ],

            "negative_sentiment": [

                "Customer grievance handling",
                "Escalate support ticket",
                "Manager-level outreach",

            ],

            "high_value": [

                "VIP customer support",
                "Exclusive premium benefits",
                "Dedicated account manager",

            ],

        }

    # ========================================================
    # GENERATE RECOMMENDATIONS
    # ========================================================

    def generate_recommendations(
        self,
        customer_data: pd.DataFrame,
        churn_probabilities,
        segment_column: Optional[str] = None,
    ) -> pd.DataFrame:

        logger.info(
            "Generating recommendations"
        )

        df = customer_data.copy()

        df["churn_probability"] = (
            churn_probabilities
        )

        df["risk_level"] = df[
            "churn_probability"
        ].apply(
            self.risk_level
        )

        recommendations = []

        for _, row in df.iterrows():

            recommendation = (
                self.customer_recommendation(
                    row
                )
            )

            recommendations.append(
                recommendation
            )

        recommendation_df = pd.DataFrame(
            recommendations
        )

        final_df = pd.concat(

            [df, recommendation_df],

            axis=1

        )

        return final_df

    # ========================================================
    # CUSTOMER RECOMMENDATION
    # ========================================================

    def customer_recommendation(
        self,
        customer_row
    ) -> Dict:

        risk = customer_row[
            "risk_level"
        ]

        probability = customer_row[
            "churn_probability"
        ]

        recommendations = []

        # ----------------------------------------------------
        # HIGH RISK
        # ----------------------------------------------------

        if risk == "high":

            recommendations.extend(

                self.recommendation_library[
                    "high_risk"
                ]

            )

        elif risk == "medium":

            recommendations.extend(

                self.recommendation_library[
                    "medium_risk"
                ]

            )

        else:

            recommendations.extend(

                self.recommendation_library[
                    "low_risk"
                ]

            )

        # ----------------------------------------------------
        # INACTIVE USERS
        # ----------------------------------------------------

        if (
            "tenure" in customer_row
            and
            customer_row["tenure"] < 3
        ):

            recommendations.extend(

                self.recommendation_library[
                    "inactive_users"
                ]

            )

        # ----------------------------------------------------
        # PAYMENT ISSUES
        # ----------------------------------------------------

        if (
            "payment_delay" in customer_row
            and
            customer_row["payment_delay"] > 2
        ):

            recommendations.extend(

                self.recommendation_library[
                    "payment_issues"
                ]

            )

        # ----------------------------------------------------
        # NEGATIVE SENTIMENT
        # ----------------------------------------------------

        if (
            "sentiment_score"
            in customer_row
            and
            customer_row[
                "sentiment_score"
            ] < 0.3
        ):

            recommendations.extend(

                self.recommendation_library[
                    "negative_sentiment"
                ]

            )

        # ----------------------------------------------------
        # HIGH VALUE
        # ----------------------------------------------------

        if (
            "monthly_revenue"
            in customer_row
            and
            customer_row[
                "monthly_revenue"
            ] > 10000
        ):

            recommendations.extend(

                self.recommendation_library[
                    "high_value"
                ]

            )

        # ----------------------------------------------------
        # REMOVE DUPLICATES
        # ----------------------------------------------------

        recommendations = list(
            set(recommendations)
        )

        # ----------------------------------------------------
        # PRIORITY
        # ----------------------------------------------------

        priority = self.priority_score(
            probability
        )

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        estimated_roi = (
            self.estimate_roi(
                customer_row,
                probability
            )
        )

        return {

            "recommendations":
                recommendations,

            "top_action":
                recommendations[0]
                if recommendations
                else "Monitor customer",

            "priority":
                priority,

            "confidence_score":
                round(
                    float(probability),
                    4
                ),

            "estimated_roi":
                estimated_roi,

            "generated_at":
                datetime.utcnow().isoformat(),

        }

    # ========================================================
    # RISK LEVEL
    # ========================================================

    def risk_level(
        self,
        probability: float
    ) -> str:

        if probability >= 0.8:

            return "high"

        elif probability >= 0.5:

            return "medium"

        return "low"

    # ========================================================
    # PRIORITY SCORE
    # ========================================================

    def priority_score(
        self,
        probability: float
    ) -> str:

        if probability >= 0.9:

            return "critical"

        elif probability >= 0.75:

            return "high"

        elif probability >= 0.5:

            return "medium"

        return "normal"

    # ========================================================
    # ESTIMATE ROI
    # ========================================================

    def estimate_roi(
        self,
        customer_row,
        probability: float
    ) -> Dict:

        monthly_revenue = float(

            customer_row.get(
                "monthly_revenue",
                1000
            )

        )

        retention_probability = (
            1 - probability
        )

        expected_saved = (

            monthly_revenue *
            12 *
            probability

        )

        estimated_cost = (
            expected_saved * 0.1
        )

        roi = (

            (expected_saved - estimated_cost)
            /
            (estimated_cost + 1e-6)

        ) * 100

        return {

            "estimated_saved_revenue":
                round(
                    expected_saved,
                    2
                ),

            "estimated_intervention_cost":
                round(
                    estimated_cost,
                    2
                ),

            "estimated_roi_percent":
                round(
                    roi,
                    2
                ),

            "retention_probability":
                round(
                    retention_probability,
                    4
                ),

        }

    # ========================================================
    # NEXT BEST ACTION
    # ========================================================

    def next_best_action(
        self,
        customer_row
    ) -> str:

        risk = self.risk_level(

            customer_row[
                "churn_probability"
            ]

        )

        if risk == "high":

            return (
                "Immediate human intervention"
            )

        elif risk == "medium":

            return (
                "Automated retention workflow"
            )

        return (
            "Upsell opportunity"
        )

    # ========================================================
    # SEGMENT RECOMMENDATIONS
    # ========================================================

    def segment_recommendations(
        self,
        df: pd.DataFrame,
        segment_column: str,
    ) -> Dict:

        segment_results = {}

        segments = df[
            segment_column
        ].unique()

        for segment in segments:

            subset = df[
                df[segment_column]
                == segment
            ]

            avg_churn = subset[
                "churn_probability"
            ].mean()

            risk = self.risk_level(
                avg_churn
            )

            actions = self.recommendation_library.get(

                f"{risk}_risk",
                []

            )

            segment_results[
                str(segment)
            ] = {

                "average_churn_probability":
                    round(
                        float(avg_churn),
                        4
                    ),

                "risk_level":
                    risk,

                "recommended_actions":
                    actions,

                "customer_count":
                    len(subset),

            }

        return segment_results

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    def executive_summary(
        self,
        recommendation_df: pd.DataFrame
    ) -> Dict:

        high_risk = len(

            recommendation_df[
                recommendation_df[
                    "risk_level"
                ] == "high"
            ]

        )

        medium_risk = len(

            recommendation_df[
                recommendation_df[
                    "risk_level"
                ] == "medium"
            ]

        )

        low_risk = len(

            recommendation_df[
                recommendation_df[
                    "risk_level"
                ] == "low"
            ]

        )

        estimated_revenue = 0

        if "estimated_roi" in recommendation_df:

            for roi in recommendation_df[
                "estimated_roi"
            ]:

                if isinstance(roi, dict):

                    estimated_revenue += roi.get(

                        "estimated_saved_revenue",
                        0

                    )

        return {

            "total_customers":
                len(recommendation_df),

            "high_risk_customers":
                high_risk,

            "medium_risk_customers":
                medium_risk,

            "low_risk_customers":
                low_risk,

            "estimated_revenue_saved":
                round(
                    estimated_revenue,
                    2
                ),

            "generated_at":
                datetime.utcnow().isoformat(),

        }

    # ========================================================
    # SAVE RECOMMENDATIONS
    # ========================================================

    def save_recommendations(
        self,
        recommendation_df: pd.DataFrame,
        filename: Optional[str] = None,
    ) -> str:

        if filename is None:

            filename = (

                f"recommendations_"
                f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                f".csv"

            )

        path = (
            self.save_dir /
            filename
        )

        recommendation_df.to_csv(

            path,
            index=False

        )

        logger.info(
            f"Recommendations saved: {path}"
        )

        return str(path)

    # ========================================================
    # PLAYBOOK GENERATION
    # ========================================================

    def generate_playbook(
        self,
        customer_row
    ) -> Dict:

        probability = customer_row[
            "churn_probability"
        ]

        risk = self.risk_level(
            probability
        )

        playbook = {

            "risk_level":
                risk,

            "timeline_days":
                7 if risk == "high"
                else 30,

            "steps":
                [],

        }

        if risk == "high":

            playbook["steps"] = [

                "Assign account manager",
                "Trigger emergency retention workflow",
                "Offer personalized discount",
                "Conduct customer success call",
                "Escalate unresolved complaints",

            ]

        elif risk == "medium":

            playbook["steps"] = [

                "Launch automated email sequence",
                "Share educational resources",
                "Offer feature tutorials",
                "Provide temporary incentives",

            ]

        else:

            playbook["steps"] = [

                "Promote loyalty program",
                "Recommend premium upgrades",
                "Collect referrals",

            ]

        return playbook


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def generate_recommendations(
    customer_data,
    churn_probabilities
):

    engine = RecommendationEngine()

    return engine.generate_recommendations(

        customer_data,
        churn_probabilities

    )


def segment_recommendations(
    df,
    segment_column
):

    engine = RecommendationEngine()

    return engine.segment_recommendations(

        df,
        segment_column

    )


def generate_playbook(
    customer_row
):

    engine = RecommendationEngine()

    return engine.generate_playbook(
        customer_row
    )