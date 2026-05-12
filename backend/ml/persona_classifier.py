"""
ChurnShield 2.0 — Hyper Persona Intelligence Engine

Purpose:
Classify customers into intelligent behavioral personas
using engagement, spending, support behavior,
product adoption, payment patterns, and lifecycle analytics.

Capabilities:
- AI-driven persona classification
- behavioral segmentation
- lifecycle persona mapping
- churn-prone identity detection
- customer psychology inference
- engagement archetypes
- business value clustering
- adaptive persona scoring
- persona confidence scoring
- hybrid rule + scoring engine
- enterprise-grade segmentation intelligence
"""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import Dict

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.persona_classifier"
)


class HyperPersonaClassifier:

    def __init__(self):

        self.personas = {
            "Power User":
                "Highly engaged heavy user with strong retention",

            "Passive Subscriber":
                "Low engagement customer likely to churn",

            "Value Seeker":
                "Price-sensitive optimization-focused customer",

            "Relationship Buyer":
                "Trust-driven loyal long-term customer",

            "ROI Tracker":
                "Business-driven customer measuring efficiency",

            "Support Heavy":
                "Customer requiring excessive support attention",

            "Expansion Ready":
                "High-growth customer likely to upgrade",

            "Silent Risk":
                "Low-noise but highly dangerous hidden churn risk",

            "At Risk Enterprise":
                "High revenue enterprise customer showing churn signs",

            "Champion":
                "Elite customer with strong engagement and advocacy",
        }

    # ─────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────

    def classify(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Starting hyper persona classification"
        )

        data = df.copy()

        persona_results = []

        for idx, row in data.iterrows():

            persona_scores = self._calculate_persona_scores(
                row
            )

            dominant_persona = max(
                persona_scores,
                key=persona_scores.get
            )

            confidence = self._calculate_confidence(
                persona_scores
            )

            secondary_persona = self._secondary_persona(
                persona_scores
            )

            behavior_summary = self._generate_behavior_summary(
                row,
                dominant_persona,
            )

            risk_level = self._risk_profile(
                row,
                dominant_persona,
            )

            expansion_score = self._expansion_score(
                row
            )

            loyalty_score = self._loyalty_score(
                row
            )

            engagement_band = self._engagement_band(
                row
            )

            persona_results.append({

                "persona":
                    dominant_persona,

                "persona_description":
                    self.personas.get(
                        dominant_persona,
                        "General customer profile"
                    ),

                "persona_confidence":
                    round(confidence, 4),

                "secondary_persona":
                    secondary_persona,

                "behavior_summary":
                    behavior_summary,

                "persona_risk_level":
                    risk_level,

                "expansion_score":
                    expansion_score,

                "loyalty_score":
                    loyalty_score,

                "engagement_band":
                    engagement_band,
            })

        persona_df = pd.DataFrame(
            persona_results
        )

        data = pd.concat(
            [
                data.reset_index(drop=True),
                persona_df.reset_index(drop=True),
            ],
            axis=1,
        )

        logger.info(
            "Persona classification completed"
        )

        return data

    # ─────────────────────────────────────────────
    # PERSONA SCORING ENGINE
    # ─────────────────────────────────────────────

    def _calculate_persona_scores(
        self,
        row,
    ) -> Dict:

        scores = {
            k: 0
            for k in self.personas
        }

        revenue = row.get(
            "monthly_revenue",
            0,
        )

        engagement = row.get(
            "engagement_score",
            0.5,
        )

        feature_usage = row.get(
            "feature_usage_score",
            0.5,
        )

        tickets = row.get(
            "support_tickets",
            0,
        )

        tenure = row.get(
            "contract_age_months",
            0,
        )

        payment_delays = row.get(
            "payment_delays",
            0,
        )

        utilization = row.get(
            "seat_utilization",
            0.5,
        )

        nps = row.get(
            "nps_score",
            5,
        )

        # ─────────────────────────────────────
        # POWER USER
        # ─────────────────────────────────────

        scores["Power User"] += (
            engagement * 35
        )

        scores["Power User"] += (
            feature_usage * 35
        )

        scores["Power User"] += (
            utilization * 15
        )

        if tickets <= 2:
            scores["Power User"] += 15

        # ─────────────────────────────────────
        # PASSIVE SUBSCRIBER
        # ─────────────────────────────────────

        scores["Passive Subscriber"] += (
            (1 - engagement) * 40
        )

        scores["Passive Subscriber"] += (
            (1 - feature_usage) * 30
        )

        if tenure <= 6:
            scores["Passive Subscriber"] += 20

        if payment_delays >= 2:
            scores["Passive Subscriber"] += 10

        # ─────────────────────────────────────
        # VALUE SEEKER
        # ─────────────────────────────────────

        if revenue < 3000:
            scores["Value Seeker"] += 35

        if payment_delays > 0:
            scores["Value Seeker"] += 25

        if feature_usage > 0.5:
            scores["Value Seeker"] += 15

        scores["Value Seeker"] += (
            (1 - tickets / 10) * 25
        )

        # ─────────────────────────────────────
        # RELATIONSHIP BUYER
        # ─────────────────────────────────────

        scores["Relationship Buyer"] += (
            min(tenure / 36, 1) * 40
        )

        scores["Relationship Buyer"] += (
            min(nps / 10, 1) * 35
        )

        if payment_delays == 0:
            scores["Relationship Buyer"] += 25

        # ─────────────────────────────────────
        # ROI TRACKER
        # ─────────────────────────────────────

        scores["ROI Tracker"] += (
            utilization * 35
        )

        scores["ROI Tracker"] += (
            feature_usage * 30
        )

        if revenue >= 10000:
            scores["ROI Tracker"] += 20

        if tickets <= 2:
            scores["ROI Tracker"] += 15

        # ─────────────────────────────────────
        # SUPPORT HEAVY
        # ─────────────────────────────────────

        scores["Support Heavy"] += (
            min(tickets / 10, 1) * 70
        )

        if engagement > 0.4:
            scores["Support Heavy"] += 15

        if revenue > 5000:
            scores["Support Heavy"] += 15

        # ─────────────────────────────────────
        # EXPANSION READY
        # ─────────────────────────────────────

        scores["Expansion Ready"] += (
            engagement * 30
        )

        scores["Expansion Ready"] += (
            feature_usage * 25
        )

        scores["Expansion Ready"] += (
            utilization * 25
        )

        if revenue >= 7000:
            scores["Expansion Ready"] += 20

        # ─────────────────────────────────────
        # SILENT RISK
        # ─────────────────────────────────────

        if engagement < 0.3:
            scores["Silent Risk"] += 35

        if tickets <= 1:
            scores["Silent Risk"] += 20

        if feature_usage < 0.3:
            scores["Silent Risk"] += 25

        if tenure > 12:
            scores["Silent Risk"] += 20

        # ─────────────────────────────────────
        # AT RISK ENTERPRISE
        # ─────────────────────────────────────

        if revenue >= 20000:
            scores["At Risk Enterprise"] += 35

        if engagement < 0.5:
            scores["At Risk Enterprise"] += 30

        if payment_delays >= 2:
            scores["At Risk Enterprise"] += 20

        if tickets >= 5:
            scores["At Risk Enterprise"] += 15

        # ─────────────────────────────────────
        # CHAMPION
        # ─────────────────────────────────────

        scores["Champion"] += (
            engagement * 30
        )

        scores["Champion"] += (
            feature_usage * 30
        )

        scores["Champion"] += (
            min(nps / 10, 1) * 20
        )

        if payment_delays == 0:
            scores["Champion"] += 10

        if tenure >= 18:
            scores["Champion"] += 10

        return scores

    # ─────────────────────────────────────────────
    # CONFIDENCE ENGINE
    # ─────────────────────────────────────────────

    def _calculate_confidence(
        self,
        scores,
    ):

        sorted_scores = sorted(
            scores.values(),
            reverse=True,
        )

        top = sorted_scores[0]

        second = sorted_scores[1]

        confidence = (
            top - second
        ) / max(top, 1)

        return np.clip(
            confidence,
            0,
            1,
        )

    # ─────────────────────────────────────────────
    # SECONDARY PERSONA
    # ─────────────────────────────────────────────

    def _secondary_persona(
        self,
        scores,
    ):

        sorted_personas = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_personas[1][0]

    # ─────────────────────────────────────────────
    # BEHAVIOR SUMMARY
    # ─────────────────────────────────────────────

    def _generate_behavior_summary(
        self,
        row,
        persona,
    ):

        revenue = row.get(
            "monthly_revenue",
            0
        )

        engagement = row.get(
            "engagement_score",
            0
        )

        tenure = row.get(
            "contract_age_months",
            0
        )

        summary = (
            f"{persona} profile | "
            f"Revenue ₹{int(revenue):,} | "
            f"Engagement {round(engagement,2)} | "
            f"Tenure {tenure} months"
        )

        return summary

    # ─────────────────────────────────────────────
    # RISK PROFILE
    # ─────────────────────────────────────────────

    def _risk_profile(
        self,
        row,
        persona,
    ):

        engagement = row.get(
            "engagement_score",
            0.5
        )

        delays = row.get(
            "payment_delays",
            0
        )

        if persona in [
            "Silent Risk",
            "Passive Subscriber",
            "At Risk Enterprise",
        ]:
            return "Critical"

        if engagement < 0.3:
            return "High"

        if delays >= 2:
            return "Moderate"

        return "Low"

    # ─────────────────────────────────────────────
    # EXPANSION SCORE
    # ─────────────────────────────────────────────

    def _expansion_score(
        self,
        row,
    ):

        score = 0

        score += (
            row.get(
                "engagement_score",
                0
            ) * 40
        )

        score += (
            row.get(
                "feature_usage_score",
                0
            ) * 40
        )

        score += (
            row.get(
                "seat_utilization",
                0
            ) * 20
        )

        return round(
            np.clip(score, 0, 100),
            2
        )

    # ─────────────────────────────────────────────
    # LOYALTY SCORE
    # ─────────────────────────────────────────────

    def _loyalty_score(
        self,
        row,
    ):

        score = 0

        tenure = row.get(
            "contract_age_months",
            0
        )

        nps = row.get(
            "nps_score",
            5
        )

        delays = row.get(
            "payment_delays",
            0
        )

        score += min(
            tenure / 36,
            1
        ) * 50

        score += (
            nps / 10
        ) * 40

        if delays == 0:
            score += 10

        return round(
            np.clip(score, 0, 100),
            2
        )

    # ─────────────────────────────────────────────
    # ENGAGEMENT BAND
    # ─────────────────────────────────────────────

    def _engagement_band(
        self,
        row,
    ):

        engagement = row.get(
            "engagement_score",
            0.5
        )

        if engagement < 0.25:
            return "Very Low"

        if engagement < 0.50:
            return "Low"

        if engagement < 0.75:
            return "Medium"

        return "High"


# ─────────────────────────────────────────────
# FUNCTIONAL INTERFACE
# ─────────────────────────────────────────────

def classify_personas(
    df: pd.DataFrame,
):

    classifier = HyperPersonaClassifier()

    return classifier.classify(df)