"""
ChurnShield 2.0 — Hyper Churn Reason Classifier

Purpose:
Identify WHY customers are likely to churn.

Capabilities:
- behavioral churn diagnosis
- AI-style churn attribution
- root-cause classification
- revenue-loss reasoning
- support frustration detection
- product adoption analysis
- hidden churn signal extraction
- customer psychology inference
- business intelligence narratives
- multi-reason weighted scoring
- executive-level retention insights
"""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import Dict

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.reason_classifier"
)


class HyperReasonClassifier:

    def __init__(self):

        self.reason_catalog = {

            "Price Sensitivity":
                "Customer perceives low value relative to pricing",

            "Low Product Adoption":
                "Customer is not fully using the platform",

            "Support Frustration":
                "Frequent issues and unresolved tickets detected",

            "Engagement Decline":
                "Usage behavior steadily decreasing",

            "Payment Risk":
                "Repeated delayed payments or failed billing",

            "Competitor Attraction":
                "Customer likely evaluating alternatives",

            "Onboarding Failure":
                "Customer never achieved activation success",

            "ROI Dissatisfaction":
                "Business outcomes not meeting expectations",

            "Relationship Breakdown":
                "Customer loyalty weakening",

            "Expansion Saturation":
                "Customer reached growth ceiling",

            "Seasonal Disengagement":
                "Temporary usage decline due to seasonal behavior",

            "Enterprise Risk":
                "High-value account showing instability",

            "Feature Complexity":
                "Product complexity causing friction",

            "Silent Churn Risk":
                "Quiet disengagement without support interaction",

            "Operational Friction":
                "Workflow/process inefficiencies affecting retention",
        }

    # ─────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────

    def classify(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Starting churn reason classification"
        )

        data = df.copy()

        reason_results = []

        for _, row in data.iterrows():

            scores = self._calculate_reason_scores(
                row
            )

            primary_reason = max(
                scores,
                key=scores.get
            )

            secondary_reasons = self._top_secondary_reasons(
                scores
            )

            confidence = self._confidence_score(
                scores
            )

            business_impact = self._business_impact(
                row,
                primary_reason,
            )

            intervention = self._recommended_intervention(
                primary_reason
            )

            urgency = self._urgency_level(
                row,
                primary_reason,
            )

            narrative = self._executive_narrative(
                row,
                primary_reason,
            )

            weighted_reasons = self._weighted_reason_vector(
                scores
            )

            reason_results.append({

                "primary_churn_reason":
                    primary_reason,

                "reason_description":
                    self.reason_catalog.get(
                        primary_reason,
                        "General churn risk"
                    ),

                "secondary_churn_reasons":
                    secondary_reasons,

                "reason_confidence":
                    round(confidence, 4),

                "business_impact":
                    business_impact,

                "recommended_retention_action":
                    intervention,

                "retention_urgency":
                    urgency,

                "executive_reason_narrative":
                    narrative,

                "weighted_reason_vector":
                    weighted_reasons,
            })

        reason_df = pd.DataFrame(
            reason_results
        )

        data = pd.concat(
            [
                data.reset_index(drop=True),
                reason_df.reset_index(drop=True),
            ],
            axis=1,
        )

        logger.info(
            "Reason classification completed"
        )

        return data

    # ─────────────────────────────────────────────
    # SCORING ENGINE
    # ─────────────────────────────────────────────

    def _calculate_reason_scores(
        self,
        row,
    ) -> Dict:

        scores = {
            k: 0
            for k in self.reason_catalog
        }

        engagement = row.get(
            "engagement_score",
            0.5
        )

        feature_usage = row.get(
            "feature_usage_score",
            0.5
        )

        tickets = row.get(
            "support_tickets",
            0
        )

        delays = row.get(
            "payment_delays",
            0
        )

        tenure = row.get(
            "contract_age_months",
            0
        )

        revenue = row.get(
            "monthly_revenue",
            0
        )

        nps = row.get(
            "nps_score",
            5
        )

        inactivity = row.get(
            "days_since_last_login",
            0
        )

        utilization = row.get(
            "seat_utilization",
            0.5
        )

        # ─────────────────────────────────────
        # PRICE SENSITIVITY
        # ─────────────────────────────────────

        if revenue < 3000:
            scores["Price Sensitivity"] += 40

        if delays > 0:
            scores["Price Sensitivity"] += 30

        if feature_usage < 0.40:
            scores["Price Sensitivity"] += 20

        if nps < 6:
            scores["Price Sensitivity"] += 10

        # ─────────────────────────────────────
        # LOW PRODUCT ADOPTION
        # ─────────────────────────────────────

        scores["Low Product Adoption"] += (
            (1 - feature_usage) * 50
        )

        scores["Low Product Adoption"] += (
            (1 - utilization) * 25
        )

        scores["Low Product Adoption"] += (
            (1 - engagement) * 25
        )

        # ─────────────────────────────────────
        # SUPPORT FRUSTRATION
        # ─────────────────────────────────────

        scores["Support Frustration"] += (
            min(tickets / 10, 1) * 70
        )

        if nps < 5:
            scores["Support Frustration"] += 20

        if engagement < 0.40:
            scores["Support Frustration"] += 10

        # ─────────────────────────────────────
        # ENGAGEMENT DECLINE
        # ─────────────────────────────────────

        scores["Engagement Decline"] += (
            (1 - engagement) * 50
        )

        if inactivity > 15:
            scores["Engagement Decline"] += 30

        if feature_usage < 0.30:
            scores["Engagement Decline"] += 20

        # ─────────────────────────────────────
        # PAYMENT RISK
        # ─────────────────────────────────────

        scores["Payment Risk"] += (
            min(delays / 5, 1) * 80
        )

        if revenue < 2000:
            scores["Payment Risk"] += 20

        # ─────────────────────────────────────
        # COMPETITOR ATTRACTION
        # ─────────────────────────────────────

        if engagement < 0.50:
            scores["Competitor Attraction"] += 30

        if nps < 6:
            scores["Competitor Attraction"] += 30

        if tenure > 12:
            scores["Competitor Attraction"] += 20

        if feature_usage < 0.40:
            scores["Competitor Attraction"] += 20

        # ─────────────────────────────────────
        # ONBOARDING FAILURE
        # ─────────────────────────────────────

        if tenure <= 3:
            scores["Onboarding Failure"] += 50

        if feature_usage < 0.25:
            scores["Onboarding Failure"] += 30

        if engagement < 0.30:
            scores["Onboarding Failure"] += 20

        # ─────────────────────────────────────
        # ROI DISSATISFACTION
        # ─────────────────────────────────────

        if revenue > 10000:
            scores["ROI Dissatisfaction"] += 30

        if utilization < 0.50:
            scores["ROI Dissatisfaction"] += 35

        if engagement < 0.50:
            scores["ROI Dissatisfaction"] += 20

        if nps < 7:
            scores["ROI Dissatisfaction"] += 15

        # ─────────────────────────────────────
        # RELATIONSHIP BREAKDOWN
        # ─────────────────────────────────────

        if tenure > 12:
            scores["Relationship Breakdown"] += 20

        if nps < 5:
            scores["Relationship Breakdown"] += 50

        if engagement < 0.40:
            scores["Relationship Breakdown"] += 30

        # ─────────────────────────────────────
        # EXPANSION SATURATION
        # ─────────────────────────────────────

        if utilization > 0.90:
            scores["Expansion Saturation"] += 40

        if feature_usage > 0.80:
            scores["Expansion Saturation"] += 30

        if engagement < 0.50:
            scores["Expansion Saturation"] += 30

        # ─────────────────────────────────────
        # SEASONAL DISENGAGEMENT
        # ─────────────────────────────────────

        if inactivity > 20:
            scores["Seasonal Disengagement"] += 50

        if tickets == 0:
            scores["Seasonal Disengagement"] += 20

        if tenure > 6:
            scores["Seasonal Disengagement"] += 30

        # ─────────────────────────────────────
        # ENTERPRISE RISK
        # ─────────────────────────────────────

        if revenue > 25000:
            scores["Enterprise Risk"] += 40

        if engagement < 0.50:
            scores["Enterprise Risk"] += 30

        if tickets >= 5:
            scores["Enterprise Risk"] += 30

        # ─────────────────────────────────────
        # FEATURE COMPLEXITY
        # ─────────────────────────────────────

        if tickets >= 4:
            scores["Feature Complexity"] += 40

        if feature_usage < 0.30:
            scores["Feature Complexity"] += 40

        if engagement < 0.50:
            scores["Feature Complexity"] += 20

        # ─────────────────────────────────────
        # SILENT CHURN RISK
        # ─────────────────────────────────────

        if inactivity > 20:
            scores["Silent Churn Risk"] += 40

        if tickets == 0:
            scores["Silent Churn Risk"] += 35

        if engagement < 0.30:
            scores["Silent Churn Risk"] += 25

        # ─────────────────────────────────────
        # OPERATIONAL FRICTION
        # ─────────────────────────────────────

        if tickets > 3:
            scores["Operational Friction"] += 35

        if utilization < 0.50:
            scores["Operational Friction"] += 30

        if engagement < 0.50:
            scores["Operational Friction"] += 35

        return scores

    # ─────────────────────────────────────────────
    # SECONDARY REASONS
    # ─────────────────────────────────────────────

    def _top_secondary_reasons(
        self,
        scores,
    ):

        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        secondary = [
            x[0]
            for x in sorted_scores[1:4]
        ]

        return secondary

    # ─────────────────────────────────────────────
    # CONFIDENCE
    # ─────────────────────────────────────────────

    def _confidence_score(
        self,
        scores,
    ):

        vals = sorted(
            scores.values(),
            reverse=True,
        )

        top = vals[0]

        second = vals[1]

        confidence = (
            top - second
        ) / max(top, 1)

        return np.clip(
            confidence,
            0,
            1,
        )

    # ─────────────────────────────────────────────
    # BUSINESS IMPACT
    # ─────────────────────────────────────────────

    def _business_impact(
        self,
        row,
        reason,
    ):

        revenue = row.get(
            "monthly_revenue",
            0
        )

        if revenue > 25000:
            return "Enterprise Revenue Risk"

        if revenue > 10000:
            return "High Revenue Impact"

        if reason in [
            "Support Frustration",
            "Relationship Breakdown",
        ]:
            return "Brand Reputation Risk"

        return "Moderate Revenue Impact"

    # ─────────────────────────────────────────────
    # RETENTION ACTIONS
    # ─────────────────────────────────────────────

    def _recommended_intervention(
        self,
        reason,
    ):

        actions = {

            "Price Sensitivity":
                "Offer smart discount or flexible pricing",

            "Low Product Adoption":
                "Launch onboarding + product education campaign",

            "Support Frustration":
                "Assign senior customer success manager",

            "Engagement Decline":
                "Run reactivation workflow",

            "Payment Risk":
                "Offer payment restructuring support",

            "Competitor Attraction":
                "Highlight differentiation + ROI",

            "Onboarding Failure":
                "Provide activation assistance",

            "ROI Dissatisfaction":
                "Deliver executive ROI review",

            "Relationship Breakdown":
                "Schedule strategic relationship meeting",

            "Expansion Saturation":
                "Introduce advanced modules/upgrades",

            "Seasonal Disengagement":
                "Use low-frequency retention nurturing",

            "Enterprise Risk":
                "Executive escalation immediately",

            "Feature Complexity":
                "Provide guided training sessions",

            "Silent Churn Risk":
                "Initiate hidden-risk outreach campaign",

            "Operational Friction":
                "Simplify workflows and processes",
        }

        return actions.get(
            reason,
            "General retention outreach"
        )

    # ─────────────────────────────────────────────
    # URGENCY
    # ─────────────────────────────────────────────

    def _urgency_level(
        self,
        row,
        reason,
    ):

        revenue = row.get(
            "monthly_revenue",
            0
        )

        if revenue > 25000:
            return "Critical"

        if reason in [
            "Enterprise Risk",
            "Relationship Breakdown",
            "Support Frustration",
        ]:
            return "High"

        return "Moderate"

    # ─────────────────────────────────────────────
    # EXECUTIVE NARRATIVE
    # ─────────────────────────────────────────────

    def _executive_narrative(
        self,
        row,
        reason,
    ):

        revenue = row.get(
            "monthly_revenue",
            0
        )

        engagement = row.get(
            "engagement_score",
            0
        )

        feature_usage = row.get(
            "feature_usage_score",
            0
        )

        return (
            f"Customer exhibits churn signals primarily driven by "
            f"{reason.lower()}. Revenue contribution ₹{int(revenue):,}, "
            f"engagement level {round(engagement,2)}, "
            f"feature adoption {round(feature_usage,2)}."
        )

    # ─────────────────────────────────────────────
    # WEIGHTED VECTOR
    # ─────────────────────────────────────────────

    def _weighted_reason_vector(
        self,
        scores,
    ):

        total = sum(
            scores.values()
        )

        if total == 0:
            total = 1

        normalized = {

            k: round(
                v / total,
                4
            )

            for k, v in scores.items()
        }

        top_reasons = dict(

            sorted(
                normalized.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

        )

        return top_reasons


# ─────────────────────────────────────────────
# FUNCTIONAL INTERFACE
# ─────────────────────────────────────────────

def classify_churn_reasons(
    df: pd.DataFrame,
):

    classifier = HyperReasonClassifier()

    return classifier.classify(df)