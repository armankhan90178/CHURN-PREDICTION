"""
ChurnShield 2.0 — Hyper AI Playbook Generator

Purpose:
Generate enterprise-grade retention playbooks
for every customer using ML + business intelligence.

Capabilities:
- personalized retention strategies
- churn-specific intervention plans
- multilingual business messaging
- revenue protection workflows
- customer success recommendations
- executive summaries
- next-best-action generation
- AI-powered retention campaigns
- lifecycle playbooks
- escalation matrix generation
- omnichannel communication plans
- retention ROI estimation
"""

import logging
import json
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, List

from config import (
    CHURN_REASONS,
    CUSTOMER_PERSONAS,
    SUPPORTED_LANGUAGES,
)

logger = logging.getLogger(
    "churnshield.playbook_generator"
)


class HyperPlaybookGenerator:

    def __init__(self):

        self.high_risk_threshold = 0.70
        self.medium_risk_threshold = 0.40

        self.playbook_templates = {

            "Price Sensitivity": [
                "Offer temporary discount or retention coupon",
                "Move customer to better-value plan",
                "Provide ROI breakdown report",
                "Bundle additional services",
            ],

            "Low Feature Adoption": [
                "Launch onboarding walkthrough",
                "Assign product adoption specialist",
                "Schedule product demo session",
                "Enable guided tutorials",
            ],

            "Support Failure": [
                "Escalate to senior customer success manager",
                "Conduct service recovery call",
                "Provide priority support access",
                "Analyze unresolved ticket history",
            ],

            "Competitor Switch": [
                "Launch competitive differentiation campaign",
                "Offer migration assistance",
                "Provide exclusive enterprise features",
                "Share customer success stories",
            ],

            "Product Dissatisfaction": [
                "Collect detailed feedback",
                "Initiate product consultation",
                "Recommend underused features",
                "Prioritize account monitoring",
            ],

            "Seasonal Disengagement": [
                "Launch re-engagement campaign",
                "Send personalized reminders",
                "Offer temporary activity incentives",
                "Schedule lifecycle nudges",
            ],

            "Life/Business Event": [
                "Offer account pause option",
                "Provide flexible payment terms",
                "Maintain low-touch engagement",
                "Schedule future follow-up",
            ],
        }

    # ─────────────────────────────────────────────
    # MAIN PLAYBOOK GENERATION
    # ─────────────────────────────────────────────

    def generate_playbooks(
        self,
        df: pd.DataFrame,
        language: str = "english",
    ) -> List[Dict]:

        logger.info(
            "Generating hyper retention playbooks"
        )

        data = df.copy()

        playbooks = []

        for _, row in data.iterrows():

            playbook = self._build_customer_playbook(
                row,
                language=language,
            )

            playbooks.append(playbook)

        logger.info(
            f"Generated {len(playbooks)} playbooks"
        )

        return playbooks

    # ─────────────────────────────────────────────
    # CUSTOMER PLAYBOOK
    # ─────────────────────────────────────────────

    def _build_customer_playbook(
        self,
        row,
        language="english",
    ):

        customer_id = row.get(
            "customer_id",
            "UNKNOWN",
        )

        customer_name = row.get(
            "customer_name",
            "Customer",
        )

        churn_probability = float(
            row.get(
                "churn_probability",
                0,
            )
        )

        churn_reason = row.get(
            "predicted_churn_reason",
            "Low Feature Adoption",
        )

        persona = row.get(
            "customer_persona",
            "Value Seeker",
        )

        monthly_revenue = float(
            row.get(
                "monthly_revenue",
                0,
            )
        )

        risk_level = self._risk_level(
            churn_probability
        )

        priority = self._priority_level(
            churn_probability,
            monthly_revenue,
        )

        retention_actions = (
            self._generate_retention_actions(
                churn_reason,
                persona,
                churn_probability,
            )
        )

        communication_plan = (
            self._generate_communication_plan(
                row,
                language,
            )
        )

        retention_offer = (
            self._generate_retention_offer(
                row
            )
        )

        success_probability = (
            self._estimate_retention_success(
                row
            )
        )

        estimated_saved_revenue = (
            monthly_revenue * 12 *
            success_probability
        )

        executive_summary = (
            self._executive_summary(
                row,
                risk_level,
            )
        )

        escalation_plan = (
            self._escalation_plan(
                risk_level,
                monthly_revenue,
            )
        )

        next_best_action = (
            self._next_best_action(
                churn_reason,
                persona,
            )
        )

        timeline = (
            self._retention_timeline(
                risk_level
            )
        )

        playbook = {

            "customer_id":
                customer_id,

            "customer_name":
                customer_name,

            "risk_level":
                risk_level,

            "priority":
                priority,

            "churn_probability":
                round(
                    churn_probability,
                    4,
                ),

            "predicted_reason":
                churn_reason,

            "customer_persona":
                persona,

            "monthly_revenue":
                round(
                    monthly_revenue,
                    2,
                ),

            "estimated_annual_revenue":
                round(
                    monthly_revenue * 12,
                    2,
                ),

            "estimated_saved_revenue":
                round(
                    estimated_saved_revenue,
                    2,
                ),

            "retention_success_probability":
                round(
                    success_probability,
                    4,
                ),

            "executive_summary":
                executive_summary,

            "next_best_action":
                next_best_action,

            "retention_actions":
                retention_actions,

            "communication_plan":
                communication_plan,

            "retention_offer":
                retention_offer,

            "escalation_plan":
                escalation_plan,

            "timeline":
                timeline,

            "generated_at":
                datetime.utcnow().isoformat(),
        }

        return playbook

    # ─────────────────────────────────────────────
    # RISK ENGINE
    # ─────────────────────────────────────────────

    def _risk_level(
        self,
        probability,
    ):

        if probability >= 0.85:
            return "CRITICAL"

        elif probability >= 0.70:
            return "HIGH"

        elif probability >= 0.40:
            return "MEDIUM"

        return "LOW"

    def _priority_level(
        self,
        probability,
        revenue,
    ):

        score = (
            probability * 0.7
            +
            min(revenue / 50000, 1) * 0.3
        )

        if score >= 0.80:
            return "P1"

        elif score >= 0.60:
            return "P2"

        elif score >= 0.40:
            return "P3"

        return "P4"

    # ─────────────────────────────────────────────
    # RETENTION ACTIONS
    # ─────────────────────────────────────────────

    def _generate_retention_actions(
        self,
        reason,
        persona,
        probability,
    ):

        actions = []

        base_actions = self.playbook_templates.get(
            reason,
            [],
        )

        actions.extend(base_actions)

        if persona == "Power User":

            actions.append(
                "Offer beta access to advanced features"
            )

        elif persona == "ROI Tracker":

            actions.append(
                "Provide measurable ROI performance dashboard"
            )

        elif persona == "Relationship Buyer":

            actions.append(
                "Assign dedicated relationship manager"
            )

        if probability >= 0.85:

            actions.append(
                "Initiate executive escalation within 24 hours"
            )

            actions.append(
                "Conduct urgent retention review meeting"
            )

        return list(dict.fromkeys(actions))

    # ─────────────────────────────────────────────
    # COMMUNICATION PLAN
    # ─────────────────────────────────────────────

    def _generate_communication_plan(
        self,
        row,
        language,
    ):

        customer_name = row.get(
            "customer_name",
            "Customer",
        )

        reason = row.get(
            "predicted_churn_reason",
            "engagement decline",
        )

        templates = {

            "english": {

                "email":

                    f"""
Dear {customer_name},

We noticed reduced engagement and want to ensure
you continue getting value from our platform.

Our team is ready to help optimize your experience.

Regards,
Customer Success Team
""",

                "sms":

                    f"""
Hi {customer_name},
we noticed some inactivity.
Our team would love to help you maximize value.
""",

                "whatsapp":

                    f"""
Hello {customer_name} 👋

We observed possible signs of disengagement related to:
{reason}

Would you like a quick optimization session?
""",
            }
        }

        return templates.get(
            language,
            templates["english"],
        )

    # ─────────────────────────────────────────────
    # RETENTION OFFER
    # ─────────────────────────────────────────────

    def _generate_retention_offer(
        self,
        row,
    ):

        revenue = float(
            row.get(
                "monthly_revenue",
                0,
            )
        )

        probability = float(
            row.get(
                "churn_probability",
                0,
            )
        )

        if revenue >= 50000:

            return {
                "type": "Enterprise Success Package",
                "discount": "15%",
                "priority_support": True,
                "executive_manager": True,
            }

        if probability >= 0.80:

            return {
                "type": "Critical Retention Offer",
                "discount": "25%",
                "priority_support": True,
                "training_sessions": 3,
            }

        return {
            "type": "Standard Retention Package",
            "discount": "10%",
            "product_training": True,
        }

    # ─────────────────────────────────────────────
    # RETENTION SUCCESS
    # ─────────────────────────────────────────────

    def _estimate_retention_success(
        self,
        row,
    ):

        engagement = float(
            row.get(
                "engagement_score",
                0.5,
            )
        )

        feature_usage = float(
            row.get(
                "feature_usage_score",
                0.5,
            )
        )

        nps = float(
            row.get(
                "nps_score",
                5,
            )
        ) / 10

        probability = float(
            row.get(
                "churn_probability",
                0.5,
            )
        )

        score = (

            engagement * 0.30

            +

            feature_usage * 0.25

            +

            nps * 0.20

            +

            (1 - probability) * 0.25

        )

        return float(
            np.clip(
                score,
                0.05,
                0.95,
            )
        )

    # ─────────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────

    def _executive_summary(
        self,
        row,
        risk_level,
    ):

        customer_name = row.get(
            "customer_name",
            "Customer",
        )

        reason = row.get(
            "predicted_churn_reason",
            "engagement decline",
        )

        revenue = row.get(
            "monthly_revenue",
            0,
        )

        return (

            f"{customer_name} is classified as "
            f"{risk_level} risk due to "
            f"{reason.lower()}. "
            f"Monthly revenue exposure estimated at "
            f"₹{int(revenue):,}. "
            f"Immediate proactive retention intervention recommended."

        )

    # ─────────────────────────────────────────────
    # ESCALATION PLAN
    # ─────────────────────────────────────────────

    def _escalation_plan(
        self,
        risk_level,
        revenue,
    ):

        if risk_level == "CRITICAL":

            return {
                "owner": "VP Customer Success",
                "sla": "4 hours",
                "priority": "Emergency",
            }

        if revenue >= 50000:

            return {
                "owner": "Enterprise Success Manager",
                "sla": "12 hours",
                "priority": "High",
            }

        return {
            "owner": "Customer Success Team",
            "sla": "24 hours",
            "priority": "Normal",
        }

    # ─────────────────────────────────────────────
    # NEXT BEST ACTION
    # ─────────────────────────────────────────────

    def _next_best_action(
        self,
        reason,
        persona,
    ):

        mapping = {

            "Price Sensitivity":
                "Provide personalized pricing optimization.",

            "Low Feature Adoption":
                "Schedule onboarding and adoption workshop.",

            "Support Failure":
                "Escalate unresolved issues immediately.",

            "Competitor Switch":
                "Deliver competitive differentiation pitch.",

            "Product Dissatisfaction":
                "Collect detailed customer feedback.",

            "Seasonal Disengagement":
                "Launch re-engagement campaign.",
        }

        action = mapping.get(
            reason,
            "Initiate proactive customer outreach.",
        )

        if persona == "Relationship Buyer":

            action += (
                " Prioritize human-led communication."
            )

        return action

    # ─────────────────────────────────────────────
    # TIMELINE
    # ─────────────────────────────────────────────

    def _retention_timeline(
        self,
        risk_level,
    ):

        if risk_level == "CRITICAL":

            return {

                "0-24h":
                    "Executive escalation",

                "24-48h":
                    "Retention negotiation",

                "3-7d":
                    "Product optimization session",

                "7-30d":
                    "Continuous monitoring",
            }

        if risk_level == "HIGH":

            return {

                "0-48h":
                    "Customer success outreach",

                "3-7d":
                    "Feature adoption campaign",

                "7-30d":
                    "Health monitoring",
            }

        return {

            "7d":
                "Automated engagement sequence",

            "30d":
                "Lifecycle review",
        }


# ─────────────────────────────────────────────
# FUNCTIONAL API
# ─────────────────────────────────────────────

def generate_playbooks(
    df: pd.DataFrame,
    language: str = "english",
):

    generator = HyperPlaybookGenerator()

    return generator.generate_playbooks(
        df=df,
        language=language,
    )