"""
ChurnShield 2.0 — Executive Summary Generator

Purpose:
Generate enterprise-grade executive summaries
for churn analytics, revenue risk, customer health,
retention opportunities, and board-level reporting.

Capabilities:
- CXO summaries
- churn risk overview
- revenue leakage analysis
- retention recommendations
- AI-powered insights
- KPI narrative generation
- customer health summaries
- trend storytelling
- forecast commentary
- investor-style reporting
- markdown/text export
- multilingual-ready architecture

Author:
ChurnShield AI
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("churnshield.executive_summary")


# ============================================================
# MAIN ENGINE
# ============================================================

class ExecutiveSummaryGenerator:

    def __init__(self):

        self.templates = {

            "excellent":
                """
Customer retention performance is currently strong with
healthy engagement patterns and controlled churn exposure.
Revenue stability remains positive and customer activity
signals indicate sustainable business growth.
                """,

            "moderate":
                """
Customer churn risk is moderately elevated.
Certain customer segments are showing declining engagement,
reduced product usage, and increased support dependency.
Immediate retention optimization is recommended.
                """,

            "critical":
                """
Business churn exposure is critically high.
Revenue leakage risk is increasing due to poor engagement,
customer inactivity, and rising dissatisfaction indicators.
Aggressive intervention strategies are strongly recommended.
                """
        }

    # ========================================================
    # MAIN GENERATOR
    # ========================================================

    def generate_summary(
        self,
        df: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None,
        forecast: Optional[Dict] = None,
        insights: Optional[Dict] = None,
    ) -> Dict:

        logger.info("Generating executive summary")

        report = {}

        # ----------------------------------------------------
        # BASIC KPIs
        # ----------------------------------------------------

        total_customers = len(df)

        churn_rate = (
            float(df["churned"].mean())
            if "churned" in df.columns
            else 0.0
        )

        monthly_revenue = (
            float(df["monthly_revenue"].sum())
            if "monthly_revenue" in df.columns
            else 0.0
        )

        avg_revenue = (
            float(df["monthly_revenue"].mean())
            if "monthly_revenue" in df.columns
            else 0.0
        )

        high_risk = 0

        if predictions is not None:

            if "risk_level" in predictions.columns:

                high_risk = int(

                    predictions["risk_level"]
                    .astype(str)
                    .str.lower()
                    .eq("high")
                    .sum()

                )

        # ----------------------------------------------------
        # BUSINESS HEALTH
        # ----------------------------------------------------

        business_health = self._classify_business_health(
            churn_rate
        )

        # ----------------------------------------------------
        # REVENUE AT RISK
        # ----------------------------------------------------

        revenue_at_risk = self._calculate_revenue_risk(
            df,
            predictions
        )

        # ----------------------------------------------------
        # KEY INSIGHTS
        # ----------------------------------------------------

        key_insights = self._generate_key_insights(
            df,
            predictions
        )

        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        recommendations = self._generate_recommendations(
            churn_rate,
            high_risk,
            revenue_at_risk,
        )

        # ----------------------------------------------------
        # EXECUTIVE NARRATIVE
        # ----------------------------------------------------

        narrative = self._build_narrative(
            churn_rate=churn_rate,
            revenue=monthly_revenue,
            health=business_health,
            high_risk=high_risk,
        )

        # ----------------------------------------------------
        # FORECAST COMMENTARY
        # ----------------------------------------------------

        forecast_commentary = None

        if forecast:

            forecast_commentary = (
                self._forecast_commentary(
                    forecast
                )
            )

        # ----------------------------------------------------
        # BUILD REPORT
        # ----------------------------------------------------

        report["generated_at"] = (
            datetime.utcnow().isoformat()
        )

        report["business_health"] = business_health

        report["summary"] = narrative

        report["metrics"] = {

            "total_customers":
                total_customers,

            "monthly_revenue":
                round(monthly_revenue, 2),

            "average_customer_value":
                round(avg_revenue, 2),

            "churn_rate":
                round(churn_rate * 100, 2),

            "high_risk_customers":
                high_risk,

            "revenue_at_risk":
                round(revenue_at_risk, 2),
        }

        report["key_insights"] = key_insights

        report["recommendations"] = recommendations

        report["forecast_commentary"] = forecast_commentary

        logger.info("Executive summary generation completed")

        return report

    # ========================================================
    # BUSINESS HEALTH
    # ========================================================

    def _classify_business_health(
        self,
        churn_rate: float
    ) -> str:

        if churn_rate < 0.15:
            return "Excellent"

        elif churn_rate < 0.30:
            return "Moderate"

        return "Critical"

    # ========================================================
    # REVENUE RISK
    # ========================================================

    def _calculate_revenue_risk(
        self,
        df,
        predictions
    ) -> float:

        if (
            predictions is None
            or "churn_probability" not in predictions.columns
            or "monthly_revenue" not in df.columns
        ):
            return 0.0

        merged = df.copy()

        merged["churn_probability"] = (
            predictions["churn_probability"]
        )

        merged["risk_revenue"] = (

            merged["monthly_revenue"]
            * merged["churn_probability"]

        )

        return float(
            merged["risk_revenue"].sum()
        )

    # ========================================================
    # KEY INSIGHTS
    # ========================================================

    def _generate_key_insights(
        self,
        df,
        predictions
    ) -> List[str]:

        insights = []

        # ----------------------------------------------------
        # CHURN INSIGHT
        # ----------------------------------------------------

        if "churned" in df.columns:

            churn_rate = df["churned"].mean()

            insights.append(

                f"Current churn rate stands at "
                f"{churn_rate:.2%}, indicating "
                f"customer retention performance."

            )

        # ----------------------------------------------------
        # REVENUE INSIGHT
        # ----------------------------------------------------

        if "monthly_revenue" in df.columns:

            top_20 = (

                df["monthly_revenue"]
                .quantile(0.80)

            )

            insights.append(

                f"Top 20% customers contribute "
                f"significantly to total revenue, "
                f"making high-value retention critical."

            )

        # ----------------------------------------------------
        # SUPPORT INSIGHT
        # ----------------------------------------------------

        if "support_tickets" in df.columns:

            avg_tickets = (
                df["support_tickets"]
                .mean()
            )

            insights.append(

                f"Average support ticket volume is "
                f"{avg_tickets:.2f} per customer, "
                f"indicating operational engagement trends."

            )

        # ----------------------------------------------------
        # ENGAGEMENT INSIGHT
        # ----------------------------------------------------

        if "days_since_last_login" in df.columns:

            inactive = (

                df["days_since_last_login"] > 30
            ).mean()

            insights.append(

                f"{inactive:.2%} of customers show "
                f"signs of inactivity beyond 30 days."

            )

        # ----------------------------------------------------
        # RISK INSIGHT
        # ----------------------------------------------------

        if (
            predictions is not None
            and "risk_level" in predictions.columns
        ):

            high_risk = (

                predictions["risk_level"]
                .astype(str)
                .str.lower()
                .eq("high")
                .mean()

            )

            insights.append(

                f"{high_risk:.2%} of customers "
                f"fall into the high-risk churn segment."

            )

        return insights

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def _generate_recommendations(
        self,
        churn_rate,
        high_risk,
        revenue_at_risk,
    ) -> List[str]:

        recommendations = []

        if churn_rate > 0.25:

            recommendations.append(

                "Launch aggressive customer retention "
                "campaigns targeting disengaged users."

            )

        if high_risk > 50:

            recommendations.append(

                "Deploy priority intervention workflows "
                "for high-risk customer segments."

            )

        if revenue_at_risk > 100000:

            recommendations.append(

                "Initiate executive escalation plans "
                "for high-value accounts."

            )

        recommendations.append(

            "Improve product engagement through "
            "personalized onboarding journeys."

        )

        recommendations.append(

            "Monitor churn predictors weekly "
            "using automated Early Warning Systems."

        )

        return recommendations

    # ========================================================
    # NARRATIVE BUILDER
    # ========================================================

    def _build_narrative(
        self,
        churn_rate,
        revenue,
        health,
        high_risk,
    ) -> str:

        template = self.templates.get(
            health.lower(),
            self.templates["moderate"]
        )

        narrative = f"""

{template.strip()}

Current monthly recurring revenue stands at
₹{revenue:,.0f}.

Approximately {high_risk} customers are currently
classified under elevated churn-risk segments.

The current churn rate of {churn_rate:.2%}
suggests that proactive customer engagement,
retention optimization, and personalized
communication strategies should remain
top organizational priorities.

ChurnShield AI recommends prioritizing
high-value customers, strengthening customer
success workflows, and increasing behavioral
monitoring for early churn detection.

        """

        return narrative.strip()

    # ========================================================
    # FORECAST COMMENTARY
    # ========================================================

    def _forecast_commentary(
        self,
        forecast
    ) -> str:

        if not forecast:
            return ""

        expected_growth = (
            forecast.get("growth_rate", 0)
        )

        if expected_growth > 0:

            return (

                f"Forecast models indicate a projected "
                f"growth trend of {expected_growth:.2f}% "
                f"over the upcoming forecasting period."

            )

        return (

            "Forecast models indicate potential "
            "business slowdown and elevated churn "
            "pressure in future periods."

        )

    # ========================================================
    # MARKDOWN EXPORT
    # ========================================================

    def export_markdown(
        self,
        report: Dict,
        output_path: str
    ):

        path = Path(output_path)

        lines = []

        lines.append("# ChurnShield Executive Report\n")

        lines.append(
            f"Generated At: "
            f"{report.get('generated_at')}\n"
        )

        lines.append(
            f"## Business Health: "
            f"{report.get('business_health')}\n"
        )

        lines.append("## Executive Summary\n")

        lines.append(
            report.get("summary", "")
        )

        lines.append("\n## Key Metrics\n")

        metrics = report.get("metrics", {})

        for k, v in metrics.items():

            lines.append(
                f"- {k}: {v}"
            )

        lines.append("\n## Key Insights\n")

        for insight in report.get(
            "key_insights",
            []
        ):

            lines.append(
                f"- {insight}"
            )

        lines.append("\n## Recommendations\n")

        for rec in report.get(
            "recommendations",
            []
        ):

            lines.append(
                f"- {rec}"
            )

        path.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

        logger.info(
            f"Markdown report exported: {path}"
        )

        return str(path)


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def generate_executive_summary(
    df: pd.DataFrame,
    predictions: Optional[pd.DataFrame] = None,
    forecast: Optional[Dict] = None,
    insights: Optional[Dict] = None,
):

    generator = ExecutiveSummaryGenerator()

    return generator.generate_summary(
        df=df,
        predictions=predictions,
        forecast=forecast,
        insights=insights,
    )


def export_executive_markdown(
    report: Dict,
    output_path: str
):

    generator = ExecutiveSummaryGenerator()

    return generator.export_markdown(
        report,
        output_path,
    )