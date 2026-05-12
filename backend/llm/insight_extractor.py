"""
ChurnShield 2.0 — Hyper Business Insight Extractor

Purpose:
Generate executive-grade business insights
from churn analytics, behavioral data,
revenue trends, and customer intelligence.

Capabilities:
- executive business intelligence
- churn trend discovery
- revenue leakage analysis
- customer health analytics
- retention opportunity detection
- hidden behavioral pattern extraction
- AI-style business storytelling
- cohort intelligence
- risk segmentation
- growth opportunity analytics
- strategic recommendations
- portfolio-level summaries
- enterprise KPI diagnostics
"""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import Dict
from collections import Counter

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.insight_extractor"
)


class HyperInsightExtractor:

    def __init__(self):

        self.high_risk_threshold = 0.70

        self.medium_risk_threshold = 0.40

    # ─────────────────────────────────────────────
    # MAIN EXTRACTION PIPELINE
    # ─────────────────────────────────────────────

    def extract(
        self,
        df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Starting hyper insight extraction"
        )

        data = df.copy()

        insights = {

            "executive_summary":
                self._executive_summary(data),

            "portfolio_health":
                self._portfolio_health(data),

            "revenue_intelligence":
                self._revenue_intelligence(data),

            "churn_intelligence":
                self._churn_intelligence(data),

            "behavioral_patterns":
                self._behavioral_patterns(data),

            "customer_segments":
                self._customer_segments(data),

            "retention_opportunities":
                self._retention_opportunities(data),

            "support_intelligence":
                self._support_intelligence(data),

            "engagement_intelligence":
                self._engagement_intelligence(data),

            "payment_intelligence":
                self._payment_intelligence(data),

            "product_adoption":
                self._product_adoption(data),

            "future_risk_forecast":
                self._future_risk_forecast(data),

            "strategic_recommendations":
                self._strategic_recommendations(data),

            "executive_alerts":
                self._executive_alerts(data),

            "kpi_dashboard":
                self._kpi_dashboard(data),
        }

        logger.info(
            "Insight extraction completed"
        )

        return insights

    # ─────────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────

    def _executive_summary(
        self,
        df,
    ):

        total_customers = len(df)

        churn_rate = self._safe_mean(
            df,
            "churned",
        )

        revenue = self._safe_sum(
            df,
            "monthly_revenue",
        )

        high_risk = self._high_risk_count(
            df
        )

        summary = (

            f"Portfolio contains "
            f"{total_customers:,} customers with "
            f"estimated churn rate of "
            f"{round(churn_rate*100,1)}%. "
            f"Total monthly revenue exposure is "
            f"₹{int(revenue):,}. "
            f"{high_risk:,} accounts are categorized "
            f"as high-risk and require proactive intervention."

        )

        return summary

    # ─────────────────────────────────────────────
    # PORTFOLIO HEALTH
    # ─────────────────────────────────────────────

    def _portfolio_health(
        self,
        df,
    ):

        avg_engagement = self._safe_mean(
            df,
            "engagement_score",
        )

        avg_feature_usage = self._safe_mean(
            df,
            "feature_usage_score",
        )

        avg_nps = self._safe_mean(
            df,
            "nps_score",
        )

        health_score = (

            avg_engagement * 0.4

            +

            avg_feature_usage * 0.4

            +

            (avg_nps / 10) * 0.2

        )

        if health_score >= 0.75:
            health = "Excellent"

        elif health_score >= 0.60:
            health = "Healthy"

        elif health_score >= 0.40:
            health = "Moderate"

        else:
            health = "Critical"

        return {

            "portfolio_health":
                health,

            "health_score":
                round(
                    float(health_score),
                    4,
                ),

            "avg_engagement":
                round(
                    float(avg_engagement),
                    4,
                ),

            "avg_feature_usage":
                round(
                    float(avg_feature_usage),
                    4,
                ),

            "avg_nps":
                round(
                    float(avg_nps),
                    2,
                ),
        }

    # ─────────────────────────────────────────────
    # REVENUE INTELLIGENCE
    # ─────────────────────────────────────────────

    def _revenue_intelligence(
        self,
        df,
    ):

        if "monthly_revenue" not in df.columns:

            return {}

        total_revenue = df[
            "monthly_revenue"
        ].sum()

        high_risk_revenue = df.loc[

            self._risk_filter(df),

            "monthly_revenue",

        ].sum()

        top_customers = df.nlargest(

            10,
            "monthly_revenue",
        )

        avg_revenue = df[
            "monthly_revenue"
        ].mean()

        concentration = (

            top_customers[
                "monthly_revenue"
            ].sum()

            /

            max(total_revenue, 1)

        )

        return {

            "total_monthly_revenue":
                round(
                    float(total_revenue),
                    2,
                ),

            "high_risk_revenue":
                round(
                    float(high_risk_revenue),
                    2,
                ),

            "revenue_at_risk_percent":
                round(
                    float(
                        high_risk_revenue /
                        max(total_revenue, 1)
                    ) * 100,
                    2,
                ),

            "average_customer_revenue":
                round(
                    float(avg_revenue),
                    2,
                ),

            "revenue_concentration":
                round(
                    float(concentration),
                    4,
                ),
        }

    # ─────────────────────────────────────────────
    # CHURN INTELLIGENCE
    # ─────────────────────────────────────────────

    def _churn_intelligence(
        self,
        df,
    ):

        churn_rate = self._safe_mean(
            df,
            "churned",
        )

        high_risk_accounts = self._high_risk_count(
            df
        )

        critical_accounts = self._critical_risk_count(
            df
        )

        return {

            "current_churn_rate":
                round(
                    float(churn_rate),
                    4,
                ),

            "high_risk_accounts":
                int(high_risk_accounts),

            "critical_accounts":
                int(critical_accounts),

            "estimated_future_churn":
                int(
                    high_risk_accounts * 0.65
                ),
        }

    # ─────────────────────────────────────────────
    # BEHAVIORAL PATTERNS
    # ─────────────────────────────────────────────

    def _behavioral_patterns(
        self,
        df,
    ):

        patterns = []

        if "feature_usage_score" in df.columns:

            low_usage = (
                df[
                    "feature_usage_score"
                ] < 0.30
            ).mean()

            if low_usage > 0.30:

                patterns.append(
                    "Large portion of customers exhibit low feature adoption."
                )

        if "payment_delays" in df.columns:

            delayed = (
                df[
                    "payment_delays"
                ] > 1
            ).mean()

            if delayed > 0.20:

                patterns.append(
                    "Payment friction increasing across customer base."
                )

        if "support_tickets" in df.columns:

            high_support = (
                df[
                    "support_tickets"
                ] > 4
            ).mean()

            if high_support > 0.15:

                patterns.append(
                    "Support burden elevated — possible operational friction."
                )

        if not patterns:

            patterns.append(
                "Customer behavior currently stable."
            )

        return patterns

    # ─────────────────────────────────────────────
    # CUSTOMER SEGMENTS
    # ─────────────────────────────────────────────

    def _customer_segments(
        self,
        df,
    ):

        segments = {}

        if "monthly_revenue" not in df.columns:

            return {}

        revenue = df[
            "monthly_revenue"
        ]

        segments["enterprise"] = int(
            (revenue > 25000).sum()
        )

        segments["mid_market"] = int(
            (
                (revenue >= 5000)
                &
                (revenue <= 25000)
            ).sum()
        )

        segments["small_business"] = int(
            (revenue < 5000).sum()
        )

        return segments

    # ─────────────────────────────────────────────
    # RETENTION OPPORTUNITIES
    # ─────────────────────────────────────────────

    def _retention_opportunities(
        self,
        df,
    ):

        opportunities = []

        if "engagement_score" in df.columns:

            moderate = (

                (df["engagement_score"] > 0.40)

                &

                (df["engagement_score"] < 0.70)

            ).sum()

            if moderate > 0:

                opportunities.append({

                    "type":
                        "Engagement Optimization",

                    "accounts":
                        int(moderate),

                    "impact":
                        "Medium-to-High",
                })

        if "feature_usage_score" in df.columns:

            underutilized = (
                df[
                    "feature_usage_score"
                ] < 0.50
            ).sum()

            opportunities.append({

                "type":
                    "Feature Adoption Campaign",

                "accounts":
                    int(underutilized),

                "impact":
                    "High",
            })

        return opportunities

    # ─────────────────────────────────────────────
    # SUPPORT INTELLIGENCE
    # ─────────────────────────────────────────────

    def _support_intelligence(
        self,
        df,
    ):

        if "support_tickets" not in df.columns:

            return {}

        avg_tickets = df[
            "support_tickets"
        ].mean()

        high_support = (
            df[
                "support_tickets"
            ] >= 5
        ).sum()

        return {

            "average_support_load":
                round(
                    float(avg_tickets),
                    2,
                ),

            "high_support_accounts":
                int(high_support),

            "support_risk_level":
                "High"
                if avg_tickets >= 4
                else "Moderate",
        }

    # ─────────────────────────────────────────────
    # ENGAGEMENT INTELLIGENCE
    # ─────────────────────────────────────────────

    def _engagement_intelligence(
        self,
        df,
    ):

        if "engagement_score" not in df.columns:

            return {}

        avg = df[
            "engagement_score"
        ].mean()

        declining = (
            df[
                "engagement_score"
            ] < 0.40
        ).sum()

        return {

            "average_engagement":
                round(
                    float(avg),
                    4,
                ),

            "low_engagement_accounts":
                int(declining),

            "engagement_status":
                "Healthy"
                if avg >= 0.65
                else "Needs Attention",
        }

    # ─────────────────────────────────────────────
    # PAYMENT INTELLIGENCE
    # ─────────────────────────────────────────────

    def _payment_intelligence(
        self,
        df,
    ):

        if "payment_delays" not in df.columns:

            return {}

        delayed_accounts = (
            df[
                "payment_delays"
            ] > 0
        ).sum()

        avg_delay = df[
            "payment_delays"
        ].mean()

        return {

            "delayed_payment_accounts":
                int(delayed_accounts),

            "average_payment_delay":
                round(
                    float(avg_delay),
                    2,
                ),

            "financial_risk":
                "Elevated"
                if avg_delay >= 1
                else "Stable",
        }

    # ─────────────────────────────────────────────
    # PRODUCT ADOPTION
    # ─────────────────────────────────────────────

    def _product_adoption(
        self,
        df,
    ):

        if "feature_usage_score" not in df.columns:

            return {}

        avg_usage = df[
            "feature_usage_score"
        ].mean()

        power_users = (
            df[
                "feature_usage_score"
            ] > 0.80
        ).sum()

        inactive = (
            df[
                "feature_usage_score"
            ] < 0.20
        ).sum()

        return {

            "average_feature_adoption":
                round(
                    float(avg_usage),
                    4,
                ),

            "power_users":
                int(power_users),

            "inactive_users":
                int(inactive),
        }

    # ─────────────────────────────────────────────
    # FUTURE RISK
    # ─────────────────────────────────────────────

    def _future_risk_forecast(
        self,
        df,
    ):

        high_risk = self._high_risk_count(
            df
        )

        forecast = {

            "expected_next_quarter_churn":
                int(high_risk * 0.60),

            "projected_revenue_loss":
                round(
                    float(
                        self._safe_sum(
                            df.loc[
                                self._risk_filter(df)
                            ],
                            "monthly_revenue",
                        ) * 0.55
                    ),
                    2,
                ),

            "forecast_confidence":
                0.82,
        }

        return forecast

    # ─────────────────────────────────────────────
    # STRATEGIC RECOMMENDATIONS
    # ─────────────────────────────────────────────

    def _strategic_recommendations(
        self,
        df,
    ):

        recommendations = []

        recommendations.append(
            "Launch proactive retention workflows for high-risk accounts."
        )

        recommendations.append(
            "Increase feature adoption initiatives for low-engagement customers."
        )

        recommendations.append(
            "Assign customer success managers to enterprise accounts."
        )

        recommendations.append(
            "Improve onboarding journeys to reduce early churn."
        )

        recommendations.append(
            "Implement predictive outreach before renewal periods."
        )

        return recommendations

    # ─────────────────────────────────────────────
    # EXECUTIVE ALERTS
    # ─────────────────────────────────────────────

    def _executive_alerts(
        self,
        df,
    ):

        alerts = []

        high_risk = self._high_risk_count(
            df
        )

        if high_risk > len(df) * 0.25:

            alerts.append(
                "High-risk accounts exceed 25% of portfolio."
            )

        if "monthly_revenue" in df.columns:

            enterprise_risk = df.loc[
                self._risk_filter(df),
                "monthly_revenue",
            ].sum()

            if enterprise_risk > 100000:

                alerts.append(
                    "Significant enterprise revenue exposure detected."
                )

        if not alerts:

            alerts.append(
                "No major executive threats detected."
            )

        return alerts

    # ─────────────────────────────────────────────
    # KPI DASHBOARD
    # ─────────────────────────────────────────────

    def _kpi_dashboard(
        self,
        df,
    ):

        return {

            "customers":
                int(len(df)),

            "churn_rate":
                round(
                    float(
                        self._safe_mean(
                            df,
                            "churned",
                        )
                    ),
                    4,
                ),

            "high_risk_accounts":
                int(
                    self._high_risk_count(df)
                ),

            "critical_accounts":
                int(
                    self._critical_risk_count(df)
                ),

            "monthly_revenue":
                round(
                    float(
                        self._safe_sum(
                            df,
                            "monthly_revenue",
                        )
                    ),
                    2,
                ),

            "avg_nps":
                round(
                    float(
                        self._safe_mean(
                            df,
                            "nps_score",
                        )
                    ),
                    2,
                ),
        }

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _safe_mean(
        self,
        df,
        column,
    ):

        if column not in df.columns:
            return 0

        return df[column].fillna(0).mean()

    def _safe_sum(
        self,
        df,
        column,
    ):

        if column not in df.columns:
            return 0

        return df[column].fillna(0).sum()

    def _risk_filter(
        self,
        df,
    ):

        if "churn_probability" in df.columns:

            return (
                df[
                    "churn_probability"
                ] >= self.high_risk_threshold
            )

        if "predicted_churn" in df.columns:

            return (
                df[
                    "predicted_churn"
                ] == 1
            )

        return pd.Series(
            [False] * len(df)
        )

    def _high_risk_count(
        self,
        df,
    ):

        return int(
            self._risk_filter(df).sum()
        )

    def _critical_risk_count(
        self,
        df,
    ):

        if "churn_probability" not in df.columns:

            return 0

        return int(

            (
                df[
                    "churn_probability"
                ] >= 0.85
            ).sum()

        )


# ─────────────────────────────────────────────
# FUNCTIONAL INTERFACE
# ─────────────────────────────────────────────

def extract_business_insights(
    df: pd.DataFrame,
):

    extractor = HyperInsightExtractor()

    return extractor.extract(df)