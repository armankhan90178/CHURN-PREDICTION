"""
ChurnShield 2.0 — Revenue Intelligence Engine

Purpose:
Enterprise-grade revenue analytics engine
for churn-driven revenue intelligence.

Capabilities:
- Revenue at risk analysis
- Churn revenue leakage detection
- Expansion opportunity detection
- MRR/ARR intelligence
- Revenue waterfall analysis
- Customer LTV estimation
- Revenue segmentation
- High-value customer monitoring
- Net revenue retention simulation
- Upsell potential scoring
- Executive revenue reporting
"""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import Dict, List
from datetime import datetime

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.analytics.revenue"
)


# ─────────────────────────────────────────────
# ENTERPRISE REVENUE ENGINE
# ─────────────────────────────────────────────

class RevenueIntelligenceEngine:

    """
    Enterprise-grade revenue intelligence engine
    """

    # ─────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────

    def __init__(self):

        self.required_columns = [

            "monthly_revenue",
        ]

    # ─────────────────────────────────────────
    # MASTER ANALYSIS
    # ─────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Starting revenue intelligence analysis..."
        )

        data = self._prepare_dataset(df)

        summary = (
            self.generate_revenue_summary(data)
        )

        revenue_segments = (
            self.segment_customers_by_revenue(data)
        )

        churn_impact = (
            self.analyze_churn_revenue_impact(data)
        )

        ltv_analysis = (
            self.calculate_ltv(data)
        )

        waterfall = (
            self.generate_revenue_waterfall(data)
        )

        upsell_analysis = (
            self.identify_upsell_opportunities(data)
        )

        retention_analysis = (
            self.calculate_retention_metrics(data)
        )

        risk_analysis = (
            self.calculate_revenue_risk(data)
        )

        executive_summary = (
            self.generate_executive_summary(
                summary,
                churn_impact,
                retention_analysis,
            )
        )

        logger.info(
            "Revenue intelligence completed"
        )

        return {

            "summary":
                summary,

            "revenue_segments":
                revenue_segments,

            "churn_impact":
                churn_impact,

            "ltv_analysis":
                ltv_analysis,

            "waterfall":
                waterfall,

            "upsell_analysis":
                upsell_analysis,

            "retention_analysis":
                retention_analysis,

            "risk_analysis":
                risk_analysis,

            "executive_summary":
                executive_summary,
        }

    # ─────────────────────────────────────────
    # PREP
    # ─────────────────────────────────────────

    def _prepare_dataset(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        data = df.copy()

        numeric_defaults = {

            "monthly_revenue": 0,
            "churned": 0,
            "contract_age_months": 6,
            "feature_usage_score": 0.5,
            "nps_score": 5,
        }

        for col, default in numeric_defaults.items():

            if col not in data.columns:

                data[col] = default

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            ).fillna(default)

        return data

    # ─────────────────────────────────────────
    # REVENUE SUMMARY
    # ─────────────────────────────────────────

    def generate_revenue_summary(
        self,
        df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Generating revenue summary..."
        )

        total_mrr = (
            df["monthly_revenue"].sum()
        )

        arr = total_mrr * 12

        avg_customer_value = (
            df["monthly_revenue"].mean()
        )

        median_customer_value = (
            df["monthly_revenue"].median()
        )

        high_value_threshold = (
            df["monthly_revenue"]
            .quantile(0.80)
        )

        high_value_customers = (
            df["monthly_revenue"]
            >= high_value_threshold
        ).sum()

        return {

            "total_mrr":
                round(total_mrr, 2),

            "total_arr":
                round(arr, 2),

            "average_customer_value":
                round(avg_customer_value, 2),

            "median_customer_value":
                round(median_customer_value, 2),

            "high_value_threshold":
                round(high_value_threshold, 2),

            "high_value_customers":
                int(high_value_customers),

            "customer_count":
                int(len(df)),
        }

    # ─────────────────────────────────────────
    # REVENUE SEGMENTS
    # ─────────────────────────────────────────

    def segment_customers_by_revenue(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Segmenting customers by revenue..."
        )

        revenue = df["monthly_revenue"]

        q1 = revenue.quantile(0.25)
        q2 = revenue.quantile(0.50)
        q3 = revenue.quantile(0.75)

        def assign_segment(value):

            if value >= q3:
                return "Enterprise"

            elif value >= q2:
                return "Growth"

            elif value >= q1:
                return "Mid-Market"

            return "Low Value"

        segmented = df.copy()

        segmented["revenue_segment"] = (
            segmented["monthly_revenue"]
            .apply(assign_segment)
        )

        summary = (

            segmented.groupby("revenue_segment")

            .agg({

                "customer_id": "count",
                "monthly_revenue": "sum",
                "churned": "mean",
            })

        )

        summary.columns = [

            "customers",
            "revenue",
            "churn_rate",
        ]

        summary["churn_rate"] = (
            summary["churn_rate"] * 100
        ).round(2)

        return summary.reset_index()

    # ─────────────────────────────────────────
    # CHURN REVENUE IMPACT
    # ─────────────────────────────────────────

    def analyze_churn_revenue_impact(
        self,
        df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Analyzing churn revenue impact..."
        )

        churned = (
            df[df["churned"] == 1]
        )

        retained = (
            df[df["churned"] == 0]
        )

        churn_revenue_loss = (
            churned["monthly_revenue"].sum()
        )

        retained_revenue = (
            retained["monthly_revenue"].sum()
        )

        total_revenue = (
            df["monthly_revenue"].sum()
        )

        revenue_churn_rate = 0

        if total_revenue > 0:

            revenue_churn_rate = (

                churn_revenue_loss
                /
                total_revenue
            ) * 100

        return {

            "revenue_lost_to_churn":
                round(churn_revenue_loss, 2),

            "retained_revenue":
                round(retained_revenue, 2),

            "revenue_churn_rate":
                round(revenue_churn_rate, 2),

            "projected_annual_loss":
                round(
                    churn_revenue_loss * 12,
                    2,
                ),
        }

    # ─────────────────────────────────────────
    # CUSTOMER LTV
    # ─────────────────────────────────────────

    def calculate_ltv(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Calculating customer LTV..."
        )

        data = df.copy()

        churn_rate = max(
            data["churned"].mean(),
            0.01,
        )

        data["estimated_ltv"] = (

            data["monthly_revenue"]

            *

            (
                1 / churn_rate
            )
        ).round(2)

        data["ltv_segment"] = pd.cut(

            data["estimated_ltv"],

            bins=[
                0,
                10000,
                50000,
                200000,
                np.inf,
            ],

            labels=[
                "Low",
                "Medium",
                "High",
                "Strategic",
            ],
        )

        return data[[
            "customer_id",
            "monthly_revenue",
            "estimated_ltv",
            "ltv_segment",
        ]]

    # ─────────────────────────────────────────
    # WATERFALL ANALYSIS
    # ─────────────────────────────────────────

    def generate_revenue_waterfall(
        self,
        df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Generating revenue waterfall..."
        )

        starting_revenue = (
            df["monthly_revenue"].sum()
        )

        churn_loss = (

            df.loc[
                df["churned"] == 1,
                "monthly_revenue",
            ].sum()
        )

        expansion_revenue = (

            df.loc[
                df["feature_usage_score"] > 0.80,
                "monthly_revenue",
            ].sum()
            * 0.10
        )

        ending_revenue = (

            starting_revenue
            -
            churn_loss
            +
            expansion_revenue
        )

        return {

            "starting_revenue":
                round(starting_revenue, 2),

            "churn_loss":
                round(churn_loss, 2),

            "expansion_revenue":
                round(expansion_revenue, 2),

            "ending_revenue":
                round(ending_revenue, 2),
        }

    # ─────────────────────────────────────────
    # UPSELL ANALYSIS
    # ─────────────────────────────────────────

    def identify_upsell_opportunities(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Detecting upsell opportunities..."
        )

        data = df.copy()

        conditions = (

            (data["feature_usage_score"] > 0.75)

            &

            (data["churned"] == 0)

            &

            (data["nps_score"] >= 8)
        )

        opportunities = (
            data[conditions]
            .copy()
        )

        opportunities["upsell_score"] = (

            opportunities["feature_usage_score"] * 0.5

            +

            (
                opportunities["nps_score"] / 10
            ) * 0.3

            +

            np.clip(

                opportunities[
                    "contract_age_months"
                ] / 24,

                0,
                1,
            ) * 0.2
        ).round(4)

        opportunities["estimated_expansion"] = (

            opportunities["monthly_revenue"]
            * 0.20

        ).round(2)

        return opportunities[[
            "customer_id",
            "monthly_revenue",
            "upsell_score",
            "estimated_expansion",
        ]]

    # ─────────────────────────────────────────
    # RETENTION METRICS
    # ─────────────────────────────────────────

    def calculate_retention_metrics(
        self,
        df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Calculating retention metrics..."
        )

        gross_revenue_retention = (

            (
                df.loc[
                    df["churned"] == 0,
                    "monthly_revenue",
                ].sum()
            )

            /

            max(
                df["monthly_revenue"].sum(),
                1,
            )
        ) * 100

        expansion = (

            df.loc[
                df["feature_usage_score"] > 0.8,
                "monthly_revenue",
            ].sum()
            * 0.10
        )

        net_revenue_retention = (

            (
                (
                    df.loc[
                        df["churned"] == 0,
                        "monthly_revenue",
                    ].sum()
                )
                +
                expansion
            )

            /

            max(
                df["monthly_revenue"].sum(),
                1,
            )
        ) * 100

        return {

            "gross_revenue_retention":
                round(
                    gross_revenue_retention,
                    2,
                ),

            "net_revenue_retention":
                round(
                    net_revenue_retention,
                    2,
                ),
        }

    # ─────────────────────────────────────────
    # REVENUE RISK
    # ─────────────────────────────────────────

    def calculate_revenue_risk(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Calculating revenue risk..."
        )

        data = df.copy()

        data["revenue_risk_score"] = (

            (
                data["churned"] * 0.5
            )

            +

            (
                1 - data["feature_usage_score"]
            ) * 0.3

            +

            (
                np.clip(
                    data["payment_delays"] / 5,
                    0,
                    1,
                )
            ) * 0.2
        ).round(4)

        data["risk_band"] = np.where(

            data["revenue_risk_score"] >= 0.70,
            "Critical",

            np.where(
                data["revenue_risk_score"] >= 0.40,
                "Moderate",
                "Low",
            )
        )

        return data[[
            "customer_id",
            "monthly_revenue",
            "revenue_risk_score",
            "risk_band",
        ]]

    # ─────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────

    def generate_executive_summary(
        self,
        summary: Dict,
        churn_impact: Dict,
        retention_analysis: Dict,
    ) -> Dict:

        logger.info(
            "Generating executive revenue summary..."
        )

        return {

            "generated_at":
                datetime.utcnow().isoformat(),

            "total_arr":
                summary["total_arr"],

            "revenue_at_risk":
                churn_impact[
                    "projected_annual_loss"
                ],

            "gross_retention":
                retention_analysis[
                    "gross_revenue_retention"
                ],

            "net_retention":
                retention_analysis[
                    "net_revenue_retention"
                ],

            "business_health":

                "Strong"

                if retention_analysis[
                    "net_revenue_retention"
                ] > 100

                else

                "Stable"

                if retention_analysis[
                    "net_revenue_retention"
                ] > 85

                else

                "At Risk",
        }


# ─────────────────────────────────────────────
# GLOBAL ENGINE
# ─────────────────────────────────────────────

revenue_engine = RevenueIntelligenceEngine()


# ─────────────────────────────────────────────
# FUNCTIONAL API
# ─────────────────────────────────────────────

def analyze_revenue(
    df: pd.DataFrame,
) -> Dict:

    return revenue_engine.analyze(df)


def calculate_revenue_health(
    df: pd.DataFrame,
) -> Dict:

    analysis = revenue_engine.analyze(df)

    return analysis["executive_summary"]


def generate_waterfall_analysis(
    df: pd.DataFrame,
) -> Dict:

    analysis = revenue_engine.analyze(df)

    return analysis["waterfall"]