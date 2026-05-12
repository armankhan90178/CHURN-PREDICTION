"""
ChurnShield 2.0 — Regional Analytics Intelligence Engine

Purpose:
Enterprise-grade geographic churn intelligence.

Capabilities:
- Regional churn analysis
- State/city performance analytics
- Revenue heatmaps
- Regional risk scoring
- Customer density analysis
- Geo-retention intelligence
- Multi-region comparison
- Expansion opportunity detection
- Territory performance monitoring
- Regional forecasting
- Executive geo insights
"""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import Dict
from datetime import datetime

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.analytics.regional"
)


# ─────────────────────────────────────────────
# REGIONAL ANALYTICS ENGINE
# ─────────────────────────────────────────────

class RegionalAnalyticsEngine:

    """
    Enterprise geographic intelligence engine
    """

    # ─────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────

    def __init__(self):

        self.default_region = "Unknown"

    # ─────────────────────────────────────────
    # MASTER ANALYSIS
    # ─────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Starting regional analytics..."
        )

        data = self._prepare_dataset(df)

        regional_summary = (
            self.generate_regional_summary(data)
        )

        churn_analysis = (
            self.analyze_regional_churn(data)
        )

        revenue_analysis = (
            self.analyze_regional_revenue(data)
        )

        risk_analysis = (
            self.analyze_regional_risk(data)
        )

        density_analysis = (
            self.customer_density_analysis(data)
        )

        opportunity_analysis = (
            self.identify_growth_regions(data)
        )

        performance_ranking = (
            self.rank_regions(data)
        )

        executive_summary = (
            self.generate_executive_summary(
                data,
                churn_analysis,
                revenue_analysis,
            )
        )

        logger.info(
            "Regional analytics completed"
        )

        return {

            "regional_summary":
                regional_summary,

            "churn_analysis":
                churn_analysis,

            "revenue_analysis":
                revenue_analysis,

            "risk_analysis":
                risk_analysis,

            "density_analysis":
                density_analysis,

            "opportunity_analysis":
                opportunity_analysis,

            "performance_ranking":
                performance_ranking,

            "executive_summary":
                executive_summary,
        }

    # ─────────────────────────────────────────
    # DATA PREP
    # ─────────────────────────────────────────

    def _prepare_dataset(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        data = df.copy()

        if "state" not in data.columns:

            if "city" in data.columns:

                data["state"] = (
                    data["city"]
                    .fillna(self.default_region)
                )

            else:

                data["state"] = self.default_region

        numeric_defaults = {

            "monthly_revenue": 0,
            "churned": 0,
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
    # REGIONAL SUMMARY
    # ─────────────────────────────────────────

    def generate_regional_summary(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Generating regional summary..."
        )

        summary = (

            df.groupby("state")

            .agg({

                "customer_id": "count",
                "monthly_revenue": [
                    "sum",
                    "mean",
                ],
                "churned": "mean",
            })

        )

        summary.columns = [

            "customers",
            "total_revenue",
            "avg_revenue",
            "churn_rate",
        ]

        summary["churn_rate"] = (
            summary["churn_rate"] * 100
        ).round(2)

        summary = (
            summary.sort_values(
                "total_revenue",
                ascending=False,
            )
        )

        return summary.reset_index()

    # ─────────────────────────────────────────
    # CHURN ANALYSIS
    # ─────────────────────────────────────────

    def analyze_regional_churn(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Analyzing regional churn..."
        )

        analysis = (

            df.groupby("state")

            .agg({

                "churned": "mean",
                "customer_id": "count",
                "monthly_revenue": "sum",
            })

        )

        analysis.columns = [

            "churn_rate",
            "customers",
            "revenue",
        ]

        analysis["churn_rate"] = (
            analysis["churn_rate"] * 100
        ).round(2)

        analysis["risk_level"] = np.where(

            analysis["churn_rate"] >= 30,
            "HIGH",

            np.where(
                analysis["churn_rate"] >= 15,
                "MEDIUM",
                "LOW",
            )
        )

        return analysis.reset_index()

    # ─────────────────────────────────────────
    # REVENUE ANALYSIS
    # ─────────────────────────────────────────

    def analyze_regional_revenue(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Analyzing regional revenue..."
        )

        revenue = (

            df.groupby("state")

            .agg({

                "monthly_revenue": [
                    "sum",
                    "mean",
                    "median",
                ]
            })

        )

        revenue.columns = [

            "total_revenue",
            "avg_revenue",
            "median_revenue",
        ]

        revenue["revenue_share"] = (

            revenue["total_revenue"]

            /

            revenue["total_revenue"].sum()

        ) * 100

        revenue["revenue_share"] = (
            revenue["revenue_share"]
            .round(2)
        )

        return revenue.reset_index()

    # ─────────────────────────────────────────
    # RISK ANALYSIS
    # ─────────────────────────────────────────

    def analyze_regional_risk(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Analyzing regional risk..."
        )

        risk = (

            df.groupby("state")

            .agg({

                "feature_usage_score": "mean",
                "nps_score": "mean",
                "payment_delays": "mean",
            })

        )

        risk["regional_risk_score"] = (

            (
                1 - risk["feature_usage_score"]
            ) * 0.5

            +

            (
                1 - (
                    risk["nps_score"] / 10
                )
            ) * 0.3

            +

            np.clip(
                risk["payment_delays"] / 5,
                0,
                1,
            ) * 0.2
        )

        risk["risk_band"] = np.where(

            risk["regional_risk_score"] >= 0.70,
            "Critical",

            np.where(
                risk["regional_risk_score"] >= 0.40,
                "Moderate",
                "Low",
            )
        )

        return risk.reset_index()

    # ─────────────────────────────────────────
    # CUSTOMER DENSITY
    # ─────────────────────────────────────────

    def customer_density_analysis(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Analyzing customer density..."
        )

        density = (

            df.groupby("state")

            .size()

            .reset_index(name="customer_density")
        )

        density["density_rank"] = (

            density["customer_density"]
            .rank(
                ascending=False
            )
        )

        return density.sort_values(
            "customer_density",
            ascending=False,
        )

    # ─────────────────────────────────────────
    # GROWTH REGIONS
    # ─────────────────────────────────────────

    def identify_growth_regions(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Identifying growth regions..."
        )

        growth = (

            df.groupby("state")

            .agg({

                "monthly_revenue": "mean",
                "feature_usage_score": "mean",
                "nps_score": "mean",
            })

        )

        growth["growth_score"] = (

            (
                growth["monthly_revenue"]

                /

                growth["monthly_revenue"].max()
            ) * 0.4

            +

            growth["feature_usage_score"] * 0.3

            +

            (
                growth["nps_score"] / 10
            ) * 0.3
        )

        growth = growth.sort_values(
            "growth_score",
            ascending=False,
        )

        return growth.reset_index()

    # ─────────────────────────────────────────
    # REGION RANKING
    # ─────────────────────────────────────────

    def rank_regions(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Ranking regional performance..."
        )

        ranking = (

            df.groupby("state")

            .agg({

                "monthly_revenue": "sum",
                "churned": "mean",
                "feature_usage_score": "mean",
            })

        )

        ranking["performance_score"] = (

            (
                ranking["monthly_revenue"]

                /

                ranking["monthly_revenue"].max()
            ) * 0.5

            +

            (
                1 - ranking["churned"]
            ) * 0.3

            +

            ranking["feature_usage_score"] * 0.2
        )

        ranking["rank"] = (

            ranking["performance_score"]
            .rank(
                ascending=False
            )
        )

        return ranking.sort_values(
            "rank"
        ).reset_index()

    # ─────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────

    def generate_executive_summary(
        self,
        df: pd.DataFrame,
        churn_analysis: pd.DataFrame,
        revenue_analysis: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Generating executive geo summary..."
        )

        top_region = (

            revenue_analysis
            .sort_values(
                "total_revenue",
                ascending=False,
            )
            .iloc[0]
        )

        highest_churn = (

            churn_analysis
            .sort_values(
                "churn_rate",
                ascending=False,
            )
            .iloc[0]
        )

        return {

            "generated_at":
                datetime.utcnow().isoformat(),

            "regions_analyzed":
                int(df["state"].nunique()),

            "top_revenue_region":
                top_region["state"],

            "top_revenue":
                float(
                    top_region["total_revenue"]
                ),

            "highest_risk_region":
                highest_churn["state"],

            "highest_churn_rate":
                float(
                    highest_churn["churn_rate"]
                ),
        }


# ─────────────────────────────────────────────
# GLOBAL ENGINE
# ─────────────────────────────────────────────

regional_engine = RegionalAnalyticsEngine()


# ─────────────────────────────────────────────
# FUNCTIONAL API
# ─────────────────────────────────────────────

def analyze_regions(
    df: pd.DataFrame,
) -> Dict:

    return regional_engine.analyze(df)


def regional_churn_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    analysis = regional_engine.analyze(df)

    return analysis["churn_analysis"]


def regional_revenue_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    analysis = regional_engine.analyze(df)

    return analysis["revenue_analysis"]