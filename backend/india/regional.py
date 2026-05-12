"""
ChurnShield 2.0 — Hyper Regional Intelligence Engine

Purpose:
Enterprise-grade regional analytics engine
for India-focused churn intelligence,
regional risk analysis, state-level forecasting,
heatmaps, commercial insights, and retention optimization.

Capabilities:
- state-wise churn intelligence
- regional revenue forecasting
- geographic risk clustering
- city-level engagement analysis
- regional customer segmentation
- commercial density scoring
- state-wise retention benchmarking
- revenue-at-risk heatmaps
- regional trend forecasting
- territory-level opportunity analysis
- geo-behavioral customer intelligence
- India growth potential analytics
"""

import logging
import numpy as np
import pandas as pd

from typing import Dict
from datetime import datetime

logger = logging.getLogger(
    "churnshield.analytics.regional"
)


class HyperRegionalEngine:

    def __init__(self):

        # ─────────────────────────────────────────────
        # STATE COMMERCIAL MULTIPLIERS
        # ─────────────────────────────────────────────

        self.state_profiles = {

            "Maharashtra": {
                "business_strength": 1.25,
                "digital_adoption": 1.30,
                "payment_reliability": 1.15,
                "enterprise_density": 1.35,
            },

            "Karnataka": {
                "business_strength": 1.22,
                "digital_adoption": 1.35,
                "payment_reliability": 1.12,
                "enterprise_density": 1.30,
            },

            "Telangana": {
                "business_strength": 1.15,
                "digital_adoption": 1.22,
                "payment_reliability": 1.08,
                "enterprise_density": 1.18,
            },

            "Tamil Nadu": {
                "business_strength": 1.18,
                "digital_adoption": 1.20,
                "payment_reliability": 1.10,
                "enterprise_density": 1.16,
            },

            "Delhi": {
                "business_strength": 1.30,
                "digital_adoption": 1.28,
                "payment_reliability": 1.20,
                "enterprise_density": 1.40,
            },

            "Gujarat": {
                "business_strength": 1.20,
                "digital_adoption": 1.12,
                "payment_reliability": 1.18,
                "enterprise_density": 1.14,
            },

            "Andhra Pradesh": {
                "business_strength": 1.02,
                "digital_adoption": 1.05,
                "payment_reliability": 1.00,
                "enterprise_density": 1.00,
            },

            "West Bengal": {
                "business_strength": 0.98,
                "digital_adoption": 0.95,
                "payment_reliability": 0.92,
                "enterprise_density": 0.90,
            },
        }

    # ─────────────────────────────────────────────
    # MAIN REGIONAL INTELLIGENCE
    # ─────────────────────────────────────────────

    def apply_regional_intelligence(
        self,
        df: pd.DataFrame,
        state_column: str = "state",
    ):

        logger.info(
            "Applying regional intelligence"
        )

        data = df.copy()

        if state_column not in data.columns:

            logger.warning(
                "State column missing"
            )

            return data

        # ─────────────────────────────────────────────
        # REGIONAL MULTIPLIERS
        # ─────────────────────────────────────────────

        data[
            "regional_business_strength"
        ] = data[state_column].apply(
            lambda x:
            self._metric(
                x,
                "business_strength",
            )
        )

        data[
            "regional_digital_adoption"
        ] = data[state_column].apply(
            lambda x:
            self._metric(
                x,
                "digital_adoption",
            )
        )

        data[
            "regional_payment_reliability"
        ] = data[state_column].apply(
            lambda x:
            self._metric(
                x,
                "payment_reliability",
            )
        )

        data[
            "regional_enterprise_density"
        ] = data[state_column].apply(
            lambda x:
            self._metric(
                x,
                "enterprise_density",
            )
        )

        # ─────────────────────────────────────────────
        # REGIONAL RISK ENGINE
        # ─────────────────────────────────────────────

        if "churn_probability" in data.columns:

            regional_factor = (

                2
                -

                (
                    data[
                        "regional_business_strength"
                    ]

                    +

                    data[
                        "regional_payment_reliability"
                    ]
                ) / 2
            )

            data[
                "regional_adjusted_churn_probability"
            ] = np.clip(

                data[
                    "churn_probability"
                ]

                *

                regional_factor,

                0,
                1,
            )

        # ─────────────────────────────────────────────
        # REGIONAL REVENUE QUALITY
        # ─────────────────────────────────────────────

        if "monthly_revenue" in data.columns:

            data[
                "regional_revenue_strength"
            ] = (

                data[
                    "monthly_revenue"
                ]

                *

                data[
                    "regional_business_strength"
                ]

            )

        # ─────────────────────────────────────────────
        # REGIONAL OPPORTUNITY SCORE
        # ─────────────────────────────────────────────

        base_score = (

            data[
                "regional_business_strength"
            ]

            *

            data[
                "regional_digital_adoption"
            ]

            *

            data[
                "regional_enterprise_density"
            ]

        )

        data[
            "regional_growth_opportunity"
        ] = np.clip(

            base_score / 2,

            0,
            2,
        )

        logger.info(
            "Regional intelligence completed"
        )

        return data

    # ─────────────────────────────────────────────
    # STATE HEATMAP
    # ─────────────────────────────────────────────

    def generate_state_heatmap(
        self,
        df: pd.DataFrame,
        state_column: str = "state",
    ):

        logger.info(
            "Generating state heatmap"
        )

        data = df.copy()

        if state_column not in data.columns:

            return pd.DataFrame()

        grouped = data.groupby(
            state_column
        ).agg({

            "customer_id":
                "count",

            "monthly_revenue":
                "sum",

            "churn_probability":
                "mean",

            "engagement_score":
                "mean",

            "payment_delays":
                "mean",
        }).reset_index()

        grouped = grouped.rename(columns={

            "customer_id":
                "customer_count",

            "monthly_revenue":
                "total_revenue",

            "churn_probability":
                "avg_churn_probability",

            "engagement_score":
                "avg_engagement",

            "payment_delays":
                "avg_payment_delay",
        })

        # Revenue at risk
        grouped[
            "revenue_at_risk"
        ] = (

            grouped[
                "total_revenue"
            ]

            *

            grouped[
                "avg_churn_probability"
            ]

        )

        # Heat score
        grouped[
            "regional_risk_score"
        ] = (

            grouped[
                "avg_churn_probability"
            ] * 0.50

            +

            (
                1 -
                grouped[
                    "avg_engagement"
                ]
            ) * 0.30

            +

            np.clip(

                grouped[
                    "avg_payment_delay"
                ] / 10,

                0,
                1,
            ) * 0.20
        )

        grouped = grouped.sort_values(

            by="regional_risk_score",

            ascending=False,
        )

        return grouped

    # ─────────────────────────────────────────────
    # CITY INTELLIGENCE
    # ─────────────────────────────────────────────

    def city_level_analysis(
        self,
        df: pd.DataFrame,
        city_column: str = "city",
    ):

        logger.info(
            "Generating city intelligence"
        )

        data = df.copy()

        if city_column not in data.columns:

            return pd.DataFrame()

        grouped = data.groupby(
            city_column
        ).agg({

            "customer_id":
                "count",

            "monthly_revenue":
                "sum",

            "churn_probability":
                "mean",

            "support_tickets":
                "mean",
        }).reset_index()

        grouped = grouped.rename(columns={

            "customer_id":
                "customer_volume",

            "monthly_revenue":
                "city_revenue",

            "churn_probability":
                "city_churn_probability",

            "support_tickets":
                "avg_support_load",
        })

        grouped[
            "city_growth_score"
        ] = (

            np.log1p(
                grouped[
                    "customer_volume"
                ]
            )

            *

            (
                1 -
                grouped[
                    "city_churn_probability"
                ]
            )

        )

        return grouped.sort_values(

            by="city_growth_score",

            ascending=False,
        )

    # ─────────────────────────────────────────────
    # REGIONAL SEGMENTATION
    # ─────────────────────────────────────────────

    def classify_region(
        self,
        state,
    ):

        profile = self.state_profiles.get(

            state,

            {}
        )

        strength = profile.get(
            "business_strength",
            1.0,
        )

        digital = profile.get(
            "digital_adoption",
            1.0,
        )

        enterprise = profile.get(
            "enterprise_density",
            1.0,
        )

        score = (
            strength
            +
            digital
            +
            enterprise
        ) / 3

        if score >= 1.28:
            return "Tier-1 Enterprise Hub"

        if score >= 1.15:
            return "High Growth Market"

        if score >= 1.00:
            return "Emerging Digital Region"

        return "Developing Commercial Region"

    # ─────────────────────────────────────────────
    # REGIONAL FORECAST
    # ─────────────────────────────────────────────

    def regional_forecast(
        self,
        df: pd.DataFrame,
        months: int = 6,
    ):

        logger.info(
            "Generating regional forecast"
        )

        heatmap = self.generate_state_heatmap(
            df
        )

        forecasts = []

        for _, row in heatmap.iterrows():

            growth_rate = (

                1
                -

                row[
                    "avg_churn_probability"
                ]
            )

            future_revenue = (

                row[
                    "total_revenue"
                ]

                *

                (
                    1 +
                    growth_rate * 0.10
                ) ** months
            )

            forecasts.append({

                "state":
                    row["state"],

                "current_revenue":
                    round(
                        row[
                            "total_revenue"
                        ],
                        2,
                    ),

                "forecast_revenue":
                    round(
                        future_revenue,
                        2,
                    ),

                "growth_rate":
                    round(
                        growth_rate,
                        4,
                    ),

                "risk_score":
                    round(
                        row[
                            "regional_risk_score"
                        ],
                        4,
                    ),
            })

        return pd.DataFrame(forecasts)

    # ─────────────────────────────────────────────
    # EXECUTIVE REGIONAL SUMMARY
    # ─────────────────────────────────────────────

    def executive_summary(
        self,
        df: pd.DataFrame,
    ):

        logger.info(
            "Generating executive regional summary"
        )

        summary = {}

        if "monthly_revenue" in df.columns:

            summary[
                "total_revenue"
            ] = float(

                df[
                    "monthly_revenue"
                ].sum()

            )

        if "churn_probability" in df.columns:

            summary[
                "avg_churn_probability"
            ] = float(

                df[
                    "churn_probability"
                ].mean()

            )

        if "state" in df.columns:

            top_states = (

                df.groupby("state")[
                    "monthly_revenue"
                ]

                .sum()

                .sort_values(
                    ascending=False
                )

                .head(5)

                .to_dict()
            )

            summary[
                "top_revenue_states"
            ] = top_states

        summary[
            "generated_at"
        ] = datetime.utcnow().isoformat()

        summary[
            "total_customers"
        ] = int(len(df))

        return summary

    # ─────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────

    def _metric(
        self,
        state,
        metric,
    ):

        if pd.isna(state):
            return 1.0

        profile = self.state_profiles.get(

            str(state),

            {}
        )

        return float(
            profile.get(
                metric,
                1.0,
            )
        )


# ─────────────────────────────────────────────
# FUNCTIONAL API
# ─────────────────────────────────────────────

regional_engine = (
    HyperRegionalEngine()
)


def apply_regional_intelligence(
    df: pd.DataFrame,
    state_column="state",
):

    return (
        regional_engine
        .apply_regional_intelligence(
            df,
            state_column,
        )
    )


def generate_state_heatmap(
    df: pd.DataFrame,
    state_column="state",
):

    return (
        regional_engine
        .generate_state_heatmap(
            df,
            state_column,
        )
    )


def city_level_analysis(
    df: pd.DataFrame,
    city_column="city",
):

    return (
        regional_engine
        .city_level_analysis(
            df,
            city_column,
        )
    )


def regional_forecast(
    df: pd.DataFrame,
    months=6,
):

    return (
        regional_engine
        .regional_forecast(
            df,
            months,
        )
    )


def executive_regional_summary(
    df: pd.DataFrame,
):

    return (
        regional_engine
        .executive_summary(df)
    )