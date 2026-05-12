"""
ChurnShield 2.0 — Hyper India Business Calendar Engine

Purpose:
Enterprise-grade Indian business calendar intelligence
for churn prediction, forecasting, retention timing,
seasonality adjustment, and customer engagement planning.

Capabilities:
- Indian holiday intelligence
- festival-aware churn prediction
- business slowdown detection
- GST & tax cycle adjustments
- state-level seasonality
- financial year intelligence
- customer activity normalization
- retention campaign timing optimization
- business event forecasting
- engagement heat scoring
- monthly risk amplification
- regional commercial behavior modeling
"""

import logging
import calendar

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from typing import Dict, List

from config import INDIA_CALENDAR

logger = logging.getLogger(
    "churnshield.india.calendar"
)


class HyperIndiaCalendarEngine:

    def __init__(self):

        self.calendar_rules = INDIA_CALENDAR

        # ─────────────────────────────────────────────
        # INDIAN FESTIVALS
        # ─────────────────────────────────────────────

        self.major_events = {

            "Diwali": {
                "month": 11,
                "risk_multiplier": 1.35,
                "engagement_drop": 0.30,
                "business_type": [
                    "B2B SaaS",
                    "Finance",
                    "Telecom",
                ],
            },

            "Navratri": {
                "month": 10,
                "risk_multiplier": 1.15,
                "engagement_drop": 0.18,
            },

            "Ramzan": {
                "month": 3,
                "risk_multiplier": 1.08,
                "engagement_drop": 0.10,
            },

            "Christmas": {
                "month": 12,
                "risk_multiplier": 1.12,
                "engagement_drop": 0.12,
            },

            "Financial Year End": {
                "month": 3,
                "risk_multiplier": 1.40,
                "engagement_drop": 0.35,
            },

            "Union Budget": {
                "month": 2,
                "risk_multiplier": 1.10,
                "engagement_drop": 0.08,
            },

            "GST Filing": {
                "month": 1,
                "risk_multiplier": 1.18,
                "engagement_drop": 0.20,
            },
        }

        # ─────────────────────────────────────────────
        # STATE-LEVEL COMMERCIAL BEHAVIOR
        # ─────────────────────────────────────────────

        self.state_profiles = {

            "Maharashtra": {
                "business_activity": 1.20,
                "payment_speed": 1.10,
                "digital_adoption": 1.25,
            },

            "Karnataka": {
                "business_activity": 1.15,
                "payment_speed": 1.05,
                "digital_adoption": 1.35,
            },

            "Telangana": {
                "business_activity": 1.08,
                "payment_speed": 1.02,
                "digital_adoption": 1.18,
            },

            "Andhra Pradesh": {
                "business_activity": 1.00,
                "payment_speed": 0.98,
                "digital_adoption": 1.05,
            },

            "Tamil Nadu": {
                "business_activity": 1.12,
                "payment_speed": 1.06,
                "digital_adoption": 1.20,
            },

            "Delhi": {
                "business_activity": 1.25,
                "payment_speed": 1.15,
                "digital_adoption": 1.28,
            },

            "Gujarat": {
                "business_activity": 1.18,
                "payment_speed": 1.12,
                "digital_adoption": 1.08,
            },
        }

    # ─────────────────────────────────────────────
    # MAIN ADJUSTMENT ENGINE
    # ─────────────────────────────────────────────

    def apply_calendar_intelligence(
        self,
        df: pd.DataFrame,
    ):

        logger.info(
            "Applying India calendar intelligence"
        )

        data = df.copy()

        current_month = datetime.now().month

        data[
            "calendar_adjustment_factor"
        ] = self._monthly_adjustment(
            current_month
        )

        data[
            "festival_risk_score"
        ] = self._festival_risk(
            current_month
        )

        data[
            "business_slowdown_score"
        ] = self._slowdown_score(
            current_month
        )

        data[
            "engagement_seasonality"
        ] = self._engagement_factor(
            current_month
        )

        data[
            "payment_cycle_risk"
        ] = self._payment_cycle_risk(
            current_month
        )

        # Revenue adjusted
        if "monthly_revenue" in data.columns:

            data[
                "seasonally_adjusted_revenue"
            ] = (

                data["monthly_revenue"]

                *

                data[
                    "calendar_adjustment_factor"
                ]

            )

        # Engagement adjusted
        if "engagement_score" in data.columns:

            data[
                "seasonally_adjusted_engagement"
            ] = (

                data["engagement_score"]

                *

                data[
                    "engagement_seasonality"
                ]

            )

        # Churn amplification
        if "churn_probability" in data.columns:

            data[
                "calendar_adjusted_churn_probability"
            ] = np.clip(

                data["churn_probability"]

                *

                data[
                    "festival_risk_score"
                ]

                *

                data[
                    "payment_cycle_risk"
                ],

                0,
                1,
            )

        logger.info(
            "Calendar intelligence completed"
        )

        return data

    # ─────────────────────────────────────────────
    # MONTHLY ADJUSTMENT
    # ─────────────────────────────────────────────

    def _monthly_adjustment(
        self,
        month,
    ):

        adjustment = (
            self.calendar_rules
            .get(month, {})
            .get("adjustment", 1.0)
        )

        return float(adjustment)

    # ─────────────────────────────────────────────
    # FESTIVAL RISK
    # ─────────────────────────────────────────────

    def _festival_risk(
        self,
        month,
    ):

        multiplier = 1.0

        for _, event in (
            self.major_events.items()
        ):

            if event["month"] == month:

                multiplier *= (
                    event[
                        "risk_multiplier"
                    ]
                )

        return round(
            multiplier,
            3,
        )

    # ─────────────────────────────────────────────
    # BUSINESS SLOWDOWN
    # ─────────────────────────────────────────────

    def _slowdown_score(
        self,
        month,
    ):

        if month in [3, 10, 11]:
            return 0.75

        if month in [1, 2, 12]:
            return 0.85

        return 1.0

    # ─────────────────────────────────────────────
    # ENGAGEMENT FACTOR
    # ─────────────────────────────────────────────

    def _engagement_factor(
        self,
        month,
    ):

        base = 1.0

        for _, event in (
            self.major_events.items()
        ):

            if event["month"] == month:

                base -= (
                    event[
                        "engagement_drop"
                    ]
                )

        return max(
            round(base, 3),
            0.50,
        )

    # ─────────────────────────────────────────────
    # PAYMENT RISK
    # ─────────────────────────────────────────────

    def _payment_cycle_risk(
        self,
        month,
    ):

        if month in [3, 4]:
            return 1.30

        if month in [10, 11]:
            return 1.18

        if month in [1, 7]:
            return 1.10

        return 1.0

    # ─────────────────────────────────────────────
    # STATE INTELLIGENCE
    # ─────────────────────────────────────────────

    def apply_state_intelligence(
        self,
        df: pd.DataFrame,
        state_column: str = "state",
    ):

        logger.info(
            "Applying state intelligence"
        )

        data = df.copy()

        if state_column not in data.columns:

            logger.warning(
                "State column missing"
            )

            return data

        data[
            "state_business_activity"
        ] = data[state_column].apply(
            lambda x:
            self._state_metric(
                x,
                "business_activity",
            )
        )

        data[
            "state_payment_speed"
        ] = data[state_column].apply(
            lambda x:
            self._state_metric(
                x,
                "payment_speed",
            )
        )

        data[
            "state_digital_adoption"
        ] = data[state_column].apply(
            lambda x:
            self._state_metric(
                x,
                "digital_adoption",
            )
        )

        # Enhanced churn
        if "churn_probability" in data.columns:

            data[
                "state_adjusted_churn_probability"
            ] = np.clip(

                data["churn_probability"]

                /

                data[
                    "state_digital_adoption"
                ]

                *

                (
                    2
                    -
                    data[
                        "state_payment_speed"
                    ]
                ),

                0,
                1,
            )

        return data

    # ─────────────────────────────────────────────
    # STATE METRIC
    # ─────────────────────────────────────────────

    def _state_metric(
        self,
        state,
        metric,
    ):

        if pd.isna(state):
            return 1.0

        profile = (
            self.state_profiles.get(
                str(state),
                {},
            )
        )

        return float(
            profile.get(
                metric,
                1.0,
            )
        )

    # ─────────────────────────────────────────────
    # RETENTION TIMING
    # ─────────────────────────────────────────────

    def best_retention_window(
        self,
        current_date=None,
    ):

        if current_date is None:

            current_date = datetime.now()

        current_month = current_date.month

        bad_months = [3, 10, 11]

        recommendations = []

        for i in range(1, 90):

            future_date = (
                current_date
                +
                timedelta(days=i)
            )

            if future_date.month not in bad_months:

                recommendations.append(
                    future_date.strftime(
                        "%Y-%m-%d"
                    )
                )

            if len(recommendations) >= 5:
                break

        return recommendations

    # ─────────────────────────────────────────────
    # EXECUTIVE CALENDAR SUMMARY
    # ─────────────────────────────────────────────

    def executive_calendar_summary(
        self,
    ):

        current_month = datetime.now().month

        month_name = calendar.month_name[
            current_month
        ]

        rule = self.calendar_rules.get(
            current_month,
            {},
        )

        return {

            "month":
                month_name,

            "event":
                rule.get(
                    "event",
                    "Normal Business Period",
                ),

            "adjustment":
                rule.get(
                    "adjustment",
                    1.0,
                ),

            "note":
                rule.get(
                    "note",
                    "",
                ),

            "festival_risk":
                self._festival_risk(
                    current_month
                ),

            "payment_cycle_risk":
                self._payment_cycle_risk(
                    current_month
                ),

            "recommended_retention_days":
                self.best_retention_window(),
        }

    # ─────────────────────────────────────────────
    # FORECAST BUSINESS HEATMAP
    # ─────────────────────────────────────────────

    def generate_business_heatmap(
        self,
    ):

        records = []

        for month in range(1, 13):

            records.append({

                "month":
                    calendar.month_name[
                        month
                    ],

                "adjustment_factor":
                    self._monthly_adjustment(
                        month
                    ),

                "festival_risk":
                    self._festival_risk(
                        month
                    ),

                "payment_risk":
                    self._payment_cycle_risk(
                        month
                    ),

                "engagement_factor":
                    self._engagement_factor(
                        month
                    ),

                "business_slowdown":
                    self._slowdown_score(
                        month
                    ),
            })

        return pd.DataFrame(records)


# ─────────────────────────────────────────────
# FUNCTIONAL API
# ─────────────────────────────────────────────

calendar_engine = (
    HyperIndiaCalendarEngine()
)


def apply_calendar_intelligence(
    df: pd.DataFrame,
):

    return (
        calendar_engine
        .apply_calendar_intelligence(df)
    )


def apply_state_intelligence(
    df: pd.DataFrame,
    state_column="state",
):

    return (
        calendar_engine
        .apply_state_intelligence(
            df,
            state_column,
        )
    )


def executive_calendar_summary():

    return (
        calendar_engine
        .executive_calendar_summary()
    )


def generate_business_heatmap():

    return (
        calendar_engine
        .generate_business_heatmap()
    )


def best_retention_window():

    return (
        calendar_engine
        .best_retention_window()
    )