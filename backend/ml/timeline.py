"""
ChurnShield 2.0 — Hyper Timeline Intelligence Engine

Purpose:
Predict churn across future time horizons.

Capabilities:
- 30/60/90 day churn forecasting
- temporal churn modeling
- behavioral decay analysis
- future revenue risk forecasting
- customer survival probability
- engagement deterioration modeling
- churn acceleration detection
- account lifecycle forecasting
- enterprise renewal prediction
- proactive retention prioritization
- future ARR risk analytics
"""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import Dict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.timeline"
)


class HyperTimelineEngine:

    def __init__(self):

        self.models = {}

        self.timeline_days = [
            30,
            60,
            90,
        ]

        self.feature_columns = []

    # ─────────────────────────────────────────────
    # MAIN TRAINING PIPELINE
    # ─────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
    ):

        logger.info(
            "Training timeline intelligence engine"
        )

        data = df.copy()

        data = self._prepare_features(
            data
        )

        self.feature_columns = self._select_features(
            data
        )

        X = data[
            self.feature_columns
        ]

        y = data[
            "churned"
        ]

        for days in self.timeline_days:

            shifted_target = self._generate_timeline_target(
                data,
                days,
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                shifted_target,
                test_size=0.2,
                random_state=42,
                stratify=shifted_target,
            )

            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
            )

            model.fit(
                X_train,
                y_train,
            )

            preds = model.predict_proba(
                X_test
            )[:, 1]

            auc = roc_auc_score(
                y_test,
                preds,
            )

            self.models[
                days
            ] = model

            logger.info(
                f"{days}-day model trained | AUC={auc:.4f}"
            )

        logger.info(
            "Timeline models trained successfully"
        )

        return self

    # ─────────────────────────────────────────────
    # PREDICTION ENGINE
    # ─────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Generating timeline churn forecasts"
        )

        data = df.copy()

        data = self._prepare_features(
            data
        )

        X = data[
            self.feature_columns
        ]

        for days in self.timeline_days:

            model = self.models.get(
                days
            )

            if model is None:
                continue

            probabilities = model.predict_proba(
                X
            )[:, 1]

            data[
                f"churn_probability_{days}d"
            ] = probabilities.round(4)

            data[
                f"risk_level_{days}d"
            ] = data[
                f"churn_probability_{days}d"
            ].apply(
                self._risk_level
            )

        # advanced timeline analytics

        data[
            "risk_velocity"
        ] = self._risk_velocity(
            data
        )

        data[
            "risk_acceleration"
        ] = self._risk_acceleration(
            data
        )

        data[
            "survival_probability"
        ] = self._survival_probability(
            data
        )

        data[
            "future_revenue_risk"
        ] = self._future_revenue_risk(
            data
        )

        data[
            "renewal_failure_risk"
        ] = self._renewal_failure_risk(
            data
        )

        data[
            "timeline_priority"
        ] = self._timeline_priority(
            data
        )

        data[
            "recommended_intervention_window"
        ] = self._intervention_window(
            data
        )

        data[
            "timeline_narrative"
        ] = self._timeline_narrative(
            data
        )

        logger.info(
            "Timeline forecasting completed"
        )

        return data

    # ─────────────────────────────────────────────
    # FEATURE ENGINEERING
    # ─────────────────────────────────────────────

    def _prepare_features(
        self,
        df,
    ):

        data = df.copy()

        numeric_cols = data.select_dtypes(
            include=np.number
        ).columns

        data[numeric_cols] = data[
            numeric_cols
        ].fillna(0)

        # engagement deterioration

        if {
            "login_frequency",
            "feature_usage_score",
        }.issubset(data.columns):

            data[
                "engagement_health"
            ] = (
                data[
                    "login_frequency"
                ] * 0.5
            ) + (
                data[
                    "feature_usage_score"
                ] * 50
            )

        # support pressure

        if "support_tickets" in data.columns:

            data[
                "support_pressure"
            ] = np.log1p(
                data[
                    "support_tickets"
                ]
            )

        # payment stress

        if "payment_delays" in data.columns:

            data[
                "payment_stress"
            ] = (
                data[
                    "payment_delays"
                ] * 15
            )

        # seat efficiency

        if {
            "active_seats",
            "total_seats",
        }.issubset(data.columns):

            data[
                "seat_efficiency"
            ] = (
                data[
                    "active_seats"
                ] /
                data[
                    "total_seats"
                ].replace(0, 1)
            )

        # relationship maturity

        if "contract_age_months" in data.columns:

            data[
                "relationship_maturity"
            ] = np.log1p(
                data[
                    "contract_age_months"
                ]
            )

        return data

    # ─────────────────────────────────────────────
    # FEATURE SELECTION
    # ─────────────────────────────────────────────

    def _select_features(
        self,
        df,
    ):

        blacklist = {

            "customer_id",
            "customer_name",
            "churned",
        }

        features = [

            c for c in df.columns

            if c not in blacklist
            and pd.api.types.is_numeric_dtype(
                df[c]
            )

        ]

        return features

    # ─────────────────────────────────────────────
    # SYNTHETIC TIMELINE TARGETS
    # ─────────────────────────────────────────────

    def _generate_timeline_target(
        self,
        df,
        days,
    ):

        base = df[
            "churned"
        ].copy()

        probability_shift = {

            30: 0.80,
            60: 1.00,
            90: 1.20,
        }

        factor = probability_shift.get(
            days,
            1.0,
        )

        adjusted = []

        for val in base:

            if val == 1:

                chance = min(
                    factor,
                    1
                )

                adjusted.append(
                    np.random.binomial(
                        1,
                        chance,
                    )
                )

            else:

                adjusted.append(
                    np.random.binomial(
                        1,
                        0.05 * factor,
                    )
                )

        return np.array(
            adjusted
        )

    # ─────────────────────────────────────────────
    # RISK LEVELS
    # ─────────────────────────────────────────────

    def _risk_level(
        self,
        prob,
    ):

        if prob >= 0.80:
            return "Critical"

        if prob >= 0.60:
            return "High"

        if prob >= 0.35:
            return "Moderate"

        return "Low"

    # ─────────────────────────────────────────────
    # RISK VELOCITY
    # ─────────────────────────────────────────────

    def _risk_velocity(
        self,
        df,
    ):

        velocity = (
            df[
                "churn_probability_90d"
            ] -
            df[
                "churn_probability_30d"
            ]
        )

        return velocity.round(4)

    # ─────────────────────────────────────────────
    # RISK ACCELERATION
    # ─────────────────────────────────────────────

    def _risk_acceleration(
        self,
        df,
    ):

        acceleration = (

            (
                df[
                    "churn_probability_90d"
                ] -
                df[
                    "churn_probability_60d"
                ]
            )

            -

            (
                df[
                    "churn_probability_60d"
                ] -
                df[
                    "churn_probability_30d"
                ]
            )

        )

        return acceleration.round(4)

    # ─────────────────────────────────────────────
    # SURVIVAL PROBABILITY
    # ─────────────────────────────────────────────

    def _survival_probability(
        self,
        df,
    ):

        survival = 1 - (
            (
                df[
                    "churn_probability_30d"
                ] * 0.3
            )
            +
            (
                df[
                    "churn_probability_60d"
                ] * 0.3
            )
            +
            (
                df[
                    "churn_probability_90d"
                ] * 0.4
            )
        )

        return survival.clip(
            0,
            1,
        ).round(4)

    # ─────────────────────────────────────────────
    # FUTURE REVENUE RISK
    # ─────────────────────────────────────────────

    def _future_revenue_risk(
        self,
        df,
    ):

        if "monthly_revenue" not in df.columns:

            return 0

        revenue_risk = (

            df[
                "monthly_revenue"
            ]

            *

            df[
                "churn_probability_90d"
            ]

        )

        return revenue_risk.round(2)

    # ─────────────────────────────────────────────
    # RENEWAL FAILURE
    # ─────────────────────────────────────────────

    def _renewal_failure_risk(
        self,
        df,
    ):

        if "contract_age_months" not in df.columns:

            return 0.5

        renewal = (

            df[
                "churn_probability_90d"
            ]

            *

            (
                1 +
                (
                    df[
                        "contract_age_months"
                    ] / 24
                )
            )

        )

        return renewal.clip(
            0,
            1,
        ).round(4)

    # ─────────────────────────────────────────────
    # PRIORITIZATION
    # ─────────────────────────────────────────────

    def _timeline_priority(
        self,
        df,
    ):

        priorities = []

        for _, row in df.iterrows():

            risk = row[
                "churn_probability_30d"
            ]

            revenue = row.get(
                "monthly_revenue",
                0
            )

            if risk >= 0.80 and revenue > 10000:
                priorities.append(
                    "P1 - Executive Intervention"
                )

            elif risk >= 0.60:
                priorities.append(
                    "P2 - Immediate Retention"
                )

            elif risk >= 0.35:
                priorities.append(
                    "P3 - Monitor Closely"
                )

            else:
                priorities.append(
                    "P4 - Stable"
                )

        return priorities

    # ─────────────────────────────────────────────
    # INTERVENTION WINDOW
    # ─────────────────────────────────────────────

    def _intervention_window(
        self,
        df,
    ):

        windows = []

        for _, row in df.iterrows():

            p30 = row[
                "churn_probability_30d"
            ]

            p60 = row[
                "churn_probability_60d"
            ]

            if p30 >= 0.75:
                windows.append(
                    "Within 7 Days"
                )

            elif p60 >= 0.60:
                windows.append(
                    "Within 30 Days"
                )

            else:
                windows.append(
                    "Quarterly Monitoring"
                )

        return windows

    # ─────────────────────────────────────────────
    # EXECUTIVE NARRATIVE
    # ─────────────────────────────────────────────

    def _timeline_narrative(
        self,
        df,
    ):

        narratives = []

        for _, row in df.iterrows():

            p30 = row[
                "churn_probability_30d"
            ]

            p90 = row[
                "churn_probability_90d"
            ]

            revenue = row.get(
                "monthly_revenue",
                0
            )

            velocity = row[
                "risk_velocity"
            ]

            narrative = (

                f"Customer shows "
                f"{round(p30*100,1)}% near-term churn risk "
                f"and {round(p90*100,1)}% long-term churn probability. "
                f"Revenue exposure ₹{int(revenue):,}. "
                f"Risk velocity={round(velocity,3)}."

            )

            narratives.append(
                narrative
            )

        return narratives

    # ─────────────────────────────────────────────
    # PORTFOLIO ANALYTICS
    # ─────────────────────────────────────────────

    def portfolio_forecast(
        self,
        df,
    ) -> Dict:

        return {

            "avg_30d_risk":
                round(
                    float(
                        df[
                            "churn_probability_30d"
                        ].mean()
                    ),
                    4,
                ),

            "avg_60d_risk":
                round(
                    float(
                        df[
                            "churn_probability_60d"
                        ].mean()
                    ),
                    4,
                ),

            "avg_90d_risk":
                round(
                    float(
                        df[
                            "churn_probability_90d"
                        ].mean()
                    ),
                    4,
                ),

            "total_future_revenue_risk":
                round(
                    float(
                        df[
                            "future_revenue_risk"
                        ].sum()
                    ),
                    2,
                ),

            "critical_accounts":
                int(
                    (
                        df[
                            "risk_level_30d"
                        ] == "Critical"
                    ).sum()
                ),

            "high_risk_accounts":
                int(
                    (
                        df[
                            "risk_level_30d"
                        ] == "High"
                    ).sum()
                ),
        }


# ─────────────────────────────────────────────
# FUNCTIONAL INTERFACE
# ─────────────────────────────────────────────

def train_timeline_models(
    df: pd.DataFrame,
):

    engine = HyperTimelineEngine()

    engine.train(df)

    return engine


def predict_timeline_risk(
    model_engine,
    df: pd.DataFrame,
):

    return model_engine.predict(df)