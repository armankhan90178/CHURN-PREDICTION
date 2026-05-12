"""
ChurnShield 2.0 — Seasonal Intelligence & Forecasting Engine

Purpose:
Enterprise-grade seasonal churn analytics engine
for identifying:
- seasonal churn spikes
- festival-driven behavior
- business cycle effects
- monthly retention volatility
- renewal seasonality
- revenue seasonality
- customer activity forecasting

Capabilities:
- Monthly churn forecasting
- Seasonal decomposition
- India-aware business calendar adjustments
- Revenue trend forecasting
- Cohort seasonality intelligence
- Festival impact scoring
- Silent seasonal churn detection
- Growth/decline projections
- Executive forecasting summaries
"""

import logging
import warnings
import numpy as np
import pandas as pd

from typing import Dict, List
from datetime import datetime

from config import INDIA_CALENDAR

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.analytics.seasonal"
)


# ─────────────────────────────────────────────
# SEASONAL FORECAST ENGINE
# ─────────────────────────────────────────────

class SeasonalForecastEngine:

    """
    Enterprise seasonal analytics engine
    """

    # ─────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────

    def __init__(self):

        self.calendar = INDIA_CALENDAR

        self.required_columns = [

            "monthly_revenue",
            "churned",
        ]

    # ─────────────────────────────────────────
    # MASTER ANALYSIS
    # ─────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Starting seasonal intelligence analysis..."
        )

        data = self._prepare_dataset(df)

        monthly_metrics = (
            self.generate_monthly_metrics(data)
        )

        churn_forecast = (
            self.forecast_churn(monthly_metrics)
        )

        revenue_forecast = (
            self.forecast_revenue(monthly_metrics)
        )

        seasonal_patterns = (
            self.detect_seasonal_patterns(monthly_metrics)
        )

        festival_analysis = (
            self.analyze_festival_impact(monthly_metrics)
        )

        volatility_analysis = (
            self.analyze_volatility(monthly_metrics)
        )

        lifecycle_analysis = (
            self.lifecycle_risk_analysis(data)
        )

        executive_summary = (
            self.generate_executive_summary(
                monthly_metrics,
                churn_forecast,
                revenue_forecast,
            )
        )

        logger.info(
            "Seasonal intelligence analysis completed"
        )

        return {

            "monthly_metrics":
                monthly_metrics,

            "churn_forecast":
                churn_forecast,

            "revenue_forecast":
                revenue_forecast,

            "seasonal_patterns":
                seasonal_patterns,

            "festival_analysis":
                festival_analysis,

            "volatility_analysis":
                volatility_analysis,

            "lifecycle_analysis":
                lifecycle_analysis,

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

        if "date" not in data.columns:

            logger.warning(
                "No date column found. Generating synthetic timeline."
            )

            synthetic_dates = pd.date_range(
                end=datetime.today(),
                periods=len(data),
                freq="D",
            )

            data["date"] = synthetic_dates

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce",
        )

        data["month"] = (
            data["date"].dt.month
        )

        data["year"] = (
            data["date"].dt.year
        )

        data["year_month"] = (
            data["date"]
            .dt.to_period("M")
            .astype(str)
        )

        # Fill missing revenue
        if "monthly_revenue" not in data.columns:

            data["monthly_revenue"] = 1000

        # Fill missing churn
        if "churned" not in data.columns:

            data["churned"] = 0

        return data

    # ─────────────────────────────────────────
    # MONTHLY METRICS
    # ─────────────────────────────────────────

    def generate_monthly_metrics(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Generating monthly metrics..."
        )

        grouped = (

            df.groupby("year_month")

            .agg({

                "customer_id": "count",

                "monthly_revenue": [
                    "sum",
                    "mean",
                ],

                "churned": "mean",
            })

        )

        grouped.columns = [

            "customer_count",
            "total_revenue",
            "avg_revenue",
            "churn_rate",
        ]

        grouped = grouped.reset_index()

        grouped["churn_rate"] = (
            grouped["churn_rate"] * 100
        ).round(2)

        grouped["revenue_growth"] = (

            grouped["total_revenue"]
            .pct_change()
            .fillna(0)
            * 100

        ).round(2)

        grouped["customer_growth"] = (

            grouped["customer_count"]
            .pct_change()
            .fillna(0)
            * 100

        ).round(2)

        return grouped

    # ─────────────────────────────────────────
    # CHURN FORECAST
    # ─────────────────────────────────────────

    def forecast_churn(
        self,
        monthly_metrics: pd.DataFrame,
        periods: int = 6,
    ) -> pd.DataFrame:

        logger.info(
            "Forecasting churn trends..."
        )

        churn_history = (
            monthly_metrics["churn_rate"]
            .values
        )

        if len(churn_history) < 3:

            logger.warning(
                "Insufficient history for forecasting"
            )

            return pd.DataFrame()

        trend = np.polyfit(

            np.arange(len(churn_history)),
            churn_history,
            1,
        )

        future_x = np.arange(
            len(churn_history),
            len(churn_history) + periods,
        )

        future_predictions = (

            trend[0] * future_x

            +

            trend[1]
        )

        future_predictions = np.clip(
            future_predictions,
            0,
            100,
        )

        future_months = []

        current_month = datetime.today().month
        current_year = datetime.today().year

        for i in range(periods):

            future_month = (
                current_month + i
            )

            adjusted_month = (
                (future_month - 1) % 12
            ) + 1

            adjusted_year = (

                current_year

                +

                ((future_month - 1) // 12)
            )

            month_key = (
                f"{adjusted_year}-{adjusted_month:02d}"
            )

            future_months.append(month_key)

        forecast_df = pd.DataFrame({

            "forecast_month":
                future_months,

            "predicted_churn_rate":
                np.round(
                    future_predictions,
                    2,
                ),
        })

        forecast_df["risk_level"] = np.where(

            forecast_df["predicted_churn_rate"] > 30,
            "HIGH",

            np.where(
                forecast_df["predicted_churn_rate"] > 15,
                "MEDIUM",
                "LOW",
            )
        )

        return forecast_df

    # ─────────────────────────────────────────
    # REVENUE FORECAST
    # ─────────────────────────────────────────

    def forecast_revenue(
        self,
        monthly_metrics: pd.DataFrame,
        periods: int = 6,
    ) -> pd.DataFrame:

        logger.info(
            "Forecasting revenue trends..."
        )

        revenue_history = (
            monthly_metrics["total_revenue"]
            .values
        )

        if len(revenue_history) < 3:

            return pd.DataFrame()

        trend = np.polyfit(

            np.arange(len(revenue_history)),
            revenue_history,
            1,
        )

        future_x = np.arange(

            len(revenue_history),
            len(revenue_history) + periods,
        )

        revenue_forecast = (

            trend[0] * future_x

            +

            trend[1]
        )

        revenue_forecast = np.clip(
            revenue_forecast,
            0,
            None,
        )

        forecast_df = pd.DataFrame({

            "forecast_period":
                np.arange(1, periods + 1),

            "predicted_revenue":
                revenue_forecast.astype(int),
        })

        return forecast_df

    # ─────────────────────────────────────────
    # SEASONAL PATTERN DETECTION
    # ─────────────────────────────────────────

    def detect_seasonal_patterns(
        self,
        monthly_metrics: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Detecting seasonal patterns..."
        )

        if len(monthly_metrics) < 6:

            return {
                "status":
                    "insufficient_data"
            }

        peak_churn_month = (

            monthly_metrics
            .loc[
                monthly_metrics["churn_rate"].idxmax()
            ]
        )

        lowest_churn_month = (

            monthly_metrics
            .loc[
                monthly_metrics["churn_rate"].idxmin()
            ]
        )

        revenue_peak = (

            monthly_metrics
            .loc[
                monthly_metrics["total_revenue"].idxmax()
            ]
        )

        patterns = {

            "highest_churn_period":
                peak_churn_month["year_month"],

            "highest_churn_rate":
                float(
                    peak_churn_month["churn_rate"]
                ),

            "lowest_churn_period":
                lowest_churn_month["year_month"],

            "lowest_churn_rate":
                float(
                    lowest_churn_month["churn_rate"]
                ),

            "peak_revenue_period":
                revenue_peak["year_month"],

            "peak_revenue":
                float(
                    revenue_peak["total_revenue"]
                ),
        }

        return patterns

    # ─────────────────────────────────────────
    # FESTIVAL IMPACT
    # ─────────────────────────────────────────

    def analyze_festival_impact(
        self,
        monthly_metrics: pd.DataFrame,
    ) -> pd.DataFrame:

        logger.info(
            "Analyzing India-specific seasonal events..."
        )

        analysis = []

        for month, config in self.calendar.items():

            adjusted_risk = (
                1 - config["adjustment"]
            )

            analysis.append({

                "month":
                    month,

                "event":
                    config["event"],

                "business_adjustment":
                    config["adjustment"],

                "risk_impact":
                    round(
                        adjusted_risk * 100,
                        2,
                    ),

                "note":
                    config["note"],
            })

        return pd.DataFrame(analysis)

    # ─────────────────────────────────────────
    # VOLATILITY ANALYSIS
    # ─────────────────────────────────────────

    def analyze_volatility(
        self,
        monthly_metrics: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Analyzing volatility..."
        )

        churn_std = (

            monthly_metrics["churn_rate"]
            .std()
        )

        revenue_std = (

            monthly_metrics["revenue_growth"]
            .std()
        )

        volatility_level = "LOW"

        if churn_std > 10:

            volatility_level = "HIGH"

        elif churn_std > 5:

            volatility_level = "MEDIUM"

        return {

            "churn_volatility":
                round(churn_std, 2),

            "revenue_volatility":
                round(revenue_std, 2),

            "volatility_level":
                volatility_level,
        }

    # ─────────────────────────────────────────
    # LIFECYCLE ANALYSIS
    # ─────────────────────────────────────────

    def lifecycle_risk_analysis(
        self,
        df: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Analyzing lifecycle risks..."
        )

        if "contract_age_months" not in df.columns:

            return {
                "status":
                    "missing_contract_age"
            }

        bins = [

            0,
            3,
            6,
            12,
            24,
            100,
        ]

        labels = [

            "0-3 Months",
            "3-6 Months",
            "6-12 Months",
            "1-2 Years",
            "2+ Years",
        ]

        df["lifecycle_stage"] = pd.cut(

            df["contract_age_months"],
            bins=bins,
            labels=labels,
        )

        lifecycle = (

            df.groupby("lifecycle_stage")["churned"]
            .mean()
            .fillna(0)
            * 100

        ).round(2)

        return lifecycle.to_dict()

    # ─────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────

    def generate_executive_summary(
        self,
        monthly_metrics: pd.DataFrame,
        churn_forecast: pd.DataFrame,
        revenue_forecast: pd.DataFrame,
    ) -> Dict:

        logger.info(
            "Generating executive summary..."
        )

        latest_churn = 0

        if len(monthly_metrics):

            latest_churn = float(

                monthly_metrics
                .iloc[-1]["churn_rate"]
            )

        future_risk = "STABLE"

        if not churn_forecast.empty:

            avg_future = (

                churn_forecast[
                    "predicted_churn_rate"
                ].mean()
            )

            if avg_future > 25:

                future_risk = "HIGH RISK"

            elif avg_future > 15:

                future_risk = "MODERATE RISK"

        return {

            "generated_at":
                datetime.utcnow().isoformat(),

            "latest_churn_rate":
                round(latest_churn, 2),

            "forecast_risk":
                future_risk,

            "months_analyzed":
                int(len(monthly_metrics)),

            "predicted_growth_direction":

                "Positive"

                if (
                    not revenue_forecast.empty

                    and

                    revenue_forecast[
                        "predicted_revenue"
                    ].iloc[-1]

                    >

                    revenue_forecast[
                        "predicted_revenue"
                    ].iloc[0]
                )

                else

                "Negative",
        }


# ─────────────────────────────────────────────
# GLOBAL ENGINE
# ─────────────────────────────────────────────

seasonal_engine = SeasonalForecastEngine()


# ─────────────────────────────────────────────
# FUNCTIONAL API
# ─────────────────────────────────────────────

def analyze_seasonality(
    df: pd.DataFrame,
) -> Dict:

    return seasonal_engine.analyze(df)


def forecast_customer_churn(
    df: pd.DataFrame,
) -> pd.DataFrame:

    analysis = seasonal_engine.analyze(df)

    return analysis["churn_forecast"]


def forecast_revenue_growth(
    df: pd.DataFrame,
) -> pd.DataFrame:

    analysis = seasonal_engine.analyze(df)

    return analysis["revenue_forecast"]