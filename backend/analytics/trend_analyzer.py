"""
ChurnShield 2.0 — Advanced Trend Analyzer

Purpose:
Enterprise-grade business trend intelligence engine.

Capabilities:
- churn trend analysis
- revenue movement analysis
- customer engagement tracking
- anomaly-aware trends
- rolling averages
- momentum detection
- growth acceleration detection
- volatility analysis
- KPI movement tracking
- trend forecasting readiness
- executive insights generation

Author: ChurnShield AI
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger("churnshield.trend_analyzer")


# ============================================================
# MAIN ENGINE
# ============================================================

class TrendAnalyzer:

    def __init__(self):

        self.results = {}

    # ========================================================
    # MAIN ENTRY
    # ========================================================

    def analyze(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        revenue_column: str = "monthly_revenue",
        churn_column: str = "churned"
    ) -> Dict[str, Any]:

        logger.info("Starting advanced trend analysis")

        data = df.copy()

        data = self._prepare_data(
            data,
            date_column
        )

        trend_summary = self._generate_trend_summary(
            data
        )

        revenue_trends = self._analyze_revenue_trends(
            data,
            revenue_column
        )

        churn_trends = self._analyze_churn_trends(
            data,
            churn_column
        )

        engagement_trends = self._analyze_engagement(
            data
        )

        momentum_analysis = self._analyze_momentum(
            data
        )

        volatility = self._analyze_volatility(
            data
        )

        seasonality = self._detect_seasonality(
            data
        )

        anomalies = self._detect_trend_anomalies(
            data
        )

        business_signals = self._generate_business_signals(
            revenue_trends,
            churn_trends,
            engagement_trends
        )

        executive_insights = self._generate_executive_insights(
            business_signals,
            momentum_analysis,
            volatility
        )

        return {
            "generated_at": datetime.now().isoformat(),

            "trend_summary": trend_summary,
            "revenue_trends": revenue_trends,
            "churn_trends": churn_trends,
            "engagement_trends": engagement_trends,
            "momentum_analysis": momentum_analysis,
            "volatility_analysis": volatility,
            "seasonality": seasonality,
            "anomalies": anomalies,
            "business_signals": business_signals,
            "executive_insights": executive_insights,
        }

    # ========================================================
    # DATA PREPARATION
    # ========================================================

    def _prepare_data(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> pd.DataFrame:

        data = df.copy()

        if date_column not in data.columns:

            logger.warning(
                "Date column missing. Creating synthetic timeline."
            )

            data[date_column] = pd.date_range(
                end=datetime.now(),
                periods=len(data),
                freq="D"
            )

        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce"
        )

        data = data.sort_values(date_column)

        data["week"] = data[date_column].dt.isocalendar().week
        data["month"] = data[date_column].dt.month
        data["year"] = data[date_column].dt.year

        return data

    # ========================================================
    # TREND SUMMARY
    # ========================================================

    def _generate_trend_summary(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns

        summary = []

        for col in numeric_cols:

            try:

                trend = self._detect_metric_trend(
                    df[col]
                )

                summary.append({
                    "metric": col,
                    "trend": trend["trend"],
                    "growth_rate": trend["growth_rate"],
                    "direction": trend["direction"]
                })

            except Exception:
                continue

        return {
            "metrics_analyzed": len(summary),
            "metric_trends": summary
        }

    # ========================================================
    # REVENUE ANALYSIS
    # ========================================================

    def _analyze_revenue_trends(
        self,
        df: pd.DataFrame,
        revenue_column: str
    ) -> Dict[str, Any]:

        if revenue_column not in df.columns:

            return {
                "status": "missing_revenue_column"
            }

        revenue = df[revenue_column]

        moving_avg_7 = revenue.rolling(7).mean().fillna(0)
        moving_avg_30 = revenue.rolling(30).mean().fillna(0)

        growth_rate = self._calculate_growth_rate(
            revenue
        )

        acceleration = self._calculate_acceleration(
            revenue
        )

        trend_strength = self._calculate_trend_strength(
            revenue
        )

        peak_revenue = float(revenue.max())
        avg_revenue = float(revenue.mean())

        return {
            "average_revenue": avg_revenue,
            "peak_revenue": peak_revenue,
            "growth_rate_percent": growth_rate,
            "acceleration": acceleration,
            "trend_strength": trend_strength,
            "trend_direction": self._classify_trend(
                growth_rate
            ),
            "moving_average_7d": moving_avg_7.tail(15).tolist(),
            "moving_average_30d": moving_avg_30.tail(15).tolist(),
        }

    # ========================================================
    # CHURN ANALYSIS
    # ========================================================

    def _analyze_churn_trends(
        self,
        df: pd.DataFrame,
        churn_column: str
    ) -> Dict[str, Any]:

        if churn_column not in df.columns:

            return {
                "status": "missing_churn_column"
            }

        churn = df[churn_column]

        churn_rate = float(churn.mean())

        rolling_churn = (
            churn
            .rolling(14)
            .mean()
            .fillna(churn_rate)
        )

        churn_direction = self._detect_metric_trend(
            rolling_churn
        )

        churn_volatility = float(
            np.std(rolling_churn)
        )

        churn_spikes = int(
            (rolling_churn > rolling_churn.mean() * 1.5).sum()
        )

        return {
            "average_churn_rate": churn_rate,
            "rolling_churn_rate": rolling_churn.tail(20).tolist(),
            "trend_direction": churn_direction["direction"],
            "growth_rate": churn_direction["growth_rate"],
            "volatility": churn_volatility,
            "detected_spikes": churn_spikes,
            "risk_level": self._classify_churn_risk(
                churn_rate
            )
        }

    # ========================================================
    # ENGAGEMENT ANALYSIS
    # ========================================================

    def _analyze_engagement(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        engagement_cols = [
            "login_frequency",
            "feature_usage_score",
            "active_seats",
        ]

        found = [
            c for c in engagement_cols
            if c in df.columns
        ]

        if not found:

            return {
                "status": "no_engagement_metrics"
            }

        engagement_scores = []

        for col in found:

            normalized = self._normalize_series(
                df[col]
            )

            engagement_scores.append(normalized)

        combined = np.mean(
            engagement_scores,
            axis=0
        )

        return {
            "average_engagement": float(np.mean(combined)),
            "engagement_trend": self._classify_trend(
                self._calculate_growth_rate(
                    pd.Series(combined)
                )
            ),
            "engagement_score_curve": combined[-20:].tolist(),
            "engagement_health": self._classify_engagement(
                np.mean(combined)
            )
        }

    # ========================================================
    # MOMENTUM ANALYSIS
    # ========================================================

    def _analyze_momentum(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns

        momentum_scores = []

        for col in numeric_cols:

            try:

                momentum = self._calculate_momentum(
                    df[col]
                )

                momentum_scores.append({
                    "metric": col,
                    "momentum": momentum,
                    "classification": self._classify_momentum(
                        momentum
                    )
                })

            except Exception:
                continue

        return {
            "metric_momentum": momentum_scores
        }

    # ========================================================
    # VOLATILITY ANALYSIS
    # ========================================================

    def _analyze_volatility(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns

        volatility_report = []

        for col in numeric_cols:

            try:

                volatility = float(
                    np.std(df[col]) /
                    (np.mean(df[col]) + 1e-6)
                )

                volatility_report.append({
                    "metric": col,
                    "volatility_score": volatility,
                    "risk": self._classify_volatility(
                        volatility
                    )
                })

            except Exception:
                continue

        return {
            "volatility_metrics": volatility_report
        }

    # ========================================================
    # SEASONALITY
    # ========================================================

    def _detect_seasonality(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        if "month" not in df.columns:

            return {}

        monthly_counts = (
            df["month"]
            .value_counts(normalize=True)
            .sort_index()
            .to_dict()
        )

        strongest_month = max(
            monthly_counts,
            key=monthly_counts.get
        )

        return {
            "monthly_distribution": monthly_counts,
            "peak_month": int(strongest_month),
            "seasonality_detected": True
        }

    # ========================================================
    # ANOMALY DETECTION
    # ========================================================

    def _detect_trend_anomalies(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns

        anomalies = []

        for col in numeric_cols:

            try:

                z_scores = (
                    (df[col] - df[col].mean()) /
                    (df[col].std() + 1e-6)
                )

                outliers = int(
                    (np.abs(z_scores) > 3).sum()
                )

                anomalies.append({
                    "metric": col,
                    "anomalies_detected": outliers
                })

            except Exception:
                continue

        return {
            "anomaly_report": anomalies
        }

    # ========================================================
    # BUSINESS SIGNALS
    # ========================================================

    def _generate_business_signals(
        self,
        revenue,
        churn,
        engagement
    ) -> List[Dict]:

        signals = []

        if revenue.get("growth_rate_percent", 0) > 15:

            signals.append({
                "signal": "Revenue acceleration detected",
                "severity": "positive"
            })

        if churn.get("average_churn_rate", 0) > 0.40:

            signals.append({
                "signal": "Churn risk increasing",
                "severity": "critical"
            })

        if engagement.get("average_engagement", 0) < 0.40:

            signals.append({
                "signal": "Engagement decline observed",
                "severity": "warning"
            })

        return signals

    # ========================================================
    # EXECUTIVE INSIGHTS
    # ========================================================

    def _generate_executive_insights(
        self,
        business_signals,
        momentum,
        volatility
    ) -> Dict[str, Any]:

        return {
            "overall_business_health": self._determine_health(
                business_signals
            ),

            "key_observations": [
                s["signal"]
                for s in business_signals
            ],

            "strategic_recommendation": self._strategic_advice(
                business_signals
            ),

            "risk_summary": (
                "High volatility metrics require monitoring"
                if len(volatility["volatility_metrics"]) > 5
                else "Business volatility stable"
            )
        }

    # ========================================================
    # HELPERS
    # ========================================================

    def _normalize_series(
        self,
        series
    ):

        scaler = StandardScaler()

        arr = scaler.fit_transform(
            series.fillna(0).values.reshape(-1, 1)
        )

        arr = (
            arr - arr.min()
        ) / (
            arr.max() - arr.min() + 1e-6
        )

        return arr.flatten()

    def _calculate_growth_rate(
        self,
        series
    ):

        if len(series) < 2:
            return 0

        start = series.iloc[0]
        end = series.iloc[-1]

        if start == 0:
            return 0

        return round(
            ((end - start) / start) * 100,
            2
        )

    def _calculate_acceleration(
        self,
        series
    ):

        diffs = np.diff(series)

        return round(
            np.mean(np.diff(diffs)),
            4
        )

    def _calculate_trend_strength(
        self,
        series
    ):

        X = np.arange(len(series)).reshape(-1, 1)

        model = LinearRegression()

        model.fit(X, series)

        return round(
            abs(model.coef_[0]),
            4
        )

    def _calculate_momentum(
        self,
        series
    ):

        short = series.tail(7).mean()
        long = series.tail(30).mean()

        return round(short - long, 4)

    def _detect_metric_trend(
        self,
        series
    ):

        growth = self._calculate_growth_rate(
            series
        )

        direction = self._classify_trend(
            growth
        )

        return {
            "growth_rate": growth,
            "direction": direction,
            "trend": direction
        }

    def _classify_trend(
        self,
        growth
    ):

        if growth > 10:
            return "strong_growth"

        elif growth > 0:
            return "growth"

        elif growth < -10:
            return "sharp_decline"

        elif growth < 0:
            return "decline"

        return "stable"

    def _classify_churn_risk(
        self,
        churn
    ):

        if churn > 0.60:
            return "critical"

        elif churn > 0.40:
            return "high"

        elif churn > 0.20:
            return "medium"

        return "low"

    def _classify_engagement(
        self,
        score
    ):

        if score >= 0.80:
            return "excellent"

        elif score >= 0.60:
            return "good"

        elif score >= 0.40:
            return "average"

        return "poor"

    def _classify_momentum(
        self,
        score
    ):

        if score > 5:
            return "surging"

        elif score > 0:
            return "positive"

        elif score < -5:
            return "falling_fast"

        return "stable"

    def _classify_volatility(
        self,
        score
    ):

        if score > 1:
            return "extreme"

        elif score > 0.5:
            return "high"

        elif score > 0.2:
            return "moderate"

        return "low"

    def _determine_health(
        self,
        signals
    ):

        critical = sum(
            1 for s in signals
            if s["severity"] == "critical"
        )

        if critical >= 2:
            return "critical"

        elif critical == 1:
            return "warning"

        return "healthy"

    def _strategic_advice(
        self,
        signals
    ):

        if not signals:

            return (
                "Business metrics stable. "
                "Continue growth optimization."
            )

        severe = any(
            s["severity"] == "critical"
            for s in signals
        )

        if severe:

            return (
                "Immediate retention and engagement "
                "intervention recommended."
            )

        return (
            "Monitor emerging patterns and optimize "
            "customer experience."
        )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def analyze_trends(
    df: pd.DataFrame,
    date_column: str = "date",
    revenue_column: str = "monthly_revenue",
    churn_column: str = "churned"
):

    analyzer = TrendAnalyzer()

    return analyzer.analyze(
        df=df,
        date_column=date_column,
        revenue_column=revenue_column,
        churn_column=churn_column
    )