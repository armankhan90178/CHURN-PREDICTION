"""
ChurnShield 2.0 — Industry Benchmark Engine

Purpose:
Compare uploaded business metrics
against industry benchmark standards.

Capabilities:
- universal industry benchmarking
- percentile ranking
- peer comparison
- churn risk benchmarking
- KPI scoring
- business maturity detection
- health scoring
- executive summaries
- AI-ready benchmark insights
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("churnshield.benchmark")


# ─────────────────────────────────────────────
# INDUSTRY BENCHMARK DATABASE
# ─────────────────────────────────────────────

INDUSTRY_BENCHMARKS = {
    "b2b saas": {
        "avg_churn_rate": 0.08,
        "avg_nps": 42,
        "avg_feature_usage": 0.68,
        "avg_ticket_rate": 1.8,
        "avg_login_frequency": 18,
        "avg_retention": 0.92,
        "avg_payment_delay": 0.4,
        "avg_seat_utilization": 0.74,
    },

    "telecom": {
        "avg_churn_rate": 0.18,
        "avg_nps": 28,
        "avg_feature_usage": 0.58,
        "avg_ticket_rate": 2.5,
        "avg_login_frequency": 11,
        "avg_retention": 0.82,
        "avg_payment_delay": 1.1,
        "avg_seat_utilization": 0.66,
    },

    "ott streaming": {
        "avg_churn_rate": 0.22,
        "avg_nps": 31,
        "avg_feature_usage": 0.63,
        "avg_ticket_rate": 1.2,
        "avg_login_frequency": 24,
        "avg_retention": 0.78,
        "avg_payment_delay": 0.2,
        "avg_seat_utilization": 0.81,
    },

    "banking": {
        "avg_churn_rate": 0.11,
        "avg_nps": 36,
        "avg_feature_usage": 0.71,
        "avg_ticket_rate": 2.0,
        "avg_login_frequency": 14,
        "avg_retention": 0.89,
        "avg_payment_delay": 0.8,
        "avg_seat_utilization": 0.70,
    },

    "healthcare": {
        "avg_churn_rate": 0.07,
        "avg_nps": 48,
        "avg_feature_usage": 0.76,
        "avg_ticket_rate": 1.0,
        "avg_login_frequency": 9,
        "avg_retention": 0.93,
        "avg_payment_delay": 0.3,
        "avg_seat_utilization": 0.72,
    },

    "ecommerce": {
        "avg_churn_rate": 0.26,
        "avg_nps": 25,
        "avg_feature_usage": 0.49,
        "avg_ticket_rate": 3.4,
        "avg_login_frequency": 8,
        "avg_retention": 0.74,
        "avg_payment_delay": 1.5,
        "avg_seat_utilization": 0.58,
    },

    "education": {
        "avg_churn_rate": 0.19,
        "avg_nps": 34,
        "avg_feature_usage": 0.60,
        "avg_ticket_rate": 1.7,
        "avg_login_frequency": 12,
        "avg_retention": 0.81,
        "avg_payment_delay": 0.5,
        "avg_seat_utilization": 0.69,
    },

    "fitness": {
        "avg_churn_rate": 0.28,
        "avg_nps": 29,
        "avg_feature_usage": 0.55,
        "avg_ticket_rate": 0.8,
        "avg_login_frequency": 7,
        "avg_retention": 0.71,
        "avg_payment_delay": 1.2,
        "avg_seat_utilization": 0.62,
    },

    "default": {
        "avg_churn_rate": 0.15,
        "avg_nps": 35,
        "avg_feature_usage": 0.60,
        "avg_ticket_rate": 2.0,
        "avg_login_frequency": 12,
        "avg_retention": 0.85,
        "avg_payment_delay": 0.7,
        "avg_seat_utilization": 0.68,
    }
}


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────

class BenchmarkAnalyzer:

    def __init__(self):
        logger.info("BenchmarkAnalyzer initialized")

    # ─────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        industry: str = "default"
    ) -> Dict:

        logger.info(f"Running benchmark analysis for: {industry}")

        industry_key = industry.lower().strip()

        benchmark = INDUSTRY_BENCHMARKS.get(
            industry_key,
            INDUSTRY_BENCHMARKS["default"]
        )

        business_metrics = self._extract_business_metrics(df)

        benchmark_scores = self._compare_against_benchmark(
            business_metrics,
            benchmark
        )

        maturity = self._detect_business_maturity(
            benchmark_scores
        )

        percentile = self._estimate_percentile(
            benchmark_scores
        )

        strengths = self._detect_strengths(
            benchmark_scores
        )

        weaknesses = self._detect_weaknesses(
            benchmark_scores
        )

        recommendations = self._generate_recommendations(
            weaknesses
        )

        executive_summary = self._generate_summary(
            percentile,
            maturity,
            strengths,
            weaknesses
        )

        result = {
            "industry": industry,
            "company_metrics": business_metrics,
            "industry_benchmark": benchmark,
            "benchmark_scores": benchmark_scores,
            "overall_score": round(
                np.mean(list(benchmark_scores.values())),
                2
            ),
            "industry_percentile": percentile,
            "business_maturity": maturity,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "executive_summary": executive_summary,
        }

        logger.info("Benchmark analysis completed")

        return result

    # ─────────────────────────────────────────

    def _extract_business_metrics(
        self,
        df: pd.DataFrame
    ) -> Dict:

        metrics = {}

        # Churn Rate
        if "churned" in df.columns:
            metrics["avg_churn_rate"] = float(
                df["churned"].mean()
            )

        # NPS
        if "nps_score" in df.columns:
            metrics["avg_nps"] = float(
                df["nps_score"].mean()
            )

        # Feature Usage
        if "feature_usage_score" in df.columns:
            metrics["avg_feature_usage"] = float(
                df["feature_usage_score"].mean()
            )

        # Ticket Rate
        if "support_tickets" in df.columns:
            metrics["avg_ticket_rate"] = float(
                df["support_tickets"].mean()
            )

        # Login Frequency
        login_col = None

        for c in [
            "login_frequency",
            "login_count_30d",
            "sessions_per_month"
        ]:
            if c in df.columns:
                login_col = c
                break

        if login_col:
            metrics["avg_login_frequency"] = float(
                df[login_col].mean()
            )

        # Retention
        if "churned" in df.columns:
            metrics["avg_retention"] = float(
                1 - df["churned"].mean()
            )

        # Payment Delays
        if "payment_delays" in df.columns:
            metrics["avg_payment_delay"] = float(
                df["payment_delays"].mean()
            )

        # Seat Utilization
        if (
            "active_seats" in df.columns and
            "total_seats" in df.columns
        ):
            util = (
                df["active_seats"] /
                df["total_seats"].replace(0, 1)
            )

            metrics["avg_seat_utilization"] = float(
                util.mean()
            )

        return metrics

    # ─────────────────────────────────────────

    def _compare_against_benchmark(
        self,
        business: Dict,
        benchmark: Dict
    ) -> Dict:

        scores = {}

        reverse_metrics = [
            "avg_churn_rate",
            "avg_ticket_rate",
            "avg_payment_delay",
        ]

        for metric, benchmark_value in benchmark.items():

            business_value = business.get(metric)

            if business_value is None:
                continue

            # Reverse metrics
            if metric in reverse_metrics:

                ratio = benchmark_value / max(
                    business_value,
                    0.0001
                )

            else:

                ratio = business_value / max(
                    benchmark_value,
                    0.0001
                )

            score = min(max(ratio * 100, 0), 150)

            scores[metric] = round(score, 2)

        return scores

    # ─────────────────────────────────────────

    def _detect_business_maturity(
        self,
        scores: Dict
    ) -> str:

        overall = np.mean(list(scores.values()))

        if overall >= 120:
            return "Market Leader"

        elif overall >= 100:
            return "Enterprise Mature"

        elif overall >= 85:
            return "Growth Stage"

        elif overall >= 70:
            return "Developing"

        return "High Risk"

    # ─────────────────────────────────────────

    def _estimate_percentile(
        self,
        scores: Dict
    ) -> int:

        overall = np.mean(list(scores.values()))

        percentile = int(
            min(max(overall / 1.2, 1), 99)
        )

        return percentile

    # ─────────────────────────────────────────

    def _detect_strengths(
        self,
        scores: Dict
    ) -> List[str]:

        strengths = []

        for metric, score in scores.items():

            if score >= 110:

                strengths.append(
                    self._humanize_metric(metric)
                )

        return strengths

    # ─────────────────────────────────────────

    def _detect_weaknesses(
        self,
        scores: Dict
    ) -> List[str]:

        weaknesses = []

        for metric, score in scores.items():

            if score < 90:

                weaknesses.append(
                    self._humanize_metric(metric)
                )

        return weaknesses

    # ─────────────────────────────────────────

    def _generate_recommendations(
        self,
        weaknesses: List[str]
    ) -> List[str]:

        recommendations = []

        mapping = {
            "Churn Rate":
                "Launch proactive retention campaigns for at-risk users.",

            "Feature Usage":
                "Improve onboarding and product adoption workflows.",

            "Support Tickets":
                "Reduce friction points causing customer complaints.",

            "Login Frequency":
                "Increase engagement through nudges and notifications.",

            "Payment Delay":
                "Improve billing reminders and payment recovery flows.",

            "Seat Utilization":
                "Increase account expansion and internal adoption.",

            "Retention":
                "Focus on customer success and loyalty programs.",

            "Nps":
                "Improve customer satisfaction touchpoints.",
        }

        for weakness in weaknesses:

            rec = mapping.get(
                weakness,
                f"Improve {weakness.lower()} performance."
            )

            recommendations.append(rec)

        return recommendations

    # ─────────────────────────────────────────

    def _generate_summary(
        self,
        percentile: int,
        maturity: str,
        strengths: List[str],
        weaknesses: List[str]
    ) -> str:

        strengths_text = (
            ", ".join(strengths[:3])
            if strengths else
            "limited competitive strengths"
        )

        weaknesses_text = (
            ", ".join(weaknesses[:3])
            if weaknesses else
            "minimal operational weaknesses"
        )

        return (
            f"The business ranks in approximately the "
            f"top {percentile}% of comparable companies. "
            f"It currently operates at '{maturity}' maturity level. "
            f"Key strengths include {strengths_text}. "
            f"Primary improvement areas include {weaknesses_text}."
        )

    # ─────────────────────────────────────────

    def _humanize_metric(
        self,
        metric: str
    ) -> str:

        metric = metric.replace("avg_", "")
        metric = metric.replace("_", " ")

        return metric.title()


# ─────────────────────────────────────────────
# PUBLIC FUNCTION
# ─────────────────────────────────────────────

def benchmark_business(
    df: pd.DataFrame,
    industry: str = "default"
) -> Dict:

    analyzer = BenchmarkAnalyzer()

    return analyzer.analyze(
        df=df,
        industry=industry
    )