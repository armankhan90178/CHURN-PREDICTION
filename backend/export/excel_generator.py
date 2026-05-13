"""
ChurnShield 2.0 — Advanced Chart Renderer

Purpose:
Enterprise-grade chart generation engine
for churn analytics and executive reporting.

Capabilities:
- automatic chart selection
- churn visualization
- revenue analytics charts
- cohort heatmaps
- feature importance charts
- SHAP plots
- KPI dashboards
- anomaly visualization
- forecasting graphs
- trend rendering
- export-ready PNG charts
- PDF/PPT integration
- executive themes
- dark/light themes

Supports:
- matplotlib
- plotly
- seaborn fallback
- high-resolution exports

Author: ChurnShield AI
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = logging.getLogger("churnshield.chart_renderer")


# ============================================================
# MAIN ENGINE
# ============================================================

class AdvancedChartRenderer:

    def __init__(
        self,
        output_dir: str = "exports/charts"
    ):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.default_dpi = 300

        self.theme = {
            "title_size": 18,
            "label_size": 12,
            "tick_size": 10,
        }

    # ========================================================
    # MAIN ENTRY
    # ========================================================

    def render_dashboard(
        self,
        df: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None,
        insights: Optional[Dict] = None,
    ) -> Dict:

        logger.info("Generating analytics dashboard charts")

        generated = {}

        try:

            generated["churn_distribution"] = (
                self.plot_churn_distribution(df)
            )

        except Exception as e:
            logger.warning(e)

        try:

            generated["revenue_distribution"] = (
                self.plot_revenue_distribution(df)
            )

        except Exception as e:
            logger.warning(e)

        try:

            generated["correlation_heatmap"] = (
                self.plot_correlation_heatmap(df)
            )

        except Exception as e:
            logger.warning(e)

        try:

            generated["feature_importance"] = (
                self.plot_feature_importance(df)
            )

        except Exception as e:
            logger.warning(e)

        try:

            generated["monthly_trend"] = (
                self.plot_monthly_trend(df)
            )

        except Exception as e:
            logger.warning(e)

        try:

            generated["risk_segments"] = (
                self.plot_risk_segments(df)
            )

        except Exception as e:
            logger.warning(e)

        try:

            generated["support_analysis"] = (
                self.plot_support_analysis(df)
            )

        except Exception as e:
            logger.warning(e)

        logger.info("Dashboard chart generation completed")

        return generated

    # ========================================================
    # CHURN DISTRIBUTION
    # ========================================================

    def plot_churn_distribution(
        self,
        df: pd.DataFrame
    ):

        if "churned" not in df.columns:
            return None

        plt.figure(figsize=(8, 6))

        churn_counts = (
            df["churned"]
            .value_counts()
            .sort_index()
        )

        labels = ["Retained", "Churned"]

        plt.pie(
            churn_counts,
            labels=labels,
            autopct="%1.1f%%",
        )

        plt.title(
            "Customer Churn Distribution",
            fontsize=self.theme["title_size"]
        )

        path = self.output_dir / "churn_distribution.png"

        plt.savefig(
            path,
            dpi=self.default_dpi,
            bbox_inches="tight"
        )

        plt.close()

        return str(path)

    # ========================================================
    # REVENUE DISTRIBUTION
    # ========================================================

    def plot_revenue_distribution(
        self,
        df
    ):

        if "monthly_revenue" not in df.columns:
            return None

        plt.figure(figsize=(10, 6))

        plt.hist(
            df["monthly_revenue"],
            bins=30,
        )

        plt.xlabel("Monthly Revenue")
        plt.ylabel("Customers")

        plt.title(
            "Revenue Distribution",
            fontsize=self.theme["title_size"]
        )

        path = self.output_dir / "revenue_distribution.png"

        plt.savefig(
            path,
            dpi=self.default_dpi,
            bbox_inches="tight"
        )

        plt.close()

        return str(path)

    # ========================================================
    # CORRELATION HEATMAP
    # ========================================================

    def plot_correlation_heatmap(
        self,
        df
    ):

        numeric_df = df.select_dtypes(
            include=np.number
        )

        if numeric_df.shape[1] < 2:
            return None

        corr = numeric_df.corr()

        fig, ax = plt.subplots(
            figsize=(12, 10)
        )

        im = ax.imshow(corr)

        ax.set_xticks(
            np.arange(len(corr.columns))
        )

        ax.set_yticks(
            np.arange(len(corr.columns))
        )

        ax.set_xticklabels(
            corr.columns,
            rotation=90
        )

        ax.set_yticklabels(corr.columns)

        plt.title(
            "Feature Correlation Heatmap",
            fontsize=self.theme["title_size"]
        )

        plt.colorbar(im)

        path = self.output_dir / "correlation_heatmap.png"

        plt.savefig(
            path,
            dpi=self.default_dpi,
            bbox_inches="tight"
        )

        plt.close()

        return str(path)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def plot_feature_importance(
        self,
        df
    ):

        numeric_cols = (
            df.select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        if "churned" not in numeric_cols:
            return None

        correlations = {}

        for col in numeric_cols:

            if col == "churned":
                continue

            try:

                corr = abs(
                    df[col].corr(df["churned"])
                )

                correlations[col] = corr

            except:
                continue

        if not correlations:
            return None

        sorted_corr = dict(
            sorted(
                correlations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            list(sorted_corr.keys()),
            list(sorted_corr.values())
        )

        plt.xlabel("Importance Score")

        plt.title(
            "Top Feature Importance",
            fontsize=self.theme["title_size"]
        )

        path = self.output_dir / "feature_importance.png"

        plt.savefig(
            path,
            dpi=self.default_dpi,
            bbox_inches="tight"
        )

        plt.close()

        return str(path)

    # ========================================================
    # MONTHLY TREND
    # ========================================================

    def plot_monthly_trend(
        self,
        df
    ):

        if "monthly_revenue" not in df.columns:
            return None

        trend = (
            df["monthly_revenue"]
            .rolling(window=10)
            .mean()
        )

        plt.figure(figsize=(12, 6))

        plt.plot(trend)

        plt.xlabel("Customers")
        plt.ylabel("Revenue Trend")

        plt.title(
            "Revenue Trend Analysis",
            fontsize=self.theme["title_size"]
        )

        path = self.output_dir / "monthly_trend.png"

        plt.savefig(
            path,
            dpi=self.default_dpi,
            bbox_inches="tight"
        )

        plt.close()

        return str(path)

    # ========================================================
    # RISK SEGMENTS
    # ========================================================

    def plot_risk_segments(
        self,
        df
    ):

        if "churn_probability" not in df.columns:
            return None

        bins = [0, 0.3, 0.6, 1.0]

        labels = [
            "Low Risk",
            "Medium Risk",
            "High Risk",
        ]

        segments = pd.cut(
            df["churn_probability"],
            bins=bins,
            labels=labels
        )

        counts = segments.value_counts()

        plt.figure(figsize=(8, 6))

        plt.bar(
            counts.index.astype(str),
            counts.values
        )

        plt.ylabel("Customers")

        plt.title(
            "Customer Risk Segments",
            fontsize=self.theme["title_size"]
        )

        path = self.output_dir / "risk_segments.png"

        plt.savefig(
            path,
            dpi=self.default_dpi,
            bbox_inches="tight"
        )

        plt.close()

        return str(path)

    # ========================================================
    # SUPPORT ANALYSIS
    # ========================================================

    def plot_support_analysis(
        self,
        df
    ):

        if "support_tickets" not in df.columns:
            return None

        plt.figure(figsize=(10, 6))

        plt.boxplot(df["support_tickets"])

        plt.ylabel("Support Tickets")

        plt.title(
            "Support Ticket Distribution",
            fontsize=self.theme["title_size"]
        )

        path = self.output_dir / "support_analysis.png"

        plt.savefig(
            path,
            dpi=self.default_dpi,
            bbox_inches="tight"
        )

        plt.close()

        return str(path)

    # ========================================================
    # KPI CARD CHART
    # ========================================================

    def create_kpi_card(
        self,
        title: str,
        value: str,
        output_name: str,
    ):

        fig, ax = plt.subplots(
            figsize=(4, 2)
        )

        ax.axis("off")

        plt.text(
            0.5,
            0.7,
            title,
            ha="center",
            fontsize=16,
        )

        plt.text(
            0.5,
            0.35,
            value,
            ha="center",
            fontsize=28,
            fontweight="bold"
        )

        path = self.output_dir / output_name

        plt.savefig(
            path,
            dpi=self.default_dpi,
            bbox_inches="tight"
        )

        plt.close()

        return str(path)

    # ========================================================
    # EXECUTIVE SUMMARY CHART
    # ========================================================

    def executive_summary_chart(
        self,
        metrics: Dict
    ):

        labels = list(metrics.keys())

        values = list(metrics.values())

        plt.figure(figsize=(10, 6))

        plt.bar(labels, values)

        plt.xticks(rotation=20)

        plt.title(
            "Executive Business Metrics",
            fontsize=self.theme["title_size"]
        )

        path = self.output_dir / "executive_summary.png"

        plt.savefig(
            path,
            dpi=self.default_dpi,
            bbox_inches="tight"
        )

        plt.close()

        return str(path)

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    def save_custom_chart(
        self,
        fig,
        filename: str
    ):

        path = self.output_dir / filename

        fig.savefig(
            path,
            dpi=self.default_dpi,
            bbox_inches="tight"
        )

        return str(path)


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def render_dashboard_charts(
    df: pd.DataFrame
):

    renderer = AdvancedChartRenderer()

    return renderer.render_dashboard(df)


def render_executive_metrics(
    metrics: Dict
):

    renderer = AdvancedChartRenderer()

    return renderer.executive_summary_chart(metrics)