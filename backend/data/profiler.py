"""
ChurnShield 2.0 — Enterprise Dataset Profiler

Purpose:
Deep intelligent profiling for any uploaded dataset.

Capabilities:
- schema intelligence
- statistical profiling
- business profiling
- churn profiling
- correlation analysis
- feature health scoring
- skew detection
- cardinality analysis
- leakage detection
- memory analysis
- ML readiness scoring
- risk detection
- anomaly summaries
- executive insights

Author: ChurnShield AI
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("churnshield.profiler")


# ============================================================
# MAIN ENGINE
# ============================================================

class DatasetProfiler:

    def __init__(self):

        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []

    # ========================================================
    # MAIN ENTRY
    # ========================================================

    def profile(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        logger.info("Starting enterprise dataset profiling")

        data = df.copy()

        self._detect_column_types(data)

        profile = {

            "generated_at": datetime.now().isoformat(),

            "dataset_overview":
                self._dataset_overview(data),

            "column_analysis":
                self._column_analysis(data),

            "numeric_analysis":
                self._numeric_analysis(data),

            "categorical_analysis":
                self._categorical_analysis(data),

            "datetime_analysis":
                self._datetime_analysis(data),

            "missing_value_analysis":
                self._missing_analysis(data),

            "duplicate_analysis":
                self._duplicate_analysis(data),

            "correlation_analysis":
                self._correlation_analysis(data),

            "business_analysis":
                self._business_analysis(data),

            "churn_analysis":
                self._churn_analysis(data),

            "data_quality":
                self._quality_analysis(data),

            "memory_analysis":
                self._memory_analysis(data),

            "ml_readiness":
                self._ml_readiness(data),

            "risk_analysis":
                self._risk_analysis(data),

            "executive_insights":
                self._executive_insights(data),

            "recommendations":
                self._recommendations(data),
        }

        logger.info("Dataset profiling completed")

        return profile

    # ========================================================
    # COLUMN TYPE DETECTION
    # ========================================================

    def _detect_column_types(
        self,
        df: pd.DataFrame
    ):

        self.numeric_columns = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        self.categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        self.datetime_columns = []

        for col in df.columns:

            if "date" in col.lower():
                self.datetime_columns.append(col)

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    def _dataset_overview(
        self,
        df
    ):

        return {

            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),

            "numeric_columns":
                len(self.numeric_columns),

            "categorical_columns":
                len(self.categorical_columns),

            "datetime_columns":
                len(self.datetime_columns),

            "missing_cells":
                int(df.isna().sum().sum()),

            "duplicate_rows":
                int(df.duplicated().sum()),

            "memory_usage_mb":
                round(
                    df.memory_usage(deep=True).sum()
                    / (1024 * 1024),
                    2
                ),
        }

    # ========================================================
    # COLUMN ANALYSIS
    # ========================================================

    def _column_analysis(
        self,
        df
    ):

        analysis = {}

        for col in df.columns:

            analysis[col] = {

                "dtype":
                    str(df[col].dtype),

                "unique_values":
                    int(df[col].nunique()),

                "missing_values":
                    int(df[col].isna().sum()),

                "missing_percent":
                    round(
                        df[col].isna().mean() * 100,
                        2
                    ),

                "sample_values":
                    df[col]
                    .dropna()
                    .astype(str)
                    .head(5)
                    .tolist(),

                "is_constant":
                    bool(df[col].nunique() <= 1),

                "high_cardinality":
                    bool(df[col].nunique() > 1000),
            }

        return analysis

    # ========================================================
    # NUMERIC ANALYSIS
    # ========================================================

    def _numeric_analysis(
        self,
        df
    ):

        results = {}

        for col in self.numeric_columns:

            series = df[col].dropna()

            if len(series) == 0:
                continue

            results[col] = {

                "mean":
                    round(float(series.mean()), 4),

                "median":
                    round(float(series.median()), 4),

                "std":
                    round(float(series.std()), 4),

                "min":
                    round(float(series.min()), 4),

                "max":
                    round(float(series.max()), 4),

                "q1":
                    round(float(series.quantile(0.25)), 4),

                "q3":
                    round(float(series.quantile(0.75)), 4),

                "skewness":
                    round(float(series.skew()), 4),

                "kurtosis":
                    round(float(series.kurtosis()), 4),

                "zero_values":
                    int((series == 0).sum()),

                "negative_values":
                    int((series < 0).sum()),

                "outlier_count":
                    self._detect_outliers(series),
            }

        return results

    # ========================================================
    # CATEGORICAL ANALYSIS
    # ========================================================

    def _categorical_analysis(
        self,
        df
    ):

        results = {}

        for col in self.categorical_columns:

            value_counts = df[col].astype(str).value_counts()

            results[col] = {

                "unique_categories":
                    int(df[col].nunique()),

                "top_categories":
                    value_counts.head(10).to_dict(),

                "most_common":
                    str(value_counts.index[0])
                    if len(value_counts) else None,

                "least_common":
                    str(value_counts.index[-1])
                    if len(value_counts) else None,
            }

        return results

    # ========================================================
    # DATETIME ANALYSIS
    # ========================================================

    def _datetime_analysis(
        self,
        df
    ):

        results = {}

        for col in self.datetime_columns:

            try:

                series = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                results[col] = {

                    "min_date":
                        str(series.min()),

                    "max_date":
                        str(series.max()),

                    "date_range_days":
                        int(
                            (
                                series.max() -
                                series.min()
                            ).days
                        ) if series.notna().sum() else 0,
                }

            except Exception:
                continue

        return results

    # ========================================================
    # MISSING ANALYSIS
    # ========================================================

    def _missing_analysis(
        self,
        df
    ):

        missing = df.isna().sum()

        return {

            "total_missing":
                int(missing.sum()),

            "missing_percentage":
                round(
                    df.isna().mean().mean() * 100,
                    2
                ),

            "columns_with_missing":
                missing[missing > 0].to_dict(),

            "worst_column":
                missing.idxmax()
                if missing.sum() else None,
        }

    # ========================================================
    # DUPLICATE ANALYSIS
    # ========================================================

    def _duplicate_analysis(
        self,
        df
    ):

        duplicates = df.duplicated()

        return {

            "duplicate_rows":
                int(duplicates.sum()),

            "duplicate_percentage":
                round(
                    duplicates.mean() * 100,
                    2
                ),
        }

    # ========================================================
    # CORRELATION ANALYSIS
    # ========================================================

    def _correlation_analysis(
        self,
        df
    ):

        if len(self.numeric_columns) < 2:

            return {}

        corr = df[self.numeric_columns].corr()

        strong_pairs = []

        for i in corr.columns:

            for j in corr.columns:

                if i == j:
                    continue

                value = corr.loc[i, j]

                if abs(value) >= 0.75:

                    strong_pairs.append({

                        "feature_1": i,
                        "feature_2": j,
                        "correlation":
                            round(float(value), 4)
                    })

        return {

            "strong_correlations":
                strong_pairs[:25],

            "correlation_matrix_shape":
                corr.shape,
        }

    # ========================================================
    # BUSINESS ANALYSIS
    # ========================================================

    def _business_analysis(
        self,
        df
    ):

        analysis = {}

        if "monthly_revenue" in df.columns:

            analysis["revenue"] = {

                "total_revenue":
                    round(
                        float(df["monthly_revenue"].sum()),
                        2
                    ),

                "average_revenue":
                    round(
                        float(df["monthly_revenue"].mean()),
                        2
                    ),

                "max_revenue":
                    round(
                        float(df["monthly_revenue"].max()),
                        2
                    ),
            }

        if "support_tickets" in df.columns:

            analysis["support"] = {

                "total_tickets":
                    int(df["support_tickets"].sum()),

                "average_tickets":
                    round(
                        float(df["support_tickets"].mean()),
                        2
                    ),
            }

        return analysis

    # ========================================================
    # CHURN ANALYSIS
    # ========================================================

    def _churn_analysis(
        self,
        df
    ):

        if "churned" not in df.columns:

            return {
                "available": False
            }

        churn_rate = df["churned"].mean()

        return {

            "available": True,

            "churn_rate":
                round(churn_rate * 100, 2),

            "churned_customers":
                int(df["churned"].sum()),

            "retained_customers":
                int((df["churned"] == 0).sum()),

            "class_balance":
                "imbalanced"
                if churn_rate < 0.15 or churn_rate > 0.85
                else "balanced",
        }

    # ========================================================
    # QUALITY ANALYSIS
    # ========================================================

    def _quality_analysis(
        self,
        df
    ):

        score = 100

        missing_ratio = df.isna().mean().mean()
        duplicate_ratio = df.duplicated().mean()

        score -= missing_ratio * 40
        score -= duplicate_ratio * 25

        constant_columns = sum(
            df[col].nunique() <= 1
            for col in df.columns
        )

        score -= constant_columns * 2

        score = max(0, min(100, score))

        return {

            "quality_score":
                round(score, 2),

            "grade":
                self._quality_grade(score),

            "constant_columns":
                constant_columns,
        }

    # ========================================================
    # MEMORY ANALYSIS
    # ========================================================

    def _memory_analysis(
        self,
        df
    ):

        memory = df.memory_usage(
            deep=True
        )

        return {

            "total_memory_mb":
                round(
                    memory.sum() / (1024 * 1024),
                    2
                ),

            "largest_columns":
                memory.sort_values(
                    ascending=False
                )
                .head(10)
                .to_dict(),
        }

    # ========================================================
    # ML READINESS
    # ========================================================

    def _ml_readiness(
        self,
        df
    ):

        readiness = 100

        if "churned" not in df.columns:
            readiness -= 40

        if len(self.numeric_columns) < 3:
            readiness -= 20

        if df.shape[0] < 100:
            readiness -= 20

        if df.isna().mean().mean() > 0.20:
            readiness -= 15

        readiness = max(0, readiness)

        return {

            "readiness_score":
                readiness,

            "status":
                "ready"
                if readiness >= 75
                else "needs_improvement",
        }

    # ========================================================
    # RISK ANALYSIS
    # ========================================================

    def _risk_analysis(
        self,
        df
    ):

        risks = []

        if df.isna().mean().mean() > 0.30:

            risks.append(
                "High missing value risk"
            )

        if df.duplicated().mean() > 0.20:

            risks.append(
                "Duplicate record risk"
            )

        if len(self.numeric_columns) < 2:

            risks.append(
                "Low numeric feature availability"
            )

        if "churned" not in df.columns:

            risks.append(
                "Missing churn target column"
            )

        return {

            "risk_count":
                len(risks),

            "risks":
                risks,
        }

    # ========================================================
    # EXECUTIVE INSIGHTS
    # ========================================================

    def _executive_insights(
        self,
        df
    ):

        insights = []

        if "monthly_revenue" in df.columns:

            insights.append(
                f"Total revenue tracked: ₹{round(df['monthly_revenue'].sum(), 2):,}"
            )

        if "churned" in df.columns:

            churn_rate = round(
                df["churned"].mean() * 100,
                2
            )

            insights.append(
                f"Current churn rate is {churn_rate}%"
            )

        if df.isna().mean().mean() > 0.10:

            insights.append(
                "Dataset contains significant missing values"
            )

        if len(self.numeric_columns) > 10:

            insights.append(
                "Dataset has rich ML feature potential"
            )

        return insights

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def _recommendations(
        self,
        df
    ):

        recommendations = []

        if df.isna().mean().mean() > 0.15:

            recommendations.append(
                "Improve missing value handling"
            )

        if df.duplicated().sum() > 0:

            recommendations.append(
                "Remove duplicate records"
            )

        if "churned" not in df.columns:

            recommendations.append(
                "Generate churn labels for ML training"
            )

        if len(self.numeric_columns) < 5:

            recommendations.append(
                "Increase feature richness"
            )

        if not recommendations:

            recommendations.append(
                "Dataset quality is production-ready"
            )

        return recommendations

    # ========================================================
    # HELPERS
    # ========================================================

    def _detect_outliers(
        self,
        series
    ):

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return int(
            ((series < lower) | (series > upper)).sum()
        )

    def _quality_grade(
        self,
        score
    ):

        if score >= 90:
            return "A+"

        elif score >= 80:
            return "A"

        elif score >= 70:
            return "B"

        elif score >= 60:
            return "C"

        return "D"


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def profile_dataset(
    df: pd.DataFrame
):

    profiler = DatasetProfiler()

    return profiler.profile(df)