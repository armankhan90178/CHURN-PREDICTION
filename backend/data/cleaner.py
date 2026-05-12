"""
ChurnShield 2.0 — Intelligent Dataset Cleaner

Enterprise-grade preprocessing engine.
Automatically fixes:
- missing values
- duplicates
- invalid datatypes
- outliers
- text corruption
- date inconsistencies
- categorical normalization
- memory inefficiencies

Goal:
Convert messy real-world business data
into ML-ready structured datasets.
"""

import logging
import warnings
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

logger = logging.getLogger("churnshield.cleaner")


class DatasetCleaner:
    """
    Intelligent enterprise-grade dataset cleaner.
    """

    def __init__(self):
        self.label_encoders = {}

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Main cleaning pipeline.
        Returns cleaned dataframe + metadata.
        """

        logger.info("Starting dataset cleaning pipeline")

        metadata = {
            "rows_before": len(df),
            "columns_before": len(df.columns),
            "operations": [],
        }

        cleaned = df.copy()

        # ─────────────────────────────────────
        # BASIC NORMALIZATION
        # ─────────────────────────────────────

        cleaned.columns = [
            c.strip().lower().replace(" ", "_")
            for c in cleaned.columns
        ]

        metadata["operations"].append("normalized_column_names")

        # ─────────────────────────────────────
        # REMOVE DUPLICATES
        # ─────────────────────────────────────

        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        removed = before - len(cleaned)

        metadata["duplicates_removed"] = int(removed)

        # ─────────────────────────────────────
        # DROP EMPTY COLUMNS
        # ─────────────────────────────────────

        empty_cols = [
            col for col in cleaned.columns
            if cleaned[col].isnull().mean() > 0.95
        ]

        cleaned.drop(columns=empty_cols, inplace=True, errors="ignore")

        metadata["empty_columns_removed"] = empty_cols

        # ─────────────────────────────────────
        # FIX DATATYPES
        # ─────────────────────────────────────

        cleaned = self._fix_data_types(cleaned)
        metadata["operations"].append("datatype_correction")

        # ─────────────────────────────────────
        # HANDLE MISSING VALUES
        # ─────────────────────────────────────

        cleaned = self._handle_missing_values(cleaned)
        metadata["operations"].append("missing_value_imputation")

        # ─────────────────────────────────────
        # HANDLE OUTLIERS
        # ─────────────────────────────────────

        cleaned = self._handle_outliers(cleaned)
        metadata["operations"].append("outlier_treatment")

        # ─────────────────────────────────────
        # CLEAN TEXT
        # ─────────────────────────────────────

        cleaned = self._clean_text_columns(cleaned)
        metadata["operations"].append("text_cleaning")

        # ─────────────────────────────────────
        # DATE PROCESSING
        # ─────────────────────────────────────

        cleaned = self._process_dates(cleaned)
        metadata["operations"].append("date_processing")

        # ─────────────────────────────────────
        # MEMORY OPTIMIZATION
        # ─────────────────────────────────────

        cleaned = self._optimize_memory(cleaned)
        metadata["operations"].append("memory_optimization")

        metadata["rows_after"] = len(cleaned)
        metadata["columns_after"] = len(cleaned.columns)

        logger.info(
            f"Cleaning completed | Shape: {cleaned.shape}"
        )

        return cleaned, metadata

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in df.columns:

            try:
                if df[col].dtype == "object":

                    # ── Numeric conversion ──────────────────────────────────
                    numeric_candidate = pd.to_numeric(
                        df[col],
                        errors="coerce"
                    )

                    success_ratio = numeric_candidate.notnull().mean()

                    if success_ratio > 0.80:
                        df[col] = numeric_candidate
                        continue  # numeric conversion succeeded — skip date check

                    # ── Date conversion (object columns only) ───────────────
                    # FIX: guard with dtype == "object" so we never call
                    # pd.to_datetime on already-numeric columns (which would
                    # silently mis-interpret integers as nanosecond timestamps).
                    if "date" in col or "time" in col:
                        parsed = pd.to_datetime(df[col], errors="coerce")

                        if parsed.notnull().mean() > 0.60:
                            df[col] = parsed

            except Exception:
                continue

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:

        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include="object").columns

        # Numeric columns
        for col in numeric_cols:
            missing_ratio = df[col].isnull().mean()

            if missing_ratio == 0:
                continue

            if missing_ratio < 0.15:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mean())

        # Categorical columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode = df[col].mode()

                if len(mode) > 0:
                    df[col] = df[col].fillna(mode[0])
                else:
                    df[col] = df[col].fillna("unknown")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:

        # FIX: collect all known target column names so none of them get
        # clipped (the original code only skipped the literal string "churned").
        target_columns = {
            "churned", "churn", "is_churned", "cancelled", "target", "label",
        }

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:

            if col in target_columns:
                continue

            try:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                df[col] = np.clip(df[col], lower, upper)

            except Exception:
                continue

        return df

    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        object_cols = df.select_dtypes(include="object").columns

        for col in object_cols:

            try:
                # FIX: fill genuine NaN with empty string BEFORE astype(str)
                # so they don't become the literal string "nan" in the data.
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                )

                # Normalize common categorical values
                df[col] = df[col].replace({
                    "yes": "Yes",
                    "YES": "Yes",
                    "no": "No",
                    "NO": "No",
                })

            except Exception:
                continue

        return df

    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:

        date_cols = [
            c for c in df.columns
            if "date" in c or "time" in c
        ]

        for col in date_cols:

            try:
                parsed = pd.to_datetime(df[col], errors="coerce")

                if parsed.notnull().mean() > 0.60:
                    df[col] = parsed

                    df[f"{col}_year"] = parsed.dt.year
                    df[f"{col}_month"] = parsed.dt.month
                    df[f"{col}_day"] = parsed.dt.day
                    df[f"{col}_weekday"] = parsed.dt.weekday

            except Exception:
                continue

        return df

    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        return df


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────


def clean_dataset(df: pd.DataFrame):
    cleaner = DatasetCleaner()
    return cleaner.clean(df)