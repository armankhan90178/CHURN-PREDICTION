"""
ChurnShield 2.0 — Advanced Dataset Validator
Enterprise-grade validation engine for churn prediction pipelines.

Responsibilities:
- Detect schema problems
- Validate upload quality
- Detect leakage
- Detect imbalance
- Detect duplicate customers
- Detect corrupted dates
- Detect suspicious distributions
- Calculate quality score
- Generate AI-ready validation report

This file should NEVER crash the pipeline.
It always returns a structured validation result.
"""

import re
import logging
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field

from config import (
    ML_MIN_ROWS_REQUIRED,
    ML_MIN_COLUMNS_REQUIRED,
    MAX_UPLOAD_COLUMNS,
    MAX_UPLOAD_ROWS,
    STANDARD_SCHEMA,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger("churnshield.validator")


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity: str
    category: str
    message: str
    affected_columns: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    is_valid: bool
    quality_score: float
    total_rows: int
    total_columns: int
    issues: List[ValidationIssue]
    warnings: List[str]
    recommendations: List[str]
    detected_target_column: str = ""
    duplicate_ratio: float = 0.0
    missing_ratio: float = 0.0
    class_balance: Dict[str, float] = field(default_factory=dict)
    schema_coverage: float = 0.0


# ─────────────────────────────────────────────
# MAIN VALIDATION ENGINE
# ─────────────────────────────────────────────

class DatasetValidator:
    """
    Master validation engine.
    Performs 20+ enterprise-grade quality checks.
    """

    def __init__(self):
        self.standard_columns = set(STANDARD_SCHEMA.keys())

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Main validation entry point.
        Never crashes.
        Always returns ValidationReport.
        """

        issues = []
        warnings_list = []
        recommendations = []

        try:
            logger.info("Starting dataset validation")

            total_rows = len(df)
            total_columns = len(df.columns)

            # ─────────────────────────────────────
            # BASIC CHECKS
            # ─────────────────────────────────────

            if total_rows < ML_MIN_ROWS_REQUIRED:
                issues.append(
                    ValidationIssue(
                        severity="critical",
                        category="dataset_size",
                        message=f"Dataset has only {total_rows} rows",
                    )
                )

            if total_columns < ML_MIN_COLUMNS_REQUIRED:
                issues.append(
                    ValidationIssue(
                        severity="critical",
                        category="dataset_columns",
                        message=f"Dataset has only {total_columns} columns",
                    )
                )

            if total_rows > MAX_UPLOAD_ROWS:
                warnings_list.append(
                    f"Large dataset detected ({total_rows:,} rows)."
                )

            if total_columns > MAX_UPLOAD_COLUMNS:
                warnings_list.append(
                    f"Very wide dataset detected ({total_columns} columns)."
                )

            # ─────────────────────────────────────
            # DUPLICATE CHECKS
            # ─────────────────────────────────────

            duplicate_rows = int(df.duplicated().sum())
            duplicate_ratio = duplicate_rows / max(len(df), 1)

            if duplicate_ratio > 0.10:
                issues.append(
                    ValidationIssue(
                        severity="high",
                        category="duplicates",
                        message=f"High duplicate ratio: {duplicate_ratio:.1%}",
                    )
                )

            # ─────────────────────────────────────
            # MISSING VALUE ANALYSIS
            # ─────────────────────────────────────

            missing_ratio = float(df.isnull().sum().sum()) / max(df.size, 1)

            if missing_ratio > 0.30:
                issues.append(
                    ValidationIssue(
                        severity="high",
                        category="missing_values",
                        message=f"Missing value ratio too high: {missing_ratio:.1%}",
                    )
                )

            # Column-level missing analysis
            for col in df.columns:
                col_missing = df[col].isnull().mean()

                if col_missing > 0.70:
                    issues.append(
                        ValidationIssue(
                            severity="medium",
                            category="column_missing",
                            message=f"Column '{col}' mostly empty ({col_missing:.1%})",
                            affected_columns=[col],
                        )
                    )

            # ─────────────────────────────────────
            # TARGET DETECTION
            # ─────────────────────────────────────

            target_column = self._detect_target_column(df)

            if not target_column:
                issues.append(
                    ValidationIssue(
                        severity="critical",
                        category="target",
                        message="No churn target column detected",
                    )
                )
            else:
                logger.info(f"Detected target column: {target_column}")

            # ─────────────────────────────────────
            # CLASS BALANCE
            # ─────────────────────────────────────

            class_balance = {}

            if target_column and target_column in df.columns:
                value_counts = df[target_column].value_counts(normalize=True)

                # FIX: convert numpy keys to str so the dict is JSON-serialisable
                class_balance = {str(k): float(v) for k, v in value_counts.items()}

                minority_ratio = value_counts.min()

                if minority_ratio < 0.05:
                    issues.append(
                        ValidationIssue(
                            severity="high",
                            category="imbalance",
                            message="Extreme class imbalance detected",
                            affected_columns=[target_column],
                        )
                    )

            # ─────────────────────────────────────
            # LEAKAGE DETECTION
            # ─────────────────────────────────────

            leakage_columns = self._detect_leakage_columns(df)

            if leakage_columns:
                issues.append(
                    ValidationIssue(
                        severity="critical",
                        category="data_leakage",
                        message="Potential leakage columns detected",
                        affected_columns=leakage_columns,
                    )
                )

            # ─────────────────────────────────────
            # DATE VALIDATION
            # ─────────────────────────────────────

            invalid_date_columns = self._detect_invalid_dates(df)

            if invalid_date_columns:
                issues.append(
                    ValidationIssue(
                        severity="medium",
                        category="invalid_dates",
                        message="Invalid or corrupted date columns detected",
                        affected_columns=invalid_date_columns,
                    )
                )

            # ─────────────────────────────────────
            # NUMERIC VALIDATION
            # ─────────────────────────────────────

            suspicious_numeric = self._detect_suspicious_numeric_columns(df)

            if suspicious_numeric:
                warnings_list.extend(suspicious_numeric)

            # ─────────────────────────────────────
            # ID VALIDATION
            # ─────────────────────────────────────

            duplicate_customer_ids = self._check_duplicate_customer_ids(df)

            if duplicate_customer_ids:
                warnings_list.append(
                    "Duplicate customer IDs detected"
                )

            # ─────────────────────────────────────
            # SCHEMA COVERAGE
            # ─────────────────────────────────────

            schema_matches = len(
                set(c.lower() for c in df.columns).intersection(self.standard_columns)
            )

            schema_coverage = schema_matches / len(self.standard_columns)

            if schema_coverage < 0.30:
                recommendations.append(
                    "Dataset schema differs significantly from ChurnShield standard schema"
                )

            # ─────────────────────────────────────
            # TEXT QUALITY CHECKS
            # ─────────────────────────────────────

            gibberish_columns = self._detect_gibberish_columns(df)

            if gibberish_columns:
                warnings_list.append(
                    f"Possible corrupted text columns: {', '.join(gibberish_columns)}"
                )

            # ─────────────────────────────────────
            # QUALITY SCORE
            # ─────────────────────────────────────

            quality_score = self._calculate_quality_score(
                df=df,
                issues=issues,
                missing_ratio=missing_ratio,
                duplicate_ratio=duplicate_ratio,
                schema_coverage=schema_coverage,
            )

            is_valid = not any(i.severity == "critical" for i in issues)

            logger.info(
                f"Validation complete | Quality Score: {quality_score:.1f}"
            )

            return ValidationReport(
                is_valid=is_valid,
                quality_score=quality_score,
                total_rows=total_rows,
                total_columns=total_columns,
                issues=issues,
                warnings=warnings_list,
                recommendations=recommendations,
                detected_target_column=target_column,
                duplicate_ratio=duplicate_ratio,
                missing_ratio=missing_ratio,
                class_balance=class_balance,
                schema_coverage=schema_coverage,
            )

        except Exception as e:
            logger.exception(f"Validator failed: {e}")

            return ValidationReport(
                is_valid=False,
                quality_score=0.0,
                total_rows=len(df),
                total_columns=len(df.columns),
                issues=[
                    ValidationIssue(
                        severity="critical",
                        category="system",
                        message=str(e),
                    )
                ],
                warnings=[],
                recommendations=[],
            )

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    def _detect_target_column(self, df: pd.DataFrame) -> str:
        target_candidates = [
            "churn",
            "churned",
            "is_churned",
            "cancelled",
            "target",
            "label",
            "status",
        ]

        lower_map = {c.lower(): c for c in df.columns}

        for candidate in target_candidates:
            if candidate in lower_map:
                return lower_map[candidate]

        return ""

    def _detect_leakage_columns(self, df: pd.DataFrame) -> List[str]:
        leakage_keywords = [
            "refund",
            "cancellation_reason",
            "closed_account",
            "termination_date",
            "exit_reason",
        ]

        suspicious = []

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in leakage_keywords):
                suspicious.append(col)

        return suspicious

    def _detect_invalid_dates(self, df: pd.DataFrame) -> List[str]:
        invalid_cols = []

        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    invalid_ratio = parsed.isnull().mean()

                    if invalid_ratio > 0.40:
                        invalid_cols.append(col)

                except Exception:
                    invalid_cols.append(col)

        return invalid_cols

    def _detect_suspicious_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        warnings_list = []

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) == 0:
                continue

            if series.std() == 0:
                warnings_list.append(
                    f"Column '{col}' has zero variance"
                )

            if series.abs().max() > 1e12:
                warnings_list.append(
                    f"Column '{col}' contains extremely large values"
                )

        return warnings_list

    def _check_duplicate_customer_ids(self, df: pd.DataFrame) -> bool:
        candidate_columns = [
            "customer_id",
            "cust_id",
            "user_id",
            "account_id",
        ]

        for col in df.columns:
            if col.lower() in candidate_columns:
                duplicate_ratio = df[col].duplicated().mean()
                return duplicate_ratio > 0.10

        return False

    def _detect_gibberish_columns(self, df: pd.DataFrame) -> List[str]:
        suspicious = []

        object_cols = df.select_dtypes(include="object").columns

        pattern = re.compile(r"^[^a-zA-Z0-9]+$")

        for col in object_cols:
            sample = df[col].dropna().astype(str).head(20)

            if len(sample) == 0:
                continue

            gibberish_count = sum(bool(pattern.match(v)) for v in sample)

            if gibberish_count / len(sample) > 0.50:
                suspicious.append(col)

        return suspicious

    def _calculate_quality_score(
        self,
        df: pd.DataFrame,
        issues: List[ValidationIssue],
        missing_ratio: float,
        duplicate_ratio: float,
        schema_coverage: float,
    ) -> float:

        score = 100.0

        critical = sum(i.severity == "critical" for i in issues)
        high = sum(i.severity == "high" for i in issues)
        medium = sum(i.severity == "medium" for i in issues)

        score -= critical * 25
        score -= high * 10
        score -= medium * 5

        score -= missing_ratio * 40
        score -= duplicate_ratio * 20

        score += schema_coverage * 15

        return round(max(0.0, min(100.0, score)), 2)


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """
    Simple functional interface.
    """
    validator = DatasetValidator()
    return validator.validate(df)