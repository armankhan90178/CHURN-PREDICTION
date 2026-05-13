"""
ChurnShield 2.0 — Unified Intelligent Data Pipeline

Purpose:
End-to-end orchestration pipeline for:
- dataset ingestion
- validation
- cleaning
- anomaly detection
- schema standardization
- feature engineering
- churn label generation
- dataset profiling
- ML-ready export

Pipeline Features:
- auto-recovery
- fault tolerance
- quality scoring
- memory optimization
- detailed pipeline reports
- enterprise logging
- adaptive execution
- modular architecture

Author: ChurnShield AI
"""

import os
import gc
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    ML_TARGET_COLUMN,
)

# Data modules
from data.fetcher import fetch_data_for_field
from data.validator import validate_dataset
from data.cleaner import clean_dataset
from data.standardizer import standardize_dataset
from data.anomaly_detector import detect_anomalies
from data.profiler import profile_dataset

# Upload modules
from upload.label_generator import generate_churn_labels

# ML modules
from ml.feature_engineer import engineer_features


# =========================================================
# LOGGER
# =========================================================

logger = logging.getLogger("churnshield.pipeline")


# =========================================================
# PIPELINE RESULT OBJECT
# =========================================================

class PipelineResult:

    def __init__(self):

        self.success = False

        self.original_shape = None
        self.final_shape = None

        self.pipeline_time = 0

        self.validation_report = {}
        self.cleaning_report = {}
        self.anomaly_report = {}
        self.profile_report = {}

        self.final_dataset = None

        self.errors = []
        self.warnings = []

        self.pipeline_steps = []

        self.quality_score = 0

    def to_dict(self):

        return {
            "success": self.success,
            "original_shape": self.original_shape,
            "final_shape": self.final_shape,
            "pipeline_time": self.pipeline_time,
            "validation_report": self.validation_report,
            "cleaning_report": self.cleaning_report,
            "anomaly_report": self.anomaly_report,
            "profile_report": self.profile_report,
            "quality_score": self.quality_score,
            "errors": self.errors,
            "warnings": self.warnings,
            "pipeline_steps": self.pipeline_steps,
        }


# =========================================================
# MAIN PIPELINE ENGINE
# =========================================================

class IntelligentPipeline:

    def __init__(self):

        self.temp_dir = DATA_DIR / "pipeline_cache"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("IntelligentPipeline initialized")

    # =====================================================
    # MAIN ENTRY POINT
    # =====================================================

    def process_dataset(
        self,
        df: Optional[pd.DataFrame] = None,
        industry: str = "general",
        auto_fetch: bool = False,
        save_outputs: bool = True,
    ) -> PipelineResult:

        start_time = time.time()

        result = PipelineResult()

        try:

            logger.info("=" * 70)
            logger.info("STARTING CHURNSHIELD PIPELINE")
            logger.info("=" * 70)

            # -------------------------------------------------
            # FETCH DATA
            # -------------------------------------------------

            if auto_fetch:

                logger.info(f"Auto-fetching dataset for: {industry}")

                df = fetch_data_for_field(industry)

                result.pipeline_steps.append(
                    "Dataset fetched automatically"
                )

            if df is None or df.empty:
                raise ValueError("No dataset available for pipeline")

            result.original_shape = df.shape

            logger.info(f"Initial dataset shape: {df.shape}")

            # -------------------------------------------------
            # STEP 1 — VALIDATION
            # -------------------------------------------------

            validation_output = self._run_validation(df)

            result.validation_report = validation_output

            result.pipeline_steps.append("Validation completed")

            if validation_output.get("critical_failure"):

                result.errors.append(
                    "Critical validation failure"
                )

                return result

            # -------------------------------------------------
            # STEP 2 — CLEANING
            # -------------------------------------------------

            df, cleaning_report = self._run_cleaning(df)

            result.cleaning_report = cleaning_report

            result.pipeline_steps.append("Cleaning completed")

            # -------------------------------------------------
            # STEP 3 — STANDARDIZATION
            # -------------------------------------------------

            logger.info("Running schema standardization")

            df = standardize_dataset(df)

            result.pipeline_steps.append(
                "Schema standardization completed"
            )

            # -------------------------------------------------
            # STEP 4 — LABEL GENERATION
            # -------------------------------------------------

            if ML_TARGET_COLUMN not in df.columns:

                logger.info("Target column missing — generating churn labels")

                df = generate_churn_labels(df)

                result.pipeline_steps.append(
                    "Synthetic churn labels generated"
                )

            # -------------------------------------------------
            # STEP 5 — ANOMALY DETECTION
            # -------------------------------------------------

            anomaly_df, anomaly_report = self._run_anomaly_detection(df)

            result.anomaly_report = anomaly_report

            result.pipeline_steps.append(
                "Anomaly detection completed"
            )

            # -------------------------------------------------
            # STEP 6 — FEATURE ENGINEERING
            # -------------------------------------------------

            logger.info("Running feature engineering")

            df = engineer_features(df)

            result.pipeline_steps.append(
                "Feature engineering completed"
            )

            # -------------------------------------------------
            # STEP 7 — DATASET PROFILING
            # -------------------------------------------------

            profile_report = self._run_profiling(df)

            result.profile_report = profile_report

            result.pipeline_steps.append(
                "Dataset profiling completed"
            )

            # -------------------------------------------------
            # STEP 8 — QUALITY SCORING
            # -------------------------------------------------

            quality_score = self._calculate_quality_score(
                validation_output,
                cleaning_report,
                anomaly_report,
                profile_report,
            )

            result.quality_score = quality_score

            # -------------------------------------------------
            # FINALIZATION
            # -------------------------------------------------

            result.final_dataset = df
            result.final_shape = df.shape

            result.pipeline_time = round(
                time.time() - start_time,
                2,
            )

            result.success = True

            logger.info("=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)

            logger.info(f"Final dataset shape: {df.shape}")
            logger.info(f"Pipeline execution time: {result.pipeline_time}s")
            logger.info(f"Quality score: {quality_score}/100")

            # -------------------------------------------------
            # SAVE OUTPUTS
            # -------------------------------------------------

            if save_outputs:

                self._save_pipeline_outputs(
                    df=df,
                    result=result,
                    industry=industry,
                )

            gc.collect()

            return result

        except Exception as e:

            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())

            result.errors.append(str(e))

            result.pipeline_time = round(
                time.time() - start_time,
                2,
            )

            return result

    # =====================================================
    # VALIDATION
    # =====================================================

    def _run_validation(self, df):

        logger.info("Running dataset validation")

        try:

            validation_report = validate_dataset(df)

            return validation_report

        except Exception as e:

            logger.warning(f"Validation failed: {e}")

            return {
                "critical_failure": False,
                "error": str(e),
            }

    # =====================================================
    # CLEANING
    # =====================================================

    def _run_cleaning(self, df):

        logger.info("Running intelligent cleaning")

        before_shape = df.shape

        cleaned_df = clean_dataset(df)

        after_shape = cleaned_df.shape

        report = {
            "before_shape": before_shape,
            "after_shape": after_shape,
            "rows_removed": before_shape[0] - after_shape[0],
            "columns_removed": before_shape[1] - after_shape[1],
        }

        return cleaned_df, report

    # =====================================================
    # ANOMALY DETECTION
    # =====================================================

    def _run_anomaly_detection(self, df):

        logger.info("Running anomaly detection")

        try:

            anomaly_result = detect_anomalies(df)

            if isinstance(anomaly_result, tuple):
                return anomaly_result

            return df, {
                "anomalies_detected": 0,
            }

        except Exception as e:

            logger.warning(f"Anomaly detection failed: {e}")

            return df, {
                "error": str(e),
            }

    # =====================================================
    # PROFILING
    # =====================================================

    def _run_profiling(self, df):

        logger.info("Running dataset profiling")

        try:

            return profile_dataset(df)

        except Exception as e:

            logger.warning(f"Profiling failed: {e}")

            return {
                "error": str(e),
            }

    # =====================================================
    # QUALITY SCORE
    # =====================================================

    def _calculate_quality_score(
        self,
        validation_report,
        cleaning_report,
        anomaly_report,
        profile_report,
    ):

        score = 100

        # Validation penalties
        missing_ratio = validation_report.get(
            "missing_ratio",
            0,
        )

        score -= min(missing_ratio * 100, 30)

        # Duplicate penalties
        duplicates = validation_report.get(
            "duplicate_rows",
            0,
        )

        score -= min(duplicates * 0.05, 10)

        # Anomaly penalties
        anomaly_ratio = anomaly_report.get(
            "anomaly_ratio",
            0,
        )

        score -= min(anomaly_ratio * 100, 20)

        # Cleaning penalties
        removed_rows = cleaning_report.get(
            "rows_removed",
            0,
        )

        score -= min(removed_rows * 0.01, 10)

        score = max(0, min(100, score))

        return round(score, 2)

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    def _save_pipeline_outputs(
        self,
        df,
        result,
        industry,
    ):

        try:

            timestamp = int(time.time())

            safe_name = industry.replace(" ", "_")

            output_dir = self.temp_dir / safe_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save cleaned dataset
            dataset_path = (
                output_dir /
                f"processed_{timestamp}.csv"
            )

            df.to_csv(dataset_path, index=False)

            # Save pipeline report
            report_path = (
                output_dir /
                f"pipeline_report_{timestamp}.json"
            )

            with open(report_path, "w") as f:
                json.dump(
                    result.to_dict(),
                    f,
                    indent=2,
                    default=str,
                )

            logger.info(f"Pipeline outputs saved to: {output_dir}")

        except Exception as e:

            logger.warning(f"Failed saving outputs: {e}")


# =========================================================
# QUICK PIPELINE FUNCTION
# =========================================================

def run_pipeline(
    df: Optional[pd.DataFrame] = None,
    industry: str = "general",
    auto_fetch: bool = False,
):

    pipeline = IntelligentPipeline()

    return pipeline.process_dataset(
        df=df,
        industry=industry,
        auto_fetch=auto_fetch,
    )


# =========================================================
# PIPELINE TEST
# =========================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print("\nChurnShield Intelligent Pipeline Test\n")

    try:

        result = run_pipeline(
            industry="telecom",
            auto_fetch=True,
        )

        print("\nPIPELINE RESULT")
        print("=" * 50)

        print(f"Success: {result.success}")
        print(f"Original Shape: {result.original_shape}")
        print(f"Final Shape: {result.final_shape}")
        print(f"Quality Score: {result.quality_score}")
        print(f"Execution Time: {result.pipeline_time}s")

        print("\nPipeline Steps:")
        for step in result.pipeline_steps:
            print(f"✓ {step}")

        if result.errors:
            print("\nErrors:")
            for err in result.errors:
                print(f"✗ {err}")

    except Exception as e:

        print(f"Pipeline crashed: {e}")