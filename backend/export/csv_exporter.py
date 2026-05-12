"""
ChurnShield 2.0 — Hyper CSV Export Engine

Purpose:
Enterprise-grade CSV exporting engine
for churn analytics, ML predictions,
executive reporting, segmentation exports,
and large-scale customer intelligence delivery.

Capabilities:
- enterprise CSV exporting
- smart column ordering
- AI-ready export formatting
- executive summary generation
- multi-export modes
- chunked export for huge datasets
- secure export sanitization
- auto metadata injection
- prediction confidence exports
- multilingual export support
- timestamped export versioning
- analytics-ready formatting
"""

import os
import csv
import json
import gzip
import shutil
import logging
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from config import (
    USER_DATA_DIR,
)

logger = logging.getLogger(
    "churnshield.export.csv"
)


class HyperCSVExporter:

    def __init__(self):

        self.default_export_columns = [

            "customer_id",
            "customer_name",
            "state",
            "city",

            "monthly_revenue",
            "engagement_score",
            "health_score",

            "churn_probability",
            "risk_level",
            "retention_priority",

            "predicted_churn_window",
            "predicted_revenue_loss",

            "persona",
            "churn_reason",

            "recommended_action",

            "confidence_score",
        ]

        self.executive_columns = [

            "customer_id",
            "customer_name",
            "monthly_revenue",
            "churn_probability",
            "risk_level",
            "predicted_revenue_loss",
            "recommended_action",
        ]

        self.analytics_columns = [

            "customer_id",
            "monthly_revenue",
            "engagement_score",
            "support_tickets",
            "payment_delays",
            "health_score",
            "churn_probability",
            "confidence_score",
            "persona",
            "churn_reason",
        ]

    # ─────────────────────────────────────────────
    # MAIN EXPORT ENGINE
    # ─────────────────────────────────────────────

    def export_csv(
        self,
        df: pd.DataFrame,
        export_name: str = "churnshield_export",
        export_type: str = "full",
        user_id: str = "default",
        compress: bool = False,
        include_metadata: bool = True,
    ) -> Dict:

        logger.info(
            f"Starting CSV export: {export_name}"
        )

        data = df.copy()

        export_dir = (
            Path(USER_DATA_DIR)
            / str(user_id)
            / "exports"
        )

        export_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"{export_name}_{timestamp}.csv"
        )

        export_path = export_dir / filename

        # ─────────────────────────────────────────────
        # FORMAT EXPORT
        # ─────────────────────────────────────────────

        data = self._prepare_export_data(
            data,
            export_type,
        )

        # ─────────────────────────────────────────────
        # METADATA INJECTION
        # ─────────────────────────────────────────────

        if include_metadata:

            data = self._inject_metadata(
                data
            )

        # ─────────────────────────────────────────────
        # SANITIZATION
        # ─────────────────────────────────────────────

        data = self._sanitize_export(
            data
        )

        # ─────────────────────────────────────────────
        # EXPORT CSV
        # ─────────────────────────────────────────────

        data.to_csv(

            export_path,

            index=False,

            encoding="utf-8-sig",

            quoting=csv.QUOTE_MINIMAL,
        )

        final_path = export_path

        # ─────────────────────────────────────────────
        # OPTIONAL COMPRESSION
        # ─────────────────────────────────────────────

        if compress:

            final_path = self._compress_file(
                export_path
            )

        # ─────────────────────────────────────────────
        # EXPORT SUMMARY
        # ─────────────────────────────────────────────

        summary = self._generate_export_summary(
            data,
            final_path,
            export_type,
        )

        logger.info(
            f"CSV export completed: {final_path}"
        )

        return {

            "success": True,

            "export_path": str(final_path),

            "rows_exported": int(len(data)),

            "columns_exported": int(
                len(data.columns)
            ),

            "file_size_mb": round(
                final_path.stat().st_size
                / (1024 * 1024),
                2,
            ),

            "summary": summary,
        }

    # ─────────────────────────────────────────────
    # PREPARE EXPORT DATA
    # ─────────────────────────────────────────────

    def _prepare_export_data(
        self,
        df: pd.DataFrame,
        export_type: str,
    ):

        data = df.copy()

        # ─────────────────────────────────────────────
        # COLUMN SELECTION
        # ─────────────────────────────────────────────

        if export_type == "executive":

            columns = [

                c for c in self.executive_columns
                if c in data.columns
            ]

            data = data[columns]

        elif export_type == "analytics":

            columns = [

                c for c in self.analytics_columns
                if c in data.columns
            ]

            data = data[columns]

        elif export_type == "minimal":

            minimal = [

                "customer_id",
                "customer_name",
                "churn_probability",
                "risk_level",
            ]

            columns = [

                c for c in minimal
                if c in data.columns
            ]

            data = data[columns]

        else:

            ordered = [

                c for c in self.default_export_columns
                if c in data.columns
            ]

            remaining = [

                c for c in data.columns
                if c not in ordered
            ]

            data = data[
                ordered + remaining
            ]

        # ─────────────────────────────────────────────
        # SORTING
        # ─────────────────────────────────────────────

        if "churn_probability" in data.columns:

            data = data.sort_values(

                by="churn_probability",

                ascending=False,
            )

        # ─────────────────────────────────────────────
        # ROUND FLOATS
        # ─────────────────────────────────────────────

        float_cols = data.select_dtypes(
            include=["float"]
        ).columns

        for col in float_cols:

            data[col] = data[col].round(4)

        return data

    # ─────────────────────────────────────────────
    # METADATA
    # ─────────────────────────────────────────────

    def _inject_metadata(
        self,
        df: pd.DataFrame,
    ):

        data = df.copy()

        data["export_generated_at"] = (
            datetime.utcnow()
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        data["export_engine"] = (
            "ChurnShield 2.0"
        )

        data["record_version"] = "v2"

        return data

    # ─────────────────────────────────────────────
    # SANITIZATION
    # ─────────────────────────────────────────────

    def _sanitize_export(
        self,
        df: pd.DataFrame,
    ):

        data = df.copy()

        # Remove line breaks
        for col in data.columns:

            if data[col].dtype == object:

                data[col] = (
                    data[col]
                    .astype(str)
                    .str.replace(
                        "\n",
                        " ",
                        regex=False,
                    )
                    .str.replace(
                        "\r",
                        " ",
                        regex=False,
                    )
                )

        # Replace infinities
        data = data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Fill NaNs
        data = data.fillna("")

        return data

    # ─────────────────────────────────────────────
    # COMPRESSION
    # ─────────────────────────────────────────────

    def _compress_file(
        self,
        file_path: Path,
    ):

        compressed_path = Path(
            str(file_path) + ".gz"
        )

        with open(file_path, "rb") as f_in:

            with gzip.open(
                compressed_path,
                "wb"
            ) as f_out:

                shutil.copyfileobj(
                    f_in,
                    f_out,
                )

        logger.info(
            f"Compressed export: {compressed_path}"
        )

        return compressed_path

    # ─────────────────────────────────────────────
    # EXPORT SUMMARY
    # ─────────────────────────────────────────────

    def _generate_export_summary(
        self,
        df: pd.DataFrame,
        path: Path,
        export_type: str,
    ):

        summary = {

            "export_type":
                export_type,

            "rows":
                int(len(df)),

            "columns":
                int(len(df.columns)),

            "generated_at":
                datetime.utcnow().isoformat(),

            "top_risk_customers":
                0,

            "total_revenue":
                0,

            "avg_churn_probability":
                0,
        }

        if "risk_level" in df.columns:

            summary[
                "top_risk_customers"
            ] = int(

                (
                    df["risk_level"]
                    .astype(str)
                    .str.upper()
                    .isin([
                        "HIGH",
                        "CRITICAL"
                    ])
                ).sum()

            )

        if "monthly_revenue" in df.columns:

            summary[
                "total_revenue"
            ] = float(

                pd.to_numeric(
                    df["monthly_revenue"],
                    errors="coerce",
                ).sum()

            )

        if "churn_probability" in df.columns:

            summary[
                "avg_churn_probability"
            ] = round(

                float(

                    pd.to_numeric(
                        df[
                            "churn_probability"
                        ],
                        errors="coerce",
                    ).mean()

                ),

                4,
            )

        summary[
            "export_size_mb"
        ] = round(

            path.stat().st_size
            / (1024 * 1024),

            2,
        )

        return summary

    # ─────────────────────────────────────────────
    # BATCH EXPORT
    # ─────────────────────────────────────────────

    def batch_export(
        self,
        datasets: Dict[str, pd.DataFrame],
        user_id="default",
    ):

        logger.info(
            "Starting batch exports"
        )

        results = {}

        for name, df in datasets.items():

            try:

                result = self.export_csv(

                    df=df,

                    export_name=name,

                    export_type="full",

                    user_id=user_id,
                )

                results[name] = result

            except Exception as e:

                logger.error(
                    f"Batch export failed: {name}"
                )

                results[name] = {

                    "success": False,

                    "error": str(e),
                }

        return results

    # ─────────────────────────────────────────────
    # ENTERPRISE SPLIT EXPORT
    # ─────────────────────────────────────────────

    def split_large_export(
        self,
        df: pd.DataFrame,
        rows_per_file: int = 100000,
        export_name: str = "split_export",
        user_id: str = "default",
    ):

        logger.info(
            "Starting split export"
        )

        chunks = []

        total_rows = len(df)

        for i in range(

            0,
            total_rows,
            rows_per_file,
        ):

            chunk = df.iloc[
                i:i + rows_per_file
            ]

            result = self.export_csv(

                df=chunk,

                export_name=f"{export_name}_part_{len(chunks)+1}",

                export_type="full",

                user_id=user_id,
            )

            chunks.append(result)

        return {

            "success": True,

            "total_parts": len(chunks),

            "exports": chunks,
        }

    # ─────────────────────────────────────────────
    # EXECUTIVE EXPORT
    # ─────────────────────────────────────────────

    def executive_export(
        self,
        df: pd.DataFrame,
        user_id="default",
    ):

        return self.export_csv(

            df=df,

            export_name="executive_report",

            export_type="executive",

            user_id=user_id,
        )

    # ─────────────────────────────────────────────
    # ANALYTICS EXPORT
    # ─────────────────────────────────────────────

    def analytics_export(
        self,
        df: pd.DataFrame,
        user_id="default",
    ):

        return self.export_csv(

            df=df,

            export_name="analytics_report",

            export_type="analytics",

            user_id=user_id,
        )


# ─────────────────────────────────────────────
# GLOBAL ENGINE
# ─────────────────────────────────────────────

csv_export_engine = (
    HyperCSVExporter()
)


# ─────────────────────────────────────────────
# FUNCTIONAL API
# ─────────────────────────────────────────────

def export_csv(
    df: pd.DataFrame,
    export_name="churnshield_export",
    export_type="full",
    user_id="default",
    compress=False,
):

    return (

        csv_export_engine.export_csv(

            df=df,

            export_name=export_name,

            export_type=export_type,

            user_id=user_id,

            compress=compress,
        )
    )


def executive_export(
    df: pd.DataFrame,
    user_id="default",
):

    return (

        csv_export_engine.executive_export(

            df,

            user_id,
        )
    )


def analytics_export(
    df: pd.DataFrame,
    user_id="default",
):

    return (

        csv_export_engine.analytics_export(

            df,

            user_id,
        )
    )


def batch_export(
    datasets,
    user_id="default",
):

    return (

        csv_export_engine.batch_export(

            datasets,

            user_id,
        )
    )