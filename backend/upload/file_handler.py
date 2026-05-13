"""
ChurnShield 2.0 — Universal File Handler

Enterprise Upload Engine

Capabilities:
- CSV / Excel / JSON ingestion
- Automatic encoding detection
- Corrupted file recovery
- Large dataset optimization
- Schema preservation
- Smart datatype inference
- Multi-sheet Excel support
- Null normalization
- Duplicate header fixing
- AI-ready dataframe output
"""

import os
import json
import logging
import traceback
import warnings

import pandas as pd
import numpy as np

from pathlib import Path
from typing import Optional

logger = logging.getLogger(
    "churnshield.upload.file_handler"
)

warnings.filterwarnings(
    "ignore"
)


# ─────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────

def load_uploaded_file(
    file_path: str,
) -> pd.DataFrame:

    """
    Universal file loader

    Supports:
    - CSV
    - XLSX
    - XLS
    - JSON

    Returns:
    Cleaned pandas DataFrame
    """

    try:

        logger.info(
            f"Loading uploaded file: {file_path}"
        )

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        extension = (
            path.suffix
            .lower()
            .strip()
        )

        # ─────────────────────────────
        # CSV
        # ─────────────────────────────

        if extension == ".csv":

            df = load_csv_file(
                file_path
            )

        # ─────────────────────────────
        # EXCEL
        # ─────────────────────────────

        elif extension in [
            ".xlsx",
            ".xls",
        ]:

            df = load_excel_file(
                file_path
            )

        # ─────────────────────────────
        # JSON
        # ─────────────────────────────

        elif extension == ".json":

            df = load_json_file(
                file_path
            )

        else:

            raise ValueError(
                f"""
                Unsupported file format:
                {extension}
                """
            )

        # ─────────────────────────────
        # POST PROCESSING
        # ─────────────────────────────

        df = universal_post_processing(
            df
        )

        logger.info(
            f"""
            File loaded successfully
            Rows: {len(df)}
            Columns: {len(df.columns)}
            """
        )

        return df

    except Exception as e:

        logger.error(
            f"Universal file loading failed: {e}"
        )

        traceback.print_exc()

        raise


# ─────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────

def load_csv_file(
    file_path: str,
) -> pd.DataFrame:

    """
    Intelligent CSV loader
    """

    encodings = [

        "utf-8",
        "utf-8-sig",
        "latin1",
        "ISO-8859-1",
        "cp1252",
    ]

    separators = [
        ",",
        ";",
        "\t",
        "|",
    ]

    last_error = None

    for encoding in encodings:

        for separator in separators:

            try:

                logger.info(
                    f"""
                    Trying CSV load
                    Encoding={encoding}
                    Separator={separator}
                    """
                )

                df = pd.read_csv(

                    file_path,

                    encoding=encoding,

                    sep=separator,

                    on_bad_lines="skip",

                    low_memory=False,
                )

                # BAD PARSE CHECK
                if len(df.columns) <= 1:

                    continue

                logger.info(
                    f"""
                    CSV loaded successfully
                    Encoding={encoding}
                    Separator={separator}
                    """
                )

                return df

            except Exception as e:

                last_error = e

                continue

    raise Exception(
        f"""
        CSV loading failed.
        Last error:
        {last_error}
        """
    )


# ─────────────────────────────────────────────
# EXCEL LOADER
# ─────────────────────────────────────────────

def load_excel_file(
    file_path: str,
) -> pd.DataFrame:

    """
    Enterprise Excel loader
    """

    try:

        excel_file = pd.ExcelFile(
            file_path
        )

        sheet_names = (
            excel_file.sheet_names
        )

        logger.info(
            f"""
            Excel sheets found:
            {sheet_names}
            """
        )

        best_df = None
        best_score = 0

        # PICK BEST SHEET
        for sheet in sheet_names:

            try:

                df = pd.read_excel(

                    file_path,

                    sheet_name=sheet,
                )

                score = (
                    len(df)
                    * len(df.columns)
                )

                if score > best_score:

                    best_score = score
                    best_df = df

            except Exception as e:

                logger.warning(
                    f"""
                    Failed reading sheet:
                    {sheet}
                    """
                )

                continue

        if best_df is None:

            raise Exception(
                "No readable Excel sheet found"
            )

        logger.info(
            f"""
            Best Excel sheet selected
            Rows={len(best_df)}
            Cols={len(best_df.columns)}
            """
        )

        return best_df

    except Exception as e:

        logger.error(
            f"Excel loading failed: {e}"
        )

        raise


# ─────────────────────────────────────────────
# JSON LOADER
# ─────────────────────────────────────────────

def load_json_file(
    file_path: str,
) -> pd.DataFrame:

    """
    Universal JSON loader
    """

    try:

        with open(

            file_path,

            "r",

            encoding="utf-8",

        ) as f:

            content = json.load(f)

        # LIST OF OBJECTS
        if isinstance(content, list):

            df = pd.DataFrame(
                content
            )

        # DICT FORMAT
        elif isinstance(content, dict):

            # nested data key
            if "data" in content:

                df = pd.DataFrame(
                    content["data"]
                )

            else:

                df = pd.DataFrame(
                    [content]
                )

        else:

            raise Exception(
                "Unsupported JSON structure"
            )

        logger.info(
            f"""
            JSON loaded successfully
            Rows={len(df)}
            """
        )

        return df

    except Exception as e:

        logger.error(
            f"JSON loading failed: {e}"
        )

        raise


# ─────────────────────────────────────────────
# UNIVERSAL POST PROCESSING
# ─────────────────────────────────────────────

def universal_post_processing(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Enterprise dataframe cleaning
    """

    try:

        data = df.copy()

        # ─────────────────────────────
        # REMOVE EMPTY ROWS/COLS
        # ─────────────────────────────

        data = data.dropna(
            how="all"
        )

        data = data.dropna(
            axis=1,
            how="all",
        )

        # ─────────────────────────────
        # FIX COLUMN NAMES
        # ─────────────────────────────

        data.columns = normalize_columns(
            data.columns
        )

        # ─────────────────────────────
        # REMOVE DUPLICATE COLUMNS
        # ─────────────────────────────

        data = data.loc[
            :,
            ~data.columns.duplicated()
        ]

        # ─────────────────────────────
        # NORMALIZE NULLS
        # ─────────────────────────────

        data = normalize_null_values(
            data
        )

        # ─────────────────────────────
        # INFER DATATYPES
        # ─────────────────────────────

        data = infer_smart_types(
            data
        )

        # ─────────────────────────────
        # RESET INDEX
        # ─────────────────────────────

        data = data.reset_index(
            drop=True
        )

        logger.info(
            "Universal post-processing completed"
        )

        return data

    except Exception as e:

        logger.error(
            f"Post-processing failed: {e}"
        )

        raise


# ─────────────────────────────────────────────
# NORMALIZE COLUMNS
# ─────────────────────────────────────────────

def normalize_columns(
    columns,
):

    """
    Enterprise column cleaner
    """

    cleaned = []

    used = set()

    for col in columns:

        col = str(col)

        col = col.strip()

        col = col.lower()

        col = col.replace(
            " ",
            "_"
        )

        col = col.replace(
            "-",
            "_"
        )

        col = col.replace(
            "/",
            "_"
        )

        col = "".join(

            c

            for c in col

            if c.isalnum()
            or c == "_"
        )

        # EMPTY COLUMN
        if col == "":

            col = "unnamed_column"

        # DUPLICATE HANDLING
        original = col
        counter = 1

        while col in used:

            col = f"""
            {original}_{counter}
            """.replace(
                "\n",
                ""
            ).replace(
                " ",
                ""
            )

            counter += 1

        used.add(col)

        cleaned.append(col)

    return cleaned


# ─────────────────────────────────────────────
# NORMALIZE NULL VALUES
# ─────────────────────────────────────────────

def normalize_null_values(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Normalize null representations
    """

    null_patterns = [

        "",

        " ",

        "nan",

        "null",

        "none",

        "na",

        "n/a",

        "missing",

        "-",

        "--",
    ]

    data = df.copy()

    for col in data.columns:

        data[col] = data[col].replace(
            null_patterns,
            np.nan,
        )

    return data


# ─────────────────────────────────────────────
# SMART TYPE INFERENCE
# ─────────────────────────────────────────────

def infer_smart_types(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Intelligent datatype detection
    """

    data = df.copy()

    for col in data.columns:

        try:

            # SKIP ALL-NULL
            if data[col].isnull().all():

                continue

            # TRY NUMERIC
            numeric_series = pd.to_numeric(

                data[col],

                errors="coerce",
            )

            numeric_ratio = (
                numeric_series.notnull().mean()
            )

            if numeric_ratio > 0.80:

                data[col] = numeric_series

                continue

            # TRY DATETIME
            datetime_series = pd.to_datetime(

                data[col],

                errors="coerce",
            )

            datetime_ratio = (
                datetime_series.notnull().mean()
            )

            if datetime_ratio > 0.80:

                data[col] = datetime_series

                continue

        except Exception:

            continue

    logger.info(
        "Smart datatype inference completed"
    )

    return data


# ─────────────────────────────────────────────
# FILE METADATA
# ─────────────────────────────────────────────

def extract_file_metadata(
    file_path: str,
):

    """
    Extract enterprise metadata
    """

    try:

        path = Path(file_path)

        metadata = {

            "filename":
                path.name,

            "extension":
                path.suffix,

            "size_mb":
                round(
                    path.stat().st_size
                    / (1024 * 1024),
                    2,
                ),

            "absolute_path":
                str(path.absolute()),

            "exists":
                path.exists(),
        }

        return metadata

    except Exception as e:

        logger.error(
            f"Metadata extraction failed: {e}"
        )

        return {}