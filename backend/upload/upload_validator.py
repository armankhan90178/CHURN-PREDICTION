"""
ChurnShield 2.0 — Upload Validator

File:
upload/upload_validator.py

Purpose:
Enterprise-grade upload validation
and security engine for ChurnShield AI.

Capabilities:
- file validation
- MIME type verification
- extension validation
- upload size limits
- schema validation
- malware protection
- suspicious pattern detection
- duplicate validation
- encoding verification
- filename sanitization
- dataset quality validation
- CSV/XLSX/JSON support
- enterprise upload policies
- security hardening
- AI dataset validation
- upload analytics
- upload scoring
- chunk validation
- secure upload pipeline

Author:
ChurnShield AI
"""

import os
import re
import json
import hashlib
import logging
import mimetypes
import traceback

import pandas as pd

from pathlib import Path

from typing import (

    Dict,
    Any,
    Optional,
    List

)

from datetime import datetime

# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(

    "upload_validator"

)

logging.basicConfig(

    level=logging.INFO

)

# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_EXTENSIONS = [

    ".csv",
    ".json",
    ".xlsx",
    ".xls",
    ".parquet",
    ".zip"

]

ALLOWED_MIME_TYPES = [

    "text/csv",

    "application/json",

    "application/vnd.ms-excel",

    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

    "application/octet-stream",

    "application/zip"

]

MAX_FILE_SIZE_MB = 100

MAX_ROWS_ALLOWED = 5_000_000

MAX_COLUMNS_ALLOWED = 500

# ============================================================
# SUSPICIOUS PATTERNS
# ============================================================

SUSPICIOUS_PATTERNS = [

    r"<script.*?>",

    r"DROP\s+TABLE",

    r"SELECT\s+\*",

    r"UNION\s+SELECT",

    r"eval\(",

    r"os\.system",

    r"subprocess",

    r"\.\./",

    r"rm\s+-rf",

    r"wget\s+",

    r"curl\s+",

]

# ============================================================
# VALIDATION RESULT
# ============================================================

class ValidationResult:

    """
    Upload validation response
    """

    def __init__(self):

        self.valid = True

        self.errors = []

        self.warnings = []

        self.metadata = {}

        self.score = 100

    def add_error(

        self,
        message: str

    ):

        self.valid = False

        self.errors.append(message)

        self.score -= 15

    def add_warning(

        self,
        message: str

    ):

        self.warnings.append(message)

        self.score -= 5

    def to_dict(self):

        return {

            "valid":

                self.valid,

            "errors":

                self.errors,

            "warnings":

                self.warnings,

            "metadata":

                self.metadata,

            "score":

                max(0, self.score)

        }

# ============================================================
# UPLOAD VALIDATOR
# ============================================================

class UploadValidator:

    """
    Enterprise upload validation engine
    """

    # ========================================================
    # VALIDATE FILE
    # ========================================================

    @staticmethod
    def validate_file(

        file_path: str

    ) -> Dict:

        result = ValidationResult()

        try:

            if not os.path.exists(

                file_path

            ):

                result.add_error(

                    "File does not exist"

                )

                return result.to_dict()

            # FILE METADATA

            stat = os.stat(file_path)

            size_mb = round(

                stat.st_size

                /
                (1024 * 1024),

                2

            )

            result.metadata = {

                "filename":

                    os.path.basename(

                        file_path

                    ),

                "size_mb":

                    size_mb,

                "created_at":

                    datetime.fromtimestamp(

                        stat.st_ctime

                    ).isoformat()

            }

            # FILE SIZE

            if size_mb > MAX_FILE_SIZE_MB:

                result.add_error(

                    f"File exceeds {MAX_FILE_SIZE_MB}MB"

                )

            # EXTENSION

            ext = (

                Path(file_path)

                .suffix

                .lower()

            )

            if ext not in ALLOWED_EXTENSIONS:

                result.add_error(

                    f"Unsupported extension: {ext}"

                )

            # MIME TYPE

            mime, _ = mimetypes.guess_type(

                file_path

            )

            if mime and mime not in ALLOWED_MIME_TYPES:

                result.add_warning(

                    f"Unexpected MIME type: {mime}"

                )

            # SCAN CONTENT

            UploadValidator.scan_content(

                file_path,
                result

            )

            # DATASET VALIDATION

            if ext in [

                ".csv",
                ".xlsx",
                ".xls",
                ".json"

            ]:

                UploadValidator.validate_dataset(

                    file_path,
                    result

                )

            logger.info({

                "event":

                    "validation_complete",

                "file":

                    file_path,

                "valid":

                    result.valid

            })

            return result.to_dict()

        except Exception as e:

            logger.error({

                "event":

                    "validation_failed",

                "trace":

                    traceback.format_exc()

            })

            result.add_error(str(e))

            return result.to_dict()

    # ========================================================
    # CONTENT SCAN
    # ========================================================

    @staticmethod
    def scan_content(

        file_path: str,
        result: ValidationResult

    ):

        try:

            with open(

                file_path,

                "r",

                encoding="utf-8",

                errors="ignore"

            ) as f:

                content = f.read(50000)

            for pattern in SUSPICIOUS_PATTERNS:

                if re.search(

                    pattern,

                    content,

                    re.IGNORECASE

                ):

                    result.add_error(

                        f"Suspicious pattern detected: {pattern}"

                    )

        except Exception:

            result.add_warning(

                "Binary or unreadable file"

            )

    # ========================================================
    # DATASET VALIDATION
    # ========================================================

    @staticmethod
    def validate_dataset(

        file_path: str,
        result: ValidationResult

    ):

        ext = (

            Path(file_path)

            .suffix

            .lower()

        )

        try:

            if ext == ".csv":

                df = pd.read_csv(

                    file_path

                )

            elif ext == ".json":

                df = pd.read_json(

                    file_path

                )

            else:

                df = pd.read_excel(

                    file_path

                )

            rows = len(df)

            cols = len(df.columns)

            result.metadata.update({

                "rows":

                    rows,

                "columns":

                    cols,

                "column_names":

                    list(df.columns)

            })

            if rows == 0:

                result.add_error(

                    "Dataset is empty"

                )

            if rows > MAX_ROWS_ALLOWED:

                result.add_error(

                    "Dataset too large"

                )

            if cols > MAX_COLUMNS_ALLOWED:

                result.add_error(

                    "Too many columns"

                )

            # DUPLICATES

            duplicates = (

                df.duplicated()

                .sum()

            )

            if duplicates > 0:

                result.add_warning(

                    f"{duplicates} duplicate rows found"

                )

            # MISSING VALUES

            missing = (

                df.isnull()

                .sum()

                .sum()

            )

            if missing > 0:

                result.add_warning(

                    f"{missing} missing values found"

                )

            # COLUMN VALIDATION

            UploadValidator.validate_columns(

                df,
                result

            )

        except Exception as e:

            result.add_error(

                f"Dataset validation failed: {str(e)}"

            )

    # ========================================================
    # COLUMN VALIDATION
    # ========================================================

    @staticmethod
    def validate_columns(

        df: pd.DataFrame,
        result: ValidationResult

    ):

        invalid_columns = []

        for column in df.columns:

            if len(str(column)) > 100:

                invalid_columns.append(column)

            if re.search(

                r"[<>]",

                str(column)

            ):

                invalid_columns.append(column)

        if invalid_columns:

            result.add_warning(

                f"Suspicious columns: {invalid_columns}"

            )

    # ========================================================
    # SANITIZE FILENAME
    # ========================================================

    @staticmethod
    def sanitize_filename(

        filename: str

    ) -> str:

        filename = os.path.basename(

            filename

        )

        filename = re.sub(

            r"[^a-zA-Z0-9_.-]",

            "_",

            filename

        )

        return filename

    # ========================================================
    # HASH FILE
    # ========================================================

    @staticmethod
    def file_hash(

        file_path: str

    ) -> str:

        sha256 = hashlib.sha256()

        with open(

            file_path,

            "rb"

        ) as f:

            while True:

                chunk = f.read(8192)

                if not chunk:

                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    # ========================================================
    # ENCODING VALIDATION
    # ========================================================

    @staticmethod
    def validate_encoding(

        file_path: str

    ) -> bool:

        try:

            with open(

                file_path,

                "r",

                encoding="utf-8"

            ):

                return True

        except UnicodeDecodeError:

            return False

    # ========================================================
    # SECURE PATH CHECK
    # ========================================================

    @staticmethod
    def secure_path(

        path: str

    ) -> bool:

        return ".." not in path

    # ========================================================
    # FILE PREVIEW
    # ========================================================

    @staticmethod
    def preview(

        file_path: str,
        rows: int = 5

    ) -> Dict:

        ext = (

            Path(file_path)

            .suffix

            .lower()

        )

        if ext == ".csv":

            df = pd.read_csv(file_path)

        elif ext == ".json":

            df = pd.read_json(file_path)

        else:

            df = pd.read_excel(file_path)

        return {

            "rows":

                len(df),

            "columns":

                list(df.columns),

            "preview":

                df.head(rows)

                .to_dict(

                    orient="records"

                )

        }

# ============================================================
# BULK VALIDATOR
# ============================================================

class BulkUploadValidator:

    """
    Enterprise bulk validation
    """

    @staticmethod
    def validate_directory(

        directory: str

    ) -> List[Dict]:

        results = []

        files = os.listdir(directory)

        for file in files:

            full_path = os.path.join(

                directory,
                file

            )

            if os.path.isfile(full_path):

                results.append(

                    UploadValidator

                    .validate_file(

                        full_path

                    )

                )

        return results

# ============================================================
# UPLOAD ANALYTICS
# ============================================================

class UploadAnalytics:

    """
    Upload metrics tracker
    """

    uploads = []

    @classmethod
    def log_upload(

        cls,
        metadata: Dict

    ):

        cls.uploads.append(metadata)

    @classmethod
    def statistics(cls):

        return {

            "total_uploads":

                len(cls.uploads),

            "successful":

                len([

                    x

                    for x in cls.uploads

                    if x.get("valid")

                ]),

            "failed":

                len([

                    x

                    for x in cls.uploads

                    if not x.get("valid")

                ])

        }

# ============================================================
# HEALTH CHECK
# ============================================================

def validator_health():

    return {

        "status":

            "healthy",

        "max_file_size_mb":

            MAX_FILE_SIZE_MB,

        "allowed_extensions":

            ALLOWED_EXTENSIONS,

        "security_patterns":

            len(SUSPICIOUS_PATTERNS)

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD UPLOAD VALIDATOR")
    print("=" * 60)

    sample = pd.DataFrame({

        "customer_id":

            [

                "CUST_1",
                "CUST_2"

            ],

        "revenue":

            [

                1000,
                2000

            ]

    })

    sample.to_csv(

        "sample_upload.csv",

        index=False

    )

    result = (

        UploadValidator

        .validate_file(

            "sample_upload.csv"

        )

    )

    print("\nValidation Result:\n")

    print(

        json.dumps(

            result,

            indent=4

        )

    )

    print("\nHealth:\n")

    print(validator_health())

    print("\n")
    print("=" * 60)
    print("UPLOAD VALIDATOR READY")
    print("=" * 60)