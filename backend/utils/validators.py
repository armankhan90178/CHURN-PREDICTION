"""
ChurnShield 2.0 — Enterprise Validation Engine

File:
utils/validators.py

Purpose:
Centralized enterprise validation system
for ChurnShield AI platform.

Capabilities:
- email validation
- password validation
- phone validation
- UUID validation
- URL validation
- IP validation
- file validation
- dataset validation
- JSON validation
- schema validation
- API payload validation
- upload validation
- ML feature validation
- business-rule validation
- SQL injection detection
- XSS protection
- regex validation
- type validation
- sanitization helpers
- enterprise security validation

Author:
ChurnShield AI
"""

import re
import json
import ipaddress
import mimetypes

from uuid import UUID

from pathlib import Path

from typing import (

    Dict,
    List,
    Any,
    Optional,
    Union

)

from urllib.parse import urlparse

# ============================================================
# CONSTANTS
# ============================================================

MAX_FILE_SIZE_MB = 500

ALLOWED_EXTENSIONS = {

    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".zip"

}

SUSPICIOUS_PATTERNS = [

    r"(\%27)|(\')|(\-\-)|(\%23)|(#)",

    r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",

    r"\b(SELECT|UNION|INSERT|DELETE|DROP|UPDATE|ALTER)\b",

    r"<script.*?>.*?</script>",

    r"javascript:",

    r"onerror=",

    r"onload="

]

# ============================================================
# VALIDATION ERROR
# ============================================================

class ValidationError(

    Exception

):

    """
    Custom validation exception
    """

    pass

# ============================================================
# EMAIL VALIDATION
# ============================================================

def validate_email(

    email: str

) -> bool:

    """
    Enterprise email validation
    """

    if not isinstance(

        email,
        str

    ):

        return False

    pattern = (

        r"^[a-zA-Z0-9_.+-]+"
        r"@[a-zA-Z0-9-]+"
        r"\.[a-zA-Z0-9-.]+$"

    )

    return bool(

        re.match(

            pattern,
            email

        )

    )

# ============================================================
# PASSWORD VALIDATION
# ============================================================

def validate_password(

    password: str,
    min_length: int = 8

) -> Dict:

    """
    Enterprise password validator
    """

    result = {

        "valid": True,

        "errors": []

    }

    if len(password) < min_length:

        result["valid"] = False

        result["errors"].append(

            "Password too short"

        )

    if not re.search(

        r"[A-Z]",
        password

    ):

        result["valid"] = False

        result["errors"].append(

            "Missing uppercase letter"

        )

    if not re.search(

        r"[a-z]",
        password

    ):

        result["valid"] = False

        result["errors"].append(

            "Missing lowercase letter"

        )

    if not re.search(

        r"\d",
        password

    ):

        result["valid"] = False

        result["errors"].append(

            "Missing number"

        )

    if not re.search(

        r"[!@#$%^&*(),.?\":{}|<>]",
        password

    ):

        result["valid"] = False

        result["errors"].append(

            "Missing special character"

        )

    return result

# ============================================================
# PHONE VALIDATION
# ============================================================

def validate_phone(

    phone: str

) -> bool:

    """
    International phone validation
    """

    pattern = r"^\+?[1-9]\d{7,14}$"

    return bool(

        re.match(

            pattern,
            phone

        )

    )

# ============================================================
# UUID VALIDATION
# ============================================================

def validate_uuid(

    value: str

) -> bool:

    """
    UUID validator
    """

    try:

        UUID(value)

        return True

    except Exception:

        return False

# ============================================================
# URL VALIDATION
# ============================================================

def validate_url(

    url: str

) -> bool:

    """
    URL validator
    """

    try:

        parsed = urlparse(url)

        return all([

            parsed.scheme,
            parsed.netloc

        ])

    except Exception:

        return False

# ============================================================
# IP VALIDATION
# ============================================================

def validate_ip(

    ip: str

) -> bool:

    """
    IPv4/IPv6 validator
    """

    try:

        ipaddress.ip_address(ip)

        return True

    except Exception:

        return False

# ============================================================
# JSON VALIDATION
# ============================================================

def validate_json(

    data: str

) -> bool:

    """
    JSON structure validation
    """

    try:

        json.loads(data)

        return True

    except Exception:

        return False

# ============================================================
# FILE EXTENSION
# ============================================================

def validate_file_extension(

    filename: str

) -> bool:

    """
    File extension validation
    """

    extension = Path(

        filename

    ).suffix.lower()

    return extension in ALLOWED_EXTENSIONS

# ============================================================
# FILE SIZE VALIDATION
# ============================================================

def validate_file_size(

    size_bytes: int

) -> bool:

    """
    File size validation
    """

    size_mb = (

        size_bytes
        /
        (1024 * 1024)

    )

    return size_mb <= MAX_FILE_SIZE_MB

# ============================================================
# MIME TYPE VALIDATION
# ============================================================

def validate_mime_type(

    filename: str

) -> bool:

    """
    MIME type validator
    """

    mime_type, _ = mimetypes.guess_type(

        filename

    )

    allowed = [

        "text/csv",

        "application/json",

        "application/zip",

        "application/vnd.ms-excel",

        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    ]

    return mime_type in allowed

# ============================================================
# SQL INJECTION DETECTION
# ============================================================

def detect_sql_injection(

    value: str

) -> bool:

    """
    SQL injection detection
    """

    for pattern in SUSPICIOUS_PATTERNS:

        if re.search(

            pattern,
            value,
            re.IGNORECASE

        ):

            return True

    return False

# ============================================================
# XSS DETECTION
# ============================================================

def detect_xss(

    value: str

) -> bool:

    """
    XSS detection
    """

    patterns = [

        r"<script.*?>",

        r"javascript:",

        r"onerror=",

        r"onload=",

        r"<iframe.*?>"

    ]

    for pattern in patterns:

        if re.search(

            pattern,
            value,
            re.IGNORECASE

        ):

            return True

    return False

# ============================================================
# SANITIZE INPUT
# ============================================================

def sanitize_input(

    value: str

) -> str:

    """
    Basic sanitization
    """

    value = re.sub(

        r"<.*?>",
        "",

        value

    )

    value = value.strip()

    return value

# ============================================================
# TYPE VALIDATION
# ============================================================

def validate_type(

    value: Any,
    expected_type

) -> bool:

    """
    Runtime type validation
    """

    return isinstance(

        value,
        expected_type

    )

# ============================================================
# REQUIRED FIELDS
# ============================================================

def validate_required_fields(

    data: Dict,
    required_fields: List[str]

) -> Dict:

    """
    Required field validator
    """

    missing = []

    for field in required_fields:

        if field not in data:

            missing.append(field)

    return {

        "valid":

            len(missing) == 0,

        "missing_fields":

            missing

    }

# ============================================================
# RANGE VALIDATION
# ============================================================

def validate_range(

    value: Union[int, float],
    minimum: Optional[float] = None,
    maximum: Optional[float] = None

) -> bool:

    """
    Numeric range validation
    """

    if minimum is not None:

        if value < minimum:

            return False

    if maximum is not None:

        if value > maximum:

            return False

    return True

# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset_structure(

    columns: List[str],
    required_columns: List[str]

) -> Dict:

    """
    Dataset schema validation
    """

    missing = [

        col

        for col in required_columns

        if col not in columns

    ]

    return {

        "valid":

            len(missing) == 0,

        "missing_columns":

            missing

    }

# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_ml_features(

    features: Dict

) -> Dict:

    """
    ML feature validation
    """

    errors = []

    for key, value in features.items():

        if value is None:

            errors.append(

                f"{key} is null"

            )

        if isinstance(

            value,
            str

        ) and len(value) > 1000:

            errors.append(

                f"{key} too large"

            )

    return {

        "valid":

            len(errors) == 0,

        "errors":

            errors

    }

# ============================================================
# API PAYLOAD VALIDATION
# ============================================================

def validate_api_payload(

    payload: Dict,
    schema: Dict

) -> Dict:

    """
    Lightweight schema validator
    """

    errors = []

    for field, expected_type in schema.items():

        if field not in payload:

            errors.append(

                f"{field} missing"

            )

            continue

        if not isinstance(

            payload[field],
            expected_type

        ):

            errors.append(

                f"{field} invalid type"

            )

    return {

        "valid":

            len(errors) == 0,

        "errors":

            errors

    }

# ============================================================
# BUSINESS RULE VALIDATION
# ============================================================

def validate_churn_probability(

    probability: float

) -> bool:

    """
    Churn probability validator
    """

    return 0 <= probability <= 1

# ============================================================
# BATCH VALIDATION
# ============================================================

def validate_batch(

    items: List[Any],
    validator_function

) -> Dict:

    """
    Batch validation engine
    """

    valid_items = []

    invalid_items = []

    for item in items:

        try:

            if validator_function(item):

                valid_items.append(item)

            else:

                invalid_items.append(item)

        except Exception:

            invalid_items.append(item)

    return {

        "valid_count":

            len(valid_items),

        "invalid_count":

            len(invalid_items),

        "valid_items":

            valid_items,

        "invalid_items":

            invalid_items

    }

# ============================================================
# HEALTH CHECK
# ============================================================

def validators_health():

    return {

        "status":

            "healthy",

        "max_file_size_mb":

            MAX_FILE_SIZE_MB,

        "allowed_extensions":

            list(ALLOWED_EXTENSIONS)

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD VALIDATION ENGINE")
    print("=" * 60)

    print("\nEmail Validation:\n")

    print(

        validate_email(

            "admin@churnshield.ai"

        )

    )

    print("\nPassword Validation:\n")

    print(

        validate_password(

            "StrongPass@123"

        )

    )

    print("\nPhone Validation:\n")

    print(

        validate_phone(

            "+919876543210"

        )

    )

    print("\nSQL Injection Detection:\n")

    print(

        detect_sql_injection(

            "' OR 1=1 --"

        )

    )

    print("\nXSS Detection:\n")

    print(

        detect_xss(

            "<script>alert(1)</script>"

        )

    )

    print("\nDataset Validation:\n")

    print(

        validate_dataset_structure(

            ["customer_id", "revenue"],

            ["customer_id", "revenue", "churn"]

        )

    )

    print("\nHealth:\n")

    print(

        validators_health()

    )

    print("\n")
    print("=" * 60)
    print("VALIDATION ENGINE READY")
    print("=" * 60)