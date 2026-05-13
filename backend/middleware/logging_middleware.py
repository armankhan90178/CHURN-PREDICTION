"""
ChurnShield 2.0 — Logging Middleware

File:
middleware/logging_middleware.py

Purpose:
Enterprise-grade API logging middleware
for FastAPI applications.

Capabilities:
- request/response logging
- execution time tracking
- structured JSON logs
- rotating file logging
- console logging
- error tracking
- request body logging
- response status tracking
- IP tracking
- user-agent tracking
- audit-ready architecture
- enterprise observability
- performance analytics
- API monitoring
- log sanitization
- async-safe middleware

Author:
ChurnShield AI
"""

import os
import json
import time
import uuid
import logging

from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

from fastapi import Request
from starlette.middleware.base import (
    BaseHTTPMiddleware
)

# ============================================================
# LOG DIRECTORY
# ============================================================

LOG_DIR = Path("logs")

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

APP_LOG_FILE = LOG_DIR / "app.log"

ERROR_LOG_FILE = LOG_DIR / "errors.log"

# ============================================================
# LOGGER CONFIGURATION
# ============================================================

logger = logging.getLogger(
    "churnshield_logger"
)

logger.setLevel(logging.INFO)

# ============================================================
# FORMATTER
# ============================================================

formatter = logging.Formatter(

    fmt="""
%(asctime)s | %(levelname)s | %(message)s
""".strip(),

    datefmt="%Y-%m-%d %H:%M:%S"

)

# ============================================================
# APP LOG HANDLER
# ============================================================

app_handler = RotatingFileHandler(

    APP_LOG_FILE,

    maxBytes=10 * 1024 * 1024,

    backupCount=10,

    encoding="utf-8"

)

app_handler.setFormatter(
    formatter
)

# ============================================================
# ERROR LOG HANDLER
# ============================================================

error_handler = RotatingFileHandler(

    ERROR_LOG_FILE,

    maxBytes=10 * 1024 * 1024,

    backupCount=10,

    encoding="utf-8"

)

error_handler.setLevel(
    logging.ERROR
)

error_handler.setFormatter(
    formatter
)

# ============================================================
# CONSOLE HANDLER
# ============================================================

console_handler = logging.StreamHandler()

console_handler.setFormatter(
    formatter
)

# ============================================================
# ATTACH HANDLERS
# ============================================================

if not logger.handlers:

    logger.addHandler(app_handler)

    logger.addHandler(error_handler)

    logger.addHandler(console_handler)

# ============================================================
# SENSITIVE HEADERS
# ============================================================

SENSITIVE_HEADERS = {

    "authorization",
    "cookie",
    "x-api-key"

}

# ============================================================
# SANITIZE HEADERS
# ============================================================

def sanitize_headers(headers):

    """
    Hide sensitive headers
    """

    sanitized = {}

    for key, value in headers.items():

        if key.lower() in SENSITIVE_HEADERS:

            sanitized[key] = "***HIDDEN***"

        else:

            sanitized[key] = value

    return sanitized

# ============================================================
# SAFE JSON SERIALIZER
# ============================================================

def safe_json(data):

    """
    Safe JSON conversion
    """

    try:

        return json.dumps(

            data,

            default=str,

            ensure_ascii=False

        )

    except Exception:

        return str(data)

# ============================================================
# REQUEST BODY READER
# ============================================================

async def read_request_body(
    request: Request
):

    """
    Read request body safely
    """

    try:

        body = await request.body()

        if not body:

            return None

        if len(body) > 5000:

            return "BODY_TOO_LARGE"

        return body.decode(
            "utf-8",
            errors="ignore"
        )

    except Exception:

        return None

# ============================================================
# LOGGING MIDDLEWARE
# ============================================================

class LoggingMiddleware(
    BaseHTTPMiddleware
):

    """
    Enterprise logging middleware
    """

    async def dispatch(

        self,
        request: Request,
        call_next

    ):

        request_id = str(
            uuid.uuid4()
        )

        start_time = time.time()

        request_body = await read_request_body(
            request
        )

        request_data = {

            "request_id":
                request_id,

            "timestamp":
                datetime.utcnow().isoformat(),

            "method":
                request.method,

            "path":
                request.url.path,

            "query_params":
                str(request.query_params),

            "client_ip":

                request.client.host

                if request.client

                else "unknown",

            "headers":

                sanitize_headers(
                    dict(request.headers)
                ),

            "body":
                request_body,

            "user_agent":

                request.headers.get(
                    "user-agent",
                    "unknown"
                )

        }

        logger.info(

            f"REQUEST | {safe_json(request_data)}"

        )

        try:

            response = await call_next(
                request
            )

            process_time = round(

                time.time()
                -
                start_time,

                4

            )

            response_data = {

                "request_id":
                    request_id,

                "status_code":
                    response.status_code,

                "process_time":
                    process_time,

                "timestamp":
                    datetime.utcnow().isoformat()

            }

            logger.info(

                f"RESPONSE | {safe_json(response_data)}"

            )

            response.headers[
                "X-Process-Time"
            ] = str(process_time)

            response.headers[
                "X-Request-ID"
            ] = request_id

            return response

        except Exception as e:

            process_time = round(

                time.time()
                -
                start_time,

                4

            )

            error_data = {

                "request_id":
                    request_id,

                "error":
                    str(e),

                "path":
                    request.url.path,

                "method":
                    request.method,

                "process_time":
                    process_time,

                "timestamp":
                    datetime.utcnow().isoformat()

            }

            logger.error(

                f"ERROR | {safe_json(error_data)}"

            )

            raise e

# ============================================================
# CUSTOM LOG FUNCTIONS
# ============================================================

def log_info(
    message,
    extra=None
):

    """
    Info logging
    """

    logger.info(

        safe_json({

            "message": message,

            "extra": extra,

            "timestamp":

                datetime.utcnow()

                .isoformat()

        })

    )

# ============================================================
# ERROR LOGGING
# ============================================================

def log_error(
    error,
    extra=None
):

    """
    Error logging
    """

    logger.error(

        safe_json({

            "error": str(error),

            "extra": extra,

            "timestamp":

                datetime.utcnow()

                .isoformat()

        })

    )

# ============================================================
# WARNING LOGGING
# ============================================================

def log_warning(
    warning,
    extra=None
):

    """
    Warning logging
    """

    logger.warning(

        safe_json({

            "warning": str(warning),

            "extra": extra,

            "timestamp":

                datetime.utcnow()

                .isoformat()

        })

    )

# ============================================================
# DEBUG LOGGING
# ============================================================

def log_debug(
    debug,
    extra=None
):

    """
    Debug logging
    """

    logger.debug(

        safe_json({

            "debug": str(debug),

            "extra": extra,

            "timestamp":

                datetime.utcnow()

                .isoformat()

        })

    )

# ============================================================
# API ANALYTICS LOGGER
# ============================================================

def log_api_metric(

    endpoint,
    latency,
    status_code

):

    """
    API metric tracking
    """

    metric = {

        "endpoint": endpoint,

        "latency": latency,

        "status_code": status_code,

        "timestamp":

            datetime.utcnow()

            .isoformat()

    }

    logger.info(

        f"API_METRIC | {safe_json(metric)}"

    )

# ============================================================
# MODEL PREDICTION LOGGER
# ============================================================

def log_prediction(

    customer_id,
    probability,
    model_name

):

    """
    ML prediction logging
    """

    payload = {

        "customer_id":
            customer_id,

        "probability":
            probability,

        "model":
            model_name,

        "timestamp":

            datetime.utcnow()

            .isoformat()

    }

    logger.info(

        f"PREDICTION | {safe_json(payload)}"

    )

# ============================================================
# HEALTH CHECK
# ============================================================

def logging_health():

    """
    Logging health status
    """

    return {

        "status": "healthy",

        "log_directory":
            str(LOG_DIR),

        "app_log_exists":
            APP_LOG_FILE.exists(),

        "error_log_exists":
            ERROR_LOG_FILE.exists()

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD LOGGING MIDDLEWARE")
    print("=" * 60)

    log_info(
        "Application started"
    )

    log_warning(
        "High latency detected"
    )

    log_error(
        "Database timeout"
    )

    log_prediction(

        customer_id=101,

        probability=0.91,

        model_name="global_model.pkl"

    )

    print("\nLogs written successfully")

    print("\nHealth Status:\n")

    print(
        logging_health()
    )

    print("\n")
    print("=" * 60)
    print("LOGGING SYSTEM READY")
    print("=" * 60)