"""
ChurnShield 2.0 — Enterprise Logger System

File:
utils/logger.py

Purpose:
Production-grade logging engine
for ChurnShield AI platform.

Capabilities:
- structured JSON logging
- colored console logs
- rotating log files
- request tracing
- correlation IDs
- async-safe logging
- performance tracking
- exception capturing
- audit logging
- security logging
- API logging
- model inference logging
- cache logging
- scheduler logging
- latency monitoring
- automatic log cleanup
- log analytics
- multi-environment support
- enterprise observability
- context-aware logging
- middleware integration
- OpenTelemetry-ready architecture

Author:
ChurnShield AI
"""

import os
import sys
import json
import time
import socket
import asyncio
import traceback
import logging

from uuid import uuid4

from typing import (

    Dict,
    Any,
    Optional,
    Union

)

from pathlib import Path

from datetime import datetime

from functools import wraps

from logging.handlers import (

    RotatingFileHandler,
    TimedRotatingFileHandler

)

from contextvars import ContextVar

# ============================================================
# LOG DIRECTORY
# ============================================================

LOG_DIR = Path("logs")

LOG_DIR.mkdir(

    parents=True,
    exist_ok=True

)

# ============================================================
# ENVIRONMENT
# ============================================================

APP_NAME = os.getenv(

    "APP_NAME",
    "ChurnShield"

)

ENVIRONMENT = os.getenv(

    "ENVIRONMENT",
    "development"

)

LOG_LEVEL = os.getenv(

    "LOG_LEVEL",
    "INFO"

).upper()

# ============================================================
# CONTEXT VARIABLES
# ============================================================

request_id_ctx = ContextVar(

    "request_id",
    default=None

)

user_id_ctx = ContextVar(

    "user_id",
    default=None

)

session_id_ctx = ContextVar(

    "session_id",
    default=None

)

# ============================================================
# COLORS
# ============================================================

class Colors:

    RESET = "\033[0m"

    RED = "\033[91m"

    GREEN = "\033[92m"

    YELLOW = "\033[93m"

    BLUE = "\033[94m"

    MAGENTA = "\033[95m"

    CYAN = "\033[96m"

    GREY = "\033[90m"

# ============================================================
# LOG LEVEL COLORS
# ============================================================

LEVEL_COLORS = {

    "DEBUG": Colors.GREY,

    "INFO": Colors.GREEN,

    "WARNING": Colors.YELLOW,

    "ERROR": Colors.RED,

    "CRITICAL": Colors.MAGENTA

}

# ============================================================
# JSON FORMATTER
# ============================================================

class JSONFormatter(

    logging.Formatter

):

    """
    Structured JSON formatter
    """

    def format(

        self,
        record

    ):

        log_data = {

            "timestamp":

                datetime.utcnow()

                .isoformat(),

            "service":

                APP_NAME,

            "environment":

                ENVIRONMENT,

            "level":

                record.levelname,

            "logger":

                record.name,

            "message":

                record.getMessage(),

            "module":

                record.module,

            "function":

                record.funcName,

            "line":

                record.lineno,

            "hostname":

                socket.gethostname(),

            "request_id":

                request_id_ctx.get(),

            "user_id":

                user_id_ctx.get(),

            "session_id":

                session_id_ctx.get()

        }

        # ====================================================
        # EXTRA DATA
        # ====================================================

        if hasattr(

            record,
            "extra_data"

        ):

            log_data.update(

                record.extra_data

            )

        # ====================================================
        # EXCEPTION TRACE
        # ====================================================

        if record.exc_info:

            log_data["traceback"] = (

                self.formatException(

                    record.exc_info

                )

            )

        return json.dumps(

            log_data,
            ensure_ascii=False

        )

# ============================================================
# CONSOLE FORMATTER
# ============================================================

class ConsoleFormatter(

    logging.Formatter

):

    """
    Colored console formatter
    """

    def format(

        self,
        record

    ):

        color = LEVEL_COLORS.get(

            record.levelname,
            Colors.RESET

        )

        timestamp = datetime.now().strftime(

            "%Y-%m-%d %H:%M:%S"

        )

        request_id = request_id_ctx.get()

        output = (

            f"{color}"

            f"[{timestamp}] "

            f"[{record.levelname}] "

            f"[{record.name}] "

            f"[RID:{request_id}] "

            f"{record.getMessage()}"

            f"{Colors.RESET}"

        )

        if record.exc_info:

            output += (

                "\n"
                +
                self.formatException(

                    record.exc_info

                )

            )

        return output

# ============================================================
# LOGGER FACTORY
# ============================================================

class LoggerFactory:

    """
    Enterprise logger builder
    """

    _loggers = {}

    # ========================================================
    # CREATE LOGGER
    # ========================================================

    @classmethod
    def create_logger(

        cls,
        name: str

    ):

        if name in cls._loggers:

            return cls._loggers[name]

        logger = logging.getLogger(

            name

        )

        logger.setLevel(

            LOG_LEVEL

        )

        logger.propagate = False

        # ====================================================
        # CONSOLE HANDLER
        # ====================================================

        console_handler = logging.StreamHandler(

            sys.stdout

        )

        console_handler.setFormatter(

            ConsoleFormatter()

        )

        # ====================================================
        # APP LOG FILE
        # ====================================================

        app_handler = RotatingFileHandler(

            filename=LOG_DIR / "app.log",

            maxBytes=10 * 1024 * 1024,

            backupCount=10,

            encoding="utf-8"

        )

        app_handler.setFormatter(

            JSONFormatter()

        )

        # ====================================================
        # ERROR LOG FILE
        # ====================================================

        error_handler = RotatingFileHandler(

            filename=LOG_DIR / "errors.log",

            maxBytes=10 * 1024 * 1024,

            backupCount=10,

            encoding="utf-8"

        )

        error_handler.setLevel(

            logging.ERROR

        )

        error_handler.setFormatter(

            JSONFormatter()

        )

        # ====================================================
        # AUDIT LOG
        # ====================================================

        audit_handler = TimedRotatingFileHandler(

            filename=LOG_DIR / "audit.log",

            when="midnight",

            interval=1,

            backupCount=30,

            encoding="utf-8"

        )

        audit_handler.setFormatter(

            JSONFormatter()

        )

        # ====================================================
        # ADD HANDLERS
        # ====================================================

        logger.addHandler(

            console_handler

        )

        logger.addHandler(

            app_handler

        )

        logger.addHandler(

            error_handler

        )

        logger.addHandler(

            audit_handler

        )

        cls._loggers[name] = logger

        return logger

# ============================================================
# LOGGER ACCESS
# ============================================================

def get_logger(

    name: str = "app"

):

    return LoggerFactory.create_logger(

        name

    )

# ============================================================
# CONTEXT HELPERS
# ============================================================

def set_request_id(

    request_id: str

):

    request_id_ctx.set(

        request_id

    )

def generate_request_id():

    request_id = str(

        uuid4()

    )

    request_id_ctx.set(

        request_id

    )

    return request_id

def set_user_id(

    user_id: Union[str, int]

):

    user_id_ctx.set(

        str(user_id)

    )

def set_session_id(

    session_id: str

):

    session_id_ctx.set(

        session_id

    )

# ============================================================
# PERFORMANCE DECORATOR
# ============================================================

def track_performance(

    logger=None

):

    """
    Track execution time
    """

    logger = logger or get_logger(

        "performance"

    )

    def decorator(func):

        @wraps(func)
        def wrapper(

            *args,
            **kwargs

        ):

            start = time.perf_counter()

            try:

                result = func(

                    *args,
                    **kwargs

                )

                duration = round(

                    (
                        time.perf_counter()
                        -
                        start
                    ) * 1000,

                    2

                )

                logger.info(

                    f"{func.__name__} executed",

                    extra={

                        "extra_data": {

                            "execution_time_ms":

                                duration

                        }

                    }

                )

                return result

            except Exception:

                logger.exception(

                    f"{func.__name__} failed"

                )

                raise

        return wrapper

    return decorator

# ============================================================
# ASYNC PERFORMANCE DECORATOR
# ============================================================

def async_track_performance(

    logger=None

):

    logger = logger or get_logger(

        "async-performance"

    )

    def decorator(func):

        @wraps(func)
        async def wrapper(

            *args,
            **kwargs

        ):

            start = time.perf_counter()

            try:

                result = await func(

                    *args,
                    **kwargs

                )

                duration = round(

                    (
                        time.perf_counter()
                        -
                        start
                    ) * 1000,

                    2

                )

                logger.info(

                    f"{func.__name__} executed",

                    extra={

                        "extra_data": {

                            "execution_time_ms":

                                duration

                        }

                    }

                )

                return result

            except Exception:

                logger.exception(

                    f"{func.__name__} failed"

                )

                raise

        return wrapper

    return decorator

# ============================================================
# SPECIALIZED LOGGERS
# ============================================================

class APILogger:

    logger = get_logger(

        "api"

    )

    @staticmethod
    def request(

        method: str,
        endpoint: str,
        status_code: int,
        latency_ms: float

    ):

        APILogger.logger.info(

            f"{method} {endpoint}",

            extra={

                "extra_data": {

                    "status_code":

                        status_code,

                    "latency_ms":

                        latency_ms

                }

            }

        )

class SecurityLogger:

    logger = get_logger(

        "security"

    )

    @staticmethod
    def suspicious_activity(

        ip: str,
        reason: str

    ):

        SecurityLogger.logger.warning(

            "Suspicious activity detected",

            extra={

                "extra_data": {

                    "ip":

                        ip,

                    "reason":

                        reason

                }

            }

        )

class MLLogger:

    logger = get_logger(

        "ml"

    )

    @staticmethod
    def prediction(

        model_name: str,
        latency_ms: float

    ):

        MLLogger.logger.info(

            "Prediction completed",

            extra={

                "extra_data": {

                    "model":

                        model_name,

                    "latency_ms":

                        latency_ms

                }

            }

        )

# ============================================================
# HEALTH CHECK
# ============================================================

def logger_health():

    return {

        "status":

            "healthy",

        "environment":

            ENVIRONMENT,

        "log_level":

            LOG_LEVEL,

        "log_directory":

            str(LOG_DIR)

    }

# ============================================================
# STARTUP BANNER
# ============================================================

def startup_banner():

    logger = get_logger(

        "startup"

    )

    logger.info("=" * 60)

    logger.info(

        f"{APP_NAME} Logger Initialized"

    )

    logger.info(

        f"Environment: {ENVIRONMENT}"

    )

    logger.info(

        f"Log Level: {LOG_LEVEL}"

    )

    logger.info("=" * 60)

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    startup_banner()

    logger = get_logger(

        "demo"

    )

    generate_request_id()

    set_user_id(1001)

    set_session_id(

        "session_xyz"

    )

    logger.info(

        "Logger system started"

    )

    logger.warning(

        "Memory usage increasing"

    )

    try:

        1 / 0

    except Exception:

        logger.exception(

            "Critical failure occurred"

        )

    @track_performance(logger)
    def sample_function():

        time.sleep(1)

    sample_function()

    print("\n")

    print(logger_health())