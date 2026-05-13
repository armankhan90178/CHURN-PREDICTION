"""
ChurnShield 2.0 — Global Error Handler

File:
middleware/error_handler.py

Purpose:
Enterprise-grade global exception
handling middleware for FastAPI.

Capabilities:
- centralized exception handling
- structured JSON error responses
- validation error handling
- database error handling
- HTTP exception handling
- unexpected crash handling
- stack trace logging
- request tracking
- enterprise audit logging
- production-safe responses
- custom exception classes
- async-compatible architecture
- security-safe error masking
- API standardization

Author:
ChurnShield AI
"""

import traceback
import logging

from datetime import datetime
from typing import Any, Dict

from fastapi import (

    Request,
    HTTPException,
    status

)

from fastapi.responses import JSONResponse

from fastapi.exceptions import (
    RequestValidationError
)

from sqlalchemy.exc import (

    SQLAlchemyError,
    IntegrityError

)

# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    "error_handler"
)

logging.basicConfig(
    level=logging.INFO
)

# ============================================================
# BASE ERROR RESPONSE
# ============================================================

def error_response(

    message: str,
    status_code: int,
    error_type: str = "ApplicationError",
    details: Any = None

):

    """
    Standardized API error response
    """

    return {

        "success": False,

        "error": {

            "type": error_type,

            "message": message,

            "details": details,

            "timestamp":

                datetime.utcnow()

                .isoformat()

        },

        "status_code": status_code

    }

# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class ChurnShieldException(
    Exception
):

    def __init__(

        self,
        message: str,
        status_code: int = 400,
        details: Any = None

    ):

        self.message = message

        self.status_code = status_code

        self.details = details

        super().__init__(message)

# ============================================================
# AUTH EXCEPTION
# ============================================================

class AuthenticationException(
    ChurnShieldException
):

    pass

# ============================================================
# AUTHORIZATION EXCEPTION
# ============================================================

class AuthorizationException(
    ChurnShieldException
):

    pass

# ============================================================
# DATA VALIDATION EXCEPTION
# ============================================================

class DataValidationException(
    ChurnShieldException
):

    pass

# ============================================================
# MODEL EXCEPTION
# ============================================================

class ModelException(
    ChurnShieldException
):

    pass

# ============================================================
# STORAGE EXCEPTION
# ============================================================

class StorageException(
    ChurnShieldException
):

    pass

# ============================================================
# LOGGER HELPER
# ============================================================

def log_exception(

    request: Request,
    exc: Exception

):

    """
    Structured exception logging
    """

    logger.error({

        "path":
            request.url.path,

        "method":
            request.method,

        "client":

            request.client.host

            if request.client

            else "unknown",

        "error_type":
            type(exc).__name__,

        "error":
            str(exc),

        "timestamp":

            datetime.utcnow()

            .isoformat(),

        "traceback":

            traceback.format_exc()

    })

# ============================================================
# HTTP EXCEPTION HANDLER
# ============================================================

async def http_exception_handler(

    request: Request,
    exc: HTTPException

):

    """
    Handle FastAPI HTTP exceptions
    """

    log_exception(
        request,
        exc
    )

    return JSONResponse(

        status_code=exc.status_code,

        content=error_response(

            message=str(exc.detail),

            status_code=exc.status_code,

            error_type="HTTPException"

        )

    )

# ============================================================
# VALIDATION ERROR HANDLER
# ============================================================

async def validation_exception_handler(

    request: Request,
    exc: RequestValidationError

):

    """
    Handle validation errors
    """

    log_exception(
        request,
        exc
    )

    formatted_errors = []

    for error in exc.errors():

        formatted_errors.append({

            "field":
                ".".join(

                    map(str, error["loc"])

                ),

            "message":
                error["msg"],

            "type":
                error["type"]

        })

    return JSONResponse(

        status_code=422,

        content=error_response(

            message="Validation failed",

            status_code=422,

            error_type="ValidationError",

            details=formatted_errors

        )

    )

# ============================================================
# DATABASE ERROR HANDLER
# ============================================================

async def database_exception_handler(

    request: Request,
    exc: SQLAlchemyError

):

    """
    Handle SQLAlchemy errors
    """

    log_exception(
        request,
        exc
    )

    message = "Database operation failed"

    if isinstance(exc, IntegrityError):

        message = "Database integrity violation"

    return JSONResponse(

        status_code=500,

        content=error_response(

            message=message,

            status_code=500,

            error_type="DatabaseError"

        )

    )

# ============================================================
# CUSTOM APP EXCEPTION HANDLER
# ============================================================

async def churnshield_exception_handler(

    request: Request,
    exc: ChurnShieldException

):

    """
    Handle custom application exceptions
    """

    log_exception(
        request,
        exc
    )

    return JSONResponse(

        status_code=exc.status_code,

        content=error_response(

            message=exc.message,

            status_code=exc.status_code,

            error_type=type(exc).__name__,

            details=exc.details

        )

    )

# ============================================================
# GENERIC EXCEPTION HANDLER
# ============================================================

async def generic_exception_handler(

    request: Request,
    exc: Exception

):

    """
    Catch-all exception handler
    """

    log_exception(
        request,
        exc
    )

    return JSONResponse(

        status_code=500,

        content=error_response(

            message="Internal server error",

            status_code=500,

            error_type="InternalServerError"

        )

    )

# ============================================================
# REGISTER HANDLERS
# ============================================================

def register_exception_handlers(
    app
):

    """
    Attach handlers to FastAPI app
    """

    app.add_exception_handler(

        HTTPException,

        http_exception_handler

    )

    app.add_exception_handler(

        RequestValidationError,

        validation_exception_handler

    )

    app.add_exception_handler(

        SQLAlchemyError,

        database_exception_handler

    )

    app.add_exception_handler(

        ChurnShieldException,

        churnshield_exception_handler

    )

    app.add_exception_handler(

        Exception,

        generic_exception_handler

    )

    logger.info(
        "Exception handlers registered"
    )

# ============================================================
# SAFE EXECUTION WRAPPER
# ============================================================

def safe_execute(

    func,
    *args,
    **kwargs

):

    """
    Safe function executor
    """

    try:

        return func(
            *args,
            **kwargs
        )

    except Exception as e:

        logger.error({

            "safe_execute_error":
                str(e),

            "function":
                func.__name__

        })

        return None

# ============================================================
# API RESPONSE HELPERS
# ============================================================

def success_response(

    data: Any = None,
    message: str = "Success"

):

    """
    Standard success response
    """

    return {

        "success": True,

        "message": message,

        "data": data,

        "timestamp":

            datetime.utcnow()

            .isoformat()

    }

# ============================================================
# NOT FOUND HELPER
# ============================================================

def not_found(

    entity: str = "Resource"

):

    """
    Raise not found exception
    """

    raise HTTPException(

        status_code=404,

        detail=f"{entity} not found"

    )

# ============================================================
# UNAUTHORIZED HELPER
# ============================================================

def unauthorized():

    """
    Raise unauthorized
    """

    raise AuthenticationException(

        message="Unauthorized access",

        status_code=401

    )

# ============================================================
# FORBIDDEN HELPER
# ============================================================

def forbidden():

    """
    Raise forbidden
    """

    raise AuthorizationException(

        message="Permission denied",

        status_code=403

    )

# ============================================================
# BAD REQUEST HELPER
# ============================================================

def bad_request(
    message="Invalid request"
):

    """
    Raise bad request
    """

    raise ChurnShieldException(

        message=message,

        status_code=400

    )

# ============================================================
# SERVICE UNAVAILABLE
# ============================================================

def service_unavailable():

    """
    Raise service unavailable
    """

    raise ChurnShieldException(

        message="Service unavailable",

        status_code=503

    )

# ============================================================
# ERROR ANALYTICS
# ============================================================

ERROR_COUNTER = {}

def track_error(
    error_name: str
):

    """
    Error analytics tracker
    """

    ERROR_COUNTER[error_name] = (

        ERROR_COUNTER.get(
            error_name,
            0
        )
        +
        1

    )

def get_error_stats():

    """
    Get error metrics
    """

    return {

        "tracked_errors":
            ERROR_COUNTER,

        "total_error_types":
            len(ERROR_COUNTER)

    }

# ============================================================
# HEALTH CHECK
# ============================================================

def error_handler_health():

    """
    Health check
    """

    return {

        "status": "healthy",

        "tracked_error_types":
            len(ERROR_COUNTER)

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD ERROR HANDLER")
    print("=" * 60)

    try:

        raise DataValidationException(

            message="Dataset missing columns",

            status_code=422,

            details={

                "missing":

                    ["customer_id"]

            }

        )

    except Exception as e:

        print("\nCaught Exception:\n")

        print(type(e).__name__)

        print(str(e))

    track_error("ValidationError")

    track_error("DatabaseError")

    track_error("ValidationError")

    print("\nError Stats:\n")

    print(get_error_stats())

    print("\n")
    print("=" * 60)
    print("ERROR HANDLER READY")
    print("=" * 60)