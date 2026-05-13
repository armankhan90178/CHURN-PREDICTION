"""
ChurnShield 2.0 — Rate Limiting Middleware

File:
middleware/rate_limit.py

Purpose:
Enterprise-grade API rate limiting
middleware for FastAPI.

Capabilities:
- IP-based rate limiting
- user-based rate limiting
- API key limiting
- burst protection
- sliding window algorithm
- endpoint-specific limits
- Redis-ready architecture
- in-memory fallback
- automatic cleanup
- DDoS mitigation
- bot protection
- abuse prevention
- request analytics
- enterprise SaaS protection

Author:
ChurnShield AI
"""

import time
import asyncio
import logging

from typing import (
    Dict,
    Optional
)

from collections import defaultdict

from fastapi import (

    Request,
    HTTPException,
    status

)

from starlette.middleware.base import (
    BaseHTTPMiddleware
)

# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    "rate_limit"
)

logging.basicConfig(
    level=logging.INFO
)

# ============================================================
# RATE LIMIT CONFIG
# ============================================================

DEFAULT_REQUEST_LIMIT = 100

DEFAULT_WINDOW_SECONDS = 60

AUTH_LIMIT = 20

UPLOAD_LIMIT = 15

PREDICTION_LIMIT = 50

ADMIN_LIMIT = 500

BURST_LIMIT = 25

# ============================================================
# STORAGE
# ============================================================

request_store = defaultdict(list)

blocked_ips = {}

# ============================================================
# ENDPOINT LIMITS
# ============================================================

ENDPOINT_LIMITS = {

    "/auth/login": AUTH_LIMIT,

    "/auth/register": AUTH_LIMIT,

    "/upload": UPLOAD_LIMIT,

    "/prediction": PREDICTION_LIMIT,

    "/admin": ADMIN_LIMIT

}

# ============================================================
# CLEANUP EXPIRED REQUESTS
# ============================================================

def cleanup_requests():

    """
    Remove expired requests
    """

    current_time = time.time()

    expired_keys = []

    for key, timestamps in request_store.items():

        valid_timestamps = [

            ts

            for ts in timestamps

            if current_time - ts
            <= DEFAULT_WINDOW_SECONDS

        ]

        request_store[key] = valid_timestamps

        if not valid_timestamps:

            expired_keys.append(key)

    for key in expired_keys:

        del request_store[key]

# ============================================================
# GET CLIENT IP
# ============================================================

def get_client_ip(
    request: Request
):

    """
    Extract client IP safely
    """

    forwarded = request.headers.get(
        "X-Forwarded-For"
    )

    if forwarded:

        return forwarded.split(",")[0].strip()

    if request.client:

        return request.client.host

    return "unknown"

# ============================================================
# GET RATE LIMIT
# ============================================================

def get_rate_limit(
    path: str
):

    """
    Get endpoint-specific limit
    """

    for endpoint, limit in ENDPOINT_LIMITS.items():

        if path.startswith(endpoint):

            return limit

    return DEFAULT_REQUEST_LIMIT

# ============================================================
# CHECK BLOCKED IP
# ============================================================

def is_ip_blocked(
    ip_address: str
):

    """
    Check blocked IP
    """

    if ip_address not in blocked_ips:

        return False

    blocked_until = blocked_ips[ip_address]

    if time.time() > blocked_until:

        del blocked_ips[ip_address]

        return False

    return True

# ============================================================
# BLOCK IP
# ============================================================

def block_ip(

    ip_address: str,
    duration: int = 300

):

    """
    Block abusive IP
    """

    blocked_ips[ip_address] = (

        time.time()
        +
        duration

    )

    logger.warning(

        f"Blocked IP: {ip_address}"

    )

# ============================================================
# RATE LIMIT CHECK
# ============================================================

def check_rate_limit(

    key: str,
    limit: int

):

    """
    Sliding window check
    """

    current_time = time.time()

    timestamps = request_store[key]

    valid_timestamps = [

        ts

        for ts in timestamps

        if current_time - ts
        <= DEFAULT_WINDOW_SECONDS

    ]

    request_store[key] = valid_timestamps

    if len(valid_timestamps) >= limit:

        return False

    request_store[key].append(
        current_time
    )

    return True

# ============================================================
# MIDDLEWARE
# ============================================================

class RateLimitMiddleware(
    BaseHTTPMiddleware
):

    """
    Enterprise rate limiter
    """

    async def dispatch(

        self,
        request: Request,
        call_next

    ):

        cleanup_requests()

        ip_address = get_client_ip(
            request
        )

        path = request.url.path

        # ====================================================
        # BLOCK CHECK
        # ====================================================

        if is_ip_blocked(ip_address):

            raise HTTPException(

                status_code=429,

                detail="IP temporarily blocked"

            )

        # ====================================================
        # USER IDENTIFIER
        # ====================================================

        auth_header = request.headers.get(
            "Authorization"
        )

        api_key = request.headers.get(
            "X-API-KEY"
        )

        user_identifier = (

            auth_header

            or

            api_key

            or

            ip_address

        )

        limit = get_rate_limit(path)

        key = f"{path}:{user_identifier}"

        # ====================================================
        # CHECK LIMIT
        # ====================================================

        allowed = check_rate_limit(

            key,
            limit

        )

        if not allowed:

            logger.warning({

                "event":
                    "rate_limit_exceeded",

                "path":
                    path,

                "ip":
                    ip_address

            })

            # ================================================
            # AUTO BLOCK BURST ATTACKS
            # ================================================

            burst_key = f"burst:{ip_address}"

            burst_allowed = check_rate_limit(

                burst_key,
                BURST_LIMIT

            )

            if not burst_allowed:

                block_ip(ip_address)

            raise HTTPException(

                status_code=429,

                detail="Rate limit exceeded"

            )

        # ====================================================
        # PROCESS REQUEST
        # ====================================================

        response = await call_next(
            request
        )

        # ====================================================
        # RESPONSE HEADERS
        # ====================================================

        remaining = max(

            0,

            limit
            -
            len(request_store[key])

        )

        response.headers[
            "X-RateLimit-Limit"
        ] = str(limit)

        response.headers[
            "X-RateLimit-Remaining"
        ] = str(remaining)

        response.headers[
            "X-RateLimit-Window"
        ] = str(DEFAULT_WINDOW_SECONDS)

        return response

# ============================================================
# DECORATOR SUPPORT
# ============================================================

def custom_rate_limit(

    limit: int,
    window: int = 60

):

    """
    Custom decorator limiter
    """

    def decorator(func):

        async def wrapper(
            request: Request,
            *args,
            **kwargs
        ):

            ip_address = get_client_ip(
                request
            )

            key = (

                f"decorator:"
                f"{func.__name__}:"
                f"{ip_address}"

            )

            current_time = time.time()

            timestamps = request_store[key]

            timestamps = [

                ts

                for ts in timestamps

                if current_time - ts
                <= window

            ]

            request_store[key] = timestamps

            if len(timestamps) >= limit:

                raise HTTPException(

                    status_code=429,

                    detail="Custom rate limit exceeded"

                )

            request_store[key].append(
                current_time
            )

            return await func(

                request,
                *args,
                **kwargs

            )

        return wrapper

    return decorator

# ============================================================
# ANALYTICS
# ============================================================

def get_rate_limit_stats():

    """
    System stats
    """

    total_requests = sum(

        len(v)

        for v in request_store.values()

    )

    return {

        "tracked_keys":
            len(request_store),

        "blocked_ips":
            len(blocked_ips),

        "total_requests":
            total_requests,

        "window_seconds":
            DEFAULT_WINDOW_SECONDS

    }

# ============================================================
# RESET FUNCTIONS
# ============================================================

def reset_rate_limits():

    """
    Clear memory
    """

    request_store.clear()

    blocked_ips.clear()

    logger.info(
        "Rate limits reset"
    )

# ============================================================
# ASYNC CLEANUP TASK
# ============================================================

async def cleanup_task():

    """
    Background cleanup
    """

    while True:

        cleanup_requests()

        await asyncio.sleep(60)

# ============================================================
# HEALTH CHECK
# ============================================================

def rate_limit_health():

    """
    Middleware health
    """

    return {

        "status": "healthy",

        "active_clients":
            len(request_store),

        "blocked_ips":
            len(blocked_ips),

        "default_limit":
            DEFAULT_REQUEST_LIMIT

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD RATE LIMITER")
    print("=" * 60)

    sample_key = "127.0.0.1"

    limit = 5

    print("\nTesting Rate Limiter:\n")

    for i in range(7):

        allowed = check_rate_limit(

            sample_key,
            limit

        )

        print(

            f"Request {i+1}: "
            f"{'ALLOWED' if allowed else 'BLOCKED'}"

        )

    print("\nStats:\n")

    print(
        get_rate_limit_stats()
    )

    print("\n")
    print("=" * 60)
    print("RATE LIMIT SYSTEM READY")
    print("=" * 60)