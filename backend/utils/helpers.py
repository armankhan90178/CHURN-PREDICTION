"""
ChurnShield 2.0 — Enterprise Helper Utilities

File:
utils/helpers.py

Purpose:
Centralized helper utilities
for ChurnShield AI platform.

Capabilities:
- UUID generation
- timestamp helpers
- async retry handling
- exponential backoff
- safe JSON parsing
- response builders
- pagination helpers
- file utilities
- memory formatting
- execution timing
- chunk processing
- validation helpers
- retry wrappers
- environment utilities
- deep dictionary merging
- flatten/unflatten utilities
- hash generation
- data normalization
- batching engine
- enterprise utility toolkit

Author:
ChurnShield AI
"""

import os
import re
import gc
import uuid
import json
import math
import time
import hashlib
import asyncio
import traceback

from typing import (

    Dict,
    List,
    Any,
    Optional,
    Callable,
    Iterable

)

from pathlib import Path

from datetime import (

    datetime,
    timezone

)

from functools import wraps

# ============================================================
# UUID HELPERS
# ============================================================

def generate_uuid():

    """
    Generate UUID4
    """

    return str(

        uuid.uuid4()

    )

def generate_short_uuid():

    """
    Short UUID
    """

    return str(

        uuid.uuid4()

    )[:8]

# ============================================================
# TIMESTAMP HELPERS
# ============================================================

def utc_now():

    """
    UTC datetime
    """

    return datetime.now(

        timezone.utc

    )

def utc_timestamp():

    """
    UTC timestamp
    """

    return int(

        time.time()

    )

def iso_timestamp():

    """
    ISO datetime string
    """

    return utc_now().isoformat()

# ============================================================
# EXECUTION TIMER
# ============================================================

def execution_timer():

    """
    Execution timer decorator
    """

    def decorator(func):

        @wraps(func)
        def wrapper(

            *args,
            **kwargs

        ):

            start = time.perf_counter()

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

            print(

                f"[TIMER] {func.__name__} "
                f"executed in {duration}ms"

            )

            return result

        return wrapper

    return decorator

# ============================================================
# ASYNC RETRY
# ============================================================

async def async_retry(

    func: Callable,
    retries: int = 3,
    delay: int = 1,
    backoff: int = 2,
    exceptions=(Exception,),
    *args,
    **kwargs

):

    """
    Async retry engine
    """

    current_delay = delay

    for attempt in range(

        retries

    ):

        try:

            return await func(

                *args,
                **kwargs

            )

        except exceptions:

            if attempt == retries - 1:

                raise

            await asyncio.sleep(

                current_delay

            )

            current_delay *= backoff

# ============================================================
# SYNC RETRY
# ============================================================

def retry(

    retries: int = 3,
    delay: int = 1,
    backoff: int = 2,
    exceptions=(Exception,)

):

    """
    Retry decorator
    """

    def decorator(func):

        @wraps(func)
        def wrapper(

            *args,
            **kwargs

        ):

            current_delay = delay

            for attempt in range(

                retries

            ):

                try:

                    return func(

                        *args,
                        **kwargs

                    )

                except exceptions:

                    if attempt == retries - 1:

                        raise

                    time.sleep(

                        current_delay

                    )

                    current_delay *= backoff

        return wrapper

    return decorator

# ============================================================
# SAFE JSON
# ============================================================

def safe_json_loads(

    data: str,
    default=None

):

    """
    Safe JSON parser
    """

    try:

        return json.loads(data)

    except Exception:

        return default

def safe_json_dumps(

    data: Any,
    default=None

):

    """
    Safe JSON serializer
    """

    try:

        return json.dumps(

            data,
            default=str

        )

    except Exception:

        return default

# ============================================================
# HASHING HELPERS
# ============================================================

def sha256_hash(

    value: str

):

    """
    SHA256 hash
    """

    return hashlib.sha256(

        value.encode()

    ).hexdigest()

def md5_hash(

    value: str

):

    """
    MD5 hash
    """

    return hashlib.md5(

        value.encode()

    ).hexdigest()

# ============================================================
# PAGINATION
# ============================================================

def paginate(

    items: List,
    page: int = 1,
    page_size: int = 50

):

    """
    Pagination helper
    """

    total = len(items)

    start = (

        page - 1

    ) * page_size

    end = start + page_size

    return {

        "page":

            page,

        "page_size":

            page_size,

        "total":

            total,

        "total_pages":

            math.ceil(

                total / page_size

            ),

        "items":

            items[start:end]

    }

# ============================================================
# RESPONSE BUILDERS
# ============================================================

def success_response(

    data=None,
    message="Success"

):

    """
    Standard success response
    """

    return {

        "success":

            True,

        "message":

            message,

        "timestamp":

            iso_timestamp(),

        "data":

            data

    }

def error_response(

    message="Error",
    errors=None

):

    """
    Standard error response
    """

    return {

        "success":

            False,

        "message":

            message,

        "timestamp":

            iso_timestamp(),

        "errors":

            errors

    }

# ============================================================
# FILE HELPERS
# ============================================================

def ensure_directory(

    path: str

):

    """
    Create directory safely
    """

    Path(path).mkdir(

        parents=True,
        exist_ok=True

    )

def file_exists(

    path: str

):

    return Path(path).exists()

def file_size_mb(

    path: str

):

    """
    File size in MB
    """

    return round(

        os.path.getsize(path)
        /
        (1024 * 1024),

        2

    )

# ============================================================
# CHUNKING
# ============================================================

def chunk_list(

    data: List,
    chunk_size: int

):

    """
    Split list into chunks
    """

    for i in range(

        0,
        len(data),
        chunk_size

    ):

        yield data[

            i:i + chunk_size

        ]

# ============================================================
# MEMORY UTILITIES
# ============================================================

def force_garbage_collection():

    """
    Force memory cleanup
    """

    gc.collect()

def bytes_to_human(

    size_bytes: int

):

    """
    Human-readable memory
    """

    if size_bytes == 0:

        return "0B"

    size_names = (

        "B",
        "KB",
        "MB",
        "GB",
        "TB"

    )

    i = int(

        math.floor(

            math.log(

                size_bytes,
                1024

            )

        )

    )

    power = math.pow(

        1024,
        i

    )

    size = round(

        size_bytes / power,
        2

    )

    return f"{size} {size_names[i]}"

# ============================================================
# STRING NORMALIZATION
# ============================================================

def normalize_text(

    text: str

):

    """
    Normalize strings
    """

    text = text.lower()

    text = re.sub(

        r"\s+",
        " ",
        text

    )

    text = re.sub(

        r"[^a-zA-Z0-9 ]",
        "",
        text

    )

    return text.strip()

# ============================================================
# EMAIL VALIDATION
# ============================================================

def is_valid_email(

    email: str

):

    """
    Email validation
    """

    pattern = (

        r"^[\w\.-]+@[\w\.-]+\.\w+$"

    )

    return bool(

        re.match(

            pattern,
            email

        )

    )

# ============================================================
# FLATTEN DICTIONARY
# ============================================================

def flatten_dict(

    d: Dict,
    parent_key="",
    sep="."

):

    """
    Flatten nested dict
    """

    items = []

    for key, value in d.items():

        new_key = (

            f"{parent_key}{sep}{key}"

            if parent_key

            else key

        )

        if isinstance(

            value,
            dict

        ):

            items.extend(

                flatten_dict(

                    value,
                    new_key,
                    sep

                ).items()

            )

        else:

            items.append(

                (new_key, value)

            )

    return dict(items)

# ============================================================
# DEEP MERGE
# ============================================================

def deep_merge(

    source: Dict,
    destination: Dict

):

    """
    Deep dictionary merge
    """

    for key, value in source.items():

        if isinstance(

            value,
            dict

        ):

            node = destination.setdefault(

                key,
                {}

            )

            deep_merge(

                value,
                node

            )

        else:

            destination[key] = value

    return destination

# ============================================================
# ENVIRONMENT HELPERS
# ============================================================

def get_env(

    key: str,
    default=None,
    cast_type=None

):

    """
    Safe environment loader
    """

    value = os.getenv(

        key,
        default

    )

    if cast_type and value:

        try:

            value = cast_type(

                value

            )

        except Exception:

            return default

    return value

# ============================================================
# TRACEBACK HELPER
# ============================================================

def traceback_string():

    """
    Get traceback as string
    """

    return traceback.format_exc()

# ============================================================
# BATCH PROCESSOR
# ============================================================

def batch_processor(

    iterable: Iterable,
    batch_size: int

):

    """
    Batch iterator
    """

    batch = []

    for item in iterable:

        batch.append(item)

        if len(batch) >= batch_size:

            yield batch

            batch = []

    if batch:

        yield batch

# ============================================================
# HEALTH CHECK
# ============================================================

def helpers_health():

    return {

        "status":

            "healthy",

        "timestamp":

            iso_timestamp()

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD HELPERS")
    print("=" * 60)

    print("\nUUID:\n")

    print(

        generate_uuid()

    )

    print("\nTimestamp:\n")

    print(

        iso_timestamp()

    )

    print("\nSHA256:\n")

    print(

        sha256_hash(

            "hello"

        )

    )

    print("\nPagination:\n")

    print(

        paginate(

            list(range(100)),
            page=2,
            page_size=10

        )

    )

    print("\nFlatten Dict:\n")

    print(

        flatten_dict({

            "a": {

                "b": {

                    "c": 1

                }

            }

        })

    )

    print("\nHealth:\n")

    print(

        helpers_health()

    )

    print("\n")
    print("=" * 60)
    print("HELPERS READY")
    print("=" * 60)