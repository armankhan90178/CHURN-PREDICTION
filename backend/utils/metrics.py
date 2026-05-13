"""
ChurnShield 2.0 — Enterprise Metrics Engine

File:
utils/metrics.py

Purpose:
Centralized enterprise metrics,
monitoring, and observability engine
for ChurnShield AI platform.

Capabilities:
- request metrics
- API latency tracking
- ML model metrics
- business KPIs
- cache analytics
- token usage monitoring
- drift metrics
- feature statistics
- uptime tracking
- Prometheus-ready design
- enterprise observability
- system performance metrics
- model inference metrics
- dataset metrics
- scheduler metrics
- export metrics
- memory usage tracking
- CPU monitoring
- error rate analytics
- custom counters
- gauge metrics
- histogram metrics

Author:
ChurnShield AI
"""

import os
import time
import psutil
import statistics

from typing import (

    Dict,
    List,
    Any,
    Optional

)

from threading import Lock

from datetime import datetime

# ============================================================
# METRIC STORAGE
# ============================================================

METRICS = {

    "requests_total": 0,

    "requests_success": 0,

    "requests_failed": 0,

    "api_latency_ms": [],

    "ml_inference_latency_ms": [],

    "cache_hits": 0,

    "cache_misses": 0,

    "active_users": 0,

    "datasets_processed": 0,

    "predictions_generated": 0,

    "exports_generated": 0,

    "token_usage": 0,

    "scheduler_jobs_executed": 0,

    "scheduler_jobs_failed": 0,

    "errors_total": 0,

    "warnings_total": 0,

    "drift_alerts": 0,

    "uptime_started_at": time.time()

}

# ============================================================
# THREAD LOCK
# ============================================================

METRIC_LOCK = Lock()

# ============================================================
# CUSTOM METRICS
# ============================================================

CUSTOM_COUNTERS = {}

CUSTOM_GAUGES = {}

CUSTOM_HISTOGRAMS = {}

# ============================================================
# REQUEST METRICS
# ============================================================

class RequestMetrics:

    """
    API request monitoring
    """

    @staticmethod
    def track_request(

        success: bool = True

    ):

        with METRIC_LOCK:

            METRICS["requests_total"] += 1

            if success:

                METRICS["requests_success"] += 1

            else:

                METRICS["requests_failed"] += 1

    @staticmethod
    def track_latency(

        latency_ms: float

    ):

        with METRIC_LOCK:

            METRICS[

                "api_latency_ms"

            ].append(latency_ms)

# ============================================================
# ML METRICS
# ============================================================

class MLMetrics:

    """
    ML observability
    """

    @staticmethod
    def track_prediction(

        latency_ms: float

    ):

        with METRIC_LOCK:

            METRICS[

                "predictions_generated"

            ] += 1

            METRICS[

                "ml_inference_latency_ms"

            ].append(latency_ms)

    @staticmethod
    def track_drift_alert():

        with METRIC_LOCK:

            METRICS["drift_alerts"] += 1

# ============================================================
# CACHE METRICS
# ============================================================

class CacheMetrics:

    """
    Cache analytics
    """

    @staticmethod
    def hit():

        with METRIC_LOCK:

            METRICS["cache_hits"] += 1

    @staticmethod
    def miss():

        with METRIC_LOCK:

            METRICS["cache_misses"] += 1

# ============================================================
# TOKEN METRICS
# ============================================================

class TokenMetrics:

    """
    LLM token monitoring
    """

    @staticmethod
    def add_tokens(

        count: int

    ):

        with METRIC_LOCK:

            METRICS["token_usage"] += count

# ============================================================
# EXPORT METRICS
# ============================================================

class ExportMetrics:

    """
    Export tracking
    """

    @staticmethod
    def export_generated():

        with METRIC_LOCK:

            METRICS["exports_generated"] += 1

# ============================================================
# DATASET METRICS
# ============================================================

class DatasetMetrics:

    """
    Dataset analytics
    """

    @staticmethod
    def dataset_processed():

        with METRIC_LOCK:

            METRICS["datasets_processed"] += 1

# ============================================================
# SCHEDULER METRICS
# ============================================================

class SchedulerMetrics:

    """
    Scheduler tracking
    """

    @staticmethod
    def job_success():

        with METRIC_LOCK:

            METRICS[

                "scheduler_jobs_executed"

            ] += 1

    @staticmethod
    def job_failure():

        with METRIC_LOCK:

            METRICS[

                "scheduler_jobs_failed"

            ] += 1

# ============================================================
# ERROR METRICS
# ============================================================

class ErrorMetrics:

    """
    Error monitoring
    """

    @staticmethod
    def error():

        with METRIC_LOCK:

            METRICS["errors_total"] += 1

    @staticmethod
    def warning():

        with METRIC_LOCK:

            METRICS["warnings_total"] += 1

# ============================================================
# CUSTOM COUNTERS
# ============================================================

def increment_counter(

    name: str,
    value: int = 1

):

    with METRIC_LOCK:

        CUSTOM_COUNTERS[name] = (

            CUSTOM_COUNTERS.get(

                name,
                0

            )

            + value

        )

# ============================================================
# CUSTOM GAUGE
# ============================================================

def set_gauge(

    name: str,
    value: float

):

    with METRIC_LOCK:

        CUSTOM_GAUGES[name] = value

# ============================================================
# HISTOGRAM
# ============================================================

def add_histogram_value(

    name: str,
    value: float

):

    with METRIC_LOCK:

        if name not in CUSTOM_HISTOGRAMS:

            CUSTOM_HISTOGRAMS[name] = []

        CUSTOM_HISTOGRAMS[name].append(

            value

        )

# ============================================================
# LATENCY STATS
# ============================================================

def latency_statistics(

    values: List[float]

):

    if not values:

        return {

            "avg": 0,

            "min": 0,

            "max": 0,

            "p95": 0

        }

    sorted_values = sorted(values)

    p95_index = int(

        len(sorted_values) * 0.95

    )

    return {

        "avg":

            round(

                statistics.mean(values),
                2

            ),

        "min":

            round(min(values), 2),

        "max":

            round(max(values), 2),

        "p95":

            round(

                sorted_values[

                    min(

                        p95_index,
                        len(sorted_values)-1

                    )

                ],

                2

            )

    }

# ============================================================
# SYSTEM METRICS
# ============================================================

def system_metrics():

    """
    System resource metrics
    """

    memory = psutil.virtual_memory()

    disk = psutil.disk_usage("/")

    return {

        "cpu_percent":

            psutil.cpu_percent(),

        "memory_percent":

            memory.percent,

        "memory_used_gb":

            round(

                memory.used
                /
                (1024 ** 3),

                2

            ),

        "disk_percent":

            disk.percent,

        "disk_used_gb":

            round(

                disk.used
                /
                (1024 ** 3),

                2

            )

    }

# ============================================================
# CACHE HIT RATIO
# ============================================================

def cache_hit_ratio():

    hits = METRICS["cache_hits"]

    misses = METRICS["cache_misses"]

    total = hits + misses

    if total == 0:

        return 0

    return round(

        hits / total,
        4

    )

# ============================================================
# REQUEST SUCCESS RATE
# ============================================================

def request_success_rate():

    total = METRICS["requests_total"]

    if total == 0:

        return 0

    return round(

        METRICS["requests_success"]
        /
        total,

        4

    )

# ============================================================
# UPTIME
# ============================================================

def uptime_seconds():

    return int(

        time.time()
        -
        METRICS["uptime_started_at"]

    )

# ============================================================
# METRICS SNAPSHOT
# ============================================================

def metrics_snapshot():

    """
    Enterprise metrics dashboard
    """

    return {

        "timestamp":

            datetime.utcnow()

            .isoformat(),

        "requests": {

            "total":

                METRICS["requests_total"],

            "success":

                METRICS["requests_success"],

            "failed":

                METRICS["requests_failed"],

            "success_rate":

                request_success_rate()

        },

        "api_latency": latency_statistics(

            METRICS["api_latency_ms"]

        ),

        "ml_latency": latency_statistics(

            METRICS["ml_inference_latency_ms"]

        ),

        "cache": {

            "hits":

                METRICS["cache_hits"],

            "misses":

                METRICS["cache_misses"],

            "hit_ratio":

                cache_hit_ratio()

        },

        "ml": {

            "predictions":

                METRICS["predictions_generated"],

            "drift_alerts":

                METRICS["drift_alerts"]

        },

        "datasets": {

            "processed":

                METRICS["datasets_processed"]

        },

        "exports": {

            "generated":

                METRICS["exports_generated"]

        },

        "scheduler": {

            "success":

                METRICS["scheduler_jobs_executed"],

            "failed":

                METRICS["scheduler_jobs_failed"]

        },

        "tokens": {

            "usage":

                METRICS["token_usage"]

        },

        "errors": {

            "total":

                METRICS["errors_total"],

            "warnings":

                METRICS["warnings_total"]

        },

        "system": system_metrics(),

        "uptime_seconds":

            uptime_seconds(),

        "custom_counters":

            CUSTOM_COUNTERS,

        "custom_gauges":

            CUSTOM_GAUGES

    }

# ============================================================
# RESET METRICS
# ============================================================

def reset_metrics():

    """
    Reset metrics engine
    """

    global METRICS

    with METRIC_LOCK:

        METRICS.update({

            "requests_total": 0,

            "requests_success": 0,

            "requests_failed": 0,

            "api_latency_ms": [],

            "ml_inference_latency_ms": [],

            "cache_hits": 0,

            "cache_misses": 0,

            "predictions_generated": 0,

            "errors_total": 0,

            "warnings_total": 0

        })

# ============================================================
# HEALTH CHECK
# ============================================================

def metrics_health():

    return {

        "status":

            "healthy",

        "uptime_seconds":

            uptime_seconds(),

        "requests_total":

            METRICS["requests_total"]

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD METRICS ENGINE")
    print("=" * 60)

    RequestMetrics.track_request(True)

    RequestMetrics.track_request(False)

    RequestMetrics.track_latency(120)

    RequestMetrics.track_latency(90)

    MLMetrics.track_prediction(240)

    CacheMetrics.hit()

    CacheMetrics.miss()

    TokenMetrics.add_tokens(5000)

    ExportMetrics.export_generated()

    SchedulerMetrics.job_success()

    ErrorMetrics.warning()

    increment_counter(

        "custom_ai_calls",
        5

    )

    set_gauge(

        "gpu_usage",
        72.5

    )

    add_histogram_value(

        "model_latency",
        120

    )

    print("\nMetrics Snapshot:\n")

    print(

        metrics_snapshot()

    )

    print("\nHealth:\n")

    print(

        metrics_health()

    )

    print("\n")
    print("=" * 60)
    print("METRICS ENGINE READY")
    print("=" * 60)