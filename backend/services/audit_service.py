"""
ChurnShield 2.0 — Audit Service

File:
services/audit_service.py

Purpose:
Enterprise-grade audit logging and
activity tracking system for ChurnShield AI.

Capabilities:
- user activity tracking
- admin action logging
- authentication audit
- API audit trails
- model operation logging
- prediction tracking
- export/download tracking
- security event logging
- file upload audit
- analytics tracking
- compliance-ready architecture
- real-time event monitoring
- enterprise traceability
- JSON structured logging

Author:
ChurnShield AI
"""

import os
import json
import uuid
import logging

from pathlib import Path
from datetime import datetime
from typing import (

    Dict,
    Any,
    Optional,
    List

)

# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(
    "audit_service"
)

logging.basicConfig(
    level=logging.INFO
)

# ============================================================
# AUDIT DIRECTORY
# ============================================================

AUDIT_DIR = Path("logs/audit")

AUDIT_DIR.mkdir(

    parents=True,
    exist_ok=True

)

# ============================================================
# AUDIT FILES
# ============================================================

MAIN_AUDIT_FILE = AUDIT_DIR / "audit_log.jsonl"

SECURITY_AUDIT_FILE = AUDIT_DIR / "security_log.jsonl"

MODEL_AUDIT_FILE = AUDIT_DIR / "model_log.jsonl"

API_AUDIT_FILE = AUDIT_DIR / "api_log.jsonl"

# ============================================================
# AUDIT LEVELS
# ============================================================

AUDIT_LEVELS = {

    "INFO",
    "WARNING",
    "CRITICAL",
    "SECURITY"

}

# ============================================================
# AUDIT SERVICE
# ============================================================

class AuditService:

    """
    Enterprise audit management system
    """

    # ========================================================
    # WRITE LOG
    # ========================================================

    @staticmethod
    def write_log(

        event_type: str,
        data: Dict,
        file_path: Path = MAIN_AUDIT_FILE

    ):

        try:

            payload = {

                "audit_id":
                    str(uuid.uuid4()),

                "event_type":
                    event_type,

                "timestamp":

                    datetime.utcnow()

                    .isoformat(),

                "data":
                    data

            }

            with open(

                file_path,
                "a",
                encoding="utf-8"

            ) as f:

                f.write(

                    json.dumps(
                        payload,
                        ensure_ascii=False
                    )

                    + "\n"

                )

            logger.info({

                "event":
                    "audit_logged",

                "type":
                    event_type

            })

            return True

        except Exception as e:

            logger.error({

                "event":
                    "audit_failed",

                "error":
                    str(e)

            })

            return False

    # ========================================================
    # USER ACTION
    # ========================================================

    @staticmethod
    def log_user_action(

        user_id: str,
        action: str,
        metadata: Optional[Dict] = None

    ):

        payload = {

            "user_id":
                user_id,

            "action":
                action,

            "metadata":
                metadata or {}

        }

        return AuditService.write_log(

            event_type="user_action",

            data=payload

        )

    # ========================================================
    # LOGIN EVENT
    # ========================================================

    @staticmethod
    def log_login(

        user_id: str,
        ip_address: str,
        success: bool

    ):

        payload = {

            "user_id":
                user_id,

            "ip_address":
                ip_address,

            "success":
                success

        }

        return AuditService.write_log(

            event_type="login_attempt",

            data=payload,

            file_path=SECURITY_AUDIT_FILE

        )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    @staticmethod
    def log_upload(

        user_id: str,
        filename: str,
        file_size_mb: float

    ):

        payload = {

            "user_id":
                user_id,

            "filename":
                filename,

            "file_size_mb":
                file_size_mb

        }

        return AuditService.write_log(

            event_type="file_upload",

            data=payload

        )

    # ========================================================
    # MODEL EVENT
    # ========================================================

    @staticmethod
    def log_model_event(

        model_name: str,
        action: str,
        metadata: Optional[Dict] = None

    ):

        payload = {

            "model_name":
                model_name,

            "action":
                action,

            "metadata":
                metadata or {}

        }

        return AuditService.write_log(

            event_type="model_event",

            data=payload,

            file_path=MODEL_AUDIT_FILE

        )

    # ========================================================
    # PREDICTION EVENT
    # ========================================================

    @staticmethod
    def log_prediction(

        user_id: str,
        model_name: str,
        prediction_count: int

    ):

        payload = {

            "user_id":
                user_id,

            "model_name":
                model_name,

            "prediction_count":
                prediction_count

        }

        return AuditService.write_log(

            event_type="prediction_generated",

            data=payload

        )

    # ========================================================
    # EXPORT EVENT
    # ========================================================

    @staticmethod
    def log_export(

        user_id: str,
        export_type: str,
        filename: str

    ):

        payload = {

            "user_id":
                user_id,

            "export_type":
                export_type,

            "filename":
                filename

        }

        return AuditService.write_log(

            event_type="data_export",

            data=payload

        )

    # ========================================================
    # API REQUEST
    # ========================================================

    @staticmethod
    def log_api_request(

        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float

    ):

        payload = {

            "endpoint":
                endpoint,

            "method":
                method,

            "status_code":
                status_code,

            "response_time_ms":
                response_time_ms

        }

        return AuditService.write_log(

            event_type="api_request",

            data=payload,

            file_path=API_AUDIT_FILE

        )

    # ========================================================
    # SECURITY EVENT
    # ========================================================

    @staticmethod
    def log_security_event(

        event_name: str,
        severity: str,
        details: Dict

    ):

        payload = {

            "event_name":
                event_name,

            "severity":
                severity,

            "details":
                details

        }

        return AuditService.write_log(

            event_type="security_event",

            data=payload,

            file_path=SECURITY_AUDIT_FILE

        )

    # ========================================================
    # ADMIN ACTION
    # ========================================================

    @staticmethod
    def log_admin_action(

        admin_id: str,
        action: str,
        target: str

    ):

        payload = {

            "admin_id":
                admin_id,

            "action":
                action,

            "target":
                target

        }

        return AuditService.write_log(

            event_type="admin_action",

            data=payload

        )

# ============================================================
# AUDIT ANALYTICS
# ============================================================

def audit_statistics():

    """
    Generate audit statistics
    """

    files = [

        MAIN_AUDIT_FILE,

        SECURITY_AUDIT_FILE,

        MODEL_AUDIT_FILE,

        API_AUDIT_FILE

    ]

    stats = {}

    for file in files:

        if file.exists():

            with open(

                file,
                "r",
                encoding="utf-8"

            ) as f:

                lines = f.readlines()

            stats[file.name] = {

                "entries":
                    len(lines),

                "size_mb":

                    round(

                        file.stat().st_size
                        /
                        (1024 * 1024),

                        4

                    )

            }

        else:

            stats[file.name] = {

                "entries": 0,

                "size_mb": 0

            }

    return stats

# ============================================================
# SEARCH AUDIT LOGS
# ============================================================

def search_audit_logs(

    keyword: str,
    limit: int = 50

) -> List[Dict]:

    """
    Search audit records
    """

    results = []

    files = [

        MAIN_AUDIT_FILE,

        SECURITY_AUDIT_FILE,

        MODEL_AUDIT_FILE,

        API_AUDIT_FILE

    ]

    for file in files:

        if not file.exists():

            continue

        with open(

            file,
            "r",
            encoding="utf-8"

        ) as f:

            for line in f:

                if keyword.lower() in line.lower():

                    try:

                        results.append(

                            json.loads(line)

                        )

                    except Exception:

                        continue

    return results[:limit]

# ============================================================
# CLEAN OLD LOGS
# ============================================================

def cleanup_old_logs(
    max_size_mb: int = 100
):

    """
    Cleanup oversized logs
    """

    deleted = []

    for file in AUDIT_DIR.glob("*.jsonl"):

        size_mb = (

            file.stat().st_size
            /
            (1024 * 1024)

        )

        if size_mb > max_size_mb:

            backup_name = (

                file.stem
                +
                "_backup_"
                +
                datetime.utcnow().strftime(
                    "%Y%m%d%H%M%S"
                )
                +
                ".jsonl"

            )

            backup_path = AUDIT_DIR / backup_name

            file.rename(backup_path)

            deleted.append(file.name)

    return {

        "cleaned":
            deleted

    }

# ============================================================
# AUDIT HEALTH
# ============================================================

def audit_health():

    return {

        "status": "healthy",

        "audit_directory":
            str(AUDIT_DIR),

        "main_log_exists":
            MAIN_AUDIT_FILE.exists(),

        "security_log_exists":
            SECURITY_AUDIT_FILE.exists(),

        "model_log_exists":
            MODEL_AUDIT_FILE.exists(),

        "api_log_exists":
            API_AUDIT_FILE.exists()

    }

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("CHURNSHIELD AUDIT SERVICE")
    print("=" * 60)

    AuditService.log_user_action(

        user_id="user_101",

        action="uploaded_dataset",

        metadata={

            "dataset": "telecom.csv"

        }

    )

    AuditService.log_login(

        user_id="admin_001",

        ip_address="192.168.1.1",

        success=True

    )

    AuditService.log_prediction(

        user_id="user_101",

        model_name="global_churn_model",

        prediction_count=1200

    )

    print("\nAudit Statistics:\n")

    print(audit_statistics())

    print("\nHealth Status:\n")

    print(audit_health())

    print("\nSearch Results:\n")

    print(

        search_audit_logs(
            "uploaded"
        )

    )

    print("\n")
    print("=" * 60)
    print("AUDIT SERVICE READY")
    print("=" * 60)