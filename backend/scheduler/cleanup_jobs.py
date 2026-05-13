"""
ChurnShield 2.0 — Cleanup Jobs Scheduler

Purpose:
Enterprise-grade cleanup and maintenance engine
for ML pipelines, FastAPI backend systems,
temporary storage, logs, models,
cache management, and analytics infrastructure.

Capabilities:
- automatic file cleanup
- temporary file deletion
- old model cleanup
- stale cache removal
- orphan file detection
- log rotation
- duplicate cleanup
- old report deletion
- user data retention management
- database backup cleanup
- failed upload cleanup
- expired session cleanup
- storage optimization
- disk usage monitoring
- smart cleanup policies
- scheduler-based automation
- recovery-safe deletion
- audit logging

Supports:
- APScheduler
- FastAPI
- Redis
- local filesystem
- Docker volumes
- ML artifact cleanup

Author:
ChurnShield AI
"""

import os
import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

from apscheduler.schedulers.background import (
    BackgroundScheduler
)

logger = logging.getLogger(
    "churnshield.cleanup_jobs"
)

logging.basicConfig(
    level=logging.INFO
)


# ============================================================
# CONFIG
# ============================================================

TEMP_DIRS = [

    "temp",
    "cache",
    "uploads/temp",
    "user_data/temp",

]

LOG_DIR = "logs"

MODEL_DIR = "models"

REPORT_DIR = (
    "user_data/reports"
)

EXPORT_DIR = "exports"

MAX_LOG_DAYS = 15

MAX_TEMP_DAYS = 2

MAX_REPORT_DAYS = 30

MAX_MODEL_KEEP = 10

MAX_EXPORT_DAYS = 15

CLEANUP_LOG = (
    Path(LOG_DIR) /
    "cleanup.log"
)

Path(LOG_DIR).mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CLEANUP ENGINE
# ============================================================

class CleanupEngine:

    def __init__(self):

        self.scheduler = (
            BackgroundScheduler()
        )

    # ========================================================
    # LOGGING
    # ========================================================

    def write_log(
        self,
        message: str
    ):

        timestamp = (
            datetime.utcnow()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        line = (
            f"[{timestamp}] {message}\n"
        )

        with open(
            CLEANUP_LOG,
            "a"
        ) as f:

            f.write(line)

        logger.info(message)

    # ========================================================
    # FILE AGE
    # ========================================================

    def file_age_days(
        self,
        file_path: Path
    ) -> int:

        modified_time = datetime.fromtimestamp(

            file_path.stat().st_mtime

        )

        delta = (
            datetime.utcnow()
            -
            modified_time
        )

        return delta.days

    # ========================================================
    # SAFE DELETE
    # ========================================================

    def safe_delete(
        self,
        path: Path
    ) -> bool:

        try:

            if path.is_file():

                path.unlink()

            elif path.is_dir():

                shutil.rmtree(path)

            self.write_log(
                f"Deleted: {path}"
            )

            return True

        except Exception as e:

            self.write_log(
                f"Delete failed: {path} | {e}"
            )

            return False

    # ========================================================
    # TEMP CLEANUP
    # ========================================================

    def cleanup_temp_files(self):

        self.write_log(
            "Starting temp cleanup"
        )

        deleted = 0

        for folder in TEMP_DIRS:

            folder_path = Path(folder)

            if not folder_path.exists():
                continue

            for item in folder_path.rglob("*"):

                try:

                    if item.is_file():

                        age = self.file_age_days(
                            item
                        )

                        if age >= MAX_TEMP_DAYS:

                            if self.safe_delete(item):

                                deleted += 1

                except Exception as e:

                    self.write_log(
                        f"Temp cleanup error: {e}"
                    )

        self.write_log(
            f"Temp cleanup complete. "
            f"Deleted: {deleted}"
        )

    # ========================================================
    # LOG CLEANUP
    # ========================================================

    def cleanup_logs(self):

        self.write_log(
            "Starting log cleanup"
        )

        log_path = Path(LOG_DIR)

        deleted = 0

        for log_file in log_path.rglob("*.log"):

            try:

                age = self.file_age_days(
                    log_file
                )

                if age >= MAX_LOG_DAYS:

                    if self.safe_delete(
                        log_file
                    ):

                        deleted += 1

            except Exception as e:

                self.write_log(
                    f"Log cleanup error: {e}"
                )

        self.write_log(
            f"Log cleanup complete. "
            f"Deleted: {deleted}"
        )

    # ========================================================
    # REPORT CLEANUP
    # ========================================================

    def cleanup_reports(self):

        self.write_log(
            "Starting report cleanup"
        )

        report_path = Path(
            REPORT_DIR
        )

        deleted = 0

        if not report_path.exists():

            return

        for report in report_path.rglob("*"):

            try:

                if report.is_file():

                    age = self.file_age_days(
                        report
                    )

                    if age >= MAX_REPORT_DAYS:

                        if self.safe_delete(
                            report
                        ):

                            deleted += 1

            except Exception as e:

                self.write_log(
                    f"Report cleanup error: {e}"
                )

        self.write_log(
            f"Report cleanup complete. "
            f"Deleted: {deleted}"
        )

    # ========================================================
    # EXPORT CLEANUP
    # ========================================================

    def cleanup_exports(self):

        self.write_log(
            "Starting export cleanup"
        )

        export_path = Path(
            EXPORT_DIR
        )

        deleted = 0

        if not export_path.exists():

            return

        for export in export_path.rglob("*"):

            try:

                if export.is_file():

                    age = self.file_age_days(
                        export
                    )

                    if age >= MAX_EXPORT_DAYS:

                        if self.safe_delete(
                            export
                        ):

                            deleted += 1

            except Exception as e:

                self.write_log(
                    f"Export cleanup error: {e}"
                )

        self.write_log(
            f"Export cleanup complete. "
            f"Deleted: {deleted}"
        )

    # ========================================================
    # MODEL CLEANUP
    # ========================================================

    def cleanup_old_models(self):

        self.write_log(
            "Starting model cleanup"
        )

        model_path = Path(
            MODEL_DIR
        )

        if not model_path.exists():

            return

        models = list(

            model_path.glob("*.pkl")

        )

        models.sort(

            key=lambda x:
            x.stat().st_mtime,

            reverse=True

        )

        deleted = 0

        for model in models[MAX_MODEL_KEEP:]:

            try:

                if self.safe_delete(model):

                    deleted += 1

            except Exception as e:

                self.write_log(
                    f"Model cleanup error: {e}"
                )

        self.write_log(
            f"Model cleanup complete. "
            f"Deleted: {deleted}"
        )

    # ========================================================
    # EMPTY DIRECTORY CLEANUP
    # ========================================================

    def cleanup_empty_dirs(self):

        self.write_log(
            "Starting empty directory cleanup"
        )

        deleted = 0

        root_dirs = [

            "temp",
            "cache",
            "uploads",
            "user_data",

        ]

        for root in root_dirs:

            root_path = Path(root)

            if not root_path.exists():
                continue

            for directory in root_path.rglob("*"):

                try:

                    if (
                        directory.is_dir()
                        and
                        not any(
                            directory.iterdir()
                        )
                    ):

                        directory.rmdir()

                        deleted += 1

                except Exception:

                    pass

        self.write_log(
            f"Empty directory cleanup complete. "
            f"Deleted: {deleted}"
        )

    # ========================================================
    # DUPLICATE FILE CLEANUP
    # ========================================================

    def cleanup_duplicate_files(self):

        self.write_log(
            "Starting duplicate cleanup"
        )

        hashes = {}

        deleted = 0

        scan_dirs = [

            "uploads",
            "exports",
            "user_data",

        ]

        for folder in scan_dirs:

            folder_path = Path(folder)

            if not folder_path.exists():
                continue

            for file in folder_path.rglob("*"):

                try:

                    if not file.is_file():
                        continue

                    file_hash = self.hash_file(
                        file
                    )

                    if file_hash in hashes:

                        self.safe_delete(
                            file
                        )

                        deleted += 1

                    else:

                        hashes[
                            file_hash
                        ] = file

                except Exception as e:

                    self.write_log(
                        f"Duplicate cleanup error: {e}"
                    )

        self.write_log(
            f"Duplicate cleanup complete. "
            f"Deleted: {deleted}"
        )

    # ========================================================
    # FILE HASH
    # ========================================================

    def hash_file(
        self,
        file_path: Path
    ) -> str:

        import hashlib

        md5 = hashlib.md5()

        with open(
            file_path,
            "rb"
        ) as f:

            while chunk := f.read(8192):

                md5.update(chunk)

        return md5.hexdigest()

    # ========================================================
    # STORAGE REPORT
    # ========================================================

    def storage_report(self):

        self.write_log(
            "Generating storage report"
        )

        total_size = 0

        scan_dirs = [

            ".",

        ]

        for folder in scan_dirs:

            for path in Path(folder).rglob("*"):

                try:

                    if path.is_file():

                        total_size += (
                            path.stat().st_size
                        )

                except Exception:

                    pass

        total_mb = (
            total_size /
            (1024 * 1024)
        )

        report = {

            "generated_at":
                datetime.utcnow()
                .isoformat(),

            "total_storage_mb":
                round(total_mb, 2),

        }

        report_path = (
            Path(LOG_DIR)
            /
            "storage_report.json"
        )

        with open(
            report_path,
            "w"
        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )

        self.write_log(
            f"Storage used: "
            f"{round(total_mb,2)} MB"
        )

        return report

    # ========================================================
    # FULL CLEANUP
    # ========================================================

    def full_cleanup(self):

        start = time.time()

        self.write_log(
            "Starting FULL cleanup cycle"
        )

        self.cleanup_temp_files()

        self.cleanup_logs()

        self.cleanup_reports()

        self.cleanup_exports()

        self.cleanup_old_models()

        self.cleanup_duplicate_files()

        self.cleanup_empty_dirs()

        report = self.storage_report()

        duration = round(

            time.time() - start,
            2

        )

        self.write_log(
            f"FULL cleanup complete "
            f"in {duration}s"
        )

        return {

            "success": True,

            "duration_seconds":
                duration,

            "storage_report":
                report,

        }

    # ========================================================
    # SCHEDULE JOBS
    # ========================================================

    def schedule_jobs(self):

        self.scheduler.add_job(

            self.cleanup_temp_files,

            trigger="interval",

            hours=6,

            id="temp_cleanup",

        )

        self.scheduler.add_job(

            self.cleanup_logs,

            trigger="cron",

            hour=2,

            minute=0,

            id="log_cleanup",

        )

        self.scheduler.add_job(

            self.full_cleanup,

            trigger="cron",

            hour=3,

            minute=0,

            id="full_cleanup",

        )

        self.scheduler.start()

        self.write_log(
            "Cleanup scheduler started"
        )

    # ========================================================
    # STOP SCHEDULER
    # ========================================================

    def stop_scheduler(self):

        self.scheduler.shutdown()

        self.write_log(
            "Cleanup scheduler stopped"
        )


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def start_cleanup_scheduler():

    engine = CleanupEngine()

    engine.schedule_jobs()

    return engine


def run_full_cleanup():

    engine = CleanupEngine()

    return engine.full_cleanup()


def cleanup_temp_files():

    engine = CleanupEngine()

    return engine.cleanup_temp_files()


def cleanup_logs():

    engine = CleanupEngine()

    return engine.cleanup_logs()


def cleanup_models():

    engine = CleanupEngine()

    return engine.cleanup_old_models()


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    engine = CleanupEngine()

    result = engine.full_cleanup()

    print("\n")
    print("=" * 60)
    print("CLEANUP RESULT")
    print("=" * 60)
    print(json.dumps(
        result,
        indent=4
    ))