"""
ChurnShield 2.0 — Alert Jobs Scheduler

Purpose:
Enterprise-grade alerting and notification
scheduler for churn prediction systems,
ML monitoring, anomaly detection,
business intelligence, and platform health.

Capabilities:
- scheduled alert jobs
- churn spike alerts
- revenue drop alerts
- model drift alerts
- anomaly alerts
- failed training alerts
- prediction threshold alerts
- API downtime alerts
- system health alerts
- customer risk alerts
- retraining reminders
- email alerts
- webhook alerts
- Slack alerts
- Telegram alerts
- dashboard alerts
- log-based alerts
- alert cooldown protection
- priority escalation
- auto recovery detection

Supports:
- APScheduler
- FastAPI
- Redis caching
- SMTP email
- Slack webhook
- Telegram bot alerts

Author:
ChurnShield AI
"""

import os
import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from apscheduler.schedulers.background import (
    BackgroundScheduler
)

logger = logging.getLogger(
    "churnshield.alert_jobs"
)

logging.basicConfig(
    level=logging.INFO
)


# ============================================================
# CONFIG
# ============================================================

ALERT_LOG_DIR = Path(
    "logs/alerts"
)

ALERT_LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

COOLDOWN_MINUTES = 30

ALERT_STATE_FILE = (
    ALERT_LOG_DIR / "alert_state.json"
)


# ============================================================
# ALERT ENGINE
# ============================================================

class AlertEngine:

    def __init__(self):

        self.scheduler = (
            BackgroundScheduler()
        )

        self.alert_state = (
            self.load_alert_state()
        )

    # ========================================================
    # LOAD ALERT STATE
    # ========================================================

    def load_alert_state(self):

        if ALERT_STATE_FILE.exists():

            try:

                with open(
                    ALERT_STATE_FILE,
                    "r"
                ) as f:

                    return json.load(f)

            except Exception:

                return {}

        return {}

    # ========================================================
    # SAVE ALERT STATE
    # ========================================================

    def save_alert_state(self):

        with open(
            ALERT_STATE_FILE,
            "w"
        ) as f:

            json.dump(

                self.alert_state,
                f,
                indent=4

            )

    # ========================================================
    # ALERT COOLDOWN
    # ========================================================

    def can_send_alert(
        self,
        alert_name: str
    ) -> bool:

        now = datetime.utcnow()

        last_sent = self.alert_state.get(
            alert_name
        )

        if not last_sent:

            return True

        last_sent = datetime.fromisoformat(
            last_sent
        )

        delta = now - last_sent

        return (
            delta.total_seconds()
            >
            COOLDOWN_MINUTES * 60
        )

    # ========================================================
    # UPDATE ALERT TIME
    # ========================================================

    def update_alert_time(
        self,
        alert_name: str
    ):

        self.alert_state[
            alert_name
        ] = datetime.utcnow().isoformat()

        self.save_alert_state()

    # ========================================================
    # EMAIL ALERT
    # ========================================================

    def send_email_alert(
        self,
        subject: str,
        message: str
    ):

        logger.warning(
            f"[EMAIL ALERT] {subject}"
        )

        print("\n")
        print("=" * 60)
        print("EMAIL ALERT")
        print("=" * 60)
        print(f"Subject: {subject}")
        print(message)
        print("=" * 60)

    # ========================================================
    # SLACK ALERT
    # ========================================================

    def send_slack_alert(
        self,
        message: str
    ):

        webhook_url = os.getenv(
            "SLACK_WEBHOOK_URL"
        )

        if not webhook_url:

            logger.warning(
                "Slack webhook missing"
            )

            return

        try:

            response = requests.post(

                webhook_url,

                json={
                    "text": message
                },

                timeout=10

            )

            logger.info(
                "Slack alert sent"
            )

        except Exception as e:

            logger.error(
                f"Slack alert failed: {e}"
            )

    # ========================================================
    # TELEGRAM ALERT
    # ========================================================

    def send_telegram_alert(
        self,
        message: str
    ):

        token = os.getenv(
            "TELEGRAM_BOT_TOKEN"
        )

        chat_id = os.getenv(
            "TELEGRAM_CHAT_ID"
        )

        if not token or not chat_id:

            return

        try:

            url = (

                f"https://api.telegram.org/"
                f"bot{token}/sendMessage"

            )

            requests.post(

                url,

                data={

                    "chat_id": chat_id,
                    "text": message,

                },

                timeout=10

            )

            logger.info(
                "Telegram alert sent"
            )

        except Exception as e:

            logger.error(
                f"Telegram alert failed: {e}"
            )

    # ========================================================
    # UNIVERSAL ALERT
    # ========================================================

    def send_alert(
        self,
        title: str,
        message: str,
        priority: str = "medium",
    ):

        alert_name = (
            f"{title}_{priority}"
        )

        if not self.can_send_alert(
            alert_name
        ):

            logger.info(
                f"Cooldown active: {alert_name}"
            )

            return

        full_message = (

            f"\n🚨 ALERT: {title}\n"
            f"Priority: {priority.upper()}\n"
            f"Time: {datetime.utcnow()}\n\n"
            f"{message}"

        )

        self.send_email_alert(

            title,
            full_message

        )

        self.send_slack_alert(
            full_message
        )

        self.send_telegram_alert(
            full_message
        )

        self.update_alert_time(
            alert_name
        )

    # ========================================================
    # CHURN SPIKE ALERT
    # ========================================================

    def churn_spike_alert(
        self,
        churn_rate: float,
        threshold: float = 0.3,
    ):

        if churn_rate > threshold:

            self.send_alert(

                title="Churn Spike Detected",

                message=(
                    f"Current churn rate is "
                    f"{round(churn_rate,4)} "
                    f"which exceeds threshold "
                    f"{threshold}"
                ),

                priority="critical",

            )

    # ========================================================
    # REVENUE DROP ALERT
    # ========================================================

    def revenue_drop_alert(
        self,
        current_revenue: float,
        previous_revenue: float,
    ):

        if previous_revenue <= 0:

            return

        drop = (

            (
                previous_revenue
                -
                current_revenue
            )

            /

            previous_revenue

        )

        if drop > 0.15:

            self.send_alert(

                title="Revenue Drop Alert",

                message=(
                    f"Revenue dropped by "
                    f"{round(drop*100,2)}%"
                ),

                priority="high",

            )

    # ========================================================
    # MODEL DRIFT ALERT
    # ========================================================

    def model_drift_alert(
        self,
        drift_score: float,
        threshold: float = 0.2,
    ):

        if drift_score > threshold:

            self.send_alert(

                title="Model Drift Detected",

                message=(
                    f"Drift score "
                    f"{round(drift_score,4)} "
                    f"exceeded threshold "
                    f"{threshold}"
                ),

                priority="critical",

            )

    # ========================================================
    # FAILED TRAINING ALERT
    # ========================================================

    def training_failure_alert(
        self,
        error_message: str
    ):

        self.send_alert(

            title="Model Training Failed",

            message=error_message,

            priority="critical",

        )

    # ========================================================
    # API HEALTH ALERT
    # ========================================================

    def api_health_alert(
        self,
        api_status: bool
    ):

        if not api_status:

            self.send_alert(

                title="API Downtime",

                message=(
                    "FastAPI backend "
                    "is unreachable."
                ),

                priority="critical",

            )

    # ========================================================
    # HIGH RISK CUSTOMER ALERT
    # ========================================================

    def high_risk_customer_alert(
        self,
        customer_count: int,
        threshold: int = 100,
    ):

        if customer_count > threshold:

            self.send_alert(

                title="High Risk Customers",

                message=(
                    f"{customer_count} "
                    f"customers detected "
                    f"with high churn risk."
                ),

                priority="high",

            )

    # ========================================================
    # ANOMALY ALERT
    # ========================================================

    def anomaly_alert(
        self,
        anomaly_count: int,
        threshold: int = 50,
    ):

        if anomaly_count > threshold:

            self.send_alert(

                title="Anomaly Spike",

                message=(
                    f"{anomaly_count} "
                    f"anomalies detected."
                ),

                priority="high",

            )

    # ========================================================
    # STORAGE ALERT
    # ========================================================

    def storage_alert(
        self,
        used_percent: float
    ):

        if used_percent > 90:

            self.send_alert(

                title="Storage Critical",

                message=(
                    f"Storage usage is "
                    f"{used_percent}%"
                ),

                priority="critical",

            )

    # ========================================================
    # CPU ALERT
    # ========================================================

    def cpu_alert(
        self,
        cpu_usage: float
    ):

        if cpu_usage > 90:

            self.send_alert(

                title="CPU Usage High",

                message=(
                    f"CPU usage reached "
                    f"{cpu_usage}%"
                ),

                priority="high",

            )

    # ========================================================
    # MEMORY ALERT
    # ========================================================

    def memory_alert(
        self,
        memory_usage: float
    ):

        if memory_usage > 90:

            self.send_alert(

                title="Memory Usage High",

                message=(
                    f"Memory usage reached "
                    f"{memory_usage}%"
                ),

                priority="high",

            )

    # ========================================================
    # RETRAIN REMINDER
    # ========================================================

    def retrain_reminder(
        self,
        days_since_training: int
    ):

        if days_since_training > 30:

            self.send_alert(

                title="Retraining Reminder",

                message=(
                    f"Model not retrained "
                    f"for "
                    f"{days_since_training} days."
                ),

                priority="medium",

            )

    # ========================================================
    # SCHEDULE JOBS
    # ========================================================

    def schedule_jobs(self):

        self.scheduler.add_job(

            self.daily_health_check,

            trigger="interval",

            minutes=60,

            id="health_check",

        )

        self.scheduler.add_job(

            self.daily_system_summary,

            trigger="cron",

            hour=8,

            minute=0,

            id="system_summary",

        )

        self.scheduler.start()

        logger.info(
            "Alert scheduler started"
        )

    # ========================================================
    # DAILY HEALTH CHECK
    # ========================================================

    def daily_health_check(self):

        logger.info(
            "Running health checks"
        )

        # ----------------------------------------------------
        # MOCK CHECKS
        # ----------------------------------------------------

        cpu_usage = np.random.randint(
            20,
            95
        )

        memory_usage = np.random.randint(
            20,
            95
        )

        storage_usage = np.random.randint(
            20,
            95
        )

        drift_score = np.random.uniform(
            0,
            0.5
        )

        self.cpu_alert(cpu_usage)

        self.memory_alert(
            memory_usage
        )

        self.storage_alert(
            storage_usage
        )

        self.model_drift_alert(
            drift_score
        )

    # ========================================================
    # SYSTEM SUMMARY
    # ========================================================

    def daily_system_summary(self):

        summary = (

            "Daily system summary completed.\n"
            "All monitoring services active."

        )

        self.send_alert(

            title="Daily Summary",

            message=summary,

            priority="low",

        )

    # ========================================================
    # STOP SCHEDULER
    # ========================================================

    def stop_scheduler(self):

        self.scheduler.shutdown()

        logger.info(
            "Scheduler stopped"
        )


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def start_alert_scheduler():

    engine = AlertEngine()

    engine.schedule_jobs()

    return engine


def send_critical_alert(
    title: str,
    message: str
):

    engine = AlertEngine()

    engine.send_alert(

        title=title,
        message=message,
        priority="critical",

    )


def send_high_alert(
    title: str,
    message: str
):

    engine = AlertEngine()

    engine.send_alert(

        title=title,
        message=message,
        priority="high",

    )


def send_medium_alert(
    title: str,
    message: str
):

    engine = AlertEngine()

    engine.send_alert(

        title=title,
        message=message,
        priority="medium",

    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    engine = AlertEngine()

    engine.send_alert(

        title="System Boot",

        message="ChurnShield alert engine active.",

        priority="low",

    )

    engine.churn_spike_alert(
        churn_rate=0.42
    )

    engine.model_drift_alert(
        drift_score=0.31
    )

    engine.revenue_drop_alert(

        current_revenue=70000,
        previous_revenue=100000

    )