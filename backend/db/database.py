"""
ChurnShield 2.0 — Database Layer
SQLite connection, table creation, and seed data.
All database operations across the app use this module.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Generator
from config import DB_PATH, INDIA_CALENDAR

logger = logging.getLogger("churnshield.db")


# ─────────────────────────────────────────────
# CONNECTION MANAGEMENT
# ─────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """
    Returns a SQLite connection with row factory set so
    results come back as dict-like objects, not plain tuples.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row          # Access columns by name
    conn.execute("PRAGMA journal_mode=WAL") # Better concurrency
    conn.execute("PRAGMA foreign_keys=ON")  # Enforce FK constraints
    return conn


def get_db() -> Generator:
    """
    FastAPI dependency — yields a DB connection and closes it after use.
    Usage in route: db: sqlite3.Connection = Depends(get_db)
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


# ─────────────────────────────────────────────
# DATABASE INITIALIZATION — CREATE ALL TABLES
# ─────────────────────────────────────────────

SCHEMA_SQL = """

-- ─────────────────────────────────────────────
-- SESSIONS — Each analysis run gets a session
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    mode            TEXT NOT NULL,        -- 'search' or 'upload'
    field_name      TEXT,                 -- 'Netflix', 'Gym', etc
    industry        TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    status          TEXT DEFAULT 'pending', -- pending/processing/complete/failed
    row_count       INTEGER DEFAULT 0,
    model_accuracy  REAL DEFAULT 0.0,
    error_message   TEXT
);

-- ─────────────────────────────────────────────
-- CACHED DATASETS — Avoid re-fetching same industry
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cached_datasets (
    field_name      TEXT PRIMARY KEY,
    search_query    TEXT,
    source          TEXT,                 -- 'kaggle', 'scrape', 'llm_generated'
    row_count       INTEGER,
    csv_path        TEXT,
    cached_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at      DATETIME,
    model_path      TEXT                  -- pre-trained model for this field
);

-- ─────────────────────────────────────────────
-- MODEL REGISTRY — Track all trained models
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    model_id        TEXT PRIMARY KEY,
    session_id      TEXT,
    field_name      TEXT,
    model_path      TEXT NOT NULL,
    feature_cols    TEXT,                 -- JSON array of feature names
    auc_score       REAL,
    precision_score REAL,
    recall_score    REAL,
    row_count       INTEGER,
    trained_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- ─────────────────────────────────────────────
-- CUSTOMERS — Every customer from every session
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    id                      TEXT PRIMARY KEY,
    session_id              TEXT NOT NULL,
    customer_name           TEXT,
    industry                TEXT,
    plan_type               TEXT,
    monthly_revenue         INTEGER DEFAULT 0,
    contract_age_months     INTEGER DEFAULT 0,
    region                  TEXT,
    city                    TEXT,
    persona                 TEXT,
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- ─────────────────────────────────────────────
-- BEHAVIORAL SIGNALS — Usage data per customer
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS behavioral_signals (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id             TEXT NOT NULL,
    recorded_date           DATE DEFAULT CURRENT_DATE,
    login_count_30d         INTEGER DEFAULT 0,
    feature_usage_score     REAL DEFAULT 0.0,
    support_tickets         INTEGER DEFAULT 0,
    days_since_last_login   INTEGER DEFAULT 0,
    payment_delays          INTEGER DEFAULT 0,
    active_seats            INTEGER DEFAULT 1,
    total_seats             INTEGER DEFAULT 1,
    nps_score               REAL,
    sentiment_score         REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

-- ─────────────────────────────────────────────
-- CHURN PREDICTIONS — ML output per customer
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS churn_predictions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id             TEXT NOT NULL,
    session_id              TEXT NOT NULL,
    predicted_at            DATETIME DEFAULT CURRENT_TIMESTAMP,
    churn_prob_30d          REAL DEFAULT 0.0,
    churn_prob_60d          REAL DEFAULT 0.0,
    churn_prob_90d          REAL DEFAULT 0.0,
    risk_level              TEXT DEFAULT 'LOW',   -- HIGH/MEDIUM/LOW
    churn_reason            TEXT,
    top_risk_factors        TEXT,                 -- JSON array of strings
    revenue_at_risk         INTEGER DEFAULT 0,
    ltv_at_risk             INTEGER DEFAULT 0,
    raw_ml_score            REAL DEFAULT 0.0,     -- Before India adjustment
    india_adjusted_score    REAL DEFAULT 0.0,     -- After India adjustment
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

-- ─────────────────────────────────────────────
-- RETENTION PLAYBOOKS — LLM-generated action plans
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retention_playbooks (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id             TEXT NOT NULL,
    session_id              TEXT NOT NULL,
    generated_at            DATETIME DEFAULT CURRENT_TIMESTAMP,
    language                TEXT DEFAULT 'english',
    playbook_text           TEXT,
    email_draft             TEXT,
    whatsapp_draft          TEXT,
    sms_draft               TEXT,
    call_script             TEXT,
    recommended_discount    INTEGER DEFAULT 0,    -- INR amount
    status                  TEXT DEFAULT 'PENDING', -- PENDING/IN_PROGRESS/RESOLVED
    outcome                 TEXT,                 -- retained/churned/partial
    actioned_at             DATETIME,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

-- ─────────────────────────────────────────────
-- BUSINESS INSIGHTS — Patterns unique to user's data
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS business_insights (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              TEXT NOT NULL,
    insight_type            TEXT,                 -- feature/regional/temporal/behavioral
    insight_text            TEXT NOT NULL,
    confidence_score        REAL DEFAULT 0.0,
    generated_at            DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- ─────────────────────────────────────────────
-- INDIA CALENDAR — Pre-seeded from config
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS india_calendar (
    month                   INTEGER PRIMARY KEY,
    event_name              TEXT,
    adjustment_multiplier   REAL DEFAULT 1.0,
    note                    TEXT,
    affected_industries     TEXT                  -- JSON array
);

-- ─────────────────────────────────────────────
-- PLAYBOOK OUTCOMES — For learning which strategy works
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS playbook_outcomes (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              TEXT,
    industry                TEXT,
    churn_reason            TEXT,
    strategy_used           TEXT,                 -- discount/call/training/email
    outcome                 TEXT,                 -- retained/churned
    recorded_at             DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ─────────────────────────────────────────────
-- INDEXES for fast queries
-- ─────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_customers_session    ON customers(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_customer ON churn_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_session  ON churn_predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_playbooks_customer   ON retention_playbooks(customer_id);
CREATE INDEX IF NOT EXISTS idx_signals_customer     ON behavioral_signals(customer_id);
CREATE INDEX IF NOT EXISTS idx_insights_session     ON business_insights(session_id);
"""


def init_db():
    """
    Creates all tables if they don't exist.
    Safe to call multiple times — uses IF NOT EXISTS.
    """
    try:
        conn = get_connection()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        conn.close()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def seed_india_calendar():
    """
    Inserts India calendar data from config.
    Uses INSERT OR REPLACE so it is safe to run multiple times.
    """
    conn = get_connection()
    try:
        for month, data in INDIA_CALENDAR.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO india_calendar
                (month, event_name, adjustment_multiplier, note, affected_industries)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    month,
                    data["event"],
                    data["adjustment"],
                    data["note"],
                    '["all"]',
                ),
            )
        conn.commit()
        logger.info("India calendar seeded successfully")
    except Exception as e:
        logger.error(f"Calendar seed failed: {e}")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# HELPER FUNCTIONS used across the app
# ─────────────────────────────────────────────

def row_to_dict(row: sqlite3.Row) -> dict:
    """Converts sqlite3.Row to plain Python dict."""
    return dict(row) if row else {}


def rows_to_list(rows) -> list:
    """Converts list of sqlite3.Row to list of dicts."""
    return [dict(row) for row in rows]


def execute_query(sql: str, params: tuple = (), fetch: str = "all"):
    """
    Utility for quick queries without managing connections manually.
    fetch: 'all', 'one', or 'none'
    """
    conn = get_connection()
    try:
        cursor = conn.execute(sql, params)
        if fetch == "all":
            return rows_to_list(cursor.fetchall())
        elif fetch == "one":
            row = cursor.fetchone()
            return row_to_dict(row) if row else None
        else:
            conn.commit()
            return cursor.lastrowid
    finally:
        conn.close()