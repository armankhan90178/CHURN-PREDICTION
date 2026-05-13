"""
ChurnShield 2.0 — Enterprise Constants Registry

File:
utils/constants.py

Purpose:
Centralized constants registry
for ChurnShield AI platform.

Capabilities:
- global application constants
- environment-safe defaults
- ML constants
- API constants
- security constants
- analytics constants
- AI/LLM constants
- cache constants
- database constants
- upload limits
- scheduler intervals
- feature flags
- retention rules
- churn thresholds
- regional intelligence
- monitoring constants
- enterprise scalability support

Author:
ChurnShield AI
"""

import os

from pathlib import Path

# ============================================================
# ROOT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models"

CACHE_DIR = ROOT_DIR / "cache"

LOG_DIR = ROOT_DIR / "logs"

UPLOAD_DIR = ROOT_DIR / "user_data"

EXPORT_DIR = ROOT_DIR / "exports"

VECTOR_DB_DIR = ROOT_DIR / "vector_db"

FEATURE_STORE_DIR = ROOT_DIR / "feature_store"

PROMPT_DIR = ROOT_DIR / "llm" / "prompts"

# ============================================================
# APPLICATION
# ============================================================

APP_NAME = "ChurnShield"

APP_VERSION = "2.0.0"

APP_DESCRIPTION = (

    "Enterprise AI Churn Intelligence Platform"

)

COMPANY_NAME = "ChurnShield AI"

DEFAULT_TIMEZONE = "Asia/Kolkata"

SUPPORTED_ENVIRONMENTS = [

    "development",
    "staging",
    "production"

]

ENVIRONMENT = os.getenv(

    "ENVIRONMENT",
    "development"

)

DEBUG_MODE = ENVIRONMENT == "development"

# ============================================================
# API CONSTANTS
# ============================================================

API_PREFIX = "/api/v1"

API_TITLE = "ChurnShield API"

API_DESCRIPTION = (

    "Enterprise Churn Intelligence APIs"

)

DEFAULT_API_TIMEOUT = 30

MAX_API_TIMEOUT = 120

DEFAULT_PAGE_SIZE = 50

MAX_PAGE_SIZE = 500

RATE_LIMIT_PER_MINUTE = 120

MAX_CONCURRENT_REQUESTS = 1000

# ============================================================
# SECURITY
# ============================================================

JWT_SECRET_KEY = os.getenv(

    "JWT_SECRET_KEY",
    "change_me"

)

JWT_ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60

REFRESH_TOKEN_EXPIRE_DAYS = 30

PASSWORD_MIN_LENGTH = 8

MAX_LOGIN_ATTEMPTS = 5

ACCOUNT_LOCK_TIME_MINUTES = 15

ENABLE_2FA = False

ALLOWED_HOSTS = [

    "*"

]

# ============================================================
# DATABASE
# ============================================================

DATABASE_POOL_SIZE = 20

DATABASE_MAX_OVERFLOW = 40

DATABASE_POOL_TIMEOUT = 30

DATABASE_POOL_RECYCLE = 3600

DEFAULT_DB_BATCH_SIZE = 5000

MAX_DB_RETRIES = 3

# ============================================================
# CACHE
# ============================================================

CACHE_DEFAULT_TTL = 3600

CACHE_MAX_ITEMS = 100000

CACHE_CLEANUP_INTERVAL = 60

CACHE_COMPRESSION_ENABLED = True

REDIS_HOST = os.getenv(

    "REDIS_HOST",
    "localhost"

)

REDIS_PORT = int(

    os.getenv(

        "REDIS_PORT",
        6379

    )

)

REDIS_DB = 0

# ============================================================
# FILE UPLOADS
# ============================================================

MAX_UPLOAD_SIZE_MB = 500

MAX_UPLOAD_FILES = 20

ALLOWED_FILE_TYPES = [

    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".zip"

]

MAX_DATASET_ROWS = 50_000_000

MAX_DATASET_COLUMNS = 1000

TEMP_FILE_EXPIRY_HOURS = 24

CHUNK_SIZE = 10000

# ============================================================
# MACHINE LEARNING
# ============================================================

DEFAULT_MODEL_NAME = "xgboost"

SUPPORTED_MODELS = [

    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest"

]

DEFAULT_TEST_SIZE = 0.2

DEFAULT_RANDOM_STATE = 42

DEFAULT_CV_FOLDS = 5

MODEL_RETRAIN_DAYS = 7

MODEL_DRIFT_THRESHOLD = 0.15

MINIMUM_MODEL_ACCURACY = 0.75

FEATURE_IMPORTANCE_THRESHOLD = 0.01

MAX_FEATURE_COLUMNS = 500

# ============================================================
# CHURN THRESHOLDS
# ============================================================

LOW_CHURN_RISK = 0.30

MEDIUM_CHURN_RISK = 0.60

HIGH_CHURN_RISK = 0.85

CRITICAL_CHURN_RISK = 0.95

# ============================================================
# TIMELINE WINDOWS
# ============================================================

CHURN_WINDOW_30 = 30

CHURN_WINDOW_60 = 60

CHURN_WINDOW_90 = 90

RETENTION_LOOKBACK_DAYS = 365

# ============================================================
# FEATURE STORE
# ============================================================

FEATURE_STORE_ONLINE_ENABLED = True

FEATURE_STORE_OFFLINE_ENABLED = True

FEATURE_REFRESH_INTERVAL_MINUTES = 60

FEATURE_VERSIONING_ENABLED = True

# ============================================================
# VECTOR DATABASE
# ============================================================

VECTOR_DIMENSION = 1536

EMBEDDING_MODEL = "text-embedding-3-small"

VECTOR_TOP_K = 10

VECTOR_DISTANCE_METRIC = "cosine"

FAISS_INDEX_TYPE = "FlatL2"

MAX_EMBEDDING_BATCH_SIZE = 128

# ============================================================
# LLM CONSTANTS
# ============================================================

DEFAULT_LLM_PROVIDER = "openai"

SUPPORTED_LLM_PROVIDERS = [

    "openai",
    "anthropic",
    "gemini"

]

DEFAULT_LLM_MODEL = "gpt-4.1-mini"

MAX_PROMPT_TOKENS = 100000

MAX_COMPLETION_TOKENS = 4000

LLM_TEMPERATURE = 0.3

RAG_CONTEXT_LIMIT = 8

ENABLE_HALLUCINATION_DETECTION = True

ENABLE_RESPONSE_GUARD = True

# ============================================================
# ANALYTICS
# ============================================================

DEFAULT_FORECAST_DAYS = 90

COHORT_MIN_USERS = 10

ANOMALY_Z_SCORE_THRESHOLD = 3.0

BENCHMARK_REFRESH_DAYS = 30

EXECUTIVE_REPORT_TOP_INSIGHTS = 10

# ============================================================
# SCHEDULER
# ============================================================

SCHEDULER_ENABLED = True

CACHE_CLEANUP_CRON = "*/5 * * * *"

MODEL_RETRAIN_CRON = "0 2 * * 0"

DRIFT_CHECK_CRON = "0 */6 * * *"

BACKUP_CRON = "0 0 * * *"

REPORT_GENERATION_CRON = "0 8 * * *"

# ============================================================
# NOTIFICATIONS
# ============================================================

EMAIL_ENABLED = True

SMS_ENABLED = False

WHATSAPP_ENABLED = False

NOTIFICATION_BATCH_SIZE = 1000

MAX_EMAIL_RECIPIENTS = 500

# ============================================================
# MONITORING
# ============================================================

PROMETHEUS_ENABLED = True

ENABLE_REQUEST_TRACING = True

LATENCY_WARNING_MS = 1000

LATENCY_CRITICAL_MS = 5000

TOKEN_USAGE_WARNING = 1000000

ERROR_RATE_THRESHOLD = 0.05

ENABLE_HEALTH_MONITORING = True

# ============================================================
# INDIA INTELLIGENCE
# ============================================================

SUPPORTED_INDIAN_LANGUAGES = [

    "English",
    "Hindi",
    "Telugu",
    "Tamil",
    "Kannada",
    "Malayalam",
    "Marathi",
    "Gujarati",
    "Punjabi",
    "Bengali"

]

SUPPORTED_INDIAN_STATES = [

    "Andhra Pradesh",
    "Telangana",
    "Tamil Nadu",
    "Karnataka",
    "Kerala",
    "Maharashtra",
    "Delhi",
    "Gujarat"

]

GST_ENABLED = True

FESTIVAL_ANALYTICS_ENABLED = True

REGIONAL_SENTIMENT_ENABLED = True

# ============================================================
# EXPORTS
# ============================================================

PDF_EXPORT_ENABLED = True

PPTX_EXPORT_ENABLED = True

EXCEL_EXPORT_ENABLED = True

CSV_EXPORT_ENABLED = True

MAX_EXPORT_ROWS = 1_000_000

EXPORT_RETENTION_DAYS = 30

# ============================================================
# FEATURE FLAGS
# ============================================================

FEATURE_FLAGS = {

    "enable_rag": True,

    "enable_llm_playbooks": True,

    "enable_real_time_predictions": True,

    "enable_auto_retraining": True,

    "enable_drift_detection": True,

    "enable_vector_search": True,

    "enable_enterprise_monitoring": True,

    "enable_behavioral_segmentation": True,

    "enable_uplift_modeling": True,

    "enable_survival_analysis": True

}

# ============================================================
# DEFAULT RESPONSE MESSAGES
# ============================================================

SUCCESS_MESSAGE = "Operation completed successfully"

ERROR_MESSAGE = "Something went wrong"

UNAUTHORIZED_MESSAGE = "Unauthorized access"

NOT_FOUND_MESSAGE = "Requested resource not found"

VALIDATION_ERROR_MESSAGE = "Validation failed"

SERVER_BUSY_MESSAGE = "Server is busy, please try again later"

# ============================================================
# SYSTEM LIMITS
# ============================================================

MAX_BACKGROUND_TASKS = 100

MAX_THREAD_POOL_WORKERS = 16

MAX_ASYNC_CONNECTIONS = 10000

MAX_LOG_FILE_SIZE_MB = 50

MAX_MEMORY_USAGE_PERCENT = 85

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv(

    "LOG_LEVEL",
    "INFO"

)

ENABLE_JSON_LOGGING = True

ENABLE_AUDIT_LOGGING = True

LOG_RETENTION_DAYS = 30

# ============================================================
# STATUS CONSTANTS
# ============================================================

STATUS_ACTIVE = "active"

STATUS_INACTIVE = "inactive"

STATUS_PENDING = "pending"

STATUS_FAILED = "failed"

STATUS_SUCCESS = "success"

STATUS_PROCESSING = "processing"

STATUS_COMPLETED = "completed"

# ============================================================
# CACHE NAMESPACES
# ============================================================

CACHE_NAMESPACE_USERS = "users"

CACHE_NAMESPACE_PREDICTIONS = "predictions"

CACHE_NAMESPACE_ANALYTICS = "analytics"

CACHE_NAMESPACE_REPORTS = "reports"

CACHE_NAMESPACE_MODELS = "models"

CACHE_NAMESPACE_EMBEDDINGS = "embeddings"

# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = MODEL_DIR / "universal_churn_model.pkl"

SCALER_FILE = MODEL_DIR / "scaler.pkl"

ENCODER_FILE = MODEL_DIR / "encoder.pkl"

MODEL_REGISTRY_FILE = MODEL_DIR / "model_registry.json"

# ============================================================
# SYSTEM HEALTH
# ============================================================

HEALTH_STATUS_HEALTHY = "healthy"

HEALTH_STATUS_WARNING = "warning"

HEALTH_STATUS_CRITICAL = "critical"

# ============================================================
# STARTUP BANNER
# ============================================================

STARTUP_BANNER = f"""

===========================================================
🚀 {APP_NAME} v{APP_VERSION}
Enterprise AI Churn Intelligence Platform
===========================================================

Environment : {ENVIRONMENT}
Debug Mode : {DEBUG_MODE}

===========================================================

"""

# ============================================================
# VALIDATE REQUIRED DIRECTORIES
# ============================================================

REQUIRED_DIRECTORIES = [

    DATA_DIR,
    MODEL_DIR,
    CACHE_DIR,
    LOG_DIR,
    UPLOAD_DIR,
    EXPORT_DIR,
    VECTOR_DB_DIR,
    FEATURE_STORE_DIR

]

# ============================================================
# AUTO CREATE DIRECTORIES
# ============================================================

for directory in REQUIRED_DIRECTORIES:

    directory.mkdir(

        parents=True,
        exist_ok=True

    )