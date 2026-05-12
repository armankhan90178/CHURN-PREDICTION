"""
ChurnShield 2.0 — Configuration File
All keys, settings, constants in one place.
Every other file imports from here. Never hardcode anything elsewhere.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
USER_DATA_DIR = BASE_DIR / "user_data"
DB_PATH = BASE_DIR / "db" / "churnshield.db"

# Create directories if they don't exist
for d in [DATA_DIR, MODELS_DIR, USER_DATA_DIR, BASE_DIR / "db"]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# API KEYS — Set these in environment variables
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-claude-api-key-here")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "your-kaggle-username")
KAGGLE_API_KEY = os.getenv("KAGGLE_API_KEY", "your-kaggle-api-key")

# ─────────────────────────────────────────────
# CLAUDE API SETTINGS
# ─────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-5"
CLAUDE_MAX_TOKENS = 1500
CLAUDE_TEMPERATURE = 0.7

# ─────────────────────────────────────────────
# ML SETTINGS
# ─────────────────────────────────────────────
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42
ML_MIN_ROWS_REQUIRED = 50
ML_MIN_COLUMNS_REQUIRED = 5
ML_TARGET_COLUMN = "churned"

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3,  # handles class imbalance
    "eval_metric": "auc",
    "random_state": ML_RANDOM_STATE,
    "use_label_encoder": False,
}

# Risk level thresholds
RISK_HIGH_THRESHOLD = 0.70
RISK_MEDIUM_THRESHOLD = 0.40

# ─────────────────────────────────────────────
# UPLOAD SETTINGS
# ─────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_ROWS = 100_000
MAX_UPLOAD_COLUMNS = 200
ALLOWED_UPLOAD_EXTENSIONS = [".csv", ".xlsx", ".xls", ".json"]
CACHE_EXPIRY_DAYS = 7

# ─────────────────────────────────────────────
# STANDARD SCHEMA — What every dataset must become
# ─────────────────────────────────────────────
STANDARD_SCHEMA = {
    "customer_id": "Unique identifier for each customer",
    "customer_name": "Name of the customer or company",
    "plan_type": "Subscription plan or tier",
    "monthly_revenue": "Monthly revenue from this customer in INR",
    "contract_age_months": "How many months customer has been with you",
    "last_activity_date": "Date of most recent activity",
    "login_frequency": "How often they log in or use the product",
    "feature_usage_score": "0-1 score of feature adoption",
    "support_tickets": "Number of support tickets raised",
    "payment_delays": "Number of times payment was delayed",
    "active_seats": "Number of active users on their account",
    "total_seats": "Total seats purchased",
    "nps_score": "Net Promoter Score if available",
    "churned": "TARGET: 1 if churned, 0 if active",
}

# ─────────────────────────────────────────────
# 100 INDUSTRY FIELDS
# ─────────────────────────────────────────────
INDUSTRY_FIELDS = {
    # Streaming & Entertainment
    "netflix": {"category": "OTT Streaming", "kaggle_query": "streaming subscription churn", "churn_type": "subscription_cancel"},
    "hotstar": {"category": "OTT Streaming", "kaggle_query": "OTT subscription churn india", "churn_type": "subscription_cancel"},
    "amazon prime": {"category": "OTT Streaming", "kaggle_query": "prime video subscription churn", "churn_type": "subscription_cancel"},
    "spotify": {"category": "Music Streaming", "kaggle_query": "music streaming churn", "churn_type": "subscription_cancel"},
    "youtube premium": {"category": "Video Streaming", "kaggle_query": "video streaming churn", "churn_type": "subscription_cancel"},
    "zee5": {"category": "OTT Streaming", "kaggle_query": "OTT churn india", "churn_type": "subscription_cancel"},
    "sonyliv": {"category": "OTT Streaming", "kaggle_query": "streaming churn dataset", "churn_type": "subscription_cancel"},
    "jiocinema": {"category": "OTT Streaming", "kaggle_query": "telecom OTT churn", "churn_type": "subscription_cancel"},
    # Automotive
    "cars": {"category": "Automotive", "kaggle_query": "automobile customer churn", "churn_type": "service_lapse"},
    "bikes": {"category": "Automotive", "kaggle_query": "vehicle subscription churn", "churn_type": "service_lapse"},
    "ev subscription": {"category": "EV", "kaggle_query": "electric vehicle subscription churn", "churn_type": "subscription_cancel"},
    "car rental": {"category": "Automotive Rental", "kaggle_query": "car rental customer churn", "churn_type": "no_booking"},
    # Food & Beverage
    "zomato pro": {"category": "Food Delivery", "kaggle_query": "food delivery subscription churn", "churn_type": "subscription_cancel"},
    "swiggy one": {"category": "Food Delivery", "kaggle_query": "food delivery churn", "churn_type": "subscription_cancel"},
    "cloud kitchen": {"category": "Food", "kaggle_query": "restaurant customer churn", "churn_type": "no_order"},
    "restaurant": {"category": "Food", "kaggle_query": "restaurant customer retention", "churn_type": "no_visit"},
    "grocery delivery": {"category": "Grocery", "kaggle_query": "grocery subscription churn", "churn_type": "no_order"},
    # Finance
    "credit card": {"category": "Finance", "kaggle_query": "credit card customer churn", "churn_type": "card_cancel"},
    "insurance": {"category": "Insurance", "kaggle_query": "insurance customer churn", "churn_type": "policy_lapse"},
    "mutual funds": {"category": "Investment", "kaggle_query": "investment customer churn", "churn_type": "withdrawal"},
    "bank": {"category": "Banking", "kaggle_query": "bank customer churn", "churn_type": "account_close"},
    "loan app": {"category": "Fintech", "kaggle_query": "fintech customer churn", "churn_type": "no_renewal"},
    # Health & Fitness
    "gym": {"category": "Fitness", "kaggle_query": "gym membership churn", "churn_type": "membership_cancel"},
    "yoga": {"category": "Fitness", "kaggle_query": "fitness subscription churn", "churn_type": "subscription_cancel"},
    "telemedicine": {"category": "Healthcare", "kaggle_query": "healthcare patient retention churn", "churn_type": "no_visit"},
    "hospital": {"category": "Healthcare", "kaggle_query": "healthcare customer churn", "churn_type": "patient_dropout"},
    # Education
    "edtech": {"category": "Education", "kaggle_query": "edtech student churn", "churn_type": "subscription_cancel"},
    "coding bootcamp": {"category": "Education", "kaggle_query": "online course churn", "churn_type": "dropout"},
    "language app": {"category": "Education", "kaggle_query": "language learning churn", "churn_type": "subscription_cancel"},
    "online tutoring": {"category": "Education", "kaggle_query": "tutoring platform churn", "churn_type": "subscription_cancel"},
    # B2B SaaS
    "crm": {"category": "B2B SaaS", "kaggle_query": "B2B SaaS customer churn", "churn_type": "subscription_cancel"},
    "hr software": {"category": "B2B SaaS", "kaggle_query": "SaaS churn dataset", "churn_type": "subscription_cancel"},
    "erp": {"category": "B2B SaaS", "kaggle_query": "enterprise software churn", "churn_type": "subscription_cancel"},
    "accounting": {"category": "B2B SaaS", "kaggle_query": "SaaS churn", "churn_type": "subscription_cancel"},
    # Retail
    "fashion": {"category": "E-Commerce", "kaggle_query": "ecommerce customer churn", "churn_type": "no_purchase"},
    "electronics": {"category": "E-Commerce", "kaggle_query": "retail customer churn", "churn_type": "no_purchase"},
    "toys": {"category": "E-Commerce", "kaggle_query": "D2C subscription churn", "churn_type": "subscription_cancel"},
    "beauty": {"category": "D2C", "kaggle_query": "D2C beauty subscription churn", "churn_type": "subscription_cancel"},
    # Telecom
    "telecom": {"category": "Telecom", "kaggle_query": "telco customer churn", "churn_type": "plan_cancel"},
    "jio": {"category": "Telecom", "kaggle_query": "telecom churn india", "churn_type": "plan_cancel"},
    "airtel": {"category": "Telecom", "kaggle_query": "telco churn dataset", "churn_type": "plan_cancel"},
    "dth": {"category": "DTH", "kaggle_query": "DTH cable TV churn", "churn_type": "subscription_cancel"},
    # Social & Gaming
    "gaming": {"category": "Gaming", "kaggle_query": "mobile gaming churn", "churn_type": "app_uninstall"},
    "fantasy sports": {"category": "Gaming", "kaggle_query": "fantasy sports user churn", "churn_type": "no_activity"},
    "dating app": {"category": "Social", "kaggle_query": "dating app subscription churn", "churn_type": "subscription_cancel"},
    # Default fallback
    "default": {"category": "General Business", "kaggle_query": "customer churn dataset", "churn_type": "subscription_cancel"},
}

# ─────────────────────────────────────────────
# INDIA CALENDAR — Month-wise adjustments
# ─────────────────────────────────────────────
INDIA_CALENDAR = {
    1:  {"event": "GST Filing Q3",     "adjustment": 0.80, "note": "Usage dip normal during filing"},
    2:  {"event": "Union Budget",       "adjustment": 0.85, "note": "Companies freeze decisions"},
    3:  {"event": "Financial Year End", "adjustment": 0.75, "note": "Businesses closing books"},
    4:  {"event": "GST Filing Q4",      "adjustment": 0.80, "note": "Annual filing month"},
    5:  {"event": "Normal Month",       "adjustment": 1.00, "note": "No major events"},
    6:  {"event": "Normal Month",       "adjustment": 1.00, "note": "No major events"},
    7:  {"event": "GST Filing Q1",      "adjustment": 0.85, "note": "Quarterly filing"},
    8:  {"event": "Independence Month", "adjustment": 0.95, "note": "Slight slowdown"},
    9:  {"event": "Normal Month",       "adjustment": 1.00, "note": "No major events"},
    10: {"event": "GST Filing + Navratri", "adjustment": 0.80, "note": "Filing + festival"},
    11: {"event": "Diwali Season",      "adjustment": 0.70, "note": "Festival — lowest activity"},
    12: {"event": "Year End Slowdown",  "adjustment": 0.90, "note": "Holiday season"},
}

# ─────────────────────────────────────────────
# CHURN REASON CATEGORIES
# ─────────────────────────────────────────────
CHURN_REASONS = [
    "Price Sensitivity",
    "Product Dissatisfaction",
    "Competitor Switch",
    "Low Feature Adoption",
    "Support Failure",
    "Life/Business Event",
    "Seasonal Disengagement",
]

# ─────────────────────────────────────────────
# CUSTOMER PERSONAS
# ─────────────────────────────────────────────
CUSTOMER_PERSONAS = [
    "Power User",
    "Passive Subscriber",
    "Value Seeker",
    "Relationship Buyer",
    "ROI Tracker",
]

# ─────────────────────────────────────────────
# SUPPORTED LANGUAGES FOR PLAYBOOK OUTPUT
# ─────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "marathi": "mr",
    "bengali": "bn",
    "gujarati": "gu",
}

# ─────────────────────────────────────────────
# SCHEDULER SETTINGS
# ─────────────────────────────────────────────
DAILY_REFRESH_HOUR = 2       # 2 AM daily score refresh
WEEKLY_REPORT_DAY = "sun"    # Sunday night report generation
CACHE_CLEANUP_HOUR = 3       # 3 AM cache cleanup

# ─────────────────────────────────────────────
# FEATURE ENGINEERING WEIGHTS
# ─────────────────────────────────────────────
ENGAGEMENT_WEIGHTS = {
    "feature_usage_score": 0.40,
    "login_health": 0.40,
    "seat_utilization": 0.20,
}

EWS_WEIGHTS = {
    "usage_health": 0.30,
    "payment_health": 0.25,
    "support_health": 0.20,
    "engagement_score": 0.15,
    "relationship_score": 0.10,
}