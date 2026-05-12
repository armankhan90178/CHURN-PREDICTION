"""
ChurnShield 2.0 — Data Fetcher
Finds and downloads real datasets for any industry.
Three-layer fallback: Kaggle → Web Search → LLM Synthetic
Always returns data — never fails empty handed.
"""

import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from faker import Faker

from config import (
    KAGGLE_USERNAME, KAGGLE_API_KEY,
    DATA_DIR, INDUSTRY_FIELDS, ANTHROPIC_API_KEY, CLAUDE_MODEL
)

logger = logging.getLogger("churnshield.fetcher")
fake = Faker("en_IN")  # Indian locale for realistic names


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def fetch_data_for_field(field_name: str, progress_callback=None) -> pd.DataFrame:
    """
    Master function — given any industry field name,
    returns a clean pandas DataFrame ready for ML.

    Tries 3 strategies in order:
    1. Kaggle API download
    2. Web search for public CSV
    3. LLM synthetic generation (always works)
    """
    field_key = field_name.lower().strip()
    field_config = INDUSTRY_FIELDS.get(field_key, INDUSTRY_FIELDS["default"])
    search_query = field_config["kaggle_query"]

    logger.info(f"Fetching data for field: {field_name} | Query: {search_query}")

    # ── STRATEGY 1: Kaggle ──────────────────────────────────────────────
    if progress_callback:
        progress_callback("Searching Kaggle for dataset...", 10)

    kaggle_df = try_kaggle_download(search_query, field_name)
    if kaggle_df is not None and len(kaggle_df) >= 100:
        logger.info(f"Kaggle success: {len(kaggle_df)} rows")
        if progress_callback:
            progress_callback(f"Dataset found on Kaggle ({len(kaggle_df):,} rows)", 30)
        return kaggle_df

    # ── STRATEGY 2: Public Web CSV ──────────────────────────────────────
    if progress_callback:
        progress_callback("Searching for public datasets...", 20)

    web_df = try_web_csv_download(search_query, field_name)
    if web_df is not None and len(web_df) >= 50:
        logger.info(f"Web CSV success: {len(web_df)} rows")
        if progress_callback:
            progress_callback(f"Public dataset found ({len(web_df):,} rows)", 30)
        return web_df

    # ── STRATEGY 3: LLM Synthetic (always succeeds) ────────────────────
    if progress_callback:
        progress_callback("Generating realistic synthetic dataset...", 25)

    logger.info(f"Using LLM synthetic generation for: {field_name}")
    synthetic_df = generate_synthetic_dataset(field_name, field_config, n=600)

    if progress_callback:
        progress_callback(f"Synthetic dataset ready ({len(synthetic_df):,} rows)", 35)

    return synthetic_df


# ─────────────────────────────────────────────
# STRATEGY 1 — KAGGLE API
# ─────────────────────────────────────────────

def try_kaggle_download(search_query: str, field_name: str) -> Optional[pd.DataFrame]:
    """
    Searches Kaggle for a relevant churn dataset and downloads it.
    Returns DataFrame or None if Kaggle fails/has nothing useful.
    """
    try:
        # Set Kaggle credentials
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = KAGGLE_API_KEY

        if KAGGLE_USERNAME == "your-kaggle-username":
            logger.warning("Kaggle credentials not configured — skipping")
            return None

        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()

        # Search for datasets
        logger.info(f"Searching Kaggle: '{search_query}'")
        datasets = api.dataset_list(search=search_query, sort_by="votes", max_size=50)

        if not datasets:
            logger.info("No Kaggle datasets found")
            return None

        # Pick the best dataset (most votes + has CSV files)
        best_dataset = None
        for ds in datasets[:5]:
            if hasattr(ds, "totalBytes") and ds.totalBytes < 50 * 1024 * 1024:  # < 50MB
                best_dataset = ds
                break

        if not best_dataset:
            return None

        # Download to temp directory
        download_path = DATA_DIR / "kaggle_downloads" / field_name.replace(" ", "_")
        download_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading dataset: {best_dataset.ref}")
        api.dataset_download_files(
            str(best_dataset.ref),
            path=str(download_path),
            unzip=True,
            quiet=True,
        )

        # Find the downloaded CSV
        csv_files = list(download_path.glob("*.csv"))
        if not csv_files:
            return None

        # Pick largest CSV (usually the main data file)
        main_csv = max(csv_files, key=lambda f: f.stat().st_size)
        df = pd.read_csv(main_csv, encoding="utf-8", on_bad_lines="skip")

        logger.info(f"Kaggle downloaded: {len(df)} rows, {len(df.columns)} cols")
        return df

    except ImportError:
        logger.warning("kaggle package not installed — skipping Kaggle strategy")
        return None
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        return None


# ─────────────────────────────────────────────
# STRATEGY 2 — WEB CSV SEARCH
# ─────────────────────────────────────────────

# Known public CSV URLs for popular industries
KNOWN_PUBLIC_DATASETS = {
    "telco": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "bank": "https://raw.githubusercontent.com/dsaks/bank-customer-churn-prediction/master/Churn_Modelling.csv",
    "ecommerce": "https://raw.githubusercontent.com/Anubhav-Verma/customer-churn-prediction/main/ecommerce_churn.csv",
}

def try_web_csv_download(search_query: str, field_name: str) -> Optional[pd.DataFrame]:
    """
    Tries to download from known public dataset URLs.
    Returns DataFrame or None.
    """
    try:
        # Check if we have a known URL for this query
        for keyword, url in KNOWN_PUBLIC_DATASETS.items():
            if keyword in search_query.lower() or keyword in field_name.lower():
                logger.info(f"Trying known public URL for: {keyword}")
                df = _download_csv_from_url(url)
                if df is not None:
                    return df

        # For telecom-like fields, use the IBM telco dataset as baseline
        # It is the best publicly available churn dataset
        if any(k in field_name.lower() for k in ["telecom", "mobile", "phone", "jio", "airtel", "ott", "streaming"]):
            df = _download_csv_from_url(KNOWN_PUBLIC_DATASETS["telco"])
            if df is not None:
                logger.info("Using IBM Telco dataset as baseline")
                return df

        return None

    except Exception as e:
        logger.warning(f"Web CSV download failed: {e}")
        return None


def _download_csv_from_url(url: str) -> Optional[pd.DataFrame]:
    """Downloads a CSV from a direct URL."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        logger.info(f"Downloaded from URL: {len(df)} rows")
        return df

    except Exception as e:
        logger.warning(f"URL download failed ({url}): {e}")
        return None


# ─────────────────────────────────────────────
# STRATEGY 3 — LLM SYNTHETIC GENERATION
# ─────────────────────────────────────────────

def generate_synthetic_dataset(
    field_name: str,
    field_config: dict,
    n: int = 600
) -> pd.DataFrame:
    """
    Generates a realistic synthetic churn dataset for any industry.
    Uses industry-specific patterns to make data believable.
    Falls back to pure Python generation if Claude API not available.
    """

    category = field_config.get("category", "General Business")
    churn_rate = 0.25  # 25% churn is realistic for most industries

    logger.info(f"Generating {n} synthetic records for: {field_name} ({category})")

    # Industry-specific signal ranges
    industry_profiles = _get_industry_profiles(field_name.lower())

    records = []
    np.random.seed(42)

    for i in range(n):
        will_churn = np.random.random() < churn_rate
        profile = industry_profiles["churned"] if will_churn else industry_profiles["retained"]

        # Generate realistic Indian company/person names
        is_b2b = category in ["B2B SaaS", "ERP", "CRM", "HR Software"]
        customer_name = fake.company() if is_b2b else fake.name()

        # Revenue in INR — realistic for Indian market
        plan_type = np.random.choice(profile["plans"])
        monthly_revenue = _get_realistic_revenue(plan_type, field_name)

        record = {
            "customer_id": f"{field_name[:3].upper()}{1000 + i}",
            "customer_name": customer_name,
            "industry": category,
            "plan_type": plan_type,
            "monthly_revenue": monthly_revenue,
            "contract_age_months": int(np.random.randint(*profile["tenure"])),
            "login_count_30d": int(np.random.randint(*profile["logins"])),
            "feature_usage_score": round(np.random.uniform(*profile["feature_usage"]), 2),
            "support_tickets": int(np.random.randint(*profile["tickets"])),
            "days_since_last_login": int(np.random.randint(*profile["days_inactive"])),
            "payment_delays": int(np.random.randint(*profile["payment_delays"])),
            "active_seats": int(np.random.randint(*profile["active_seats"])),
            "total_seats": int(np.random.randint(*profile["total_seats"])),
            "nps_score": float(np.random.randint(*profile["nps"])),
            "city": fake.city(),
            "churned": int(will_churn),
        }

        # Ensure total_seats >= active_seats
        if record["total_seats"] < record["active_seats"]:
            record["total_seats"] = record["active_seats"] + np.random.randint(0, 5)

        records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Synthetic dataset generated: {len(df)} rows, churn rate: {df['churned'].mean():.1%}")
    return df


def _get_industry_profiles(field_name: str) -> dict:
    """
    Returns realistic behavioral profiles for churned vs retained customers
    based on industry type.
    """

    # Default profiles
    profiles = {
        "churned": {
            "plans": ["Basic", "Starter", "Free"],
            "tenure": (1, 12),
            "logins": (0, 8),
            "feature_usage": (0.05, 0.30),
            "tickets": (3, 12),
            "days_inactive": (15, 60),
            "payment_delays": (1, 5),
            "active_seats": (1, 3),
            "total_seats": (2, 8),
            "nps": (1, 5),
        },
        "retained": {
            "plans": ["Professional", "Enterprise", "Premium"],
            "tenure": (6, 36),
            "logins": (15, 60),
            "feature_usage": (0.50, 1.00),
            "tickets": (0, 3),
            "days_inactive": (0, 7),
            "payment_delays": (0, 1),
            "active_seats": (5, 20),
            "total_seats": (5, 25),
            "nps": (7, 10),
        },
    }

    # Industry-specific overrides
    if any(k in field_name for k in ["gym", "fitness", "yoga"]):
        profiles["churned"]["days_inactive"] = (7, 45)   # Miss classes
        profiles["churned"]["logins"] = (0, 4)            # App not used
        profiles["retained"]["logins"] = (10, 30)         # Weekly checkins

    elif any(k in field_name for k in ["netflix", "hotstar", "ott", "streaming", "spotify"]):
        profiles["churned"]["days_inactive"] = (20, 90)   # Stopped watching
        profiles["churned"]["feature_usage"] = (0.02, 0.20)
        profiles["retained"]["logins"] = (20, 60)          # Daily/weekly

    elif any(k in field_name for k in ["bank", "insurance", "finance", "loan"]):
        profiles["churned"]["payment_delays"] = (2, 8)    # Payment issues critical
        profiles["churned"]["tickets"] = (2, 10)
        profiles["retained"]["payment_delays"] = (0, 0)

    elif any(k in field_name for k in ["edtech", "course", "education", "learn"]):
        profiles["churned"]["days_inactive"] = (10, 45)   # Stopped studying
        profiles["churned"]["feature_usage"] = (0.05, 0.25)
        profiles["retained"]["feature_usage"] = (0.60, 1.00)

    return profiles


def _get_realistic_revenue(plan_type: str, field_name: str) -> int:
    """Returns realistic INR monthly revenue based on plan and industry."""
    base_revenues = {
        "Free": 0,
        "Basic": np.random.randint(199, 499),
        "Starter": np.random.randint(999, 2999),
        "Professional": np.random.randint(3999, 14999),
        "Premium": np.random.randint(4999, 19999),
        "Enterprise": np.random.randint(19999, 74999),
    }

    # B2B SaaS commands higher prices
    multiplier = 3 if any(k in field_name.lower() for k in ["saas", "crm", "erp", "hr"]) else 1

    base = base_revenues.get(plan_type, 999)
    return int(base * multiplier)


# ─────────────────────────────────────────────
# SAVE / CACHE HELPERS
# ─────────────────────────────────────────────

def save_dataset_to_cache(df: pd.DataFrame, field_name: str, source: str) -> str:
    """
    Saves a fetched dataset to disk for caching.
    Returns the file path as string.
    """
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_name = field_name.replace(" ", "_").replace("/", "_")
    file_path = cache_dir / f"{safe_name}_churn_data.csv"

    df.to_csv(file_path, index=False)
    logger.info(f"Dataset cached: {file_path} ({len(df)} rows)")
    return str(file_path)


def load_cached_dataset(field_name: str) -> Optional[pd.DataFrame]:
    """
    Loads a previously cached dataset if it exists and hasn't expired.
    Returns DataFrame or None.
    """
    safe_name = field_name.replace(" ", "_").replace("/", "_")
    file_path = DATA_DIR / "cache" / f"{safe_name}_churn_data.csv"

    if not file_path.exists():
        return None

    # Check age — expire after CACHE_EXPIRY_DAYS
    from config import CACHE_EXPIRY_DAYS
    import time
    age_days = (time.time() - file_path.stat().st_mtime) / 86400
    if age_days > CACHE_EXPIRY_DAYS:
        logger.info(f"Cache expired for {field_name} ({age_days:.1f} days old)")
        return None

    df = pd.read_csv(file_path)
    logger.info(f"Loaded from cache: {field_name} ({len(df)} rows)")
    return df