"""
ChurnShield 2.0 — Enterprise Template Builder

Purpose:
Generate enterprise-grade upload templates
for ANY business industry.

Capabilities:
- AI-ready schema generation
- realistic demo data
- industry-specific examples
- multi-sector template creation
- intelligent column guidance
- churn-compatible structure
- Excel-safe formatting
- onboarding-ready datasets
"""

import logging
import numpy as np
import pandas as pd

from faker import Faker
from datetime import datetime, timedelta

from config import (
    STANDARD_SCHEMA,
    INDUSTRY_FIELDS,
)

logger = logging.getLogger(
    "churnshield.upload.template_builder"
)

fake = Faker("en_IN")

np.random.seed(42)


# ─────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────

def build_template_dataset(

    industry: str = "general",

    rows: int = 25,
):

    """
    Enterprise template generator
    """

    logger.info(
        f"""
        Building enterprise upload template
        Industry={industry}
        Rows={rows}
        """
    )

    industry_profile = (
        detect_industry_profile(
            industry
        )
    )

    records = []

    for i in range(rows):

        customer = generate_customer_record(

            index=i,

            industry=industry,

            profile=industry_profile,
        )

        records.append(customer)

    df = pd.DataFrame(records)

    # ─────────────────────────────
    # ADD HELP ROW
    # ─────────────────────────────

    df = attach_schema_guidance(
        df
    )

    logger.info(
        f"""
        Enterprise template generated
        Rows={len(df)}
        Cols={len(df.columns)}
        """
    )

    return df


# ─────────────────────────────────────────────
# INDUSTRY DETECTION
# ─────────────────────────────────────────────

def detect_industry_profile(
    industry: str,
):

    """
    Detect business profile
    """

    industry = (
        industry
        .lower()
        .strip()
    )

    for key, value in INDUSTRY_FIELDS.items():

        if key in industry:

            return value

    return INDUSTRY_FIELDS["default"]


# ─────────────────────────────────────────────
# CUSTOMER RECORD GENERATION
# ─────────────────────────────────────────────

def generate_customer_record(

    index: int,

    industry: str,

    profile: dict,
):

    """
    Generate realistic enterprise row
    """

    category = profile.get(
        "category",
        "General Business",
    )

    churn_type = profile.get(
        "churn_type",
        "subscription_cancel",
    )

    is_b2b = any(

        x in category.lower()

        for x in [

            "saas",
            "crm",
            "erp",
            "software",
            "b2b",
        ]
    )

    # ─────────────────────────────
    # CUSTOMER ID
    # ─────────────────────────────

    customer_id = f"""
    CUST-{1000 + index}
    """.replace(
        "\n",
        ""
    ).replace(
        " ",
        ""
    )

    # ─────────────────────────────
    # CUSTOMER NAME
    # ─────────────────────────────

    if is_b2b:

        customer_name = fake.company()

    else:

        customer_name = fake.name()

    # ─────────────────────────────
    # PLAN TYPE
    # ─────────────────────────────

    plan_type = np.random.choice(

        [

            "Starter",
            "Professional",
            "Premium",
            "Enterprise",
        ],

        p=[0.30, 0.35, 0.25, 0.10],
    )

    # ─────────────────────────────
    # REVENUE
    # ─────────────────────────────

    revenue_ranges = {

        "Starter": (499, 2999),

        "Professional": (3000, 12000),

        "Premium": (12000, 40000),

        "Enterprise": (40000, 200000),
    }

    min_rev, max_rev = (
        revenue_ranges[plan_type]
    )

    monthly_revenue = int(
        np.random.randint(
            min_rev,
            max_rev,
        )
    )

    # ─────────────────────────────
    # CONTRACT AGE
    # ─────────────────────────────

    contract_age_months = int(
        np.random.randint(
            1,
            48,
        )
    )

    # ─────────────────────────────
    # LAST ACTIVITY
    # ─────────────────────────────

    last_active_days = int(
        np.random.randint(
            0,
            45,
        )
    )

    last_activity_date = (

        datetime.now()

        - timedelta(
            days=last_active_days
        )

    ).strftime(
        "%Y-%m-%d"
    )

    # ─────────────────────────────
    # LOGIN FREQUENCY
    # ─────────────────────────────

    login_frequency = int(
        np.random.randint(
            1,
            60,
        )
    )

    # ─────────────────────────────
    # FEATURE USAGE
    # ─────────────────────────────

    feature_usage_score = round(

        np.random.uniform(
            0.10,
            1.00,
        ),

        2,
    )

    # ─────────────────────────────
    # SUPPORT TICKETS
    # ─────────────────────────────

    support_tickets = int(
        np.random.randint(
            0,
            10,
        )
    )

    # ─────────────────────────────
    # PAYMENT DELAYS
    # ─────────────────────────────

    payment_delays = int(
        np.random.randint(
            0,
            5,
        )
    )

    # ─────────────────────────────
    # SEATS
    # ─────────────────────────────

    total_seats = int(
        np.random.randint(
            1,
            100,
        )
    )

    active_seats = int(
        np.random.randint(
            1,
            total_seats + 1,
        )
    )

    # ─────────────────────────────
    # NPS SCORE
    # ─────────────────────────────

    nps_score = int(
        np.random.randint(
            1,
            10,
        )
    )

    # ─────────────────────────────
    # CHURN LABEL
    # ─────────────────────────────

    churn_probability = calculate_demo_churn_probability(

        login_frequency,

        feature_usage_score,

        payment_delays,

        support_tickets,

        active_seats,

        total_seats,
    )

    churned = int(
        churn_probability > 0.60
    )

    # ─────────────────────────────
    # CITY
    # ─────────────────────────────

    city = fake.city()

    # ─────────────────────────────
    # BUILD RECORD
    # ─────────────────────────────

    record = {

        "customer_id":
            customer_id,

        "customer_name":
            customer_name,

        "industry":
            category,

        "plan_type":
            plan_type,

        "monthly_revenue":
            monthly_revenue,

        "contract_age_months":
            contract_age_months,

        "last_activity_date":
            last_activity_date,

        "login_frequency":
            login_frequency,

        "feature_usage_score":
            feature_usage_score,

        "support_tickets":
            support_tickets,

        "payment_delays":
            payment_delays,

        "active_seats":
            active_seats,

        "total_seats":
            total_seats,

        "nps_score":
            nps_score,

        "city":
            city,

        "churned":
            churned,
    }

    return record


# ─────────────────────────────────────────────
# DEMO CHURN LOGIC
# ─────────────────────────────────────────────

def calculate_demo_churn_probability(

    login_frequency,

    feature_usage_score,

    payment_delays,

    support_tickets,

    active_seats,

    total_seats,
):

    """
    Simulated enterprise churn logic
    """

    seat_utilization = (
        active_seats
        / max(total_seats, 1)
    )

    risk = (

        (
            1 - (
                login_frequency / 60
            )
        ) * 0.25

        +

        (
            1 - feature_usage_score
        ) * 0.25

        +

        (
            payment_delays / 5
        ) * 0.20

        +

        (
            support_tickets / 10
        ) * 0.15

        +

        (
            1 - seat_utilization
        ) * 0.15
    )

    return risk


# ─────────────────────────────────────────────
# ATTACH SCHEMA GUIDANCE
# ─────────────────────────────────────────────

def attach_schema_guidance(
    df: pd.DataFrame,
):

    """
    Add enterprise guidance row
    """

    guidance = {}

    for col in df.columns:

        guidance[col] = STANDARD_SCHEMA.get(

            col,

            f"Business field: {col}"
        )

    guidance_df = pd.DataFrame(
        [guidance]
    )

    final_df = pd.concat(

        [

            guidance_df,

            df,
        ],

        ignore_index=True,
    )

    return final_df


# ─────────────────────────────────────────────
# MULTI-INDUSTRY TEMPLATE
# ─────────────────────────────────────────────

def build_multi_industry_template():

    """
    Mega onboarding template
    """

    industries = [

        "telecom",

        "saas",

        "bank",

        "edtech",

        "fitness",
    ]

    frames = []

    for industry in industries:

        df = build_template_dataset(

            industry=industry,

            rows=10,
        )

        df["template_industry"] = (
            industry
        )

        frames.append(df)

    combined = pd.concat(

        frames,

        ignore_index=True,
    )

    logger.info(
        "Multi-industry template created"
    )

    return combined


# ─────────────────────────────────────────────
# SCHEMA EXPORT
# ─────────────────────────────────────────────

def export_schema_dictionary():

    """
    Export schema documentation
    """

    schema_rows = []

    for column, meaning in STANDARD_SCHEMA.items():

        schema_rows.append({

            "column":
                column,

            "description":
                meaning,
        })

    return pd.DataFrame(
        schema_rows
    )