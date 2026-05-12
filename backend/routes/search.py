"""
ChurnShield 2.0 — Search Routes

Enterprise Intelligence Search Engine

Capabilities:
- Universal customer search
- Smart fuzzy search
- Risk-based filtering
- Revenue segmentation
- AI-powered recommendations
- Churn pattern discovery
- Multi-field querying
- Similar customer matching
- Advanced analytics search
"""

import re
import logging
import traceback
import pandas as pd
import numpy as np

from difflib import SequenceMatcher
from typing import Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from db.database import (
    get_database_connection,
)

from ml.predictor import (
    predict_customer_churn,
)

from ml.persona_classifier import (
    classify_customer_persona,
)

from ml.reason_classifier import (
    classify_churn_reason,
)

logger = logging.getLogger(
    "churnshield.routes.search"
)

router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────

def load_dataset():

    """
    Loads customer dataset
    """

    try:

        conn = get_database_connection()

        query = """
        SELECT * FROM customers
        """

        df = pd.read_sql_query(
            query,
            conn,
        )

        conn.close()

        logger.info(
            f"Loaded {len(df)} rows for search"
        )

        return df

    except Exception as e:

        logger.error(
            f"Dataset load failed: {e}"
        )

        raise


# ─────────────────────────────────────────────
# SAFE STRING NORMALIZER
# ─────────────────────────────────────────────

def normalize_text(value):

    """
    Normalize text safely
    """

    if pd.isna(value):

        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-zA-Z0-9 ]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ─────────────────────────────────────────────
# FUZZY MATCH
# ─────────────────────────────────────────────

def fuzzy_match(
    query,
    text,
):

    """
    Computes similarity score
    """

    return SequenceMatcher(

        None,

        normalize_text(query),

        normalize_text(text),

    ).ratio()


# ─────────────────────────────────────────────
# UNIVERSAL SEARCH
# ─────────────────────────────────────────────

@router.get("/")
async def universal_search(

    q: str = Query(...),

    limit: int = Query(
        25,
        ge=1,
        le=200,
    ),
):

    """
    Universal enterprise search
    """

    try:

        logger.info(
            f"Universal search query: {q}"
        )

        df = load_dataset()

        searchable_columns = [

            col

            for col in [

                "customer_id",
                "customer_name",
                "email",
                "company",
                "city",
                "state",
                "industry",
                "plan_type",
            ]

            if col in df.columns
        ]

        results = []

        for _, row in df.iterrows():

            best_score = 0

            matched_column = None

            for col in searchable_columns:

                score = fuzzy_match(
                    q,
                    row.get(col, ""),
                )

                if score > best_score:

                    best_score = score

                    matched_column = col

            if best_score > 0.45:

                customer = row.to_dict()

                customer["match_score"] = round(
                    best_score,
                    3,
                )

                customer["matched_column"] = (
                    matched_column
                )

                results.append(customer)

        results = sorted(

            results,

            key=lambda x: x["match_score"],

            reverse=True,
        )

        return {

            "status":
                "success",

            "query":
                q,

            "matches_found":
                len(results),

            "results":
                results[:limit],
        }

    except Exception as e:

        logger.error(
            f"Universal search failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# HIGH RISK SEARCH
# ─────────────────────────────────────────────

@router.get("/high-risk")
async def high_risk_search(

    threshold: float = Query(
        0.70,
        ge=0.0,
        le=1.0,
    ),

    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),
):

    """
    Find highest-risk customers
    """

    try:

        df = load_dataset()

        risky_customers = []

        for _, row in df.iterrows():

            prediction = predict_customer_churn(
                row.to_dict()
            )

            probability = prediction.get(
                "churn_probability",
                0,
            )

            if probability >= threshold:

                customer = row.to_dict()

                customer["risk_probability"] = (
                    probability
                )

                customer["risk_band"] = (
                    prediction.get(
                        "risk_band",
                        "UNKNOWN",
                    )
                )

                risky_customers.append(customer)

        risky_customers = sorted(

            risky_customers,

            key=lambda x: x["risk_probability"],

            reverse=True,
        )

        return {

            "status":
                "success",

            "threshold":
                threshold,

            "high_risk_customers":
                risky_customers[:limit],
        }

    except Exception as e:

        logger.error(
            f"High-risk search failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# PERSONA SEARCH
# ─────────────────────────────────────────────

@router.get("/persona/{persona_name}")
async def search_by_persona(
    persona_name: str,
):

    """
    Find customers by persona
    """

    try:

        df = load_dataset()

        matches = []

        for _, row in df.iterrows():

            persona = (
                classify_customer_persona(
                    row.to_dict()
                )
            )

            detected = persona.get(
                "persona",
                "",
            )

            if normalize_text(persona_name) in normalize_text(detected):

                customer = row.to_dict()

                customer["persona"] = detected

                customer["persona_confidence"] = (
                    persona.get(
                        "confidence",
                        0,
                    )
                )

                matches.append(customer)

        return {

            "status":
                "success",

            "persona":
                persona_name,

            "matches":
                matches,
        }

    except Exception as e:

        logger.error(
            f"Persona search failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# REASON SEARCH
# ─────────────────────────────────────────────

@router.get("/reason/{reason_name}")
async def search_by_reason(
    reason_name: str,
):

    """
    Find customers with same churn reason
    """

    try:

        df = load_dataset()

        matches = []

        for _, row in df.iterrows():

            reason = (
                classify_churn_reason(
                    row.to_dict()
                )
            )

            detected = reason.get(
                "primary_reason",
                "",
            )

            if normalize_text(reason_name) in normalize_text(detected):

                customer = row.to_dict()

                customer["reason"] = detected

                customer["confidence"] = (
                    reason.get(
                        "confidence",
                        0,
                    )
                )

                matches.append(customer)

        return {

            "status":
                "success",

            "reason":
                reason_name,

            "matches":
                matches,
        }

    except Exception as e:

        logger.error(
            f"Reason search failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# REVENUE SEARCH
# ─────────────────────────────────────────────

@router.get("/revenue")
async def revenue_search(

    min_revenue: float = Query(
        0,
        ge=0,
    ),

    max_revenue: float = Query(
        999999999,
        ge=0,
    ),
):

    """
    Revenue-based customer search
    """

    try:

        df = load_dataset()

        if "monthly_revenue" not in df.columns:

            raise HTTPException(

                status_code=400,

                detail="""
                monthly_revenue column missing
                """,
            )

        filtered = df[

            (
                df["monthly_revenue"]
                >= min_revenue
            )

            &

            (
                df["monthly_revenue"]
                <= max_revenue
            )
        ]

        return {

            "status":
                "success",

            "customers_found":
                len(filtered),

            "results":

                filtered.to_dict(
                    orient="records"
                ),
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Revenue search failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# INDUSTRY SEARCH
# ─────────────────────────────────────────────

@router.get("/industry/{industry_name}")
async def industry_search(
    industry_name: str,
):

    """
    Search by business industry
    """

    try:

        df = load_dataset()

        if "industry" not in df.columns:

            raise HTTPException(

                status_code=400,

                detail="Industry column missing",
            )

        filtered = df[

            df["industry"]
            .astype(str)
            .str.lower()
            .str.contains(
                industry_name.lower(),
                na=False,
            )
        ]

        return {

            "status":
                "success",

            "industry":
                industry_name,

            "count":
                len(filtered),

            "results":

                filtered.to_dict(
                    orient="records"
                ),
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Industry search failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# SIMILAR CUSTOMERS
# ─────────────────────────────────────────────

@router.get("/similar/{customer_id}")
async def similar_customers(
    customer_id: str,
):

    """
    Find similar customer profiles
    """

    try:

        df = load_dataset()

        base = df[

            df["customer_id"]
            .astype(str)
            ==
            str(customer_id)
        ]

        if len(base) == 0:

            raise HTTPException(

                status_code=404,

                detail="Customer not found",
            )

        base_customer = base.iloc[0]

        similarities = []

        numeric_columns = [

            col

            for col in [

                "monthly_revenue",
                "feature_usage_score",
                "support_tickets",
                "active_seats",
                "payment_delays",
            ]

            if col in df.columns
        ]

        for _, row in df.iterrows():

            if str(row["customer_id"]) == str(customer_id):

                continue

            score = 0

            for col in numeric_columns:

                try:

                    v1 = float(
                        base_customer[col]
                    )

                    v2 = float(
                        row[col]
                    )

                    similarity = 1 / (
                        1 + abs(v1 - v2)
                    )

                    score += similarity

                except:

                    continue

            customer = row.to_dict()

            customer["similarity_score"] = round(
                score,
                4,
            )

            similarities.append(customer)

        similarities = sorted(

            similarities,

            key=lambda x: x["similarity_score"],

            reverse=True,
        )

        return {

            "status":
                "success",

            "base_customer":
                customer_id,

            "similar_customers":
                similarities[:15],
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Similar customer search failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# ADVANCED FILTERS
# ─────────────────────────────────────────────

@router.get("/advanced/filter")
async def advanced_filters(

    state: Optional[str] = None,

    plan_type: Optional[str] = None,

    min_nps: Optional[float] = None,

    max_tickets: Optional[int] = None,
):

    """
    Advanced customer filtering
    """

    try:

        df = load_dataset()

        filtered = df.copy()

        if state and "state" in filtered.columns:

            filtered = filtered[

                filtered["state"]
                .astype(str)
                .str.lower()
                ==
                state.lower()
            ]

        if plan_type and "plan_type" in filtered.columns:

            filtered = filtered[

                filtered["plan_type"]
                .astype(str)
                .str.lower()
                ==
                plan_type.lower()
            ]

        if min_nps is not None and "nps_score" in filtered.columns:

            filtered = filtered[
                filtered["nps_score"] >= min_nps
            ]

        if max_tickets is not None and "support_tickets" in filtered.columns:

            filtered = filtered[
                filtered["support_tickets"] <= max_tickets
            ]

        return {

            "status":
                "success",

            "filtered_count":
                len(filtered),

            "results":

                filtered.to_dict(
                    orient="records"
                ),
        }

    except Exception as e:

        logger.error(
            f"Advanced filter failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# GLOBAL ANALYTICS SEARCH
# ─────────────────────────────────────────────

@router.get("/analytics/summary")
async def analytics_summary():

    """
    Smart search analytics
    """

    try:

        df = load_dataset()

        summary = {

            "total_customers":
                len(df),

            "columns":
                list(df.columns),

            "missing_values":

                df.isnull()
                .sum()
                .to_dict(),

            "numeric_summary":

                df.describe(
                    include=[np.number]
                )

                .to_dict(),

            "top_states":

                df["state"]
                .value_counts()
                .head(10)
                .to_dict()

                if "state" in df.columns
                else {},

            "top_plans":

                df["plan_type"]
                .value_counts()
                .head(10)
                .to_dict()

                if "plan_type" in df.columns
                else {},
        }

        return {

            "status":
                "success",

            "analytics":
                summary,
        }

    except Exception as e:

        logger.error(
            f"Analytics summary failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )