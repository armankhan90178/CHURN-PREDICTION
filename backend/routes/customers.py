"""
ChurnShield 2.0 — Customer Intelligence Routes

Enterprise-grade customer APIs.

Capabilities:
- Customer profile intelligence
- Churn prediction APIs
- Persona analysis
- Health scoring
- Customer search
- Segmentation
- Timeline forecasting
- Playbook generation
- Customer insights
- Retention recommendations
"""

import uuid
import logging
import traceback
import pandas as pd

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

from ml.timeline import (
    generate_timeline_forecast,
)

from analytics.ews import (
    analyze_ews,
)

from llm.playbook_generator import (
    generate_playbook,
)

from llm.insight_extractor import (
    extract_business_insights,
)

logger = logging.getLogger(
    "churnshield.routes.customers"
)

router = APIRouter(
    prefix="/customers",
    tags=["Customers"],
)


# ─────────────────────────────────────────────
# LOAD CUSTOMERS
# ─────────────────────────────────────────────

def load_customer_dataset():

    """
    Loads customer dataset from DB
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
            f"Loaded {len(df)} customers"
        )

        return df

    except Exception as e:

        logger.error(
            f"Customer load failed: {e}"
        )

        raise


# ─────────────────────────────────────────────
# CUSTOMER LIST
# ─────────────────────────────────────────────

@router.get("/")
async def get_customers(

    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),

    offset: int = Query(
        0,
        ge=0,
    ),
):

    """
    Paginated customer listing
    """

    try:

        df = load_customer_dataset()

        total = len(df)

        paginated = df.iloc[
            offset:offset + limit
        ]

        return {

            "status":
                "success",

            "total_customers":
                total,

            "returned":
                len(paginated),

            "customers":

                paginated.to_dict(
                    orient="records"
                ),
        }

    except Exception as e:

        logger.error(
            f"Customer listing failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# CUSTOMER PROFILE
# ─────────────────────────────────────────────

@router.get("/{customer_id}")
async def get_customer_profile(
    customer_id: str,
):

    """
    Full customer intelligence profile
    """

    try:

        df = load_customer_dataset()

        customer = df[
            df["customer_id"]
            .astype(str)
            ==
            str(customer_id)
        ]

        if len(customer) == 0:

            raise HTTPException(

                status_code=404,

                detail="Customer not found",
            )

        customer_row = customer.iloc[0]

        prediction = (
            predict_customer_churn(
                customer_row.to_dict()
            )
        )

        persona = (
            classify_customer_persona(
                customer_row.to_dict()
            )
        )

        reasons = (
            classify_churn_reason(
                customer_row.to_dict()
            )
        )

        timeline = (
            generate_timeline_forecast(
                customer_row.to_dict()
            )
        )

        return {

            "status":
                "success",

            "customer":

                customer_row.to_dict(),

            "prediction":
                prediction,

            "persona":
                persona,

            "reasons":
                reasons,

            "timeline":
                timeline,
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Customer profile failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# CUSTOMER SEARCH
# ─────────────────────────────────────────────

@router.get("/search/query")
async def search_customers(

    keyword: str,

    limit: int = Query(
        20,
        ge=1,
        le=100,
    ),
):

    """
    Intelligent customer search
    """

    try:

        df = load_customer_dataset()

        search_columns = [

            col

            for col in [

                "customer_id",
                "customer_name",
                "email",
                "company",
                "city",
                "state",
            ]

            if col in df.columns
        ]

        if not search_columns:

            raise HTTPException(
                status_code=400,
                detail="Searchable columns unavailable",
            )

        mask = pd.Series(
            False,
            index=df.index,
        )

        for col in search_columns:

            mask |= (

                df[col]
                .astype(str)
                .str.contains(
                    keyword,
                    case=False,
                    na=False,
                )
            )

        results = df[mask].head(limit)

        return {

            "status":
                "success",

            "query":
                keyword,

            "results_found":
                len(results),

            "customers":

                results.to_dict(
                    orient="records"
                ),
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Customer search failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# HIGH RISK CUSTOMERS
# ─────────────────────────────────────────────

@router.get("/segment/high-risk")
async def high_risk_customers(

    limit: int = Query(
        25,
        ge=1,
        le=200,
    ),
):

    """
    Highest-risk customer accounts
    """

    try:

        df = load_customer_dataset()

        predictions = []

        for _, row in df.iterrows():

            result = predict_customer_churn(
                row.to_dict()
            )

            result["customer_id"] = (
                row.get(
                    "customer_id",
                    str(uuid.uuid4())
                )
            )

            result["customer_name"] = (
                row.get(
                    "customer_name",
                    "Unknown"
                )
            )

            predictions.append(result)

        prediction_df = pd.DataFrame(
            predictions
        )

        prediction_df = (

            prediction_df.sort_values(
                "churn_probability",
                ascending=False,
            )

            .head(limit)
        )

        return {

            "status":
                "success",

            "customers":

                prediction_df.to_dict(
                    orient="records"
                ),
        }

    except Exception as e:

        logger.error(
            f"High-risk query failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# PERSONA SEGMENTATION
# ─────────────────────────────────────────────

@router.get("/segment/personas")
async def customer_personas():

    """
    Persona distribution intelligence
    """

    try:

        df = load_customer_dataset()

        personas = []

        for _, row in df.iterrows():

            result = (
                classify_customer_persona(
                    row.to_dict()
                )
            )

            personas.append({

                "customer_id":
                    row.get(
                        "customer_id"
                    ),

                "persona":
                    result.get(
                        "persona"
                    ),
            })

        persona_df = pd.DataFrame(
            personas
        )

        summary = (

            persona_df["persona"]
            .value_counts()

            .reset_index()
        )

        summary.columns = [

            "persona",
            "count",
        ]

        return {

            "status":
                "success",

            "distribution":

                summary.to_dict(
                    orient="records"
                ),
        }

    except Exception as e:

        logger.error(
            f"Persona segmentation failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# CUSTOMER HEALTH
# ─────────────────────────────────────────────

@router.get("/{customer_id}/health")
async def customer_health(
    customer_id: str,
):

    """
    Customer health intelligence
    """

    try:

        df = load_customer_dataset()

        customer = df[
            df["customer_id"]
            .astype(str)
            ==
            str(customer_id)
        ]

        if len(customer) == 0:

            raise HTTPException(
                status_code=404,
                detail="Customer not found",
            )

        customer_row = customer.iloc[0]

        ews_result = analyze_ews(
            pd.DataFrame([customer_row])
        )

        return {

            "status":
                "success",

            "customer_id":
                customer_id,

            "health":
                ews_result,
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Customer health failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# CUSTOMER PLAYBOOK
# ─────────────────────────────────────────────

@router.get("/{customer_id}/playbook")
async def customer_playbook(
    customer_id: str,
):

    """
    AI retention playbook
    """

    try:

        df = load_customer_dataset()

        customer = df[
            df["customer_id"]
            .astype(str)
            ==
            str(customer_id)
        ]

        if len(customer) == 0:

            raise HTTPException(
                status_code=404,
                detail="Customer not found",
            )

        customer_row = customer.iloc[0]

        playbook = generate_playbook(
            customer_row.to_dict()
        )

        return {

            "status":
                "success",

            "customer_id":
                customer_id,

            "playbook":
                playbook,
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Playbook generation failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# CUSTOMER INSIGHTS
# ─────────────────────────────────────────────

@router.get("/{customer_id}/insights")
async def customer_insights(
    customer_id: str,
):

    """
    AI-generated customer insights
    """

    try:

        df = load_customer_dataset()

        customer = df[
            df["customer_id"]
            .astype(str)
            ==
            str(customer_id)
        ]

        if len(customer) == 0:

            raise HTTPException(
                status_code=404,
                detail="Customer not found",
            )

        customer_row = customer.iloc[0]

        insights = extract_business_insights(
            pd.DataFrame([customer_row])
        )

        return {

            "status":
                "success",

            "customer_id":
                customer_id,

            "insights":
                insights,
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Insight extraction failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# TIMELINE FORECAST
# ─────────────────────────────────────────────

@router.get("/{customer_id}/timeline")
async def customer_timeline(
    customer_id: str,
):

    """
    30/60/90 day churn forecast
    """

    try:

        df = load_customer_dataset()

        customer = df[
            df["customer_id"]
            .astype(str)
            ==
            str(customer_id)
        ]

        if len(customer) == 0:

            raise HTTPException(
                status_code=404,
                detail="Customer not found",
            )

        customer_row = customer.iloc[0]

        timeline = generate_timeline_forecast(
            customer_row.to_dict()
        )

        return {

            "status":
                "success",

            "customer_id":
                customer_id,

            "forecast":
                timeline,
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Timeline forecast failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# CUSTOMER SEGMENTS
# ─────────────────────────────────────────────

@router.get("/segments/revenue")
async def revenue_segments():

    """
    Revenue-based customer segmentation
    """

    try:

        df = load_customer_dataset()

        if "monthly_revenue" not in df.columns:

            raise HTTPException(
                status_code=400,
                detail="Revenue column missing",
            )

        revenue = df["monthly_revenue"]

        q1 = revenue.quantile(0.25)
        q2 = revenue.quantile(0.50)
        q3 = revenue.quantile(0.75)

        def classify(value):

            if value >= q3:
                return "Enterprise"

            elif value >= q2:
                return "Growth"

            elif value >= q1:
                return "Mid-Market"

            return "Low Value"

        df["segment"] = (
            df["monthly_revenue"]
            .apply(classify)
        )

        summary = (

            df.groupby("segment")

            .agg({

                "customer_id": "count",
                "monthly_revenue": "sum",
            })

            .reset_index()
        )

        return {

            "status":
                "success",

            "segments":

                summary.to_dict(
                    orient="records"
                ),
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Revenue segmentation failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )