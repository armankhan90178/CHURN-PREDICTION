"""
ChurnShield 2.0 — Analytics API Routes

Enterprise analytics orchestration routes.

Capabilities:
- Unified analytics pipeline
- Revenue intelligence
- Cohort analysis
- Regional analytics
- Seasonal forecasting
- Early warning system
- Executive dashboard APIs
- KPI aggregation
- Customer health analytics
- Business intelligence exports
"""

import traceback
import logging
import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from typing import Optional

from db.database import get_database_connection

from analytics.cohort import (
    analyze_cohorts,
)

from analytics.revenue import (
    analyze_revenue,
)

from analytics.regional import (
    analyze_regions,
)

from analytics.seasonal import (
    analyze_seasonality,
)

from analytics.ews import (
    analyze_ews,
)

logger = logging.getLogger(
    "churnshield.routes.analytics"
)

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
)


# ─────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────

def load_dataset_from_db(
    table_name: str = "customers",
) -> pd.DataFrame:

    """
    Loads analytics dataset from database
    """

    try:

        conn = get_database_connection()

        query = f"""
        SELECT * FROM {table_name}
        """

        df = pd.read_sql_query(
            query,
            conn,
        )

        conn.close()

        logger.info(
            f"Loaded dataset: {len(df)} rows"
        )

        return df

    except Exception as e:

        logger.error(
            f"Database load failed: {e}"
        )

        raise


# ─────────────────────────────────────────────
# MASTER ANALYTICS
# ─────────────────────────────────────────────

@router.get("/full-analysis")
async def full_business_analysis():

    """
    Runs complete enterprise analytics pipeline
    """

    try:

        logger.info(
            "Starting full business analytics..."
        )

        df = load_dataset_from_db()

        if len(df) == 0:

            raise HTTPException(
                status_code=404,
                detail="No customer data found",
            )

        # ─────────────────────────────────
        # ANALYTICS PIPELINE
        # ─────────────────────────────────

        revenue_analysis = (
            analyze_revenue(df)
        )

        cohort_analysis = (
            analyze_cohorts(df)
        )

        regional_analysis = (
            analyze_regions(df)
        )

        seasonal_analysis = (
            analyze_seasonality(df)
        )

        ews_analysis = (
            analyze_ews(df)
        )

        logger.info(
            "Full analytics completed"
        )

        return {

            "status":
                "success",

            "analytics": {

                "revenue":
                    revenue_analysis,

                "cohort":
                    cohort_analysis,

                "regional":
                    regional_analysis,

                "seasonal":
                    seasonal_analysis,

                "ews":
                    ews_analysis,
            }
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Analytics pipeline failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(

            status_code=500,

            detail=f"""
            Analytics pipeline failed:
            {str(e)}
            """,
        )


# ─────────────────────────────────────────────
# REVENUE ANALYTICS
# ─────────────────────────────────────────────

@router.get("/revenue")
async def revenue_analytics():

    """
    Revenue intelligence API
    """

    try:

        df = load_dataset_from_db()

        results = analyze_revenue(df)

        return {

            "status":
                "success",

            "module":
                "revenue",

            "results":
                results,
        }

    except Exception as e:

        logger.error(
            f"Revenue analytics failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# COHORT ANALYTICS
# ─────────────────────────────────────────────

@router.get("/cohort")
async def cohort_analytics():

    """
    Cohort retention intelligence
    """

    try:

        df = load_dataset_from_db()

        results = analyze_cohorts(df)

        return {

            "status":
                "success",

            "module":
                "cohort",

            "results":
                results,
        }

    except Exception as e:

        logger.error(
            f"Cohort analysis failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# REGIONAL ANALYTICS
# ─────────────────────────────────────────────

@router.get("/regional")
async def regional_analytics():

    """
    Regional churn intelligence
    """

    try:

        df = load_dataset_from_db()

        results = analyze_regions(df)

        return {

            "status":
                "success",

            "module":
                "regional",

            "results":
                results,
        }

    except Exception as e:

        logger.error(
            f"Regional analytics failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# SEASONAL ANALYTICS
# ─────────────────────────────────────────────

@router.get("/seasonal")
async def seasonal_analytics():

    """
    Seasonal churn forecasting
    """

    try:

        df = load_dataset_from_db()

        results = analyze_seasonality(df)

        return {

            "status":
                "success",

            "module":
                "seasonal",

            "results":
                results,
        }

    except Exception as e:

        logger.error(
            f"Seasonal analytics failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# EARLY WARNING SYSTEM
# ─────────────────────────────────────────────

@router.get("/ews")
async def ews_analytics():

    """
    Customer health intelligence API
    """

    try:

        df = load_dataset_from_db()

        results = analyze_ews(df)

        return {

            "status":
                "success",

            "module":
                "ews",

            "results":
                results,
        }

    except Exception as e:

        logger.error(
            f"EWS analysis failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────

@router.get("/dashboard")
async def executive_dashboard():

    """
    Executive KPI dashboard
    """

    try:

        logger.info(
            "Generating executive dashboard..."
        )

        df = load_dataset_from_db()

        revenue = analyze_revenue(df)
        regional = analyze_regions(df)
        ews = analyze_ews(df)

        total_customers = len(df)

        churn_rate = round(

            (
                df["churned"].mean()
                * 100
            ),

            2,
        ) if "churned" in df.columns else 0

        total_revenue = round(

            df["monthly_revenue"].sum(),

            2,

        ) if "monthly_revenue" in df.columns else 0

        at_risk_customers = 0

        if "risk_band" in ews.get(
            "summary", {}
        ):

            at_risk_customers = (

                ews["summary"]
                .get(
                    "critical_customers",
                    0,
                )
            )

        return {

            "status":
                "success",

            "dashboard": {

                "generated_at":
                    str(pd.Timestamp.utcnow()),

                "total_customers":
                    total_customers,

                "overall_churn_rate":
                    churn_rate,

                "total_monthly_revenue":
                    total_revenue,

                "annual_revenue":
                    total_revenue * 12,

                "at_risk_customers":
                    at_risk_customers,

                "top_region":

                    regional[
                        "executive_summary"
                    ][
                        "top_revenue_region"
                    ],

                "highest_risk_region":

                    regional[
                        "executive_summary"
                    ][
                        "highest_risk_region"
                    ],

                "gross_retention":

                    revenue[
                        "retention_analysis"
                    ][
                        "gross_revenue_retention"
                    ],

                "net_retention":

                    revenue[
                        "retention_analysis"
                    ][
                        "net_revenue_retention"
                    ],
            }
        }

    except Exception as e:

        logger.error(
            f"Dashboard generation failed: {e}"
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# CUSTOMER HEALTH API
# ─────────────────────────────────────────────

@router.get("/health-summary")
async def customer_health_summary():

    """
    Customer health summary API
    """

    try:

        df = load_dataset_from_db()

        ews = analyze_ews(df)

        return {

            "status":
                "success",

            "summary":
                ews,
        }

    except Exception as e:

        logger.error(
            f"Health summary failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# KPI SNAPSHOT
# ─────────────────────────────────────────────

@router.get("/kpis")
async def kpi_snapshot():

    """
    Lightweight KPI endpoint
    """

    try:

        df = load_dataset_from_db()

        customer_count = len(df)

        churn_rate = round(

            (
                df["churned"].mean()
                * 100
            ),

            2,

        ) if "churned" in df.columns else 0

        mrr = round(

            df["monthly_revenue"].sum(),

            2,

        ) if "monthly_revenue" in df.columns else 0

        avg_revenue = round(

            df["monthly_revenue"].mean(),

            2,

        ) if "monthly_revenue" in df.columns else 0

        avg_nps = round(

            df["nps_score"].mean(),

            2,

        ) if "nps_score" in df.columns else 0

        return {

            "status":
                "success",

            "kpis": {

                "customers":
                    customer_count,

                "monthly_revenue":
                    mrr,

                "annual_revenue":
                    mrr * 12,

                "average_revenue":
                    avg_revenue,

                "churn_rate":
                    churn_rate,

                "average_nps":
                    avg_nps,
            }
        }

    except Exception as e:

        logger.error(
            f"KPI snapshot failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# CUSTOM REGION FILTER
# ─────────────────────────────────────────────

@router.get("/region/{region_name}")
async def region_specific_analysis(
    region_name: str,
):

    """
    Region-specific intelligence
    """

    try:

        df = load_dataset_from_db()

        if "state" not in df.columns:

            raise HTTPException(
                status_code=400,
                detail="State column missing",
            )

        regional_df = df[
            df["state"]
            .astype(str)
            .str.lower()
            ==
            region_name.lower()
        ]

        if len(regional_df) == 0:

            raise HTTPException(
                status_code=404,
                detail="Region not found",
            )

        revenue = analyze_revenue(
            regional_df
        )

        ews = analyze_ews(
            regional_df
        )

        return {

            "status":
                "success",

            "region":
                region_name,

            "customers":
                len(regional_df),

            "analytics": {

                "revenue":
                    revenue,

                "ews":
                    ews,
            }
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Regional drilldown failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ─────────────────────────────────────────────
# TOP RISK CUSTOMERS
# ─────────────────────────────────────────────

@router.get("/top-risk-customers")
async def top_risk_customers(

    limit: Optional[int] = Query(
        20,
        ge=1,
        le=200,
    )
):

    """
    Highest-risk customer accounts
    """

    try:

        df = load_dataset_from_db()

        ews = analyze_ews(df)

        if "customer_scores" not in ews:

            raise HTTPException(
                status_code=500,
                detail="Risk scoring unavailable",
            )

        scores = pd.DataFrame(
            ews["customer_scores"]
        )

        scores = scores.sort_values(

            "health_score"

        ).head(limit)

        return {

            "status":
                "success",

            "count":
                len(scores),

            "customers":
                scores.to_dict(
                    orient="records"
                ),
        }

    except HTTPException:

        raise

    except Exception as e:

        logger.error(
            f"Top-risk query failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )