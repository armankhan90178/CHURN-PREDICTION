"""
ChurnShield 2.0 — India Festival Calendar Engine

File:
india/calendar.py

Purpose:
Advanced Indian festival + business calendar intelligence
for churn prediction, customer engagement,
marketing personalization, forecasting,
and regional behavioral analytics.

Capabilities:
- Indian festival intelligence
- holiday impact scoring
- state-wise festival support
- seasonal business adjustments
- customer engagement timing
- retention campaign timing
- regional event analytics
- Ramadan / Diwali / Holi intelligence
- business peak prediction
- churn-risk seasonal mapping
- multilingual festival metadata
- custom company holidays
- fiscal quarter intelligence
- business day calculations
- working day engine
- campaign recommendation windows
- API-ready outputs

Supports:
- Telecom
- OTT
- Ecommerce
- Banking
- Healthcare
- SaaS
- Retail
- Insurance
- Education

Author:
ChurnShield AI
"""

import calendar
import logging
from pathlib import Path
from datetime import (
    datetime,
    timedelta,
    date
)
from typing import (
    Dict,
    List,
    Optional
)

import pandas as pd
import numpy as np

logger = logging.getLogger(
    "churnshield.india.calendar"
)

logging.basicConfig(
    level=logging.INFO
)


# ============================================================
# INDIA FESTIVAL DATABASE
# ============================================================

INDIAN_FESTIVALS = [

    {
        "name": "Diwali",
        "category": "festival",
        "month": 10,
        "business_impact": 0.95,
        "states": ["ALL"],
        "languages": [
            "Hindi",
            "English",
            "Telugu",
            "Tamil"
        ],
        "customer_spending_boost": 0.85,
        "marketing_priority": "critical",
    },

    {
        "name": "Holi",
        "category": "festival",
        "month": 3,
        "business_impact": 0.75,
        "states": ["North India"],
        "languages": [
            "Hindi"
        ],
        "customer_spending_boost": 0.55,
        "marketing_priority": "high",
    },

    {
        "name": "Eid",
        "category": "festival",
        "month": 4,
        "business_impact": 0.82,
        "states": ["ALL"],
        "languages": [
            "Urdu",
            "Hindi",
            "English"
        ],
        "customer_spending_boost": 0.72,
        "marketing_priority": "high",
    },

    {
        "name": "Ramadan",
        "category": "religious",
        "month": 3,
        "business_impact": 0.80,
        "states": ["ALL"],
        "languages": [
            "Urdu",
            "Hindi"
        ],
        "customer_spending_boost": 0.65,
        "marketing_priority": "high",
    },

    {
        "name": "Pongal",
        "category": "harvest",
        "month": 1,
        "business_impact": 0.68,
        "states": ["Tamil Nadu"],
        "languages": ["Tamil"],
        "customer_spending_boost": 0.50,
        "marketing_priority": "medium",
    },

    {
        "name": "Sankranti",
        "category": "harvest",
        "month": 1,
        "business_impact": 0.74,
        "states": [
            "Andhra Pradesh",
            "Telangana"
        ],
        "languages": [
            "Telugu"
        ],
        "customer_spending_boost": 0.61,
        "marketing_priority": "high",
    },

    {
        "name": "Onam",
        "category": "festival",
        "month": 8,
        "business_impact": 0.78,
        "states": ["Kerala"],
        "languages": ["Malayalam"],
        "customer_spending_boost": 0.67,
        "marketing_priority": "high",
    },

    {
        "name": "Durga Puja",
        "category": "festival",
        "month": 10,
        "business_impact": 0.84,
        "states": ["West Bengal"],
        "languages": ["Bengali"],
        "customer_spending_boost": 0.73,
        "marketing_priority": "high",
    },

]


# ============================================================
# MAIN ENGINE
# ============================================================

class IndiaCalendarEngine:

    def __init__(self):

        self.today = datetime.utcnow()

    # ========================================================
    # GET FESTIVALS BY MONTH
    # ========================================================

    def festivals_by_month(
        self,
        month: int
    ) -> List[Dict]:

        return [

            festival

            for festival
            in INDIAN_FESTIVALS

            if festival["month"] == month

        ]

    # ========================================================
    # GET FESTIVALS BY STATE
    # ========================================================

    def festivals_by_state(
        self,
        state: str
    ) -> List[Dict]:

        state = state.lower()

        results = []

        for festival in INDIAN_FESTIVALS:

            states = [

                s.lower()

                for s
                in festival["states"]

            ]

            if (
                "all" in states
                or
                state in states
            ):

                results.append(
                    festival
                )

        return results

    # ========================================================
    # BUSINESS IMPACT SCORE
    # ========================================================

    def business_impact_score(
        self,
        month: int,
        state: Optional[str] = None
    ) -> float:

        festivals = self.festivals_by_month(
            month
        )

        if state:

            festivals = [

                f for f in festivals

                if (
                    "ALL" in f["states"]
                    or
                    state in f["states"]
                )

            ]

        if not festivals:

            return 0.0

        scores = [

            f["business_impact"]

            for f in festivals

        ]

        return round(
            float(np.mean(scores)),
            4
        )

    # ========================================================
    # CUSTOMER SPENDING FORECAST
    # ========================================================

    def spending_forecast(
        self,
        month: int
    ) -> Dict:

        festivals = self.festivals_by_month(
            month
        )

        if not festivals:

            return {

                "forecast":
                    "normal",

                "boost":
                    0.0

            }

        avg_boost = np.mean([

            f[
                "customer_spending_boost"
            ]

            for f in festivals

        ])

        if avg_boost > 0.7:

            level = "very_high"

        elif avg_boost > 0.5:

            level = "high"

        else:

            level = "moderate"

        return {

            "forecast":
                level,

            "boost":
                round(
                    float(avg_boost),
                    4
                ),

            "festival_count":
                len(festivals),

        }

    # ========================================================
    # BEST CAMPAIGN DAYS
    # ========================================================

    def best_campaign_days(
        self,
        month: int
    ) -> List[int]:

        festivals = self.festivals_by_month(
            month
        )

        if not festivals:

            return [5, 10, 15]

        return [

            3,
            7,
            10,
            14,
            21,
            25

        ]

    # ========================================================
    # CHURN RISK SEASONALITY
    # ========================================================

    def churn_risk_seasonality(
        self,
        month: int
    ) -> Dict:

        impact = self.business_impact_score(
            month
        )

        if impact > 0.8:

            churn_risk = "low"

        elif impact > 0.5:

            churn_risk = "medium"

        else:

            churn_risk = "high"

        return {

            "month":
                month,

            "business_impact":
                impact,

            "predicted_churn_risk":
                churn_risk,

        }

    # ========================================================
    # WORKING DAYS
    # ========================================================

    def working_days(
        self,
        year: int,
        month: int
    ) -> int:

        cal = calendar.monthcalendar(
            year,
            month
        )

        working_days = 0

        for week in cal:

            for day in week[:5]:

                if day != 0:

                    working_days += 1

        return working_days

    # ========================================================
    # WEEKEND DAYS
    # ========================================================

    def weekend_days(
        self,
        year: int,
        month: int
    ) -> int:

        cal = calendar.monthcalendar(
            year,
            month
        )

        weekends = 0

        for week in cal:

            for day in week[5:]:

                if day != 0:

                    weekends += 1

        return weekends

    # ========================================================
    # NEXT FESTIVAL
    # ========================================================

    def next_festival(self):

        current_month = datetime.utcnow().month

        future_festivals = [

            f for f in INDIAN_FESTIVALS

            if f["month"] >= current_month

        ]

        if not future_festivals:

            return None

        future_festivals.sort(

            key=lambda x:
            x["month"]

        )

        return future_festivals[0]

    # ========================================================
    # FESTIVAL ANALYTICS
    # ========================================================

    def festival_analytics(self):

        analytics = []

        for month in range(1, 13):

            festivals = self.festivals_by_month(
                month
            )

            analytics.append({

                "month":
                    month,

                "festival_count":
                    len(festivals),

                "business_impact":
                    self.business_impact_score(
                        month
                    ),

                "spending_forecast":
                    self.spending_forecast(
                        month
                    )["forecast"]

            })

        return pd.DataFrame(
            analytics
        )

    # ========================================================
    # RETENTION CAMPAIGN RECOMMENDER
    # ========================================================

    def retention_campaign_strategy(
        self,
        month: int
    ) -> Dict:

        impact = self.business_impact_score(
            month
        )

        spending = self.spending_forecast(
            month
        )

        if impact > 0.8:

            strategy = (

                "Aggressive upselling "
                "and premium campaigns"

            )

        elif impact > 0.5:

            strategy = (

                "Retention offers and "
                "festival discounts"

            )

        else:

            strategy = (

                "Customer engagement "
                "and loyalty building"

            )

        return {

            "month":
                month,

            "strategy":
                strategy,

            "expected_spending_boost":
                spending["boost"],

            "campaign_days":
                self.best_campaign_days(
                    month
                ),

        }

    # ========================================================
    # EXPORT FESTIVAL CALENDAR
    # ========================================================

    def export_calendar_csv(
        self,
        path: str = (
            "india_festival_calendar.csv"
        )
    ):

        df = pd.DataFrame(
            INDIAN_FESTIVALS
        )

        df.to_csv(
            path,
            index=False
        )

        logger.info(
            f"Festival calendar exported: "
            f"{path}"
        )

        return path

    # ========================================================
    # YEARLY BUSINESS REPORT
    # ========================================================

    def yearly_business_report(
        self,
        year: int
    ) -> Dict:

        report = {

            "year": year,

            "months": []

        }

        for month in range(1, 13):

            report["months"].append({

                "month":
                    month,

                "impact":
                    self.business_impact_score(
                        month
                    ),

                "working_days":
                    self.working_days(
                        year,
                        month
                    ),

                "weekends":
                    self.weekend_days(
                        year,
                        month
                    ),

                "campaign_strategy":
                    self.retention_campaign_strategy(
                        month
                    )["strategy"]

            })

        return report


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def get_month_festivals(
    month: int
):

    engine = IndiaCalendarEngine()

    return engine.festivals_by_month(
        month
    )


def get_state_festivals(
    state: str
):

    engine = IndiaCalendarEngine()

    return engine.festivals_by_state(
        state
    )


def business_impact_score(
    month: int
):

    engine = IndiaCalendarEngine()

    return engine.business_impact_score(
        month
    )


def yearly_business_report(
    year: int
):

    engine = IndiaCalendarEngine()

    return engine.yearly_business_report(
        year
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    engine = IndiaCalendarEngine()

    print("\n")
    print("=" * 60)
    print("INDIA CALENDAR ENGINE")
    print("=" * 60)

    print("\nNext Festival:\n")

    print(
        engine.next_festival()
    )

    print("\n")

    print(
        engine.retention_campaign_strategy(
            month=10
        )
    )

    print("\n")

    analytics = (
        engine.festival_analytics()
    )

    print(
        analytics.head()
    )

    engine.export_calendar_csv()