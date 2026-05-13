"""
ChurnShield 2.0 — India Demographics Intelligence Engine

Purpose:
India-specific demographic intelligence
for churn prediction, segmentation,
regional analytics, and personalization.

Capabilities:
- age segmentation
- income segmentation
- urban/rural classification
- family size intelligence
- education classification
- occupation intelligence
- gender analytics
- lifestyle segmentation
- spending power estimation
- customer value scoring
- India region mapping
- Bharat vs Metro analysis
- tier city classification
- digital maturity scoring
- affordability scoring
- demographic churn risk analysis

Works For:
- telecom
- fintech
- edtech
- healthcare
- ecommerce
- OTT
- insurance
- SaaS
- banking

Author:
ChurnShield AI
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(
    "churnshield.india.demographics"
)


# ============================================================
# MAIN ENGINE
# ============================================================

class IndiaDemographicsEngine:

    def __init__(self):

        # ----------------------------------------------------
        # TIER CITIES
        # ----------------------------------------------------

        self.tier_1 = [

            "mumbai",
            "delhi",
            "bangalore",
            "hyderabad",
            "chennai",
            "kolkata",
            "pune",
            "ahmedabad",

        ]

        self.tier_2 = [

            "lucknow",
            "indore",
            "bhopal",
            "visakhapatnam",
            "patna",
            "nagpur",
            "coimbatore",
            "vijayawada",
            "warangal",
            "surat",

        ]

        # ----------------------------------------------------
        # OCCUPATION SEGMENTS
        # ----------------------------------------------------

        self.occupation_map = {

            "student":
                "Education",

            "engineer":
                "Technology",

            "developer":
                "Technology",

            "doctor":
                "Healthcare",

            "teacher":
                "Education",

            "business":
                "Business",

            "sales":
                "Sales",

            "marketing":
                "Marketing",

            "bank":
                "Finance",

            "government":
                "Government",

            "farmer":
                "Agriculture",

        }

    # ========================================================
    # MAIN PROCESSOR
    # ========================================================

    def enrich_demographics(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "Running India demographics intelligence"
        )

        data = df.copy()

        # ----------------------------------------------------
        # AGE SEGMENTS
        # ----------------------------------------------------

        if "age" in data.columns:

            data["age_segment"] = (
                data["age"]
                .apply(self._age_segment)
            )

        # ----------------------------------------------------
        # INCOME SEGMENTS
        # ----------------------------------------------------

        if "monthly_income" in data.columns:

            data["income_segment"] = (
                data["monthly_income"]
                .apply(self._income_segment)
            )

        # ----------------------------------------------------
        # CITY TIER
        # ----------------------------------------------------

        if "city" in data.columns:

            data["city_tier"] = (
                data["city"]
                .astype(str)
                .str.lower()
                .apply(self._city_tier)
            )

        # ----------------------------------------------------
        # DIGITAL MATURITY
        # ----------------------------------------------------

        data["digital_maturity_score"] = (
            self._digital_maturity(data)
        )

        # ----------------------------------------------------
        # SPENDING POWER
        # ----------------------------------------------------

        data["spending_power_score"] = (
            self._spending_power(data)
        )

        # ----------------------------------------------------
        # BHARAT VS METRO
        # ----------------------------------------------------

        if "city_tier" in data.columns:

            data["india_market_segment"] = (

                data["city_tier"]
                .apply(self._bharat_segment)

            )

        # ----------------------------------------------------
        # OCCUPATION CATEGORY
        # ----------------------------------------------------

        if "occupation" in data.columns:

            data["occupation_category"] = (

                data["occupation"]
                .astype(str)
                .str.lower()
                .apply(self._occupation_category)

            )

        # ----------------------------------------------------
        # EDUCATION LEVEL
        # ----------------------------------------------------

        if "education" in data.columns:

            data["education_level"] = (

                data["education"]
                .astype(str)
                .str.lower()
                .apply(self._education_level)

            )

        # ----------------------------------------------------
        # FAMILY CATEGORY
        # ----------------------------------------------------

        if "family_size" in data.columns:

            data["family_segment"] = (

                data["family_size"]
                .apply(self._family_segment)

            )

        # ----------------------------------------------------
        # AFFORDABILITY INDEX
        # ----------------------------------------------------

        data["affordability_index"] = (
            self._affordability_index(data)
        )

        # ----------------------------------------------------
        # DEMOGRAPHIC CHURN RISK
        # ----------------------------------------------------

        data["demographic_risk_score"] = (
            self._demographic_risk(data)
        )

        # ----------------------------------------------------
        # CUSTOMER VALUE CLASS
        # ----------------------------------------------------

        data["customer_value_segment"] = (
            self._customer_value_segment(data)
        )

        logger.info(
            "India demographics enrichment completed"
        )

        return data

    # ========================================================
    # AGE SEGMENT
    # ========================================================

    def _age_segment(
        self,
        age
    ):

        try:

            age = float(age)

            if age < 18:
                return "Minor"

            elif age <= 24:
                return "Gen Z"

            elif age <= 34:
                return "Young Adult"

            elif age <= 45:
                return "Professional"

            elif age <= 60:
                return "Mature"

            return "Senior"

        except:
            return "Unknown"

    # ========================================================
    # INCOME SEGMENT
    # ========================================================

    def _income_segment(
        self,
        income
    ):

        try:

            income = float(income)

            if income < 15000:
                return "Low Income"

            elif income < 50000:
                return "Middle Class"

            elif income < 150000:
                return "Upper Middle"

            return "Affluent"

        except:
            return "Unknown"

    # ========================================================
    # CITY TIER
    # ========================================================

    def _city_tier(
        self,
        city
    ):

        city = str(city).lower()

        if city in self.tier_1:
            return "Tier 1"

        elif city in self.tier_2:
            return "Tier 2"

        return "Tier 3"

    # ========================================================
    # DIGITAL MATURITY
    # ========================================================

    def _digital_maturity(
        self,
        df
    ):

        score = np.zeros(len(df))

        # Usage signals
        if "login_frequency" in df.columns:

            score += (

                df["login_frequency"]
                .fillna(0)
                .clip(0, 50)

            ) * 0.5

        # Feature adoption
        if "feature_usage_score" in df.columns:

            score += (

                df["feature_usage_score"]
                .fillna(0)

            ) * 50

        # Younger users
        if "age" in df.columns:

            score += np.where(
                df["age"] < 35,
                20,
                5
            )

        return np.clip(score, 0, 100)

    # ========================================================
    # SPENDING POWER
    # ========================================================

    def _spending_power(
        self,
        df
    ):

        score = np.zeros(len(df))

        if "monthly_income" in df.columns:

            income = (
                df["monthly_income"]
                .fillna(0)
            )

            score += income / max(
                income.max(),
                1
            ) * 70

        if "city_tier" in df.columns:

            score += np.where(
                df["city_tier"] == "Tier 1",
                20,
                10
            )

        return np.clip(score, 0, 100)

    # ========================================================
    # BHARAT SEGMENT
    # ========================================================

    def _bharat_segment(
        self,
        tier
    ):

        if tier == "Tier 1":
            return "Metro India"

        elif tier == "Tier 2":
            return "Emerging India"

        return "Bharat"

    # ========================================================
    # OCCUPATION CATEGORY
    # ========================================================

    def _occupation_category(
        self,
        occupation
    ):

        occupation = str(occupation).lower()

        for keyword, category in (
            self.occupation_map.items()
        ):

            if keyword in occupation:
                return category

        return "Other"

    # ========================================================
    # EDUCATION LEVEL
    # ========================================================

    def _education_level(
        self,
        education
    ):

        education = str(education).lower()

        if any(
            x in education
            for x in [
                "phd",
                "doctorate"
            ]
        ):
            return "Doctorate"

        elif any(
            x in education
            for x in [
                "master",
                "mba",
                "mtech"
            ]
        ):
            return "Postgraduate"

        elif any(
            x in education
            for x in [
                "btech",
                "degree",
                "graduate"
            ]
        ):
            return "Graduate"

        elif any(
            x in education
            for x in [
                "12",
                "intermediate"
            ]
        ):
            return "Intermediate"

        return "School"

    # ========================================================
    # FAMILY SEGMENT
    # ========================================================

    def _family_segment(
        self,
        size
    ):

        try:

            size = int(size)

            if size <= 2:
                return "Small Family"

            elif size <= 5:
                return "Medium Family"

            return "Large Family"

        except:
            return "Unknown"

    # ========================================================
    # AFFORDABILITY INDEX
    # ========================================================

    def _affordability_index(
        self,
        df
    ):

        score = np.zeros(len(df))

        if "monthly_income" in df.columns:

            score += (

                df["monthly_income"]
                .fillna(0)

            ) / 2000

        if "monthly_revenue" in df.columns:

            revenue_ratio = (

                df["monthly_revenue"]
                /
                (
                    df["monthly_income"]
                    .replace(0, 1)
                )

            )

            score -= revenue_ratio * 30

        return np.clip(score, 0, 100)

    # ========================================================
    # DEMOGRAPHIC CHURN RISK
    # ========================================================

    def _demographic_risk(
        self,
        df
    ):

        risk = np.zeros(len(df))

        # Younger users churn more
        if "age" in df.columns:

            risk += np.where(
                df["age"] < 25,
                20,
                5
            )

        # Low income users
        if "monthly_income" in df.columns:

            risk += np.where(
                df["monthly_income"] < 20000,
                25,
                10
            )

        # Rural/Tier3
        if "city_tier" in df.columns:

            risk += np.where(
                df["city_tier"] == "Tier 3",
                20,
                5
            )

        return np.clip(risk, 0, 100)

    # ========================================================
    # CUSTOMER VALUE SEGMENT
    # ========================================================

    def _customer_value_segment(
        self,
        df
    ):

        score = np.zeros(len(df))

        if "monthly_revenue" in df.columns:

            score += (

                df["monthly_revenue"]
                /
                max(
                    df["monthly_revenue"].max(),
                    1
                )

            ) * 60

        if "digital_maturity_score" in df.columns:

            score += (

                df["digital_maturity_score"]
                * 0.4

            )

        segments = []

        for s in score:

            if s >= 75:
                segments.append("Premium")

            elif s >= 45:
                segments.append("High Value")

            elif s >= 20:
                segments.append("Mid Value")

            else:
                segments.append("Low Value")

        return segments

    # ========================================================
    # SUMMARY REPORT
    # ========================================================

    def generate_summary(
        self,
        df
    ) -> Dict:

        summary = {}

        if "city_tier" in df.columns:

            summary["city_distribution"] = (

                df["city_tier"]
                .value_counts()
                .to_dict()

            )

        if "age_segment" in df.columns:

            summary["age_distribution"] = (

                df["age_segment"]
                .value_counts()
                .to_dict()

            )

        if "income_segment" in df.columns:

            summary["income_distribution"] = (

                df["income_segment"]
                .value_counts()
                .to_dict()

            )

        if "customer_value_segment" in df.columns:

            summary["customer_value_distribution"] = (

                df["customer_value_segment"]
                .value_counts()
                .to_dict()

            )

        return summary


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def enrich_india_demographics(
    df: pd.DataFrame
) -> pd.DataFrame:

    engine = IndiaDemographicsEngine()

    return engine.enrich_demographics(df)


def demographics_summary(
    df: pd.DataFrame
) -> Dict:

    engine = IndiaDemographicsEngine()

    return engine.generate_summary(df)