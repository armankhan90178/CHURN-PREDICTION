"""
ChurnShield 2.0 — Hyper Sentiment Intelligence Engine

Purpose:
Analyze customer communication sentiment,
emotion, escalation probability,
and churn psychology from support tickets,
emails, reviews, chats, and feedback.

Capabilities:
- sentiment analysis
- emotion intelligence
- churn emotion detection
- escalation prediction
- urgency detection
- frustration scoring
- loyalty estimation
- psychological risk analysis
- enterprise complaint intelligence
- executive customer mood insights
- multilingual-ready architecture
- behavioral NLP analytics
"""

import re
import logging
import warnings
import numpy as np
import pandas as pd

from typing import Dict
from collections import Counter

warnings.filterwarnings("ignore")

logger = logging.getLogger(
    "churnshield.sentiment"
)


class HyperSentimentEngine:

    def __init__(self):

        # ─────────────────────────────────────
        # POSITIVE SIGNALS
        # ─────────────────────────────────────

        self.positive_words = {

            "good", "great", "excellent",
            "awesome", "amazing", "fantastic",
            "love", "liked", "happy",
            "satisfied", "smooth", "fast",
            "helpful", "perfect", "wonderful",
            "easy", "resolved", "thanks",
            "thankyou", "best", "valuable",
            "efficient", "impressive",
            "brilliant", "nice", "super",
            "reliable", "stable", "trusted",
            "recommended", "success",
            "improved", "responsive",
        }

        # ─────────────────────────────────────
        # NEGATIVE SIGNALS
        # ─────────────────────────────────────

        self.negative_words = {

            "bad", "worst", "terrible",
            "awful", "hate", "angry",
            "slow", "issue", "problem",
            "delay", "frustrated", "annoying",
            "useless", "broken", "bug",
            "refund", "cancel", "cancelling",
            "disappointed", "poor",
            "failure", "pathetic",
            "unhappy", "confusing",
            "difficult", "complicated",
            "expensive", "overpriced",
            "unresponsive", "spam",
            "switching", "competitor",
            "downtime", "error",
            "crash", "unstable",
            "fake", "misleading",
            "waste", "quit",
        }

        # ─────────────────────────────────────
        # HIGH RISK CHURN SIGNALS
        # ─────────────────────────────────────

        self.churn_keywords = {

            "cancel",
            "unsubscribe",
            "leaving",
            "switching",
            "moving",
            "competitor",
            "refund",
            "close account",
            "not renewing",
            "terminate",
            "stop using",
            "discontinue",
            "frustrated",
            "disappointed",
        }

        # ─────────────────────────────────────
        # EMOTION CLASSES
        # ─────────────────────────────────────

        self.emotion_map = {

            "anger": [
                "angry", "worst", "hate",
                "terrible", "pathetic",
                "useless", "annoying",
                "frustrated",
            ],

            "sadness": [
                "disappointed", "unhappy",
                "sad", "upset",
                "bad experience",
            ],

            "trust": [
                "reliable", "trusted",
                "great", "excellent",
                "satisfied",
            ],

            "joy": [
                "awesome", "amazing",
                "fantastic", "love",
                "happy",
            ],

            "fear": [
                "worried", "unsafe",
                "risk", "concern",
                "scared",
            ],
        }

    # ─────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
    ) -> pd.DataFrame:

        logger.info(
            "Starting hyper sentiment analysis"
        )

        data = df.copy()

        if text_column not in data.columns:

            logger.warning(
                f"Missing text column: {text_column}"
            )

            data[text_column] = ""

        results = []

        for _, row in data.iterrows():

            text = str(
                row.get(
                    text_column,
                    ""
                )
            )

            cleaned = self._clean_text(
                text
            )

            sentiment_score = self._sentiment_score(
                cleaned
            )

            sentiment_label = self._sentiment_label(
                sentiment_score
            )

            emotion = self._detect_emotion(
                cleaned
            )

            escalation_risk = self._escalation_risk(
                cleaned,
                sentiment_score,
            )

            churn_intent = self._churn_intent_score(
                cleaned
            )

            frustration_score = self._frustration_score(
                cleaned
            )

            urgency_level = self._urgency_level(
                cleaned
            )

            loyalty_signal = self._loyalty_signal(
                cleaned
            )

            confidence = self._confidence(
                cleaned,
                sentiment_score,
            )

            executive_summary = self._executive_summary(
                sentiment_label,
                emotion,
                churn_intent,
            )

            business_risk = self._business_risk(
                sentiment_score,
                churn_intent,
            )

            results.append({

                "sentiment_score":
                    sentiment_score,

                "sentiment_label":
                    sentiment_label,

                "dominant_emotion":
                    emotion,

                "escalation_risk":
                    escalation_risk,

                "churn_intent_score":
                    churn_intent,

                "frustration_score":
                    frustration_score,

                "urgency_level":
                    urgency_level,

                "loyalty_signal":
                    loyalty_signal,

                "sentiment_confidence":
                    confidence,

                "business_risk":
                    business_risk,

                "executive_sentiment_summary":
                    executive_summary,
            })

        sentiment_df = pd.DataFrame(
            results
        )

        data = pd.concat(
            [
                data.reset_index(drop=True),
                sentiment_df.reset_index(drop=True),
            ],
            axis=1,
        )

        logger.info(
            "Sentiment analysis completed"
        )

        return data

    # ─────────────────────────────────────────────
    # CLEAN TEXT
    # ─────────────────────────────────────────────

    def _clean_text(
        self,
        text,
    ):

        text = text.lower()

        text = re.sub(
            r"http\S+",
            "",
            text
        )

        text = re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    # ─────────────────────────────────────────────
    # SENTIMENT SCORE
    # ─────────────────────────────────────────────

    def _sentiment_score(
        self,
        text,
    ):

        words = text.split()

        positive = sum(
            1 for w in words
            if w in self.positive_words
        )

        negative = sum(
            1 for w in words
            if w in self.negative_words
        )

        total = max(
            positive + negative,
            1
        )

        score = (
            positive - negative
        ) / total

        normalized = (
            score + 1
        ) / 2

        return round(
            normalized,
            4
        )

    # ─────────────────────────────────────────────
    # LABEL
    # ─────────────────────────────────────────────

    def _sentiment_label(
        self,
        score,
    ):

        if score >= 0.75:
            return "Very Positive"

        if score >= 0.60:
            return "Positive"

        if score >= 0.45:
            return "Neutral"

        if score >= 0.25:
            return "Negative"

        return "Highly Negative"

    # ─────────────────────────────────────────────
    # EMOTION DETECTION
    # ─────────────────────────────────────────────

    def _detect_emotion(
        self,
        text,
    ):

        emotion_scores = {}

        for emotion, keywords in self.emotion_map.items():

            score = sum(
                text.count(k)
                for k in keywords
            )

            emotion_scores[emotion] = score

        dominant = max(
            emotion_scores,
            key=emotion_scores.get
        )

        if emotion_scores[dominant] == 0:
            return "Neutral"

        return dominant.capitalize()

    # ─────────────────────────────────────────────
    # ESCALATION RISK
    # ─────────────────────────────────────────────

    def _escalation_risk(
        self,
        text,
        sentiment_score,
    ):

        risk = 0

        if sentiment_score < 0.30:
            risk += 50

        if any(
            k in text
            for k in [
                "manager",
                "legal",
                "complaint",
                "lawsuit",
                "escalate",
                "refund",
            ]
        ):
            risk += 35

        if "!!!" in text:
            risk += 15

        if risk >= 70:
            return "Critical"

        if risk >= 40:
            return "High"

        if risk >= 20:
            return "Moderate"

        return "Low"

    # ─────────────────────────────────────────────
    # CHURN INTENT
    # ─────────────────────────────────────────────

    def _churn_intent_score(
        self,
        text,
    ):

        count = sum(
            1
            for k in self.churn_keywords
            if k in text
        )

        score = min(
            count / 5,
            1
        )

        return round(
            score,
            4
        )

    # ─────────────────────────────────────────────
    # FRUSTRATION SCORE
    # ─────────────────────────────────────────────

    def _frustration_score(
        self,
        text,
    ):

        frustration_terms = [

            "issue",
            "problem",
            "delay",
            "bad",
            "slow",
            "angry",
            "hate",
            "terrible",
            "worst",
            "annoying",
        ]

        count = sum(
            text.count(t)
            for t in frustration_terms
        )

        score = min(
            count / 10,
            1
        )

        return round(
            score,
            4
        )

    # ─────────────────────────────────────────────
    # URGENCY
    # ─────────────────────────────────────────────

    def _urgency_level(
        self,
        text,
    ):

        urgent_words = [

            "immediately",
            "urgent",
            "asap",
            "today",
            "critical",
            "now",
            "serious",
            "emergency",
        ]

        count = sum(
            1 for w in urgent_words
            if w in text
        )

        if count >= 3:
            return "Critical"

        if count >= 2:
            return "High"

        if count >= 1:
            return "Moderate"

        return "Low"

    # ─────────────────────────────────────────────
    # LOYALTY SIGNAL
    # ─────────────────────────────────────────────

    def _loyalty_signal(
        self,
        text,
    ):

        loyalty_terms = [

            "love",
            "trusted",
            "loyal",
            "using for years",
            "happy",
            "recommended",
            "great experience",
        ]

        score = sum(
            1
            for t in loyalty_terms
            if t in text
        )

        if score >= 3:
            return "Strong"

        if score >= 1:
            return "Moderate"

        return "Weak"

    # ─────────────────────────────────────────────
    # CONFIDENCE
    # ─────────────────────────────────────────────

    def _confidence(
        self,
        text,
        score,
    ):

        word_count = len(
            text.split()
        )

        density = abs(
            score - 0.5
        ) * 2

        confidence = (
            min(word_count / 20, 1) * 0.5
        ) + (
            density * 0.5
        )

        return round(
            confidence,
            4
        )

    # ─────────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────

    def _executive_summary(
        self,
        sentiment,
        emotion,
        churn_intent,
    ):

        return (
            f"Customer communication shows "
            f"{sentiment.lower()} sentiment with "
            f"{emotion.lower()} emotional dominance. "
            f"Detected churn intent level: "
            f"{round(churn_intent * 100, 1)}%."
        )

    # ─────────────────────────────────────────────
    # BUSINESS RISK
    # ─────────────────────────────────────────────

    def _business_risk(
        self,
        sentiment_score,
        churn_intent,
    ):

        combined = (
            (1 - sentiment_score) * 0.6
        ) + (
            churn_intent * 0.4
        )

        if combined >= 0.80:
            return "Critical"

        if combined >= 0.60:
            return "High"

        if combined >= 0.35:
            return "Moderate"

        return "Low"

    # ─────────────────────────────────────────────
    # PORTFOLIO ANALYTICS
    # ─────────────────────────────────────────────

    def portfolio_summary(
        self,
        analyzed_df,
    ) -> Dict:

        total = len(
            analyzed_df
        )

        negative = len(
            analyzed_df[
                analyzed_df[
                    "sentiment_label"
                ].isin([
                    "Negative",
                    "Highly Negative"
                ])
            ]
        )

        escalation = len(
            analyzed_df[
                analyzed_df[
                    "escalation_risk"
                ].isin([
                    "High",
                    "Critical"
                ])
            ]
        )

        avg_sentiment = analyzed_df[
            "sentiment_score"
        ].mean()

        avg_churn_intent = analyzed_df[
            "churn_intent_score"
        ].mean()

        emotion_distribution = dict(

            Counter(
                analyzed_df[
                    "dominant_emotion"
                ]
            )

        )

        return {

            "records_analyzed":
                int(total),

            "negative_sentiment_cases":
                int(negative),

            "high_escalation_cases":
                int(escalation),

            "average_sentiment_score":
                round(
                    float(avg_sentiment),
                    4,
                ),

            "average_churn_intent":
                round(
                    float(avg_churn_intent),
                    4,
                ),

            "emotion_distribution":
                emotion_distribution,
        }


# ─────────────────────────────────────────────
# FUNCTIONAL INTERFACE
# ─────────────────────────────────────────────

def analyze_sentiment(
    df: pd.DataFrame,
    text_column: str = "text",
):

    engine = HyperSentimentEngine()

    return engine.analyze(
        df=df,
        text_column=text_column,
    )