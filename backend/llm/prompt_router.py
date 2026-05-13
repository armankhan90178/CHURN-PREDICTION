"""
ChurnShield 2.0 — Intelligent Prompt Router

Purpose:
Dynamically select and optimize prompts
for churn prediction, retention intelligence,
executive reporting, multilingual communication,
and AI reasoning workflows.

Capabilities:
- industry prompt routing
- smart prompt selection
- dynamic template injection
- customer-context prompting
- multilingual prompt adaptation
- retention strategy prompts
- RAG-aware prompt building
- executive insight prompts
- communication prompt routing
- token optimization
- hallucination-safe prompt generation
- semantic intent classification
- fallback routing
- universal business support

Supports:
- OpenAI
- Claude
- Gemini
- Local LLMs
- Ollama
- HuggingFace

Author:
ChurnShield AI
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(
    "churnshield.prompt_router"
)


# ============================================================
# MAIN ENGINE
# ============================================================

class PromptRouter:

    def __init__(
        self,
        prompt_dir: str = "llm/prompts"
    ):

        self.prompt_dir = Path(prompt_dir)

        # ----------------------------------------------------
        # PROMPT TYPES
        # ----------------------------------------------------

        self.prompt_map = {

            "executive_summary":
                "insight_extractor.txt",

            "playbook":
                "playbook_base.txt",

            "column_mapping":
                "column_mapper.txt",

            "email":
                "communication/email.txt",

            "sms":
                "communication/sms.txt",

            "whatsapp":
                "communication/whatsapp.txt",

        }

        # ----------------------------------------------------
        # INDUSTRY KEYWORDS
        # ----------------------------------------------------

        self.industry_keywords = {

            "telecom": [
                "sim",
                "network",
                "recharge",
                "postpaid",
                "prepaid",
                "data usage",
            ],

            "ott": [
                "streaming",
                "subscription",
                "watch time",
                "movies",
                "shows",
            ],

            "ecommerce": [
                "orders",
                "cart",
                "purchase",
                "delivery",
                "shopping",
            ],

            "healthcare": [
                "patient",
                "hospital",
                "doctor",
                "appointment",
            ],

            "banking": [
                "loan",
                "credit",
                "account",
                "bank",
                "emi",
            ],

            "insurance": [
                "policy",
                "claim",
                "premium",
            ],

            "b2b_saas": [
                "mrr",
                "subscription",
                "seats",
                "workspace",
                "api",
            ],

            "edtech": [
                "course",
                "student",
                "learning",
                "education",
            ],

            "gaming": [
                "matches",
                "battle",
                "coins",
                "gameplay",
            ],

        }

        # ----------------------------------------------------
        # TASK INTENTS
        # ----------------------------------------------------

        self.intent_map = {

            "summary": [
                "summary",
                "overview",
                "executive",
                "insight",
            ],

            "playbook": [
                "retention",
                "strategy",
                "playbook",
                "save customers",
            ],

            "communication": [
                "email",
                "sms",
                "whatsapp",
                "message",
            ],

            "prediction": [
                "predict",
                "forecast",
                "churn",
            ],

            "explanation": [
                "why",
                "reason",
                "explain",
            ],

        }

    # ========================================================
    # DETECT INDUSTRY
    # ========================================================

    def detect_industry(
        self,
        text: str
    ) -> str:

        text = str(text).lower()

        scores = {}

        for industry, keywords in (
            self.industry_keywords.items()
        ):

            score = 0

            for keyword in keywords:

                if keyword in text:
                    score += 1

            scores[industry] = score

        best = max(
            scores,
            key=scores.get
        )

        if scores[best] == 0:
            return "general"

        return best

    # ========================================================
    # DETECT INTENT
    # ========================================================

    def detect_intent(
        self,
        query: str
    ) -> str:

        query = str(query).lower()

        scores = {}

        for intent, keywords in (
            self.intent_map.items()
        ):

            score = 0

            for keyword in keywords:

                if keyword in query:
                    score += 1

            scores[intent] = score

        best = max(
            scores,
            key=scores.get
        )

        if scores[best] == 0:
            return "summary"

        return best

    # ========================================================
    # LOAD PROMPT
    # ========================================================

    def load_prompt(
        self,
        relative_path: str
    ) -> str:

        path = self.prompt_dir / relative_path

        if not path.exists():

            logger.warning(
                f"Prompt file missing: {path}"
            )

            return ""

        try:

            return path.read_text(
                encoding="utf-8"
            )

        except Exception as e:

            logger.error(
                f"Prompt load failed: {e}"
            )

            return ""

    # ========================================================
    # INDUSTRY PROMPT
    # ========================================================

    def load_industry_prompt(
        self,
        industry: str
    ) -> str:

        industry_path = (

            self.prompt_dir
            /
            "industry_prompt.txt"

        )

        if not industry_path.exists():
            return ""

        try:

            template = (
                industry_path.read_text(
                    encoding="utf-8"
                )
            )

            return template.replace(
                "{{industry}}",
                industry
            )

        except Exception:

            return ""

    # ========================================================
    # SMART ROUTER
    # ========================================================

    def route_prompt(
        self,
        query: str,
        context: Optional[Dict] = None,
        customer_data: Optional[Dict] = None,
    ) -> Dict:

        logger.info(
            "Routing intelligent prompt"
        )

        context = context or {}

        # ----------------------------------------------------
        # DETECT
        # ----------------------------------------------------

        industry = self.detect_industry(
            query
        )

        intent = self.detect_intent(
            query
        )

        # ----------------------------------------------------
        # SELECT BASE PROMPT
        # ----------------------------------------------------

        if intent == "summary":

            prompt_file = (
                self.prompt_map[
                    "executive_summary"
                ]
            )

        elif intent == "playbook":

            prompt_file = (
                self.prompt_map[
                    "playbook"
                ]
            )

        elif intent == "communication":

            communication_type = (
                context.get(
                    "channel",
                    "email"
                )
            )

            prompt_file = (
                self.prompt_map.get(
                    communication_type,
                    "communication/email.txt"
                )
            )

        elif intent == "prediction":

            prompt_file = (
                self.prompt_map[
                    "executive_summary"
                ]
            )

        else:

            prompt_file = (
                self.prompt_map[
                    "executive_summary"
                ]
            )

        # ----------------------------------------------------
        # LOAD PROMPTS
        # ----------------------------------------------------

        base_prompt = self.load_prompt(
            prompt_file
        )

        industry_prompt = (
            self.load_industry_prompt(
                industry
            )
        )

        # ----------------------------------------------------
        # BUILD
        # ----------------------------------------------------

        final_prompt = self.build_prompt(

            base_prompt=base_prompt,
            industry_prompt=industry_prompt,
            query=query,
            context=context,
            customer_data=customer_data,
            industry=industry,
            intent=intent,
        )

        return {

            "industry":
                industry,

            "intent":
                intent,

            "prompt_file":
                prompt_file,

            "final_prompt":
                final_prompt,

        }

    # ========================================================
    # BUILD FINAL PROMPT
    # ========================================================

    def build_prompt(
        self,
        base_prompt: str,
        industry_prompt: str,
        query: str,
        context: Dict,
        customer_data: Optional[Dict],
        industry: str,
        intent: str,
    ) -> str:

        sections = []

        # ----------------------------------------------------
        # SYSTEM
        # ----------------------------------------------------

        sections.append(
            "You are ChurnShield AI, "
            "an enterprise churn intelligence system."
        )

        # ----------------------------------------------------
        # INDUSTRY
        # ----------------------------------------------------

        sections.append(
            f"Industry Context: {industry}"
        )

        if industry_prompt:

            sections.append(
                industry_prompt
            )

        # ----------------------------------------------------
        # TASK
        # ----------------------------------------------------

        sections.append(
            f"Task Intent: {intent}"
        )

        # ----------------------------------------------------
        # BASE PROMPT
        # ----------------------------------------------------

        if base_prompt:

            sections.append(
                base_prompt
            )

        # ----------------------------------------------------
        # CUSTOMER DATA
        # ----------------------------------------------------

        if customer_data:

            sections.append(
                "Customer Data:"
            )

            for k, v in (
                customer_data.items()
            ):

                sections.append(
                    f"{k}: {v}"
                )

        # ----------------------------------------------------
        # EXTRA CONTEXT
        # ----------------------------------------------------

        if context:

            sections.append(
                "Additional Context:"
            )

            for k, v in context.items():

                sections.append(
                    f"{k}: {v}"
                )

        # ----------------------------------------------------
        # QUERY
        # ----------------------------------------------------

        sections.append(
            f"User Query: {query}"
        )

        # ----------------------------------------------------
        # RULES
        # ----------------------------------------------------

        sections.append(

            """
Rules:
- Be business focused
- Avoid hallucinations
- Give actionable insights
- Explain churn drivers clearly
- Use concise executive language
- Focus on retention outcomes
- Prioritize revenue preservation
            """

        )

        return "\n\n".join(sections)

    # ========================================================
    # TOKEN OPTIMIZER
    # ========================================================

    def optimize_prompt(
        self,
        prompt: str,
        max_chars: int = 12000,
    ) -> str:

        if len(prompt) <= max_chars:
            return prompt

        logger.warning(
            "Prompt exceeded limit. Trimming."
        )

        return prompt[:max_chars]

    # ========================================================
    # DATAFRAME CONTEXT
    # ========================================================

    def dataframe_context(
        self,
        df: pd.DataFrame
    ) -> Dict:

        context = {}

        context["rows"] = len(df)

        context["columns"] = (
            list(df.columns)
        )

        if "churned" in df.columns:

            context["churn_rate"] = round(

                float(
                    df["churned"].mean()
                ) * 100,

                2
            )

        if "monthly_revenue" in df.columns:

            context["monthly_revenue"] = round(

                float(
                    df["monthly_revenue"]
                    .sum()
                ),

                2
            )

        return context

    # ========================================================
    # RAG QUERY BUILDER
    # ========================================================

    def rag_prompt(
        self,
        query: str,
        retrieved_docs: list,
    ) -> str:

        sections = []

        sections.append(

            "Answer the query using only "
            "the provided context."

        )

        sections.append(
            f"User Query: {query}"
        )

        sections.append(
            "Retrieved Context:"
        )

        for idx, doc in enumerate(
            retrieved_docs
        ):

            sections.append(

                f"[Document {idx+1}]"

            )

            sections.append(str(doc))

        sections.append(

            """
Instructions:
- Use retrieved context
- Avoid fabricated information
- Give precise insights
- Mention uncertainty if needed
            """

        )

        return "\n\n".join(sections)

    # ========================================================
    # COMMUNICATION PROMPT
    # ========================================================

    def communication_prompt(
        self,
        customer_name: str,
        churn_reason: str,
        language: str = "English",
        channel: str = "email",
    ) -> str:

        template = f"""

Generate a {channel} retention communication.

Customer Name:
{customer_name}

Primary Churn Reason:
{churn_reason}

Language:
{language}

Requirements:
- Professional tone
- Personalized
- Persuasive but respectful
- Encourage re-engagement
- Avoid spammy language
- Include retention incentive
        """

        return template.strip()

    # ========================================================
    # EXECUTIVE INSIGHT PROMPT
    # ========================================================

    def executive_prompt(
        self,
        metrics: Dict
    ) -> str:

        sections = []

        sections.append(

            "Generate a board-level executive "
            "summary for churn analytics."

        )

        sections.append(
            "Business Metrics:"
        )

        for k, v in metrics.items():

            sections.append(
                f"{k}: {v}"
            )

        sections.append(

            """
Focus Areas:
- churn risk
- revenue impact
- customer health
- retention strategy
- operational recommendations
- executive storytelling
            """

        )

        return "\n".join(sections)


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def route_prompt(
    query: str,
    context: Optional[Dict] = None,
    customer_data: Optional[Dict] = None,
):

    router = PromptRouter()

    return router.route_prompt(
        query=query,
        context=context,
        customer_data=customer_data,
    )


def build_rag_prompt(
    query: str,
    retrieved_docs: list,
):

    router = PromptRouter()

    return router.rag_prompt(
        query,
        retrieved_docs
    )


def communication_prompt(
    customer_name: str,
    churn_reason: str,
    language: str = "English",
    channel: str = "email",
):

    router = PromptRouter()

    return router.communication_prompt(
        customer_name,
        churn_reason,
        language,
        channel
    )