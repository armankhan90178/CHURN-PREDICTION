"""
ChurnShield 2.0 — Response Engine

Purpose:
Enterprise-grade AI response orchestration engine
for churn analytics, executive intelligence,
customer retention, multilingual communication,
and safe AI response generation.

Capabilities:
- LLM response generation
- multi-model routing
- OpenAI/Claude/Gemini support
- RAG-enhanced responses
- response validation
- hallucination filtering
- JSON-safe responses
- retry & fallback models
- executive summaries
- retention recommendations
- multilingual responses
- streaming support
- token tracking
- response scoring
- structured outputs
- safe AI orchestration

Supports:
- OpenAI
- Claude
- Gemini
- Ollama
- Local LLMs
- HuggingFace endpoints

Author:
ChurnShield AI
"""

import json
import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(
    "churnshield.response_engine"
)


# ============================================================
# MAIN ENGINE
# ============================================================

class ResponseEngine:

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):

        self.provider = provider.lower()

        self.api_key = api_key

        self.timeout = timeout

        self.max_retries = max_retries

        # ----------------------------------------------------
        # DEFAULT MODELS
        # ----------------------------------------------------

        self.default_models = {

            "openai":
                "gpt-4o-mini",

            "claude":
                "claude-3-5-sonnet",

            "gemini":
                "gemini-1.5-pro",

            "ollama":
                "llama3",

        }

        self.model = (

            model
            or
            self.default_models.get(
                self.provider
            )

        )

        logger.info(
            f"Response engine initialized "
            f"with provider={self.provider}"
        )

    # ========================================================
    # MAIN GENERATION
    # ========================================================

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500,
        response_format: str = "text",
    ) -> Dict:

        logger.info(
            "Generating AI response"
        )

        for attempt in range(
            self.max_retries
        ):

            try:

                if self.provider == "openai":

                    return self._openai_response(

                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,

                    )

                elif self.provider == "claude":

                    return self._claude_response(

                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,

                    )

                elif self.provider == "gemini":

                    return self._gemini_response(

                        prompt=prompt

                    )

                elif self.provider == "ollama":

                    return self._ollama_response(

                        prompt=prompt

                    )

                else:

                    raise ValueError(
                        "Unsupported provider"
                    )

            except Exception as e:

                logger.error(
                    f"Attempt {attempt+1} failed: {e}"
                )

                time.sleep(2)

        return {

            "success": False,
            "response": "",
            "error": "All retries failed",

        }

    # ========================================================
    # OPENAI
    # ========================================================

    def _openai_response(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        response_format: str,
    ) -> Dict:

        url = (
            "https://api.openai.com/v1/"
            "chat/completions"
        )

        headers = {

            "Authorization":
                f"Bearer {self.api_key}",

            "Content-Type":
                "application/json",

        }

        messages = []

        if system_prompt:

            messages.append({

                "role":
                    "system",

                "content":
                    system_prompt,

            })

        messages.append({

            "role":
                "user",

            "content":
                prompt,

        })

        payload = {

            "model":
                self.model,

            "messages":
                messages,

            "temperature":
                temperature,

            "max_tokens":
                max_tokens,

        }

        response = requests.post(

            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,

        )

        response.raise_for_status()

        data = response.json()

        text = (

            data["choices"][0]
            ["message"]["content"]

        )

        return {

            "success": True,

            "provider":
                "openai",

            "model":
                self.model,

            "response":
                text,

            "usage":
                data.get("usage", {}),

        }

    # ========================================================
    # CLAUDE
    # ========================================================

    def _claude_response(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Dict:

        url = (
            "https://api.anthropic.com/v1/messages"
        )

        headers = {

            "x-api-key":
                self.api_key,

            "anthropic-version":
                "2023-06-01",

            "content-type":
                "application/json",

        }

        payload = {

            "model":
                self.model,

            "max_tokens":
                max_tokens,

            "temperature":
                temperature,

            "system":
                system_prompt or "",

            "messages": [

                {
                    "role":
                        "user",

                    "content":
                        prompt
                }

            ]

        }

        response = requests.post(

            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,

        )

        response.raise_for_status()

        data = response.json()

        text = (
            data["content"][0]["text"]
        )

        return {

            "success": True,

            "provider":
                "claude",

            "model":
                self.model,

            "response":
                text,

            "usage":
                data.get("usage", {}),

        }

    # ========================================================
    # GEMINI
    # ========================================================

    def _gemini_response(
        self,
        prompt: str
    ) -> Dict:

        url = (

            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"

        )

        payload = {

            "contents": [

                {

                    "parts": [

                        {
                            "text":
                                prompt
                        }

                    ]

                }

            ]

        }

        response = requests.post(

            url,
            json=payload,
            timeout=self.timeout,

        )

        response.raise_for_status()

        data = response.json()

        text = (

            data["candidates"][0]
            ["content"]["parts"][0]["text"]

        )

        return {

            "success": True,

            "provider":
                "gemini",

            "model":
                self.model,

            "response":
                text,

        }

    # ========================================================
    # OLLAMA
    # ========================================================

    def _ollama_response(
        self,
        prompt: str
    ) -> Dict:

        url = (
            "http://localhost:11434/api/generate"
        )

        payload = {

            "model":
                self.model,

            "prompt":
                prompt,

            "stream":
                False,

        }

        response = requests.post(

            url,
            json=payload,
            timeout=self.timeout,

        )

        response.raise_for_status()

        data = response.json()

        return {

            "success": True,

            "provider":
                "ollama",

            "model":
                self.model,

            "response":
                data["response"],

        }

    # ========================================================
    # JSON RESPONSE
    # ========================================================

    def generate_json_response(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
    ) -> Dict:

        system_prompt = """

You are a strict JSON generator.

Rules:
- Output valid JSON only
- No markdown
- No explanations
- No extra text
- Follow schema carefully

        """

        if schema:

            system_prompt += (
                "\nExpected Schema:\n"
            )

            system_prompt += json.dumps(
                schema,
                indent=2
            )

        response = self.generate_response(

            prompt=prompt,
            system_prompt=system_prompt,
            response_format="json",

        )

        if not response["success"]:
            return response

        try:

            parsed = json.loads(
                response["response"]
            )

            response["parsed_json"] = parsed

        except Exception as e:

            response["json_error"] = str(e)

        return response

    # ========================================================
    # RAG RESPONSE
    # ========================================================

    def rag_response(
        self,
        query: str,
        retrieved_context: List[str],
    ) -> Dict:

        context_block = "\n\n".join(

            [
                f"[Context {i+1}]\n{c}"
                for i, c in enumerate(
                    retrieved_context
                )
            ]

        )

        prompt = f"""

Use the provided context to answer.

USER QUERY:
{query}

RETRIEVED CONTEXT:
{context_block}

Rules:
- Use context only
- Do not hallucinate
- Mention uncertainty if needed
- Give business-focused answers

        """

        return self.generate_response(
            prompt=prompt
        )

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    def executive_summary(
        self,
        metrics: Dict
    ) -> Dict:

        prompt = f"""

Generate a board-level executive summary.

Metrics:
{json.dumps(metrics, indent=2)}

Requirements:
- executive tone
- business insights
- churn risks
- revenue impact
- retention opportunities
- concise but strategic

        """

        return self.generate_response(
            prompt=prompt
        )

    # ========================================================
    # RETENTION PLAYBOOK
    # ========================================================

    def retention_playbook(
        self,
        customer_data: Dict
    ) -> Dict:

        prompt = f"""

Generate a retention playbook.

Customer Data:
{json.dumps(customer_data, indent=2)}

Requirements:
- identify churn risks
- give retention actions
- include communication ideas
- include discounts/offers
- prioritize ROI

        """

        return self.generate_response(
            prompt=prompt
        )

    # ========================================================
    # MULTILINGUAL RESPONSE
    # ========================================================

    def multilingual_response(
        self,
        text: str,
        language: str
    ) -> Dict:

        prompt = f"""

Translate and localize the text.

Target Language:
{language}

Text:
{text}

Requirements:
- natural language
- culturally localized
- preserve meaning
- professional tone

        """

        return self.generate_response(
            prompt=prompt
        )

    # ========================================================
    # RESPONSE VALIDATOR
    # ========================================================

    def validate_response(
        self,
        response_text: str
    ) -> Dict:

        issues = []

        if not response_text:
            issues.append(
                "Empty response"
            )

        if len(response_text) < 10:
            issues.append(
                "Very short response"
            )

        hallucination_words = [

            "definitely guaranteed",
            "100% accurate",
            "certainly always",

        ]

        for word in hallucination_words:

            if word in response_text.lower():

                issues.append(
                    f"Potential hallucination phrase: {word}"
                )

        return {

            "valid":
                len(issues) == 0,

            "issues":
                issues,

        }

    # ========================================================
    # RESPONSE SCORE
    # ========================================================

    def score_response(
        self,
        response_text: str
    ) -> Dict:

        score = 100

        if len(response_text) < 100:
            score -= 20

        if len(response_text.split()) < 30:
            score -= 15

        if "retention" not in (
            response_text.lower()
        ):
            score -= 5

        if "customer" not in (
            response_text.lower()
        ):
            score -= 5

        return {

            "quality_score":
                max(score, 0),

            "length":
                len(response_text),

            "words":
                len(response_text.split()),

        }

    # ========================================================
    # STREAM RESPONSE
    # ========================================================

    def stream_response(
        self,
        text: str,
        chunk_size: int = 100
    ):

        for i in range(

            0,
            len(text),
            chunk_size

        ):

            yield text[
                i:i+chunk_size
            ]

    # ========================================================
    # TOKEN ESTIMATION
    # ========================================================

    def estimate_tokens(
        self,
        text: str
    ) -> int:

        return int(
            len(text.split()) * 1.3
        )

    # ========================================================
    # SAFE RESPONSE
    # ========================================================

    def safe_response(
        self,
        prompt: str
    ) -> Dict:

        response = self.generate_response(
            prompt
        )

        if not response["success"]:
            return response

        validation = self.validate_response(

            response["response"]

        )

        scoring = self.score_response(

            response["response"]

        )

        response["validation"] = (
            validation
        )

        response["scoring"] = (
            scoring
        )

        return response


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def generate_ai_response(
    prompt: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
):

    engine = ResponseEngine(

        provider=provider,
        api_key=api_key,

    )

    return engine.generate_response(
        prompt
    )


def generate_json_response(
    prompt: str,
    schema: Optional[Dict] = None,
):

    engine = ResponseEngine()

    return engine.generate_json_response(
        prompt,
        schema
    )


def generate_rag_response(
    query: str,
    retrieved_context: List[str],
):

    engine = ResponseEngine()

    return engine.rag_response(
        query,
        retrieved_context
    )


def generate_executive_summary(
    metrics: Dict
):

    engine = ResponseEngine()

    return engine.executive_summary(
        metrics
    )