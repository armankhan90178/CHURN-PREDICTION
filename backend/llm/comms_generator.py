"""
ChurnShield 2.0 — llm/comms_generator.py

Generates ready-to-send communication drafts for every at-risk customer.
CS team should NOT have to write anything — just click send.

For each HIGH/MEDIUM risk customer generates:
  1. Email         — formal, personalized, with subject line
  2. WhatsApp      — short, casual, conversational
  3. SMS           — under 160 characters
  4. Call Script   — bullet points for phone call
  5. LinkedIn DM   — professional outreach (B2B only)

Tone adapts based on:
  - Customer persona (Power User / Value Seeker / etc.)
  - Churn reason (Price / Support Failure / Low Adoption / etc.)
  - Industry (gym vs SaaS vs OTT vs banking)
  - Contract age (new customer vs long-term)
  - Risk level (CRITICAL = personal tone / MEDIUM = softer)
  - India cultural context (relationship-first, regional language option)
  - Current India calendar event (Diwali offer / GST season empathy)
"""

import json
import logging
import re
from datetime import datetime
from typing import Optional
import anthropic

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MAX_TOKENS,
    INDIA_CALENDAR,
    SUPPORTED_LANGUAGES,
)

logger = logging.getLogger("churnshield.comms_generator")

# ─────────────────────────────────────────────────────────────
# PERSONA → TONE MAPPING
# ─────────────────────────────────────────────────────────────
PERSONA_TONE = {
    "Power User": {
        "style":    "technical and data-driven",
        "lead_with":"usage insights and advanced features they are missing",
        "avoid":    "generic marketing language or basic tips",
    },
    "Passive Subscriber": {
        "style":    "simple, warm, benefit-focused",
        "lead_with":"the core value they are getting but not using",
        "avoid":    "overwhelming them with features or data",
    },
    "Value Seeker": {
        "style":    "ROI-focused, cost-saving angle",
        "lead_with":"money saved, ROI achieved, or annual plan savings",
        "avoid":    "feature talk without connecting to cost/value",
    },
    "Relationship Buyer": {
        "style":    "personal, warm, relationship-first",
        "lead_with":"personal check-in, genuine care, no hard sell",
        "avoid":    "automated-sounding language or pushy offers",
    },
    "ROI Tracker": {
        "style":    "data-heavy, metrics-focused, business case",
        "lead_with":"numbers, ROI report, measurable outcomes",
        "avoid":    "emotional appeals without data backing",
    },
}

# ─────────────────────────────────────────────────────────────
# CHURN REASON → STRATEGY MAPPING
# ─────────────────────────────────────────────────────────────
REASON_STRATEGY = {
    "Price Sensitivity": {
        "primary_action": "Lead with annual plan savings or discount offer",
        "avoid":          "Feature talk before addressing cost concern",
        "offer_hint":     "Calculate exact rupee savings on annual plan",
    },
    "Product Dissatisfaction": {
        "primary_action": "Acknowledge issues first, then show resolution path",
        "avoid":          "Ignoring complaints and jumping to retention offer",
        "offer_hint":     "Free training session or dedicated support upgrade",
    },
    "Competitor Switch": {
        "primary_action": "Understand what competitor offers, highlight your unique value",
        "avoid":          "Badmouthing competitor or panicking with heavy discount",
        "offer_hint":     "Feature comparison, unique capabilities they cannot get elsewhere",
    },
    "Low Feature Adoption": {
        "primary_action": "Offer free onboarding/training session for their team",
        "avoid":          "Discount — they are not getting value yet, money is not the issue",
        "offer_hint":     "Personal onboarding call, quick-win tutorial for their use case",
    },
    "Support Failure": {
        "primary_action": "Apologize sincerely, escalate open tickets immediately",
        "avoid":          "Any upsell or discount before resolving the support issue",
        "offer_hint":     "Service recovery: free month, dedicated support agent",
    },
    "Life/Business Event": {
        "primary_action": "Show empathy, offer flexible payment or pause option",
        "avoid":          "Aggressive retention push during personal/business difficulty",
        "offer_hint":     "Subscription pause, payment deferral, or downgrade path",
    },
    "Seasonal Disengagement": {
        "primary_action": "Acknowledge the season, light re-engagement touch",
        "avoid":          "Treating seasonal dip as full churn — over-reacting",
        "offer_hint":     "Content/tips for their industry during this season",
    },
}


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def generate_all_comms(
    customer: dict,
    prediction: dict,
    playbook_summary: str,
    language:  str = "english",
    is_b2b:    bool = True,
) -> dict:
    """
    Master function — generates all 4 communication formats for one customer.

    Args:
        customer:         Customer profile dict (name, industry, plan, revenue, etc.)
        prediction:       Churn prediction dict (probability, risk_level, reason, factors)
        playbook_summary: 2-3 sentence summary of what the playbook recommends
        language:         Output language (english/hindi/tamil/telugu/marathi/bengali/gujarati)
        is_b2b:           True for business customers, False for individual consumers

    Returns:
        {
            "email":        {"subject": "...", "body": "..."},
            "whatsapp":     {"message": "..."},
            "sms":          {"message": "..."},     # always English, max 160 chars
            "call_script":  {"opening": "...", "key_points": [...], "closing": "..."},
            "linkedin_dm":  {"message": "..."},     # only if is_b2b=True
            "language":     "english",
            "generated_at": "2024-01-15T10:30:00",
            "tone_used":    "ROI-focused, data-driven",
        }
    """
    logger.info(
        f"Generating comms for: {customer.get('customer_name')} | "
        f"Risk: {prediction.get('risk_level')} | "
        f"Language: {language}"
    )

    # Build the full context package for Claude
    context = _build_context(customer, prediction, playbook_summary)

    # Generate all formats in ONE Claude call (saves tokens + latency)
    raw_comms = _call_claude_for_comms(context, language, is_b2b)

    # Post-process and validate each format
    result = _post_process_comms(raw_comms, customer, language, is_b2b)

    return result


# ─────────────────────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────

def _build_context(
    customer: dict,
    prediction: dict,
    playbook_summary: str,
) -> dict:
    """
    Assembles all context Claude needs to write great communications.
    Includes India calendar awareness and persona/reason strategy hints.
    """
    # India calendar context
    current_month = datetime.now().month
    calendar_event = INDIA_CALENDAR.get(current_month, {})

    # Persona strategy
    persona = customer.get("persona", "Relationship Buyer")
    persona_hints = PERSONA_TONE.get(persona, PERSONA_TONE["Relationship Buyer"])

    # Churn reason strategy
    churn_reason = prediction.get("churn_reason", "Price Sensitivity")
    reason_hints = REASON_STRATEGY.get(churn_reason, REASON_STRATEGY["Price Sensitivity"])

    # Revenue calculations for discount framing
    monthly_rev = customer.get("monthly_revenue", 0)
    annual_rev  = monthly_rev * 12
    revenue_at_risk = prediction.get("revenue_at_risk", annual_rev)

    # Days urgency framing
    prob_30d = prediction.get("churn_prob_30d", 0)
    if prob_30d > 0.75:
        urgency = "CRITICAL — act within 48 hours"
    elif prob_30d > 0.50:
        urgency = "HIGH — reach out this week"
    else:
        urgency = "MEDIUM — follow up within 2 weeks"

    return {
        "customer_name":     customer.get("customer_name", "Valued Customer"),
        "industry":          customer.get("industry", "General"),
        "plan_type":         customer.get("plan_type", "Standard"),
        "monthly_revenue":   f"₹{monthly_rev:,}",
        "annual_revenue":    f"₹{annual_rev:,}",
        "revenue_at_risk":   f"₹{revenue_at_risk:,}",
        "contract_months":   customer.get("contract_age_months", 0),
        "city":              customer.get("city", "India"),
        "persona":           persona,
        "churn_probability": f"{prediction.get('churn_prob_30d', 0) * 100:.0f}%",
        "risk_level":        prediction.get("risk_level", "HIGH"),
        "churn_reason":      churn_reason,
        "top_risk_factors":  prediction.get("top_risk_factors", []),
        "urgency":           urgency,
        "playbook_summary":  playbook_summary,
        "persona_style":     persona_hints["style"],
        "persona_lead":      persona_hints["lead_with"],
        "persona_avoid":     persona_hints["avoid"],
        "reason_action":     reason_hints["primary_action"],
        "reason_avoid":      reason_hints["avoid"],
        "offer_hint":        reason_hints["offer_hint"],
        "india_event":       calendar_event.get("event", "Normal Month"),
        "india_note":        calendar_event.get("note", ""),
        "is_gst_month":      calendar_event.get("adjustment", 1.0) < 0.85,
        "is_festival_month": "festival" in calendar_event.get("event", "").lower()
                             or "diwali" in calendar_event.get("event", "").lower(),
    }


# ─────────────────────────────────────────────────────────────
# CLAUDE API CALL
# ─────────────────────────────────────────────────────────────

def _call_claude_for_comms(
    ctx: dict,
    language: str,
    is_b2b: bool,
) -> dict:
    """
    Single Claude API call that generates ALL communication formats.
    Returns parsed JSON dict or fallback templates if API fails.
    """

    linkedin_instruction = """
  "linkedin_dm": "A short professional LinkedIn message (max 100 words) for B2B outreach. Connect personally before mentioning product."
""" if is_b2b else ""

    festival_note = ""
    if ctx["is_festival_month"]:
        festival_note = f"\nNote: It is {ctx['india_event']} season — use a festive angle if appropriate (e.g., Diwali offer)."

    gst_note = ""
    if ctx["is_gst_month"]:
        gst_note = "\nNote: It is GST filing season — show empathy for their busy period, don't push hard."

    language_note = ""
    if language != "english":
        lang_name = language.title()
        language_note = f"""
LANGUAGE REQUIREMENT:
Write the email body, WhatsApp message, and call script in {lang_name}.
Keep the JSON keys in English. Keep the email subject line in English.
The SMS must stay in English (SMS character limits work best in English).
Adapt cultural tone for {lang_name}-speaking Indian business audience.
"""

    prompt = f"""
You are an expert Customer Success Manager at an Indian B2B/B2C company.
Your job: write personalized, effective retention communications for an at-risk customer.
These must feel HUMAN-written, not automated. They should get a response.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CUSTOMER PROFILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name:           {ctx["customer_name"]}
Industry:       {ctx["industry"]}
Plan:           {ctx["plan_type"]} ({ctx["monthly_revenue"]}/month)
Tenure:         {ctx["contract_months"]} months
City:           {ctx["city"]}
Persona:        {ctx["persona"]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK SIGNALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Churn Probability (30 days): {ctx["churn_probability"]}
Risk Level:     {ctx["risk_level"]}
Urgency:        {ctx["urgency"]}
Primary Reason: {ctx["churn_reason"]}
Top Signals:    {json.dumps(ctx["top_risk_factors"], ensure_ascii=False)}
Revenue at Risk:{ctx["revenue_at_risk"]} annually

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THE PLAYBOOK RECOMMENDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ctx["playbook_summary"]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Style:      {ctx["persona_style"]}
Lead with:  {ctx["persona_lead"]}
Avoid:      {ctx["persona_avoid"]}
Strategy:   {ctx["reason_action"]}
Do NOT:     {ctx["reason_avoid"]}
Offer hint: {ctx["offer_hint"]}
{festival_note}
{gst_note}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INDIA CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use INR (₹) for all amounts
- WhatsApp is preferred over email for urgent outreach in India
- Relationship matters more than transaction — be personal
- Address as "ji" suffix if using Hindi (e.g., "Sharma ji")
- Current business context: {ctx["india_event"]}

{language_note}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond ONLY with valid JSON. No markdown. No explanation outside the JSON.

{{
  "email": {{
    "subject": "Short compelling subject line in English (max 8 words)",
    "body": "Full professional email body (150-250 words). No [brackets] placeholders — use the actual customer name and real details above. Sign off as 'Customer Success Team'."
  }},
  "whatsapp": {{
    "message": "Conversational WhatsApp message (50-80 words). Warm opening. One clear ask. End with a question to invite reply. No formal email language."
  }},
  "sms": {{
    "message": "SMS under 160 characters. English only. Include customer first name, one benefit, one action. No links."
  }},
  "call_script": {{
    "opening": "First 2 sentences to say when they pick up. Warm, not salesy.",
    "key_points": [
      "Point 1 to cover during the call",
      "Point 2 to cover during the call",
      "Point 3 to cover during the call",
      "Point 4 — what to offer if they seem hesitant",
      "Point 5 — how to close the call positively"
    ],
    "objection_handlers": {{
      "too expensive": "What to say if they say it costs too much",
      "not using it": "What to say if they admit low usage",
      "switching to competitor": "What to say if they mention a competitor",
      "bad support experience": "What to say if they raise a complaint"
    }},
    "closing": "Last 2 sentences to end the call. Get a commitment or next step."
  }}{linkedin_instruction}
}}
"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)
        logger.info(f"Claude comms generated successfully for: {ctx['customer_name']}")
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f"Claude returned invalid JSON for comms: {e}")
        return _fallback_comms(ctx, is_b2b)

    except anthropic.APIError as e:
        logger.error(f"Claude API error in comms generator: {e}")
        return _fallback_comms(ctx, is_b2b)

    except Exception as e:
        logger.error(f"Unexpected error in comms generator: {e}")
        return _fallback_comms(ctx, is_b2b)


# ─────────────────────────────────────────────────────────────
# POST-PROCESSING & VALIDATION
# ─────────────────────────────────────────────────────────────

def _post_process_comms(
    raw: dict,
    customer: dict,
    language: str,
    is_b2b: bool,
) -> dict:
    """
    Validates and cleans the raw Claude output.
    Enforces SMS character limit.
    Adds metadata.
    Returns the final clean dict.
    """
    result = {}

    # ── Email ────────────────────────────────────────────────
    email = raw.get("email", {})
    result["email"] = {
        "subject": email.get("subject", _default_subject(customer)),
        "body":    email.get("body",    _default_email_body(customer)),
    }

    # ── WhatsApp ─────────────────────────────────────────────
    wa = raw.get("whatsapp", {})
    result["whatsapp"] = {
        "message": wa.get("message", _default_whatsapp(customer)),
    }

    # ── SMS — enforce 160 char limit ─────────────────────────
    sms = raw.get("sms", {})
    sms_text = sms.get("message", _default_sms(customer))
    if len(sms_text) > 160:
        sms_text = sms_text[:157] + "..."
        logger.warning(f"SMS truncated to 160 chars for: {customer.get('customer_name')}")
    result["sms"] = {"message": sms_text, "char_count": len(sms_text)}

    # ── Call Script ──────────────────────────────────────────
    script = raw.get("call_script", {})
    result["call_script"] = {
        "opening":            script.get("opening", "Hi, this is a quick check-in call."),
        "key_points":         script.get("key_points", []),
        "objection_handlers": script.get("objection_handlers", {}),
        "closing":            script.get("closing", "Thank you for your time. We will follow up."),
    }

    # ── LinkedIn DM (B2B only) ────────────────────────────────
    if is_b2b:
        li = raw.get("linkedin_dm", {})
        result["linkedin_dm"] = {
            "message": li.get("message", _default_linkedin(customer)) if isinstance(li, dict)
                       else str(li),
        }

    # ── Metadata ─────────────────────────────────────────────
    result["language"]     = language
    result["generated_at"] = datetime.utcnow().isoformat()
    result["tone_used"]    = PERSONA_TONE.get(
        customer.get("persona", ""),
        {}
    ).get("style", "professional and empathetic")
    result["is_b2b"]       = is_b2b

    return result


# ─────────────────────────────────────────────────────────────
# FALLBACK TEMPLATES — used when Claude API fails
# ─────────────────────────────────────────────────────────────

def _fallback_comms(ctx: dict, is_b2b: bool) -> dict:
    """
    Rule-based fallback templates when Claude API is unavailable.
    Not as good as Claude output but always works.
    """
    name = ctx["customer_name"]
    first_name = name.split()[0] if name else "there"
    monthly = ctx["monthly_revenue"]
    reason = ctx["churn_reason"]

    comms = {
        "email": {
            "subject": f"Quick check-in — {first_name}",
            "body": (
                f"Hi {name},\n\n"
                f"I wanted to personally reach out and check in on your experience with us.\n\n"
                f"We noticed some changes in your usage patterns and wanted to make sure "
                f"everything is going well. We truly value your business and want to ensure "
                f"you are getting the most out of your {ctx['plan_type']} plan.\n\n"
                f"Could we schedule a quick 15-minute call this week? I would love to understand "
                f"your current needs and see how we can better support you.\n\n"
                f"Please reply to this email or WhatsApp me directly.\n\n"
                f"Warm regards,\nCustomer Success Team"
            ),
        },
        "whatsapp": {
            "message": (
                f"Hi {first_name}! 👋 This is a quick check-in from our Customer Success team. "
                f"We noticed you haven't been as active lately and wanted to make sure everything "
                f"is going well. Can we schedule a quick call? We have some updates that might be "
                f"really useful for you. When works best? 🙏"
            ),
        },
        "sms": {
            "message": f"Hi {first_name}, quick check-in from ChurnShield CS team. "
                       f"Can we connect for 15 min this week? Reply YES to confirm.",
        },
        "call_script": {
            "opening": (
                f"Hi, am I speaking with someone from {name}? "
                f"Great — this is [Your Name] from the Customer Success team. "
                f"I am calling for a quick 5-minute check-in, is now a good time?"
            ),
            "key_points": [
                f"Thank them for their time and acknowledge their {ctx['contract_months']}-month relationship",
                f"Ask open question: 'How has your experience been with us lately?'",
                f"Listen actively — let them share concerns before offering anything",
                f"Address the primary concern: {reason}",
                f"Present relevant offer: {ctx['offer_hint']}",
            ],
            "objection_handlers": {
                "too expensive":          "I completely understand cost is important. Let me show you our annual plan — you save significantly and get priority support.",
                "not using it":           "That's really helpful to know. Let's schedule a quick training session — I'll personally walk your team through the top 3 features for your use case.",
                "switching to competitor":"I appreciate you telling me. Can I ask what specifically they offer that you feel we're missing? I'd like to understand.",
                "bad support experience": "I sincerely apologize for that experience. Let me personally escalate your open tickets right now while we're on the call.",
            },
            "closing": (
                f"Thank you so much for your time, {first_name}. "
                f"I will follow up with an email summarizing what we discussed. "
                f"Looking forward to continuing our relationship!"
            ),
        },
    }

    if is_b2b:
        comms["linkedin_dm"] = {
            "message": (
                f"Hi {first_name}, hope you're well! "
                f"I wanted to personally reach out — we've been working together for "
                f"{ctx['contract_months']} months now and I wanted to check in on your team's experience. "
                f"Would love to connect for a quick catch-up. Any time work for you this week?"
            ),
        }

    return comms


# ─────────────────────────────────────────────────────────────
# DEFAULT CONTENT HELPERS
# ─────────────────────────────────────────────────────────────

def _default_subject(customer: dict) -> str:
    name = customer.get("customer_name", "").split()[0]
    return f"Quick check-in — {name}"

def _default_email_body(customer: dict) -> str:
    return (
        f"Hi {customer.get('customer_name', 'there')},\n\n"
        f"We wanted to personally check in on your experience. "
        f"Please reply to this email or WhatsApp us — we are here to help.\n\n"
        f"Warm regards,\nCustomer Success Team"
    )

def _default_whatsapp(customer: dict) -> str:
    name = customer.get("customer_name", "").split()[0]
    return f"Hi {name}! 👋 Quick check-in from our CS team. Can we connect for 10 minutes this week? When works for you? 🙏"

def _default_sms(customer: dict) -> str:
    name = customer.get("customer_name", "").split()[0]
    return f"Hi {name}, CS team here. Can we connect briefly this week? Reply YES to confirm a call."

def _default_linkedin(customer: dict) -> str:
    return (
        f"Hi, hope you're doing well! "
        f"Wanted to personally check in on your team's experience. "
        f"Would love a quick 15-min catch-up — any time work for you this week?"
    )


# ─────────────────────────────────────────────────────────────
# BATCH GENERATION — for multiple customers at once
# ─────────────────────────────────────────────────────────────

def generate_comms_batch(
    customers_with_predictions: list,
    language: str = "english",
    risk_filter: str = "HIGH",
) -> dict:
    """
    Generates communications for multiple customers.
    Only processes customers at or above the risk_filter level.

    Args:
        customers_with_predictions: list of dicts, each with 'customer' and 'prediction' keys
        language:    Output language
        risk_filter: Only generate for 'HIGH' or 'HIGH'+'MEDIUM'

    Returns:
        {customer_id: comms_dict, ...}
    """
    risk_levels_to_process = (
        ["HIGH"] if risk_filter == "HIGH"
        else ["HIGH", "MEDIUM"]
    )

    results = {}
    processed = 0
    skipped = 0

    for item in customers_with_predictions:
        customer   = item.get("customer", {})
        prediction = item.get("prediction", {})
        customer_id = customer.get("id", customer.get("customer_id", "unknown"))

        risk = prediction.get("risk_level", "LOW")
        if risk not in risk_levels_to_process:
            skipped += 1
            continue

        playbook_summary = item.get(
            "playbook_summary",
            f"Customer showing {risk} churn risk. Reach out personally and address their concerns."
        )

        is_b2b = customer.get("industry", "") in [
            "B2B SaaS", "ERP", "CRM", "HR Software", "Manufacturing",
            "Logistics", "Finance", "Healthcare", "Education",
        ]

        try:
            comms = generate_all_comms(
                customer=customer,
                prediction=prediction,
                playbook_summary=playbook_summary,
                language=language,
                is_b2b=is_b2b,
            )
            results[customer_id] = comms
            processed += 1
            logger.info(f"Batch comms generated: {customer_id} ({processed} done)")

        except Exception as e:
            logger.error(f"Failed to generate comms for {customer_id}: {e}")
            results[customer_id] = {"error": str(e)}

    logger.info(
        f"Batch complete — processed: {processed} | skipped (low risk): {skipped}"
    )
    return results


# ─────────────────────────────────────────────────────────────
# QUICK REFRESH — regenerate one format only
# ─────────────────────────────────────────────────────────────

def regenerate_single_format(
    customer: dict,
    prediction: dict,
    playbook_summary: str,
    format_type: str,      # "email" | "whatsapp" | "sms" | "call_script"
    language: str = "english",
) -> dict:
    """
    Regenerates just one communication format for a customer.
    Useful when CS team wants to refresh only the email, for example.
    """
    valid_formats = ["email", "whatsapp", "sms", "call_script", "linkedin_dm"]
    if format_type not in valid_formats:
        raise ValueError(f"Invalid format_type. Must be one of: {valid_formats}")

    all_comms = generate_all_comms(
        customer=customer,
        prediction=prediction,
        playbook_summary=playbook_summary,
        language=language,
        is_b2b=True,
    )

    return {
        format_type:     all_comms.get(format_type, {}),
        "language":      language,
        "generated_at":  all_comms.get("generated_at"),
        "refreshed":     True,
    }