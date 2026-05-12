"""
ChurnShield 2.0 — llm/column_mapper.py

When a user uploads their own CSV/Excel, their column names can be ANYTHING.
This file uses Claude to automatically understand what each column means
and maps it to ChurnShield's standard 14-column schema.

Example:
    User has: "last_seen_dt", "tix_count", "arr_inr", "is_cancelled"
    We map to: "last_activity_date", "support_tickets", "monthly_revenue", "churned"

Flow:
    1. Read column names + first 5 rows of data
    2. Send to Claude with standard schema
    3. Claude returns JSON mapping
    4. Validate and return mapping to user for confirmation
    5. User confirms or corrects
    6. Apply mapping to full dataframe
"""

import json
import logging
import re
import pandas as pd
import anthropic
from typing import Optional
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS, STANDARD_SCHEMA

logger = logging.getLogger("churnshield.column_mapper")

# ─────────────────────────────────────────────────────────────
# STANDARD TARGET COLUMNS — what every dataset must have
# ─────────────────────────────────────────────────────────────
TARGET_COLUMNS = list(STANDARD_SCHEMA.keys())

# Columns that MUST be found for the system to work
REQUIRED_COLUMNS = ["customer_id", "churned"]

# Columns that are very important but optional
IMPORTANT_COLUMNS = [
    "monthly_revenue",
    "contract_age_months",
    "support_tickets",
    "payment_delays",
    "login_frequency",
    "feature_usage_score",
    "days_since_last_login",
]

# ─────────────────────────────────────────────────────────────
# RULE-BASED PRE-MAPPING — catches obvious cases without LLM
# ─────────────────────────────────────────────────────────────
RULE_BASED_MAPPINGS = {
    # customer_id variations
    "customer_id":      "customer_id",
    "cust_id":          "customer_id",
    "client_id":        "customer_id",
    "user_id":          "customer_id",
    "account_id":       "customer_id",
    "id":               "customer_id",
    "customerid":       "customer_id",
    "userid":           "customer_id",

    # customer_name variations
    "customer_name":    "customer_name",
    "company_name":     "customer_name",
    "name":             "customer_name",
    "client_name":      "customer_name",
    "account_name":     "customer_name",
    "business_name":    "customer_name",
    "org_name":         "customer_name",

    # plan_type variations
    "plan_type":        "plan_type",
    "plan":             "plan_type",
    "subscription_type":"plan_type",
    "tier":             "plan_type",
    "package":          "plan_type",
    "product":          "plan_type",
    "plan_name":        "plan_type",

    # monthly_revenue variations
    "monthly_revenue":  "monthly_revenue",
    "mrr":              "monthly_revenue",
    "monthly_charges":  "monthly_revenue",
    "monthlycharges":   "monthly_revenue",
    "arr_inr":          "monthly_revenue",
    "revenue":          "monthly_revenue",
    "amount":           "monthly_revenue",
    "subscription_amount": "monthly_revenue",
    "monthly_fee":      "monthly_revenue",
    "plan_amount":      "monthly_revenue",

    # contract_age_months variations
    "contract_age_months": "contract_age_months",
    "tenure":           "contract_age_months",
    "months_active":    "contract_age_months",
    "customer_age":     "contract_age_months",
    "account_age":      "contract_age_months",
    "months_since_join":"contract_age_months",
    "subscription_age": "contract_age_months",
    "contract_months":  "contract_age_months",

    # last_activity_date variations
    "last_activity_date":    "last_activity_date",
    "last_login":            "last_activity_date",
    "last_seen":             "last_activity_date",
    "last_seen_dt":          "last_activity_date",
    "last_active":           "last_activity_date",
    "most_recent_activity":  "last_activity_date",
    "date_last_seen":        "last_activity_date",
    "final_access_timestamp":"last_activity_date",
    "last_access_date":      "last_activity_date",
    "last_used_date":        "last_activity_date",

    # login_frequency / days_since_last_login variations
    "login_count_30d":       "login_frequency",
    "login_count":           "login_frequency",
    "login_frequency":       "login_frequency",
    "logins":                "login_frequency",
    "monthly_logins":        "login_frequency",
    "session_count":         "login_frequency",
    "days_since_last_login": "days_since_last_login",
    "days_inactive":         "days_since_last_login",
    "inactivity_days":       "days_since_last_login",

    # feature_usage_score variations
    "feature_usage_score":  "feature_usage_score",
    "feature_usage":        "feature_usage_score",
    "usage_score":          "feature_usage_score",
    "product_usage":        "feature_usage_score",
    "adoption_score":       "feature_usage_score",
    "engagement_score":     "feature_usage_score",
    "usage_rate":           "feature_usage_score",

    # support_tickets variations
    "support_tickets":      "support_tickets",
    "support_tickets_30d":  "support_tickets",
    "tix_count":            "support_tickets",
    "ticket_count":         "support_tickets",
    "num_tickets":          "support_tickets",
    "open_tickets":         "support_tickets",
    "complaints":           "support_tickets",
    "issues_raised":        "support_tickets",

    # payment_delays variations
    "payment_delays":       "payment_delays",
    "payment_delay_count":  "payment_delays",
    "late_payments":        "payment_delays",
    "missed_payments":      "payment_delays",
    "overdue_count":        "payment_delays",
    "num_failed_payments":  "payment_delays",
    "failed_transactions":  "payment_delays",

    # active_seats variations
    "active_seats":         "active_seats",
    "active_users":         "active_seats",
    "num_users":            "active_seats",
    "user_count":           "active_seats",
    "seats_used":           "active_seats",
    "licensed_users_active":"active_seats",

    # total_seats variations
    "total_seats":          "total_seats",
    "purchased_seats":      "total_seats",
    "total_licenses":       "total_seats",
    "licensed_seats":       "total_seats",
    "seats_purchased":      "total_seats",
    "max_seats":            "total_seats",

    # nps_score variations
    "nps_score":            "nps_score",
    "nps":                  "nps_score",
    "net_promoter_score":   "nps_score",
    "satisfaction_score":   "nps_score",
    "csat":                 "nps_score",

    # churned — TARGET COLUMN — many variations
    "churned":              "churned",
    "churn":                "churned",
    "is_churned":           "churned",
    "is_cancelled":         "churned",
    "cancelled":            "churned",
    "canceled":             "churned",
    "subscription_status":  "churned",
    "status":               "churned",
    "active":               "churned",      # note: inverted — 0=active=not churned
    "retained":             "churned",      # note: inverted
    "has_churned":          "churned",
    "exited":               "churned",
    "left":                 "churned",
    "dropped":              "churned",
    "churn_flag":           "churned",
}


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def auto_map_columns(df: pd.DataFrame) -> dict:
    """
    Main function — given any DataFrame, returns a column mapping dict.

    Returns:
        {
            "mapping": {
                "original_col_name": "standard_col_name",
                ...
            },
            "unmapped": ["col1", "col2"],      # columns we couldn't map
            "missing_required": ["churned"],   # required cols not found
            "missing_important": ["nps_score"],# important cols not found
            "confidence": 0.87,                # how confident we are
            "needs_user_confirmation": True,   # should UI show confirmation?
            "inversion_needed": [],            # cols where values are inverted
            "source": "rules+llm"              # how mapping was determined
        }
    """
    logger.info(f"Auto-mapping {len(df.columns)} columns: {list(df.columns)}")

    # Step 1 — Rule-based mapping (fast, no API call)
    rule_mapping = _apply_rule_based_mapping(df.columns.tolist())
    logger.info(f"Rule-based mapped: {len(rule_mapping['mapped'])} columns")

    # Step 2 — Find unmapped columns
    unmapped = [
        col for col in df.columns
        if col not in rule_mapping["mapped"]
    ]

    # Step 3 — Send unmapped columns to LLM if any remain
    llm_mapping = {}
    if unmapped:
        logger.info(f"Sending {len(unmapped)} unmapped columns to Claude")
        llm_mapping = _llm_map_columns(unmapped, df, rule_mapping["mapped"])
    else:
        logger.info("All columns mapped by rules — no LLM call needed")

    # Step 4 — Merge rule + LLM mappings
    final_mapping = {**rule_mapping["mapped"], **llm_mapping}

    # Step 5 — Detect columns needing value inversion
    inversion_needed = _detect_inversion_needed(df, final_mapping)

    # Step 6 — Check what's missing
    mapped_targets = set(final_mapping.values())
    missing_required = [c for c in REQUIRED_COLUMNS  if c not in mapped_targets]
    missing_important = [c for c in IMPORTANT_COLUMNS if c not in mapped_targets]

    # Step 7 — Calculate confidence
    total_important = len(REQUIRED_COLUMNS) + len(IMPORTANT_COLUMNS)
    found_important = sum(1 for c in REQUIRED_COLUMNS + IMPORTANT_COLUMNS if c in mapped_targets)
    confidence = round(found_important / total_important, 2)

    still_unmapped = [
        col for col in df.columns
        if col not in final_mapping
    ]

    result = {
        "mapping":                final_mapping,
        "unmapped":               still_unmapped,
        "missing_required":       missing_required,
        "missing_important":      missing_important,
        "confidence":             confidence,
        "needs_user_confirmation": confidence < 0.80 or bool(missing_required),
        "inversion_needed":       inversion_needed,
        "source":                 "rules" if not llm_mapping else "rules+llm",
        "total_original_columns": len(df.columns),
        "total_mapped":           len(final_mapping),
    }

    logger.info(
        f"Mapping complete — confidence: {confidence:.0%} | "
        f"missing required: {missing_required} | "
        f"source: {result['source']}"
    )

    return result


# ─────────────────────────────────────────────────────────────
# STEP 1 — RULE-BASED MAPPING
# ─────────────────────────────────────────────────────────────

def _apply_rule_based_mapping(columns: list) -> dict:
    """
    Applies the static lookup table to map obvious column names.
    Case-insensitive and strips whitespace.
    Returns {"mapped": {original: target}, "unmapped": [cols]}
    """
    mapped = {}
    used_targets = set()  # prevent duplicate mappings to same target

    for col in columns:
        normalized = col.lower().strip().replace(" ", "_").replace("-", "_")

        if normalized in RULE_BASED_MAPPINGS:
            target = RULE_BASED_MAPPINGS[normalized]

            # Don't map two source cols to same target
            if target not in used_targets:
                mapped[col] = target
                used_targets.add(target)
            else:
                logger.debug(f"Skipping duplicate target '{target}' for column '{col}'")

    return {"mapped": mapped}


# ─────────────────────────────────────────────────────────────
# STEP 2 — LLM MAPPING for unmapped columns
# ─────────────────────────────────────────────────────────────

def _llm_map_columns(
    unmapped_cols: list,
    df: pd.DataFrame,
    already_mapped: dict
) -> dict:
    """
    Sends unmapped column names + sample data to Claude.
    Claude returns a JSON mapping of column → standard name.
    Returns dict of {original_col: standard_col} for newly mapped ones.
    """

    # Build sample data for context — first 5 rows of unmapped cols only
    sample_rows = []
    for _, row in df[unmapped_cols].head(5).iterrows():
        sample_rows.append({col: str(row[col]) for col in unmapped_cols})

    # Already-mapped targets — so Claude doesn't duplicate them
    already_mapped_targets = list(already_mapped.values())

    # Available targets Claude can map to
    available_targets = [t for t in TARGET_COLUMNS if t not in already_mapped_targets]

    prompt = f"""
You are a data schema expert for ChurnShield, a customer churn prediction platform.

A user has uploaded a dataset with some columns we couldn't automatically identify.
Your job: map each unidentified column to the correct standard column name.

## STANDARD SCHEMA (map to one of these):
{json.dumps({k: v for k, v in STANDARD_SCHEMA.items() if k in available_targets}, indent=2)}

## ALREADY MAPPED (do NOT use these targets again):
{json.dumps(already_mapped_targets)}

## UNIDENTIFIED COLUMNS with sample data:
Columns: {unmapped_cols}

Sample values (first 5 rows):
{json.dumps(sample_rows, indent=2)}

## YOUR TASK:
For each unidentified column, decide:
1. Which standard column does it best represent? (must be from available targets above)
2. Or should it be ignored? (use null if it has no relevant meaning)

## RULES:
- Each standard column can only be used ONCE across all mappings
- If a column clearly has no relevance to churn prediction, map it to null
- "churned" target is the most important — look for any cancel/exit/status column
- Revenue columns: if annual, divide by 12 mentally — still map to monthly_revenue
- Date columns showing last activity → map to last_activity_date
- If a column name is in a language other than English, use the sample values to infer meaning

## RESPOND IN THIS EXACT JSON FORMAT (no extra text, no markdown):
{{
  "mappings": {{
    "original_column_name": "standard_column_name_or_null",
    "another_column": "standard_column_name_or_null"
  }},
  "reasoning": {{
    "original_column_name": "brief reason for this mapping",
    "another_column": "brief reason"
  }}
}}
"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        logger.debug(f"Claude raw response: {raw[:200]}...")

        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)
        mappings = parsed.get("mappings", {})
        reasoning = parsed.get("reasoning", {})

        # Filter out nulls and log reasoning
        result = {}
        used_targets = set(already_mapped.values())

        for original_col, target in mappings.items():
            if target is None or target == "null":
                logger.info(f"LLM: '{original_col}' → ignored (no relevant meaning)")
                continue

            if target not in TARGET_COLUMNS:
                logger.warning(f"LLM returned invalid target '{target}' for '{original_col}' — skipping")
                continue

            if target in used_targets:
                logger.warning(f"LLM tried to duplicate target '{target}' for '{original_col}' — skipping")
                continue

            result[original_col] = target
            used_targets.add(target)
            reason = reasoning.get(original_col, "")
            logger.info(f"LLM mapped: '{original_col}' → '{target}' ({reason})")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Claude returned invalid JSON: {e}")
        return {}
    except anthropic.APIError as e:
        logger.error(f"Claude API error during column mapping: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in LLM mapping: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# STEP 3 — DETECT VALUE INVERSION
# ─────────────────────────────────────────────────────────────

def _detect_inversion_needed(df: pd.DataFrame, mapping: dict) -> list:
    """
    Some columns are semantically inverted relative to our standard.
    Example: "active" column where 1 = active (NOT churned).
             We need churned = 1 - active.
    Example: "retained" column where 1 = retained (NOT churned).

    Returns list of original column names that need inversion.
    """
    inversion_needed = []
    inversion_hints = {"active", "retained", "is_active", "is_retained", "still_customer"}

    for original_col, target in mapping.items():
        if target == "churned":
            col_lower = original_col.lower()
            if any(hint in col_lower for hint in inversion_hints):
                logger.info(f"Inversion needed: '{original_col}' is active-style — will flip to churned")
                inversion_needed.append(original_col)

    return inversion_needed


# ─────────────────────────────────────────────────────────────
# APPLY MAPPING TO DATAFRAME
# ─────────────────────────────────────────────────────────────

def apply_mapping_to_dataframe(
    df: pd.DataFrame,
    mapping_result: dict
) -> pd.DataFrame:
    """
    Applies the confirmed column mapping to the full DataFrame.
    Returns a new DataFrame with standard column names.

    Steps:
    1. Rename mapped columns to standard names
    2. Invert columns that need it (active → churned)
    3. Add any missing columns with sensible defaults
    4. Drop columns that were not mapped to any standard name
    """

    mapping       = mapping_result["mapping"]
    inversion_cols = mapping_result.get("inversion_needed", [])

    # 1 — Rename columns
    df_mapped = df.rename(columns=mapping)

    # 2 — Handle value inversion for active/retained columns
    for original_col in inversion_cols:
        standard_col = mapping.get(original_col, "churned")
        if standard_col in df_mapped.columns:
            logger.info(f"Inverting values in '{standard_col}' (was '{original_col}')")
            df_mapped[standard_col] = (df_mapped[standard_col] == 0).astype(int)

    # 3 — Normalize the churned column
    if "churned" in df_mapped.columns:
        df_mapped["churned"] = _normalize_churned_column(df_mapped["churned"])

    # 4 — Add missing columns with safe defaults
    df_mapped = _fill_missing_columns(df_mapped)

    # 5 — Keep only standard + leftover columns (drop unmapped)
    standard_cols = [c for c in TARGET_COLUMNS if c in df_mapped.columns]
    extra_cols    = [c for c in df_mapped.columns if c not in TARGET_COLUMNS]

    # Keep extra columns too — they may be useful as custom features
    final_cols = standard_cols + extra_cols
    df_mapped  = df_mapped[final_cols]

    logger.info(
        f"Mapping applied: {len(df_mapped.columns)} columns remaining | "
        f"Standard: {len(standard_cols)} | Extra: {len(extra_cols)}"
    )

    return df_mapped


def _normalize_churned_column(series: pd.Series) -> pd.Series:
    """
    Churned column can come in many formats.
    Normalize everything to 0 (not churned) or 1 (churned).

    Handles:
    - Boolean True/False
    - String "Yes"/"No", "True"/"False", "1"/"0"
    - String "Churned"/"Active", "Left"/"Retained"
    - Integer 0/1
    """
    if series.dtype == bool:
        return series.astype(int)

    # Try direct numeric conversion first
    try:
        numeric = pd.to_numeric(series, errors="raise")
        return (numeric > 0).astype(int)
    except (ValueError, TypeError):
        pass

    # String-based normalization
    churn_positive_values = {
        "yes", "true", "1", "churned", "churn", "cancelled",
        "canceled", "inactive", "left", "dropped", "exited",
        "lost", "terminated", "closed",
    }

    normalized = series.astype(str).str.lower().str.strip()
    return normalized.isin(churn_positive_values).astype(int)


def _fill_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds missing standard columns with sensible defaults
    so downstream ML doesn't break on missing features.
    """
    defaults = {
        "customer_name":        lambda: "Unknown",
        "plan_type":            lambda: "Unknown",
        "monthly_revenue":      lambda: 0,
        "contract_age_months":  lambda: 0,
        "login_frequency":      lambda: 0,
        "feature_usage_score":  lambda: 0.0,
        "support_tickets":      lambda: 0,
        "payment_delays":       lambda: 0,
        "active_seats":         lambda: 1,
        "total_seats":          lambda: 1,
        "nps_score":            lambda: None,
        "last_activity_date":   lambda: None,
    }

    for col, default_fn in defaults.items():
        if col not in df.columns:
            df[col] = default_fn()
            logger.debug(f"Added missing column '{col}' with default value")

    return df


# ─────────────────────────────────────────────────────────────
# UTILITY — Format mapping for UI display
# ─────────────────────────────────────────────────────────────

def format_mapping_for_ui(df: pd.DataFrame, mapping_result: dict) -> list:
    """
    Returns a list of dicts formatted for the frontend column mapping screen.

    Each item represents one row in the mapping confirmation table:
    {
        "original_column":   "tix_count",
        "sample_values":     ["3", "0", "7", "1", "2"],
        "mapped_to":         "support_tickets",
        "mapped_to_label":   "Support Tickets",
        "description":       "Number of support tickets raised",
        "confidence":        "high",       # high / medium / low / unknown
        "source":            "rules",      # rules / llm / unknown
        "needs_attention":   False,
        "inversion_needed":  False,
    }
    """
    ui_rows = []
    mapping = mapping_result["mapping"]
    inversion_cols = mapping_result.get("inversion_needed", [])

    for col in df.columns:
        sample = df[col].dropna().head(5).astype(str).tolist()
        mapped_to = mapping.get(col)

        row = {
            "original_column":  col,
            "sample_values":    sample,
            "mapped_to":        mapped_to,
            "mapped_to_label":  mapped_to.replace("_", " ").title() if mapped_to else None,
            "description":      STANDARD_SCHEMA.get(mapped_to, "Custom column — kept as extra feature"),
            "confidence":       _get_confidence_level(col, mapped_to, mapping_result),
            "source":           mapping_result.get("source", "unknown"),
            "needs_attention":  mapped_to in REQUIRED_COLUMNS and mapped_to is None,
            "inversion_needed": col in inversion_cols,
        }
        ui_rows.append(row)

    return ui_rows


def _get_confidence_level(original_col: str, mapped_to: Optional[str], result: dict) -> str:
    """Returns 'high', 'medium', 'low', or 'unknown' confidence for a mapping."""
    if mapped_to is None:
        return "unknown"

    normalized = original_col.lower().strip().replace(" ", "_").replace("-", "_")

    if normalized in RULE_BASED_MAPPINGS and RULE_BASED_MAPPINGS[normalized] == mapped_to:
        return "high"    # Exact rule match — very confident

    if normalized.replace("_", "") == mapped_to.replace("_", ""):
        return "high"    # Essentially same name

    if result.get("source") == "rules+llm":
        return "medium"  # LLM mapped it — less certain

    return "low"


# ─────────────────────────────────────────────────────────────
# APPLY USER CORRECTIONS from UI
# ─────────────────────────────────────────────────────────────

def apply_user_corrections(
    original_mapping_result: dict,
    user_corrections: dict
) -> dict:
    """
    Merges user corrections from the UI into the auto-mapped result.

    user_corrections format:
    {
        "tix_count": "support_tickets",    # user changed this mapping
        "region": null,                    # user said this column is irrelevant
    }

    Returns updated mapping_result dict.
    """
    mapping = dict(original_mapping_result["mapping"])

    for original_col, corrected_target in user_corrections.items():
        if corrected_target is None:
            # User said this column should be ignored
            mapping.pop(original_col, None)
            logger.info(f"User removed mapping for: '{original_col}'")
        else:
            mapping[original_col] = corrected_target
            logger.info(f"User corrected: '{original_col}' → '{corrected_target}'")

    # Recalculate what's missing after corrections
    mapped_targets = set(mapping.values())
    missing_required  = [c for c in REQUIRED_COLUMNS  if c not in mapped_targets]
    missing_important = [c for c in IMPORTANT_COLUMNS if c not in mapped_targets]

    return {
        **original_mapping_result,
        "mapping":          mapping,
        "missing_required": missing_required,
        "missing_important":missing_important,
        "user_corrected":   True,
    }