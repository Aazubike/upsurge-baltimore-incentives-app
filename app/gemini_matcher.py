"""
Stage 2 of the matching pipeline: Gemini reasons over whatever the rules
engine already narrowed down. It never re-decides a hard eligibility gate
(county, stage bucket, etc.) — that already happened in rules_engine.py.
What it DOES do:
  - Ranks the shortlist by genuine fit, using the messy free-text fields
    (Business Size Requirement, Revenue / Income Cap, Stacking Notes, etc.)
    that the rules engine deliberately left unparsed.
  - Writes a short plain-English reason for each ranking.
  - Flags anything where the free text suggests a real eligibility question
    the rules engine couldn't catch (e.g. Needs_Manual_Review == True).
"""
import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL_NAME = "gemini-3.1-flash-lite"
MAX_RETRIES = 2

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")
        _client = genai.Client(api_key=api_key)
    return _client


def _build_prompt(answers: dict, shortlist_df: pd.DataFrame) -> str:
    company_profile = {
        "county": answers.get("county"),
        "stage": answers.get("stage"),
        "employee_count": answers.get("employee_count"),
        "annual_revenue": answers.get("annual_revenue"),
        "industry": answers.get("industry"),
        "ownership_groups": answers.get("mwbe_groups", []),
    }

    programs = []
    for _, row in shortlist_df.iterrows():
        programs.append({
            "program_name": row.get("Program Name"),
            "description": row.get("Program Description"),
            "business_size_requirement": row.get("Business Size Requirement"),
            "revenue_income_cap": row.get("Revenue / Income Cap"),
            "mwbe_flag": row.get("MWBE / DEI Flag?"),
            "industry_exclusions": row.get("Industry Exclusions"),
            "needs_manual_review": bool(row.get("Needs_Manual_Review")),
        })

    return f"""You are ranking Maryland/Baltimore-region business incentive programs
for a specific company. These programs already passed a hard eligibility
filter on county, business stage, and industry exclusions — do not re-reject
a program for those reasons. Your job is to rank by genuine fit using the
free-text fields provided, and flag real eligibility concerns you notice.

Company profile:
{json.dumps(company_profile, indent=2)}

Candidate programs (already passed hard filters):
{json.dumps(programs, indent=2)}

Return ONLY a JSON array, no markdown fences, no commentary. Each element:
{{
  "program_name": "<exact name from input>",
  "fit_score": <integer 1-100, higher = better fit>,
  "reasoning": "<one plain-English sentence on why this fits, under 25 words>",
  "flag": "<short note if needs_manual_review or the free text raises a real eligibility concern, else null>"
}}

Order the array by fit_score descending."""


def rank_shortlist(answers: dict, shortlist_df: pd.DataFrame):
    """
    Returns (ranked_records, error_message).
    ranked_records: list of dicts merging original program data with
                     fit_score / reasoning / flag, sorted by fit_score desc.
    error_message: None on success, else a short string to show in the UI.
    """
    if shortlist_df.empty:
        return [], None

    try:
        client = _get_client()
        prompt = _build_prompt(answers, shortlist_df)

        raw_text = None
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
                raw_text = response.text.strip()
                break
            except Exception as e:
                last_error = e
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    if attempt < MAX_RETRIES:
                        time.sleep(2 * (attempt + 1))
                        continue
                raise
        if raw_text is None:
            raise last_error

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
        rankings = json.loads(raw_text)
    except Exception as e:
        return shortlist_df.to_dict("records"), f"Gemini ranking failed ({e}). Showing unranked shortlist."

    rank_by_name = {r["program_name"]: r for r in rankings}
    merged = []
    for _, row in shortlist_df.iterrows():
        record = row.to_dict()
        rank_info = rank_by_name.get(row.get("Program Name"), {})
        record["fit_score"] = rank_info.get("fit_score")
        record["reasoning"] = rank_info.get("reasoning")
        record["flag"] = rank_info.get("flag")
        merged.append(record)

    merged.sort(key=lambda r: r.get("fit_score") or 0, reverse=True)
    return merged, None

