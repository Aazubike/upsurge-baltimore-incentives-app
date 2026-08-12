"""
Stage 2 of the matching pipeline: Gemini reasons over whatever the rules
engine already narrowed down. It never re-decides a hard eligibility gate
(county, stage bucket, etc.) -- that already happened in rules_engine.py.

WEIGHTED FIT-SCORE FRAMEWORK (the rubric Gemini is instructed to use):
    Location fit ............... 20%
    Business stage fit ......... 20%
    Size (employees/revenue) ... 15%
    Industry fit ................ 15%
    Ownership / MWBE fit ........ 10%
    Overall program relevance ... 20%

For every scored program, Gemini also returns a per-criterion breakdown
(status + short note for each of the 6 dimensions above) so the UI can show
exactly WHY a program scored the way it did when someone clicks into it --
not just a single number.

If a criterion has NO DATA for a given program (e.g. the MD Business Compass
dataset has no stage/size/MWBE fields at all), that dimension is marked
"no_data" rather than penalized or guessed.

ANTI-HALLUCINATION RULE: if a REQUIRED criterion is clearly stated in the
program's data and clearly NOT met by the company's profile, Gemini must
mark that program "likely_ineligible" with a specific reason instead of
inventing a fit percentage.
"""
import os
import json
import re
import time
import hashlib
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.rules_engine import locality_tier, enterprise_zone_note, opportunity_zone_note

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL_NAME = "gemini-3.1-flash-lite"
MAX_RETRIES = 2
BATCH_SIZE = 8  # smaller batches = less output per call = each parallel batch
                # finishes faster (generation time scales with output length).
                # With MAX_PARALLEL_BATCHES=10 there's plenty of headroom to run
                # more, smaller batches concurrently instead of fewer, larger ones.
MAX_PARALLEL_BATCHES = 10  # raised from 5 now that credit exhaustion (not burst
                           # concurrency) looks like the real cause of the earlier
                           # 429s -- Tier 1 RPM/TPM usage never actually got close
                           # to its ceiling, so there's real headroom here. Still
                           # well under the 25 that caused problems originally.
SAFETY_NET_MAX_CANDIDATES = 300  # not a normal operating limit -- just prevents a pathological
                                  # worst-case query (e.g. an almost-unrestricted profile matching
                                  # hundreds of statewide programs) from generating a runaway bill.
                                  # Under normal use, everything eligible gets scored, no fixed cap.

CACHE_TTL_SECONDS = 600  # identical resubmissions within 10 min skip Gemini entirely
_result_cache = {}  # {cache_key: (merged, dropped_count, error_message, timestamp)}


def _make_cache_key(answers: dict, shortlist_df: pd.DataFrame) -> str:
    """Stable key from the answers + exact set of candidate program names --
    if either changes (new answers, or the underlying data changes), it's a
    fresh key and won't hit a stale cached result."""
    normalized_answers = {
        k: (tuple(sorted(v)) if isinstance(v, list) else v)
        for k, v in sorted(answers.items())
    }
    program_names = tuple(sorted(shortlist_df["Program Name"].tolist()))
    raw = json.dumps({"answers": normalized_answers, "programs": program_names}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")
        _client = genai.Client(api_key=api_key)
    return _client


def _cap_candidates(shortlist_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if len(shortlist_df) <= SAFETY_NET_MAX_CANDIDATES:
        return shortlist_df, 0
    df = shortlist_df.copy()
    df["_tier"] = df.apply(locality_tier, axis=1)
    df = df.sort_values("_tier")
    capped = df.head(SAFETY_NET_MAX_CANDIDATES).drop(columns=["_tier"])
    return capped, len(shortlist_df) - SAFETY_NET_MAX_CANDIDATES


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
            "data_source": row.get("Data_Source"),
            "description": row.get("Program Description"),
            "business_size_requirement": row.get("Business Size Requirement"),
            "revenue_income_cap": row.get("Revenue / Income Cap"),
            "mwbe_flag": row.get("MWBE / DEI Flag?"),
            "industry_exclusions": row.get("Industry Exclusions"),
            "eligible_industries_naics": row.get("Eligible_Industries_Raw"),
            "scope": row.get("Scope"),
            "status": row.get("Status"),
            "needs_manual_review": bool(row.get("Needs_Manual_Review")),
        })

    return f"""You are ranking Maryland business incentive programs for a specific company.
These programs already passed a hard eligibility filter on county, business stage,
and industry exclusions -- do not re-reject a program for those reasons alone.

Company profile:
{json.dumps(company_profile, indent=2)}

WEIGHTED FIT-SCORE RUBRIC (use these weights when a dimension has data):
- Location fit: 20%
- Business stage fit: 20%
- Size (employees/revenue) fit: 15%
- Industry fit: 15% (note: "eligible_industries_naics" uses a different
  category system than the company's industry tag -- use judgment on
  whether they plausibly overlap, don't require an exact string match)
- Ownership/MWBE fit: 10%
- Overall program relevance (award usefulness, how well free-text criteria fit): 20%

CRITICAL RULES:
1. Some programs (data_source = "MD Business Compass") have NO data for stage,
   size, or MWBE -- those fields will be null. Mark that dimension's status as
   "no_data" and exclude it from the weighted score, redistributing its weight
   proportionally across dimensions that DO have data. Never penalize missing data.
2. If a field explicitly states a requirement and the company clearly fails it,
   do NOT compute a fit_score. Instead set fit_score to null, eligibility to
   "likely_ineligible", and reasoning must state exactly which requirement is unmet.
3. Never invent a requirement that isn't stated in the data.
4. If needs_manual_review is true, treat the free text as authoritative but
   flag genuine ambiguity in the "flag" field.
5. For EVERY scored program, return a "breakdown" object covering all 6 rubric
   dimensions (location, stage, size, industry, mwbe, overall). Each dimension
   needs a "status" of exactly one of: "match", "partial", "no_data", "unmet".
   Each needs a "note": 2-4 words max, terse tag style, not a sentence
   (e.g. "County match", "Not specified", "Tech services fit" -- not
   "Baltimore City, exact match for this program").
6. CRITICAL -- MWBE ANTI-HALLUCINATION RULE: mwbe_flag text almost never names
   a specific race, ethnicity, or gender (it usually just says generic terms
   like "MWBE", "veteran", "disadvantaged", "SEDI"). NEVER state a specific
   race, ethnicity, or gender in your mwbe note or reasoning unless that exact
   word appears in mwbe_flag. If mwbe_flag says "MWBE" generically, your note
   must also stay generic ("MWBE required", not "Black-owned business match"
   or "Women-owned match" or any invented specificity). Inventing demographic
   detail not in the source data is a serious error.

Candidate programs (already passed hard filters):
{json.dumps(programs, indent=2)}

Return ONLY a JSON array, no markdown fences, no commentary. Each element:
{{
  "program_name": "<exact name from input>",
  "eligibility": "<'eligible' or 'likely_ineligible'>",
  "fit_score": <integer 1-100, or null if likely_ineligible>,
  "reasoning": "<one plain-English sentence, under 25 words>",
  "flag": "<short note if needs_manual_review or a real eligibility concern, else null>",
  "breakdown": {{
    "location": {{"status": "match|partial|no_data|unmet", "note": "<short note>"}},
    "stage": {{"status": "match|partial|no_data|unmet", "note": "<short note>"}},
    "size": {{"status": "match|partial|no_data|unmet", "note": "<short note>"}},
    "industry": {{"status": "match|partial|no_data|unmet", "note": "<short note>"}},
    "mwbe": {{"status": "match|partial|no_data|unmet", "note": "<short note>"}},
    "overall": {{"status": "match|partial|no_data|unmet", "note": "<short note>"}}
  }}
}}

Order the array by fit_score descending (nulls last)."""


def _call_batch(answers: dict, batch_df: pd.DataFrame):
    """Calls Gemini for one batch. Returns (rankings_list_or_None, error_or_None)."""
    try:
        client = _get_client()
        prompt = _build_prompt(answers, batch_df)

        raw_text = None
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        # Constrains generation straight to JSON tokens instead of
                        # letting the model spend output tokens on markdown fences /
                        # prose framing that we were just stripping afterward anyway.
                        # This is the single biggest latency win here, since
                        # generation time scales with output length.
                        response_mime_type="application/json",
                        # "low" is Google's own recommendation for high-throughput,
                        # simple-instruction-following structured output. Pinned
                        # explicitly rather than relying on the model's current
                        # ("minimal") default so a future default change can't
                        # silently reintroduce latency here.
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                    ),
                )
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
        raw_text = re.sub(r",(\s*[}\]])", r"\1", raw_text)  # strip trailing commas before } or ]
        return json.loads(raw_text), None
    except Exception as e:
        return None, str(e)


def rank_shortlist(answers: dict, shortlist_df: pd.DataFrame):
    """
    Returns (ranked_records, dropped_count, error_message).
    Each record includes: fit_score, reasoning, flag, eligibility, breakdown
    (dict of 6 rubric dimensions each with status + note), and is_low_score
    (True if fit_score <= LOW_SCORE_THRESHOLD, used to hide it behind a
    "show less likely matches" toggle in the UI).

    Candidates are split into small batches and sent to Gemini CONCURRENTLY
    (not one at a time) -- this is the main latency win, since generation
    time scales with output size and running N batches in parallel takes
    roughly as long as the single slowest batch, not the sum of all of them.
    """
    if shortlist_df.empty:
        return [], 0, None

    cache_key = _make_cache_key(answers, shortlist_df)
    cached = _result_cache.get(cache_key)
    if cached and (time.time() - cached[3]) < CACHE_TTL_SECONDS:
        return cached[0], cached[1], cached[2]

    capped_df, dropped_count = _cap_candidates(shortlist_df)

    batches = [capped_df.iloc[i:i + BATCH_SIZE] for i in range(0, len(capped_df), BATCH_SIZE)]

    all_rankings = []
    batch_errors = []
    failed_program_names = set()

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_BATCHES, len(batches))) as executor:
        future_to_batch = {executor.submit(_call_batch, answers, batch): batch for batch in batches}
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            rankings, error = future.result()
            if error:
                batch_errors.append(error)
                failed_program_names.update(batch["Program Name"].tolist())
            else:
                all_rankings.extend(rankings)

    rank_by_name = {r["program_name"]: r for r in all_rankings}
    merged = []
    for _, row in capped_df.iterrows():
        record = row.to_dict()
        rank_info = rank_by_name.get(row.get("Program Name"), {})
        fit_score = rank_info.get("fit_score")
        record["fit_score"] = fit_score
        record["reasoning"] = rank_info.get("reasoning")
        record["flag"] = rank_info.get("flag")
        record["eligibility"] = rank_info.get("eligibility", "eligible")
        record["breakdown"] = rank_info.get("breakdown", {})
        record["match_tier"] = "match" if (fit_score is not None and fit_score >= 90) else \
                                "possible" if (fit_score is not None and fit_score >= 75) else "below_threshold"
        record["_tier"] = locality_tier(row)

        zone_note = enterprise_zone_note(row, answers.get("zip_code"))
        if zone_note:
            record["flag"] = zone_note

        oz_note = opportunity_zone_note(row, answers.get("oz_eligible", False), answers.get("oz_tract"))
        if oz_note:
            record["flag"] = oz_note

        merged.append(record)

    # HARD CUTOFF: anything scoring below 70% (or unscored due to an error)
    # is dropped entirely here -- never passed to the template, never shown
    # behind a toggle. The "likely_ineligible" (verification-needed) bucket
    # is kept regardless, since that's a different category (not a low score).
    merged = [r for r in merged if r.get("eligibility") == "likely_ineligible" or r.get("match_tier") in ("match", "possible")]

    def sort_key(r):
        is_ineligible = r.get("eligibility") == "likely_ineligible" or r.get("fit_score") is None
        return (is_ineligible, r.get("_tier", 1), -(r.get("fit_score") or 0))

    merged.sort(key=sort_key)
    for r in merged:
        r.pop("_tier", None)

    error_message = None
    if batch_errors:
        error_message = (
            f"Some programs couldn't be scored due to a temporary error "
            f"({batch_errors[0]}). Try again in a moment for full coverage."
        )

    _result_cache[cache_key] = (merged, dropped_count, error_message, time.time())
    return merged, dropped_count, error_message
