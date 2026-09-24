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

PERSISTENT CACHE: before any program is sent to Gemini, we check the
Postgres-backed cache in match_cache.py. A program is only re-scored if
either the company's relevant answers changed or that program's own data
changed since it was last cached. This is what keeps Gemini credit usage
down on repeat runs. See match_cache.py for the details.
"""
import os
import json
import re
import time
import random
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.rules_engine import (
    locality_tier, enterprise_zone_note, opportunity_zone_note,
    _is_named_enterprise_zone_program, _is_named_opportunity_zone_program,
)
from app import match_cache

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL_NAME = "gemini-3.1-flash-lite"
MAX_RETRIES = 3
RETRYABLE_MARKERS = (
    "503", "UNAVAILABLE",
    "429", "RESOURCE_EXHAUSTED",
    "DEADLINE_EXCEEDED", "timeout", "Timeout",
    "500", "INTERNAL",
)
BATCH_SIZE = 8
MAX_PARALLEL_BATCHES = 10
MAX_OUTPUT_TOKENS = 4096

# OUTER retry layer, separate from the per-call retries inside _call_batch.
# A batch that exhausts its internal retries gets retried again in a fresh
# round, up to OUTER_MAX_ROUNDS times, with a pause between rounds so a
# rate limit or transient outage has time to clear. Correctness over speed:
# only after every round is exhausted do we surface an error for whatever's
# still failing.
OUTER_MAX_ROUNDS = 5
OUTER_ROUND_PAUSE_SECONDS = 5

FAST_MODE = os.environ.get("FAST_MODE", "0") == "1"
SAFETY_NET_MAX_CANDIDATES = 300

# Cache lookups can fail (network hiccup, Supabase briefly unavailable).
# When that happens we log it and fall back to treating everything as a
# cache miss for that run, rather than blowing up the whole match.
def _safe_get_cached(company_hash, program_names):
    try:
        return match_cache.get_cached_results(company_hash, program_names)
    except Exception as e:
        print(f"[gemini_matcher] cache lookup failed, continuing without cache: {e!r}")
        return {}


def _safe_save(company_hash, results, program_hashes):
    try:
        match_cache.save_results(company_hash, results, program_hashes)
    except Exception as e:
        print(f"[gemini_matcher] cache save failed (non-fatal): {e!r}")


_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")
        _client = genai.Client(api_key=api_key)
    return _client


def _cap_priority(row) -> int:
    """
    Order used when the shortlist has to be cut down to the cap.
      -1 = Enterprise Zone or Opportunity Zone program. These only reach the
           shortlist when the rules engine found real evidence the company is
           in a zone, so they are never cut.
       0 = explicitly names a Baltimore-region county
       1 = statewide program
    """
    if _is_named_enterprise_zone_program(row) or _is_named_opportunity_zone_program(row):
        return -1
    return locality_tier(row)


def _cap_candidates(shortlist_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if len(shortlist_df) <= SAFETY_NET_MAX_CANDIDATES:
        return shortlist_df, 0
    df = shortlist_df.copy()
    df["_tier"] = df.apply(_cap_priority, axis=1)
    df = df.sort_values("_tier", kind="stable")
    capped = df.head(SAFETY_NET_MAX_CANDIDATES).drop(columns=["_tier"])
    return capped, len(shortlist_df) - SAFETY_NET_MAX_CANDIDATES


def verified_location_status(answers: dict) -> list:
    """
    Plain-English facts about zones the company's address has been VERIFIED
    to be inside, for Gemini to see. Only address-level verified results are
    included. ZIP-only matches are not, since those aren't confirmed.
    Shared with match_cache.make_company_hash so the cache key changes
    whenever these facts change.
    """
    facts = []
    ez = answers.get("ez_result") or {}
    if ez.get("status") == "in_zone":
        for z in ez.get("zones", []):
            if z.get("name") and not z.get("expired"):
                facts.append(
                    f"Address verified inside the {z['name']} (official Maryland "
                    f"Department of Commerce Enterprise Zone boundaries)."
                )
        for f in ez.get("focus_areas", []):
            if f.get("name"):
                facts.append(f"Address verified inside the {f['name']} Enterprise Zone Focus Area.")
    if answers.get("oz_eligible") and answers.get("oz_tract"):
        facts.append(
            f"Address verified in census tract {answers['oz_tract']}, a designated "
            f"Qualified Opportunity Zone."
        )
    return facts


def _build_prompt(answers: dict, shortlist_df: pd.DataFrame) -> str:
    company_profile = {
        "county": answers.get("county"),
        "stage": answers.get("stage"),
        "employee_count": answers.get("employee_count"),
        "annual_revenue": answers.get("annual_revenue"),
        "industry": answers.get("industry"),
        "ownership_groups": answers.get("mwbe_groups", []),
    }
    location_facts = verified_location_status(answers)
    if location_facts:
        company_profile["verified_location_status"] = location_facts

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

    breakdown_rule = (
        "Do not include a \"breakdown\" field in this response -- it is disabled for speed."
        if FAST_MODE else
        'For EVERY scored program, return a "breakdown" object covering all 6 rubric '
        'dimensions (location, stage, size, industry, mwbe, overall). Each dimension '
        'needs a "status" of exactly one of: "match", "partial", "no_data", "unmet". '
        'Each needs a "note": 2-4 words max, terse tag style, not a sentence '
        '(e.g. "County match", "Not specified", "Tech services fit" -- not '
        '"Baltimore City, exact match for this program").'
    )

    base_prompt = f"""You are ranking Maryland business incentive programs for a specific company.
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
5. {breakdown_rule}
6. CRITICAL -- MWBE ANTI-HALLUCINATION RULE: mwbe_flag text almost never names
   a specific race, ethnicity, or gender (it usually just says generic terms
   like "MWBE", "veteran", "disadvantaged", "SEDI"). NEVER state a specific
   race, ethnicity, or gender in your mwbe note or reasoning unless that exact
   word appears in mwbe_flag. If mwbe_flag says "MWBE" generically, your note
   must also stay generic ("MWBE required", not "Black-owned business match"
   or "Women-owned match" or any invented specificity). Inventing demographic
   detail not in the source data is a serious error.
7. VERIFIED ZONES: if the company profile includes "verified_location_status",
   those facts were confirmed by checking the company's street address against
   official government zone boundaries. For any Enterprise Zone or Opportunity
   Zone program matching a verified zone, treat the zone location requirement
   as MET. Do not describe it as uncertain, "likely", or "if situated in a
   zone", and score the location dimension as a full match.

Candidate programs (already passed hard filters):
{json.dumps(programs, indent=2)}
"""

    if FAST_MODE:
        base_prompt += """Return ONLY a JSON array, no markdown fences, no commentary. Each element:
{
  "program_name": "<exact name from input>",
  "eligibility": "<'eligible' or 'likely_ineligible'>",
  "fit_score": <integer 1-100, or null if likely_ineligible>,
  "reasoning": "<one plain-English sentence, under 25 words>",
  "flag": "<short note if needs_manual_review or a real eligibility concern, else null>"
}

Order the array by fit_score descending (nulls last)."""
    else:
        base_prompt += """Return ONLY a JSON array, no markdown fences, no commentary. Each element:
{
  "program_name": "<exact name from input>",
  "eligibility": "<'eligible' or 'likely_ineligible'>",
  "fit_score": <integer 1-100, or null if likely_ineligible>,
  "reasoning": "<one plain-English sentence, under 25 words>",
  "flag": "<short note if needs_manual_review or a real eligibility concern, else null>",
  "breakdown": {
    "location": {"status": "match|partial|no_data|unmet", "note": "<short note>"},
    "stage": {"status": "match|partial|no_data|unmet", "note": "<short note>"},
    "size": {"status": "match|partial|no_data|unmet", "note": "<short note>"},
    "industry": {"status": "match|partial|no_data|unmet", "note": "<short note>"},
    "mwbe": {"status": "match|partial|no_data|unmet", "note": "<short note>"},
    "overall": {"status": "match|partial|no_data|unmet", "note": "<short note>"}
  }
}

Order the array by fit_score descending (nulls last)."""

    return base_prompt


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
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                )
                raw_text = response.text.strip()
                break
            except Exception as e:
                last_error = e
                is_retryable = any(marker in str(e) for marker in RETRYABLE_MARKERS)
                if is_retryable and attempt < MAX_RETRIES:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                raise
        if raw_text is None:
            raise last_error

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
        raw_text = re.sub(r",(\s*[}\]])", r"\1", raw_text)

        try:
            return json.loads(raw_text), None
        except json.JSONDecodeError as e:
            if e.msg == "Extra data":
                try:
                    salvaged, _ = json.JSONDecoder().raw_decode(raw_text)
                    return salvaged, None
                except json.JSONDecodeError:
                    pass
            raise
    except Exception as e:
        print(f"[gemini_matcher] batch failed: {e!r}")
        return None, "temporary scoring error"


def _run_batches(answers: dict, batches: list, progress_callback=None):
    """
    Fires the given batches concurrently. Returns (rankings, failed_batches,
    error_reasons): failed_batches is the subset of the input batch DataFrames
    that came back with an error, so the caller can retry just those.

    progress_callback, if given, is called as progress_callback("progress",
    delta=N) each time a batch of N programs finishes SUCCESSFULLY -- failed
    batches don't count toward progress yet, since they may still succeed on
    a later retry round.
    """
    rankings = []
    failed_batches = []
    error_reasons = []

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_BATCHES, len(batches))) as executor:
        future_to_batch = {executor.submit(_call_batch, answers, batch): batch for batch in batches}
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            batch_rankings, error = future.result()
            if error:
                failed_batches.append(batch)
                error_reasons.append(error)
            else:
                rankings.extend(batch_rankings)
                if progress_callback:
                    progress_callback("progress", delta=len(batch))

    return rankings, failed_batches, error_reasons


def rank_shortlist(answers: dict, shortlist_df: pd.DataFrame, progress_callback=None):
    """
    Returns (ranked_records, dropped_count, error_message).

    progress_callback, if given, is called with:
      progress_callback("total", total=N)      -- once, right after capping,
                                                    with the real candidate count
      progress_callback("progress", delta=N)   -- each time N more programs
                                                    finish scoring (success or,
                                                    at the very end, permanent
                                                    failure after all retry
                                                    rounds are exhausted)
    This lets the caller (main.py) show real "X of Y checked" progress
    instead of a fake or indeterminate bar.

    Batches that still fail after their internal retries are retried again
    in a fresh round (see OUTER_MAX_ROUNDS) instead of being given up on --
    correctness wins over speed here: a slow, fully-scored result beats a
    fast one with programs silently missing.

    CACHING: before anything is sent to Gemini, each program in the
    shortlist is checked against the persistent cache (match_cache.py).
    Only cache misses go to Gemini; hits are reused as-is. New results are
    saved back to the cache once Gemini returns them.
    """
    if shortlist_df.empty:
        return [], 0, None

    capped_df, dropped_count = _cap_candidates(shortlist_df)

    if progress_callback:
        progress_callback("total", total=len(capped_df))

    company_hash = match_cache.make_company_hash(answers)

    # Compute each candidate program's current data hash, and split into
    # cache hits (reuse) vs misses (need to ask Gemini).
    program_hashes = {}
    row_by_name = {}
    for _, row in capped_df.iterrows():
        name = row.get("Program Name")
        row_dict = row.to_dict()
        program_hashes[name] = match_cache.make_program_hash(row_dict)
        row_by_name[name] = row_dict

    all_names = list(row_by_name.keys())
    cached = _safe_get_cached(company_hash, all_names)

    cache_hit_results = []
    miss_names = []
    for name in all_names:
        hit = cached.get(name)
        if hit and hit.get("program_hash") == program_hashes.get(name):
            cache_hit_results.append({
                "program_name": name,
                "fit_score": hit.get("fit_score"),
                "reasoning": hit.get("reasoning"),
                "flag": hit.get("flag"),
                "eligibility": hit.get("eligibility"),
                "breakdown": hit.get("breakdown") or {},
            })
        else:
            miss_names.append(name)

    if progress_callback and cache_hit_results:
        progress_callback("progress", delta=len(cache_hit_results))

    miss_df = capped_df[capped_df["Program Name"].isin(miss_names)]
    remaining_batches = [miss_df.iloc[i:i + BATCH_SIZE] for i in range(0, len(miss_df), BATCH_SIZE)]

    all_rankings = list(cache_hit_results)
    last_error_reason = None

    for round_num in range(OUTER_MAX_ROUNDS):
        if not remaining_batches:
            break
        if round_num > 0:
            time.sleep(min(OUTER_ROUND_PAUSE_SECONDS * round_num, 20))
        rankings, remaining_batches, error_reasons = _run_batches(
            answers, remaining_batches, progress_callback=progress_callback
        )
        all_rankings.extend(rankings)
        if rankings:
            _safe_save(company_hash, rankings, program_hashes)
        if error_reasons:
            last_error_reason = error_reasons[0]

    # Whatever's still failing after every round is permanently given up on --
    # count those programs toward progress now so the bar still reaches 100%
    # instead of stalling short of it.
    if remaining_batches and progress_callback:
        leftover = sum(len(b) for b in remaining_batches)
        progress_callback("progress", delta=leftover)

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

        # Zone notes. An address-verified zone match is good news, so it goes
        # in "verified_note" (shown as a confirmation, not a warning) and
        # Gemini's own flag, if any, is kept. A ZIP-only Enterprise Zone
        # match is still unconfirmed, so it stays in "flag" as a caveat.
        verified_notes = []
        ez_result = answers.get("ez_result") or {}
        zone_note = enterprise_zone_note(row, answers.get("zip_code"), ez_result)
        if zone_note:
            if ez_result.get("status") == "in_zone":
                verified_notes.append(zone_note)
            else:
                record["flag"] = zone_note

        oz_note = opportunity_zone_note(row, answers.get("oz_eligible", False), answers.get("oz_tract"))
        if oz_note:
            verified_notes.append(oz_note)

        record["verified_note"] = " · ".join(verified_notes) if verified_notes else None

        merged.append(record)

    merged = [r for r in merged if r.get("eligibility") == "likely_ineligible" or r.get("match_tier") in ("match", "possible")]

    def sort_key(r):
        is_ineligible = r.get("eligibility") == "likely_ineligible" or r.get("fit_score") is None
        return (is_ineligible, r.get("_tier", 1), -(r.get("fit_score") or 0))

    merged.sort(key=sort_key)
    for r in merged:
        r.pop("_tier", None)

    error_message = None
    if remaining_batches:
        error_message = (
            f"Some programs couldn't be scored after several attempts "
            f"({last_error_reason}). Try again in a moment for full coverage."
        )

    return merged, dropped_count, error_message
