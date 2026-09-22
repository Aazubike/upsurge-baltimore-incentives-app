"""
Persistent per-company/program match cache, backed by Postgres (Supabase).

This replaces the in-memory whole-shortlist cache that used to live in
gemini_matcher.py. That old cache had two problems: it disappeared every
time the app restarted or redeployed, and it cached the entire shortlist
as one unit, so a single changed answer meant re-scoring every program
from scratch.

This cache instead stores one row per (company, program) pair. A pair is
only re-scored by Gemini if either the company's relevant answers changed
or the program's own data changed since it was last cached. Everything
else is served straight from Postgres, at zero Gemini cost.
"""
import os
import json
import hashlib
from contextlib import contextmanager

import psycopg2
import psycopg2.extras


def _get_connection():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL not found. Check your .env file or Render environment.")
    return psycopg2.connect(database_url)


@contextmanager
def _cursor():
    conn = _get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        conn.commit()
    finally:
        conn.close()


def make_company_hash(answers: dict) -> str:
    """
    A fingerprint of the parts of the company's profile that Gemini actually
    sees (see _build_prompt's company_profile in gemini_matcher.py). If any
    of these change, cached results for this company are no longer valid.
    """
    relevant = {
        "county": answers.get("county"),
        "stage": answers.get("stage"),
        "employee_count": answers.get("employee_count"),
        "annual_revenue": answers.get("annual_revenue"),
        "industry": answers.get("industry"),
        "ownership_groups": sorted(answers.get("mwbe_groups", []) or []),
    }
    raw = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def make_program_hash(program_row: dict) -> str:
    """
    A fingerprint of the parts of a single program's data that get sent to
    Gemini (see _build_prompt's programs list in gemini_matcher.py). If the
    program's data changes (updated description, new revenue cap, etc.),
    this hash changes and the old cached result is treated as stale.
    """
    relevant = {
        "description": program_row.get("Program Description"),
        "business_size_requirement": program_row.get("Business Size Requirement"),
        "revenue_income_cap": program_row.get("Revenue / Income Cap"),
        "mwbe_flag": program_row.get("MWBE / DEI Flag?"),
        "industry_exclusions": program_row.get("Industry Exclusions"),
        "eligible_industries_naics": program_row.get("Eligible_Industries_Raw"),
        "scope": program_row.get("Scope"),
        "status": program_row.get("Status"),
    }
    raw = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached_results(company_hash: str, program_names: list) -> dict:
    """
    Looks up whichever of the given program names already have a cached
    result for this exact company_hash. Returns {program_name: result_dict}
    for hits only; anything not in the returned dict is a cache miss and
    needs to be sent to Gemini.

    Note: this does NOT check program_hash here, that's the caller's job
    (see gemini_matcher.py), since the caller already has the current
    program data in hand and can compare it directly against what's stored.
    """
    if not program_names:
        return {}

    with _cursor() as cur:
        cur.execute(
            """
            SELECT program_name, program_hash, fit_score, reasoning, flag,
                   eligibility, breakdown
            FROM gemini_match_cache
            WHERE company_hash = %s AND program_name = ANY(%s)
            """,
            (company_hash, program_names),
        )
        rows = cur.fetchall()

    return {row["program_name"]: dict(row) for row in rows}


def save_results(company_hash: str, results: list, program_hashes: dict) -> None:
    """
    Saves newly-scored results to the cache. `results` is the list of
    ranking dicts Gemini returned (program_name, fit_score, reasoning, flag,
    eligibility, breakdown). `program_hashes` maps program_name -> the hash
    computed by make_program_hash for that program's current data.

    Uses ON CONFLICT to overwrite any existing row for the same
    (company_hash, program_name, program_hash) combination, so re-saving
    the same result twice is harmless.
    """
    if not results:
        return

    # Gemini occasionally returns the same program twice in one response
    # (most often when a truncated/malformed response gets partially
    # salvaged). Postgres's ON CONFLICT DO UPDATE can't affect the same
    # row twice within a single INSERT, so we keep only the last entry
    # for each (company_hash, program_name, program_hash) combination.
    deduped = {}
    for r in results:
        program_name = r.get("program_name")
        program_hash = program_hashes.get(program_name)
        if not program_hash:
            continue
        key = (company_hash, program_name, program_hash)
        deduped[key] = (
            company_hash,
            program_name,
            program_hash,
            r.get("fit_score"),
            r.get("reasoning"),
            r.get("flag"),
            r.get("eligibility"),
            json.dumps(r.get("breakdown") or {}),
        )
    rows = list(deduped.values())

    if not rows:
        return

    with _cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO gemini_match_cache
                (company_hash, program_name, program_hash, fit_score,
                 reasoning, flag, eligibility, breakdown)
            VALUES %s
            ON CONFLICT (company_hash, program_name, program_hash)
            DO UPDATE SET
                fit_score = EXCLUDED.fit_score,
                reasoning = EXCLUDED.reasoning,
                flag = EXCLUDED.flag,
                eligibility = EXCLUDED.eligibility,
                breakdown = EXCLUDED.breakdown
            """,
            rows,
        )
