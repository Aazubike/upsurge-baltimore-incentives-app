"""
Logs match submissions and feedback to Postgres (Supabase), replacing the
old Google Sheets logger. Same function names and signatures as the old
sheets_logger.py so main.py only needs its import line changed, not its
call sites.
"""
import os
from datetime import datetime, timezone

import psycopg2


def _get_connection():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL not found. Check your .env file or Render environment.")
    return psycopg2.connect(database_url)


def _bucket_matches(matched_programs, match_scores):
    """
    Splits matched programs into 3 tiers by fit_score. Each entry is
    "Program Name (93%)" so the score travels with the name. Programs with
    no score (likely_ineligible) are skipped -- they were never a scored
    match to begin with. Same logic as the old Sheets version.
    """
    tier_90_plus, tier_80_89, tier_75_79 = [], [], []
    for name, score in zip(matched_programs or [], match_scores or []):
        if score is None:
            continue
        entry = f"{name} ({score}%)"
        if score >= 90:
            tier_90_plus.append(entry)
        elif score >= 80:
            tier_80_89.append(entry)
        elif score >= 75:
            tier_75_79.append(entry)
    return tier_90_plus, tier_80_89, tier_75_79


def log_submission(
    submission_id: str,
    flow_type: str,
    company_name: str = "",
    region: str = "",
    stage: str = "",
    employee_count=None,
    annual_revenue=None,
    industry: str = "",
    ownership: str = "",
    zip_code: str = "",
    street_address: str = "",
    oz_eligible: bool = False,
    oz_tract: str = "",
    matched_programs: list = None,
    match_scores: list = None,
) -> None:
    """
    Writes one row to match_submissions. Called from a background task in
    main.py, wrapped in a try/except there so a logging failure never
    takes down the actual match results a user is waiting on.
    """
    tier_90_plus, tier_80_89, tier_75_79 = _bucket_matches(matched_programs, match_scores)

    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO match_submissions
                (submission_id, created_at, flow_type, company_name, region,
                 stage, employee_count, annual_revenue, industry, ownership,
                 zip_code, street_address, oz_eligible, oz_tract,
                 tier_90_plus, tier_80_89, tier_75_79)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                submission_id,
                datetime.now(timezone.utc),
                flow_type,
                company_name,
                region,
                stage,
                str(employee_count) if employee_count is not None else "",
                str(annual_revenue) if annual_revenue is not None else "",
                industry,
                ownership,
                zip_code,
                street_address,
                oz_eligible,
                oz_tract,
                "|".join(tier_90_plus),
                "|".join(tier_80_89),
                "|".join(tier_75_79),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_feedback(submission_id: str, thumbs: str = "", comment: str = "") -> bool:
    """
    Call this from the /feedback endpoint when the user reacts to their
    results. thumbs should be "up" or "down". Returns False if the
    submission_id wasn't found, so the caller can decide how to handle that.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        updates = []
        values = []
        if thumbs:
            updates.append("thumbs = %s")
            values.append(thumbs)
        if comment:
            updates.append("comment = %s")
            values.append(comment)

        if not updates:
            return True

        values.append(submission_id)
        cur.execute(
            f"UPDATE match_submissions SET {', '.join(updates)} WHERE submission_id = %s",
            values,
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
