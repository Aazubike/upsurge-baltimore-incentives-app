"""
Logs match submissions and all related activity to Postgres (Supabase):
the original submission and its full results, outbound link clicks,
per-program "was this match good" reactions, and the end-of-results
survey. Also provides the lookups the internal dashboard page uses.
"""
import os
import json
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras


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
    match to begin with. This is the quick-glance summary; full_results
    (see log_submission) carries the complete detail per program.
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
    full_results: list = None,
    wants_contact: bool = False,
    contact_email: str = "",
    contact_phone: str = "",
) -> None:
    """
    Writes one row to match_submissions: every intake answer, the tiered
    summary, and (in full_results) the complete result for every scored
    program -- name, score, eligibility, reasoning, flag. wants_contact,
    contact_email, and contact_phone capture whether the person asked to be
    connected with someone who can help identify incentives faster. Called
    from a background task in main.py, wrapped in a try/except there so a
    logging failure never takes down the actual match results a user is
    waiting on.
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
                 tier_90_plus, tier_80_89, tier_75_79, full_results,
                 wants_contact, contact_email, contact_phone)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                json.dumps(full_results or []),
                wants_contact,
                contact_email,
                contact_phone,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_feedback(submission_id: str, thumbs: str = "", comment: str = "") -> bool:
    """
    Overall (not per-program) thumbs + comment on match_submissions itself.
    Kept for compatibility; not currently wired to any UI element.
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


def log_link_click(submission_id: str, program_name: str) -> None:
    """Records that someone clicked "Apply / learn more" for a specific
    program on a specific match. Called from the /go redirect route."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO link_clicks (submission_id, program_name) VALUES (%s, %s)",
            (submission_id, program_name),
        )
        conn.commit()
    finally:
        conn.close()


def log_program_feedback(submission_id: str, program_name: str, thumbs: str) -> None:
    """Records whether someone applied to a specific program after clicking
    out to it ('applied' or 'not_applied'), from the popup on results.html."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO program_feedback (submission_id, program_name, thumbs) VALUES (%s, %s, %s)",
            (submission_id, program_name, thumbs),
        )
        conn.commit()
    finally:
        conn.close()


def save_feedback_response(
    submission_id: str,
    relevance_rating=None,
    found_what_needed: str = "",
    improve_notes: str = "",
    missing_data_notes: str = "",
) -> None:
    """
    Saves feedback for a submission: the quick relevance rating (1-3, from
    the buttons on the results page) and/or the fuller survey answers
    (overall read, what could improve, missing data/category notes). These
    can arrive separately -- a quick rating now, a fuller survey later, or
    vice versa -- so each field only overwrites when something new is
    actually provided; an unset field never blanks out a value saved earlier.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback_responses
                (submission_id, relevance_rating, found_what_needed, improve_notes, missing_data_notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (submission_id) DO UPDATE SET
                relevance_rating = COALESCE(EXCLUDED.relevance_rating, feedback_responses.relevance_rating),
                found_what_needed = COALESCE(NULLIF(EXCLUDED.found_what_needed, ''), feedback_responses.found_what_needed),
                improve_notes = COALESCE(NULLIF(EXCLUDED.improve_notes, ''), feedback_responses.improve_notes),
                missing_data_notes = COALESCE(NULLIF(EXCLUDED.missing_data_notes, ''), feedback_responses.missing_data_notes)
            """,
            (submission_id, relevance_rating, found_what_needed, improve_notes, missing_data_notes),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_submissions(limit: int = 50) -> list:
    """Powers the internal dashboard's list view: newest matches first,
    with click and feedback counts joined in."""
    conn = _get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT
                s.submission_id, s.created_at, s.company_name, s.flow_type,
                s.region, s.stage, s.industry,
                s.tier_90_plus, s.tier_80_89, s.tier_75_79,
                s.wants_contact, s.contact_email, s.contact_phone,
                (SELECT COUNT(*) FROM link_clicks c WHERE c.submission_id = s.submission_id) AS click_count,
                (SELECT COUNT(*) FROM program_feedback pf WHERE pf.submission_id = s.submission_id) AS feedback_count,
                fr.relevance_rating, fr.found_what_needed
            FROM match_submissions s
            LEFT JOIN feedback_responses fr ON fr.submission_id = s.submission_id
            ORDER BY s.created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_submission_detail(submission_id: str) -> dict:
    """Powers the internal dashboard's detail view for one submission:
    full intake, full results, every click, every program reaction, and
    the survey response, if any."""
    conn = _get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM match_submissions WHERE submission_id = %s", (submission_id,))
        submission = cur.fetchone()
        if submission is None:
            return None
        submission = dict(submission)

        cur.execute(
            "SELECT program_name, clicked_at FROM link_clicks WHERE submission_id = %s ORDER BY clicked_at",
            (submission_id,),
        )
        submission["clicks"] = [dict(row) for row in cur.fetchall()]

        cur.execute(
            "SELECT program_name, thumbs, created_at FROM program_feedback WHERE submission_id = %s ORDER BY created_at",
            (submission_id,),
        )
        submission["program_feedback"] = [dict(row) for row in cur.fetchall()]

        cur.execute(
            "SELECT relevance_rating, found_what_needed, missing_data_notes, created_at FROM feedback_responses WHERE submission_id = %s",
            (submission_id,),
        )
        row = cur.fetchone()
        submission["survey"] = dict(row) if row else None

        return submission
    finally:
        conn.close()
