"""
sheets_logger.py

Logs every app submission (every intake field + every matched program,
split into 3 score tiers) to a Google Sheet, then updates that same row
later when thumbs up/down + comment feedback comes in.

Requires:
    pip install gspread google-auth

Env vars required:
    GOOGLE_SHEET_ID     - the ID from your sheet's URL
    GOOGLE_SHEETS_CREDS - full service account JSON key, minified to one line
"""

import os
import json
import uuid
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Column order — must match your sheet's header row exactly
COLUMNS = [
    "timestamp",
    "submission_id",
    "flow_type",
    "company_name",
    "region",
    "stage",
    "employee_count",
    "annual_revenue",
    "industry",
    "ownership",
    "zip_code",
    "street_address",
    "oz_eligible",
    "oz_tract",
    "matches_90_plus",
    "matches_80_89",
    "matches_75_79",
    "thumbs",
    "comment",
]

_client = None
_sheet = None


def _get_sheet():
    """Lazily authenticate and cache the worksheet handle."""
    global _client, _sheet
    if _sheet is not None:
        return _sheet

    creds_json = os.environ["GOOGLE_SHEETS_CREDS"]
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)

    _client = gspread.authorize(creds)
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    _sheet = _client.open_by_key(sheet_id).sheet1
    return _sheet


def _bucket_matches(matched_programs, match_scores):
    """
    Splits matched programs into 3 tiers by fit_score, so the Sheet has one
    clean column per tier instead of one cluttered blob. Each entry is
    "Program Name (93%)" so the score travels with the name without needing
    its own separate column. Programs with no score (likely_ineligible) are
    skipped here -- they were never a scored match to begin with.
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
    matched_programs: list | None = None,
    match_scores: list | None = None,
) -> str:
    """
    Call this right after Gemini ranking completes.
    Returns a submission_id — hang onto it (e.g. in the session or pass to the
    frontend) so you can attach feedback to this exact row later.

    matched_programs / match_scores should be the FULL lists from
    ranked_shortlist -- every program that came back, in the same order
    (row N in each list refers to the same program). This function buckets
    them into 3 tier columns automatically.
    """
    sheet = _get_sheet()
    submission_id = str(uuid.uuid4())

    tier_90_plus, tier_80_89, tier_75_79 = _bucket_matches(matched_programs, match_scores)

    row = [
        datetime.now(timezone.utc).isoformat(),
        submission_id,
        flow_type,
        company_name,
        region,
        stage,
        employee_count if employee_count is not None else "",
        annual_revenue if annual_revenue is not None else "",
        industry,
        ownership,
        zip_code,
        street_address,
        "yes" if oz_eligible else "no",
        oz_tract,
        "|".join(tier_90_plus),
        "|".join(tier_80_89),
        "|".join(tier_75_79),
        "",  # thumbs - filled in later
        "",  # comment - filled in later
    ]
    sheet.append_row(row, value_input_option="USER_ENTERED")
    return submission_id


def update_feedback(submission_id: str, thumbs: str = "", comment: str = "") -> bool:
    """
    Call this from your /feedback endpoint when the user reacts to their results.
    thumbs should be "up" or "down". Returns False if the submission_id wasn't found
    (e.g. sheet got manually edited) so you can decide how to handle that.
    """
    sheet = _get_sheet()
    try:
        cell = sheet.find(submission_id, in_column=COLUMNS.index("submission_id") + 1)
    except gspread.exceptions.CellNotFound:
        return False

    row_num = cell.row
    thumbs_col = COLUMNS.index("thumbs") + 1
    comment_col = COLUMNS.index("comment") + 1

    if thumbs:
        sheet.update_cell(row_num, thumbs_col, thumbs)
    if comment:
        sheet.update_cell(row_num, comment_col, comment)

    return True
