"""
sheets_logger.py

Logs every app submission (intake answers + matched programs) to a Google Sheet,
then updates that same row later when thumbs up/down + comment feedback comes in.

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
    "industry",
    "ownership",
    "zip_code",
    "matched_programs",
    "match_tiers",
    "match_scores",
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


def log_submission(
    flow_type: str,
    company_name: str = "",
    region: str = "",
    stage: str = "",
    industry: str = "",
    ownership: str = "",
    zip_code: str = "",
    matched_programs: list[str] | None = None,
    match_tiers: list[str] | None = None,
    match_scores: list | None = None,
) -> str:
    """
    Call this right after Gemini ranking completes.
    Returns a submission_id — hang onto it (e.g. in the session or pass to the
    frontend) so you can attach feedback to this exact row later.

    matched_programs / match_tiers / match_scores should all be the FULL lists
    from ranked_shortlist -- every program that came back (both "match" 90%+
    and "possible" 75-89%), not just the top ones. Pass them in the same
    order so row N in each list refers to the same program.
    """
    sheet = _get_sheet()
    submission_id = str(uuid.uuid4())

    row = [
        datetime.now(timezone.utc).isoformat(),
        submission_id,
        flow_type,
        company_name,
        region,
        stage,
        industry,
        ownership,
        zip_code,
        "|".join(matched_programs or []),
        "|".join(match_tiers or []),
        "|".join(str(s) for s in (match_scores or [])),
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
