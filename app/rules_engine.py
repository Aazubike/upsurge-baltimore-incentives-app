"""
Stage 1 of the matching pipeline: deterministic rules-based filter.

This runs BEFORE Gemini touches anything. It uses only the structured
Norm_* columns built during data cleaning. Gemini never re-decides a hard
eligibility gate here — it only reasons over whatever survives this filter,
per the schematic. Anything the normalization couldn't parse cleanly
(Needs_Manual_Review == True) is never hard-excluded by this filter; it's
passed through and left for Gemini to reason about using the raw text.

Answers dict shape (all 5 questions):
{
    "county": "Baltimore City",                # one of the 7 counties
    "stage": "seed",                            # pre-seed | seed | early | growth | established
    "employee_count": 8,
    "annual_revenue": 450000,                    # dollars, or None if unknown/declined
    "industry": "Enterprise Technology",         # free text, matched loosely against exclusions
    "mwbe_groups": ["women-owned"],              # list, possibly empty
}
"""
import pandas as pd
from app.data_loader import get_incentives


def _county_ok(row, county: str) -> bool:
    counties = row.get("Norm_Counties")
    if pd.isna(counties) or counties is None:
        return True  # unparsed -> don't hard-exclude, let Gemini judge from raw text
    if counties == "ALL":
        return True
    return county in counties.split(",")


def _stage_ok(row, stage: str) -> bool:
    if row.get("Institutional_Only"):
        return False  # Government/Higher-Ed-only programs never match a business
    tags = row.get("Norm_Stage_Tags")
    if pd.isna(tags) or tags is None:
        return True  # unparsed -> pass through for Gemini to judge
    return stage in tags.split(",")


def _mwbe_ok(row, mwbe_groups: list) -> bool:
    level = row.get("Norm_MWBE_Level")
    if level in (None, "No Requirement", "Not Applicable") or pd.isna(level):
        return True
    if level in ("Required", "Encouraged"):
        # "Required" should really be a stricter gate than "Encouraged," but since MWBE
        # status isn't verified anywhere in the underlying data, we don't hard-exclude here.
        # Instead this becomes a strong signal Gemini uses when ranking, with the group
        # requirement passed through as context.
        return True
    return True


def _employee_ok(row, employee_count) -> bool:
    cap = row.get("Norm_Employee_Max")
    if pd.isna(cap) or cap is None or employee_count is None:
        return True
    return employee_count <= cap


def _industry_ok(row, industry: str) -> bool:
    exclusions = row.get("Industry Exclusions")
    if pd.isna(exclusions) or not exclusions or not industry:
        return True
    return industry.lower() not in exclusions.lower()


def filter_eligible(answers: dict) -> pd.DataFrame:
    """
    Returns the shortlist of programs that pass every parseable hard gate.
    This shortlist (not the full 117) is what gets sent to Gemini for ranking.
    """
    df = get_incentives()

    keep_mask = df.apply(
        lambda row: (
            _county_ok(row, answers.get("county"))
            and _stage_ok(row, answers.get("stage"))
            and _mwbe_ok(row, answers.get("mwbe_groups", []))
            and _employee_ok(row, answers.get("employee_count"))
            and _industry_ok(row, answers.get("industry"))
        ),
        axis=1,
    )
    return df[keep_mask].copy()


def filter_summary(answers: dict) -> dict:
    """Quick counts for debugging/demoing the filter step before Gemini runs."""
    full = get_incentives()
    shortlist = filter_eligible(answers)
    return {
        "total_programs": len(full),
        "eligible_after_rules": len(shortlist),
        "excluded": len(full) - len(shortlist),
        "shortlist_names": shortlist["Program Name"].tolist(),
    }
