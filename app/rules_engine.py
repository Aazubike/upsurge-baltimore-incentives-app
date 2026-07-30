"""
Stage 1 of the matching pipeline: deterministic rules-based filter.

This runs BEFORE Gemini touches anything. It uses only the structured
Norm_* columns built during data cleaning. Gemini never re-decides a hard
eligibility gate here — it only reasons over whatever survives this filter,
per the schematic. Anything the normalization couldn't parse cleanly
(Needs_Manual_Review == True) is never hard-excluded by this filter; it's
passed through and left for Gemini to reason about using the raw text.

Answers dict shape:
{
    "county": "Baltimore City",                # one of the 7 counties
    "stage": "seed",                            # pre-seed | seed | early | growth | established
    "employee_count": 8,
    "annual_revenue": 450000,                    # dollars, or None if unknown/declined
    "industry": "Enterprise Technology",         # free text, matched loosely against exclusions
    "mwbe_groups": ["women-owned"],              # list, possibly empty
    "zip_code": "21201",                         # optional, used for Enterprise Zone matching
    "oz_eligible": True,                         # optional, from opportunity_zones.check_opportunity_zone()
    "oz_tract": "24001000800",                   # optional, the matched census tract
}
"""
import pandas as pd
from app.data_loader import get_incentives
from app.industry_map import map_industry_to_naics_buckets, is_consumer_service_only
from app.enterprise_zones import zone_names_for_zip, zip_has_enterprise_zone


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


SPECIFIC_MWBE_CATEGORIES = {"Black", "Hispanic/Latino", "Asian", "Native American", "Women", "Veteran", "Disabled"}


def _mwbe_ok(row, mwbe_groups: list) -> bool:
    level = row.get("Norm_MWBE_Level")
    if level in (None, "No Requirement", "Not Applicable") or pd.isna(level):
        return True
    if level == "Required":
        group_text = row.get("Norm_MWBE_Group") or ""
        named = [g.strip() for g in str(group_text).split(",") if g.strip() in SPECIFIC_MWBE_CATEGORIES]
        if named:
            # We know the exact category this program requires -- exact match only.
            return any(g in mwbe_groups for g in named)
        # Generic requirement (just "MWBE"/"disadvantaged"/"SEDI" with no named
        # category) -- our data can't be more specific, so any qualifying
        # ownership selection is treated as sufficient.
        return len(mwbe_groups) > 0
    # "Encouraged" is a soft preference, not a hard requirement -- Gemini
    # reasons over it, but it doesn't hard-exclude.
    return True


def _employee_ok(row, employee_count) -> bool:
    cap = row.get("Norm_Employee_Max")
    if pd.isna(cap) or cap is None or employee_count is None:
        return True
    return employee_count <= cap


def _industry_ok(row, industry: str) -> bool:
    exclusions = row.get("Industry Exclusions")
    if not pd.isna(exclusions) and exclusions and industry:
        if industry.lower() in exclusions.lower():
            return False

    # Consumer-service-only programs (nail salons, restaurants, retail shops)
    # shouldn't surface for a tech/biotech/etc. startup just because nothing
    # technically excludes them. Hard exclude unless the company's own
    # mapped industry buckets genuinely overlap with that consumer-service set.
    eligible_industries_raw = row.get("Eligible_Industries_Raw")
    if is_consumer_service_only(eligible_industries_raw):
        company_buckets = set(map_industry_to_naics_buckets(industry))
        program_buckets = {c.strip() for c in eligible_industries_raw.split(";")}
        if not company_buckets & program_buckets:
            return False

    return True


def locality_tier(row) -> int:
    """0 = explicitly names a Baltimore-region county (most locally specific),
    1 = statewide 'ALL' program (still eligible, but more general)."""
    counties = row.get("Norm_Counties")
    if pd.isna(counties) or counties is None:
        return 1
    return 0 if counties != "ALL" else 1


def _is_named_enterprise_zone_program(row) -> bool:
    name = row.get("Program Name", "")
    return isinstance(name, str) and "enterprise zone" in name.lower()


def _enterprise_zone_ok(row, zip_code) -> bool:
    """
    Hard exclude Enterprise Zone programs unless we have a zip code AND it
    genuinely matches a real designated zone. No zip = no evidence = excluded
    (we don't show unverifiable claims). Zip matches = let it through to be
    scored normally, with a caveat note attached (see enterprise_zone_note).
    """
    if not _is_named_enterprise_zone_program(row):
        return True
    if not zip_code:
        return False
    return zip_has_enterprise_zone(zip_code)


def enterprise_zone_note(row, zip_code):
    """
    For programs that passed _enterprise_zone_ok (so a zip match is already
    confirmed), returns a specific caveat naming the actual zone(s) -- since
    zones are defined by exact site acreage, not the whole zip code. Returns
    None for anything that isn't an Enterprise Zone program.
    """
    if not _is_named_enterprise_zone_program(row) or not zip_code:
        return None
    zones = zone_names_for_zip(zip_code)
    if not zones:
        return None
    zone_list = ", ".join(zones)
    return (
        f"Your ZIP code ({zip_code}) includes: {zone_list}. Zones are defined by exact site "
        f"acreage, not the whole ZIP code -- confirm your address falls inside the boundary."
    )


def _is_named_opportunity_zone_program(row) -> bool:
    name = row.get("Program Name", "")
    return isinstance(name, str) and "opportunity zone" in name.lower()


def _opportunity_zone_ok(row, oz_eligible: bool) -> bool:
    """
    Hard exclude Opportunity Zone programs unless the submitted address
    geocoded to a census tract on the eligible-tracts list. Mirrors
    _enterprise_zone_ok -- no verified match = excluded, never shown as an
    unverifiable guess. oz_eligible is computed ONCE per submission (see
    opportunity_zones.check_opportunity_zone), not recomputed per row.
    """
    if not _is_named_opportunity_zone_program(row):
        return True
    return bool(oz_eligible)


def opportunity_zone_note(row, oz_eligible: bool, oz_tract):
    """
    For programs that passed _opportunity_zone_ok, returns a caveat naming
    the matched census tract so the person can verify it themselves. Returns
    None for anything that isn't an Opportunity Zone program.
    """
    if not _is_named_opportunity_zone_program(row) or not oz_eligible or not oz_tract:
        return None
    return (
        f"Your address matched census tract {oz_tract}, a designated Qualified "
        f"Opportunity Zone."
    )


def filter_eligible(answers: dict) -> pd.DataFrame:
    """
    Returns the shortlist of programs that pass every parseable hard gate.
    This shortlist (not the full 117) is what gets sent to Gemini for ranking.
    """
    df = get_incentives()
    zip_code = answers.get("zip_code")
    oz_eligible = answers.get("oz_eligible", False)

    keep_mask = df.apply(
        lambda row: (
            _county_ok(row, answers.get("county"))
            and _stage_ok(row, answers.get("stage"))
            and _mwbe_ok(row, answers.get("mwbe_groups", []))
            and _employee_ok(row, answers.get("employee_count"))
            and _industry_ok(row, answers.get("industry"))
            and _enterprise_zone_ok(row, zip_code)
            and _opportunity_zone_ok(row, oz_eligible)
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
