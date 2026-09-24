"""
Loads the three source datasets once at startup and keeps them in memory.
No database needed at this scale (117 programs, ~560 companies, ~480 rounds).
"""
import re
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

_incentives_df = None
_companies_df = None
_rounds_df = None


def _extract_zip_from_address(address) -> str | None:
    """Address SoT is free text like '822 guilford avenue, baltimore, md 21202'
    (sometimes with a trailing source tag like ' - Li'). Pull out a 5-digit
    zip, preferring one in Maryland's real range (206xx-219xx) if there's
    more than one number that looks zip-shaped, since some addresses include
    stray numbers (suite numbers, etc)."""
    if not address or not isinstance(address, str):
        return None
    matches = re.findall(r"\b(\d{5})\b", address)
    for m in matches:
        if 20600 <= int(m) <= 21999:
            return m
    return matches[0] if matches else None


CATEGORY_KEYWORDS = {
    "Tax": ["tax"],
    "Grants": ["grant"],
    "Loans": ["loan", "line of credit"],
    "Workforce / Training": ["training"],
    "Equity / Venture Capital": ["venture capital", "equity"],
    "Property / Development": [
        "property development", "property acquisition",
        "real estate development", "waterfront development",
    ],
    "Technical Assistance": ["technical assistance", "tech transfer", "technical support"],
}


# Zone categories are assigned by program NAME, using the same rule as
# rules_engine._is_named_enterprise_zone_program / _is_named_opportunity_zone_program,
# so the results filter and the eligibility check always agree on which
# programs count as zone programs. (Not imported from rules_engine because
# rules_engine imports this file, which would be a circular import.)
ZONE_CATEGORIES = {
    "Enterprise Zones": "enterprise zone",
    "Opportunity Zones": "opportunity zone",
}


def _classify_categories(row) -> list[str]:
    """
    Tags each program with one or more clean, human-facing categories for
    the results page filter, based on keyword matches across Instrument
    Type, Incentive Purpose, and Incentive Area. These source columns are
    inconsistent -- two different data sources use different formats and
    casing -- so this normalizes them into a small fixed set rather than
    filtering on the raw values directly. A program can land in more than
    one category (many combine grant + loan + tax credit, for example).
    """
    haystack = " ".join(
        str(row.get(col, "") or "")
        for col in ("Instrument Type", "Incentive Purpose", "Incentive Area")
    ).lower()
    categories = [
        category for category, keywords in CATEGORY_KEYWORDS.items()
        if any(kw in haystack for kw in keywords)
    ]
    name = row.get("Program Name", "")
    if isinstance(name, str):
        for category, phrase in ZONE_CATEGORIES.items():
            if phrase in name.lower():
                categories.append(category)
    return categories


def load_all():
    global _incentives_df, _companies_df, _rounds_df
    _incentives_df = pd.read_excel(DATA_DIR / "Incentives_Master_Combined.xlsx")
    _incentives_df["Categories"] = _incentives_df.apply(_classify_categories, axis=1)

    # This export has a title block before the real header row (row 12, 0-indexed).
    companies = pd.read_excel(DATA_DIR / "Known_Companies_v2.xlsx", header=12)
    companies = companies.loc[:, ~companies.columns.str.startswith("Unnamed")]
    companies = companies[companies["Account Name"].notna()].copy()
    companies["Derived_Zip"] = companies["Address SoT"].apply(_extract_zip_from_address)
    _companies_df = companies

    _rounds_df = pd.read_excel(DATA_DIR / "Venture_Rounds_Clean.xlsx")
    return _incentives_df, _companies_df, _rounds_df


def get_incentives() -> pd.DataFrame:
    if _incentives_df is None:
        load_all()
    return _incentives_df


def get_companies() -> pd.DataFrame:
    if _companies_df is None:
        load_all()
    return _companies_df


def get_rounds() -> pd.DataFrame:
    if _rounds_df is None:
        load_all()
    return _rounds_df


def search_companies(query: str, limit: int = 10):
    """Used by the known-company search-as-you-type field on the home screen."""
    df = get_companies()
    if not query:
        return []
    mask = df["Account Name"].str.contains(query, case=False, na=False)
    return df[mask].head(limit).to_dict("records")


def get_company_by_name(account_name: str) -> dict | None:
    df = get_companies()
    match = df[df["Account Name"] == account_name]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def get_rounds_for_company(account_id: str):
    """Returns funding rounds for a company, most recent first, for the stage-suggestion logic."""
    df = get_rounds()
    if account_id is None or pd.isna(account_id):
        return []
    matches = df[df["Account ID"] == account_id]
    if matches.empty:
        return []
    if "Fiscal Year" in matches.columns:
        matches = matches.sort_values("Fiscal Year", ascending=False)
    return matches.to_dict("records")


def suggest_stage_from_rounds(account_id: str) -> str | None:
    """Returns the suggested stage from the most recent funding round, or None if no rounds on file."""
    rounds = get_rounds_for_company(account_id)
    for r in rounds:
        stage = r.get("Suggested_Stage")
        if stage and not pd.isna(stage):
            return stage
    return None


def get_industry_options():
    """Clean, deduped industry list for the intake form dropdown."""
    # Known spelling/pluralization variants from different source data --
    # these are the SAME category, just written differently across sources.
    CANONICAL_MERGE = {
        "aerospace and defence": "Aerospace and Defense",
        "aerospace and defense": "Aerospace and Defense",
        "medical device": "Medical Devices",
        "medical devices": "Medical Devices",
    }
    df = get_companies()
    raw = df["Industry SoT"].dropna().tolist()
    cleaned = set()
    for v in raw:
        v = str(v).split(" - ")[0].strip()
        if v and v.lower() != "no value":
            v = CANONICAL_MERGE.get(v.lower(), v)
            cleaned.add(v)
    return sorted(cleaned)


def parse_employee_count(value) -> int | None:
    """Accounts data stores employee count as a range string like '11-50 - Li'.
    Returns the midpoint as an int, or None if unparseable."""
    import re
    if value is None or pd.isna(value):
        return None
    s = str(value)
    m = re.search(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) // 2
    m = re.search(r"(\d+)\+", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return None

