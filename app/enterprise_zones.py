"""
Real Enterprise Zone data (from Maryland's official designated zones, provided
by GBC's Patrick Hosford). This is zip-code-level, NOT exact address/parcel
boundaries -- a zip code can be much larger than the actual zone (which is
measured in acres). So this data supports two honest, asymmetric conclusions:

  - Zip code NOT in this list -> confident the business is NOT in any
    Enterprise Zone. Hard-excluded from Enterprise Zone programs entirely.
  - Zip code IS in this list -> the business MIGHT be in one of the named
    zones, but exact site boundaries still matter (acreage, not the whole
    zip). Shown as a flagged "needs verification" match, naming the specific
    zone(s) so the person knows exactly what to check.
  - No zip code provided at all -> we have no basis for either conclusion,
    stays in the generic "can't verify" bucket.
"""
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "Enterprise_Zones.csv"

_zip_to_zones = None


def _load():
    global _zip_to_zones
    if _zip_to_zones is not None:
        return _zip_to_zones
    df = pd.read_csv(DATA_PATH)
    df["zip"] = df["zip"].astype(str).str.strip()
    df["sitename"] = df["sitename"].astype(str).str.replace(r"[\r\n]+", "", regex=True).str.strip()
    _zip_to_zones = df.groupby("zip")["sitename"].apply(list).to_dict()
    return _zip_to_zones


def zone_names_for_zip(zip_code: str):
    """Returns a list of Enterprise Zone names whose zip matches, or [] if none."""
    if not zip_code:
        return []
    zip_to_zones = _load()
    return zip_to_zones.get(str(zip_code).strip(), [])


def zip_has_enterprise_zone(zip_code: str) -> bool:
    return len(zone_names_for_zip(zip_code)) > 0
