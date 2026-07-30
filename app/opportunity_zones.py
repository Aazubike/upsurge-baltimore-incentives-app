"""
Real Opportunity Zone data (U.S. Treasury/IRS-designated Qualified Opportunity
Zones, tracked at the census-tract level -- provided as
Maryland_Qualifying_OZ_Census_Tracts.xlsx).

Unlike Enterprise Zones (zip-code level), Opportunity Zone eligibility is
census-TRACT level, which is much smaller than a zip code -- there's no way
to approximate it from a zip alone. So this module geocodes the actual
street address (via the free U.S. Census Bureau Geocoder) to get its exact
census tract, then checks that tract against the eligible-tracts list.

Same honest, asymmetric pattern as Enterprise Zones (see enterprise_zones.py):
  - Address geocodes to a tract NOT in this list -> confidently NOT eligible.
    Hard-excluded from the Opportunity Zone program entirely.
  - Address geocodes to a tract IN this list -> eligible. Shown with a note
    naming the matched tract so the person can verify it themselves.
  - No address provided, or geocoding fails/times out -> no basis for either
    conclusion, so treated as NOT eligible ("no evidence = excluded" -- we
    never show an unverifiable claim).
"""
import pandas as pd
import requests
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "Maryland_Qualifying_OZ_Census_Tracts.xlsx"
GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
GEOCODER_TIMEOUT_SECONDS = 6

_eligible_tracts = None
_geocode_cache = {}  # {normalized_address: tract_or_None} -- avoids re-geocoding
                      # the same address if it's submitted more than once


def _load_eligible_tracts():
    global _eligible_tracts
    if _eligible_tracts is not None:
        return _eligible_tracts
    df = pd.read_excel(DATA_PATH)
    _eligible_tracts = set(df["Census Tract"].astype(str).str.strip())
    return _eligible_tracts


def geocode_to_tract(street_address: str):
    """
    Calls the free Census Bureau Geocoder to turn a street address into an
    11-digit census tract GEOID (state+county+tract). Returns None if the
    address doesn't geocode, or if the API call fails/times out for any
    reason -- callers should treat None the same as "not eligible."
    """
    if not street_address:
        return None

    key = street_address.strip().lower()
    if key in _geocode_cache:
        return _geocode_cache[key]

    tract = None
    try:
        response = requests.get(
            GEOCODER_URL,
            params={
                "address": street_address,
                "benchmark": "Public_AR_Current",
                "vintage": "Current_Current",
                "format": "json",
            },
            timeout=GEOCODER_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        matches = data.get("result", {}).get("addressMatches", [])
        if matches:
            geographies = matches[0].get("geographies", {})
            tracts = geographies.get("Census Tracts", [])
            if tracts:
                tract = str(tracts[0].get("GEOID", "")).strip()
    except Exception:
        # Network error, timeout, unexpected response shape, etc. -- fail
        # safe to None rather than raising and breaking the whole submission.
        tract = None

    _geocode_cache[key] = tract
    return tract


def check_opportunity_zone(street_address: str):
    """
    Returns (is_eligible: bool, tract: str | None).
    Call this ONCE per submission (not once per program) -- geocoding is a
    network call, not a cheap local lookup.
    """
    tract = geocode_to_tract(street_address)
    if tract is None:
        return False, None
    eligible_tracts = _load_eligible_tracts()
    return tract in eligible_tracts, tract
