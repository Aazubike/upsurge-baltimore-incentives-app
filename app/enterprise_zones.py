"""
Real Enterprise Zone data, checked two ways.

1. ZIP-CODE LEVEL (original, unchanged)
   From Maryland's official designated zones, provided by GBC's Patrick
   Hosford. A zip code can be much larger than the actual zone (which is
   measured in acres), so this supports two honest, asymmetric conclusions:

     - Zip code NOT in this list -> confident the business is NOT in any
       Enterprise Zone. Hard-excluded from Enterprise Zone programs entirely.
     - Zip code IS in this list -> the business MIGHT be in one of the named
       zones, but exact site boundaries still matter. Shown as a flagged
       "needs verification" match, naming the specific zone(s).
     - No zip code provided at all -> no basis for either conclusion,
       stays in the generic "can't verify" bucket.

2. ADDRESS LEVEL (new)
   Geocodes the street address with the free U.S. Census Bureau Geocoder
   (same service as opportunity_zones.py), then asks the State's official
   Enterprise Zone boundary map (MD iMAP, maintained by the Maryland
   Department of Commerce) whether that exact point is inside a zone.
   Also checks Enterprise Zone Focus Areas.

   Possible "status" values from check_enterprise_zone_address():
     - "in_zone"             address is inside at least one active zone
     - "not_in_zone"         address geocoded fine but is outside every zone
     - "address_not_found"   no address given, or Census couldn't match it
     - "service_unavailable" a government service didn't respond. Callers
                             should fall back to the ZIP-level check.

   Call check_enterprise_zone_address() ONCE per submission (not once per
   program). It makes network calls, not a cheap local lookup.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "Enterprise_Zones.csv"

_zip_to_zones = None


# ---------------------------------------------------------------------------
# 1. ZIP-code level (original, unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Address level (new)
# ---------------------------------------------------------------------------

GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
GEOCODER_TIMEOUT_SECONDS = 5
IMAP_TIMEOUT_SECONDS = 5

# Two hostnames serve the same MD iMAP layers. Try the first, then the second.
IMAP_HOSTS = [
    "https://mdgeodata.md.gov",
    "https://geodata.md.gov",
]
IMAP_LAYER_PATH = "/imap/rest/services/BusinessEconomy/MD_IncentiveZones/MapServer/{layer}/query"
ENTERPRISE_ZONE_LAYER = 4
FOCUS_AREA_LAYER = 5

_address_cache = {}  # {normalized_address: result_dict}, avoids repeat lookups


def _geocode_to_point(street_address: str):
    """
    Returns (longitude, latitude, matched_address), or None if Census
    couldn't match the address. Raises if the service itself fails.
    """
    response = requests.get(
        GEOCODER_URL,
        params={
            "address": street_address,
            "benchmark": "Public_AR_Current",
            "format": "json",
        },
        timeout=GEOCODER_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    matches = response.json().get("result", {}).get("addressMatches", [])
    if not matches:
        return None
    coords = matches[0].get("coordinates", {})
    lon, lat = coords.get("x"), coords.get("y")
    if lon is None or lat is None:
        return None
    return lon, lat, matches[0].get("matchedAddress", street_address)


def _query_layer(layer: int, lon: float, lat: float, out_fields: str):
    """Returns the zone records containing the point. Raises if all hosts fail."""
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields,
        "returnGeometry": "false",
        "f": "json",
    }
    last_error = None
    for host in IMAP_HOSTS:
        try:
            response = requests.get(
                host + IMAP_LAYER_PATH.format(layer=layer),
                params=params,
                timeout=IMAP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise RuntimeError(f"MD iMAP error: {data['error']}")
            return [f.get("attributes", {}) for f in data.get("features", [])]
        except Exception as exc:
            last_error = exc
            logger.warning("MD iMAP query failed on %s: %s", host, exc)
    raise RuntimeError(f"All MD iMAP hosts failed: {last_error}")


def _format_expiration(value):
    """MD iMAP stores dates as milliseconds since 1970. Returns (YYYY-MM-DD, expired)."""
    if value in (None, ""):
        return None, False
    try:
        dt = datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None, False
    return dt.strftime("%Y-%m-%d"), dt < datetime.now(tz=timezone.utc)


def check_enterprise_zone_address(street_address: str):
    """
    Checks whether a street address is inside a Maryland Enterprise Zone.

    Returns a dict:
        {
          "status": "in_zone" | "not_in_zone" | "address_not_found" | "service_unavailable",
          "match_level": "address",
          "matched_address": str or None,
          "zones": [ {"name", "county", "expiration", "expired"} ],
          "focus_areas": [ {"name", "county"} ],
        }
    """
    result = {
        "status": "service_unavailable",
        "match_level": "address",
        "matched_address": None,
        "zones": [],
        "focus_areas": [],
    }

    if not street_address or not street_address.strip():
        result["status"] = "address_not_found"
        return result

    key = street_address.strip().lower()
    if key in _address_cache:
        return _address_cache[key]

    try:
        point = _geocode_to_point(street_address.strip())
    except Exception as exc:
        logger.warning("Census Geocoder failed: %s", exc)
        return result  # not cached, so a later retry can succeed

    if point is None:
        result["status"] = "address_not_found"
        _address_cache[key] = result
        return result

    lon, lat, matched = point
    result["matched_address"] = matched

    try:
        zone_rows = _query_layer(ENTERPRISE_ZONE_LAYER, lon, lat, "sitename,county,Expiration")
    except Exception as exc:
        logger.warning("Enterprise Zone lookup failed: %s", exc)
        return result  # not cached, so a later retry can succeed

    for row in zone_rows:
        expiration, expired = _format_expiration(row.get("Expiration"))
        result["zones"].append({
            "name": (row.get("sitename") or "").strip(),
            "county": row.get("county"),
            "expiration": expiration,
            "expired": expired,
        })

    # Focus Areas are a bonus layer. If it fails, keep the main zone result.
    try:
        for row in _query_layer(FOCUS_AREA_LAYER, lon, lat, "sitename,county"):
            result["focus_areas"].append({
                "name": (row.get("sitename") or "").strip(),
                "county": row.get("county"),
            })
    except Exception as exc:
        logger.warning("Focus Area lookup failed: %s", exc)

    active = [z for z in result["zones"] if not z["expired"]]
    result["status"] = "in_zone" if active else "not_in_zone"
    _address_cache[key] = result
    return result
