"""
Maps UpSurge's ~50 startup-style industry tags to the NAICS-sector-style
vocabulary used in the Maryland Business Compass dataset. Shared between
normalize_compass.py (the one-time data build) and rules_engine.py (the
live filter), so both use the exact same mapping.
"""

INDUSTRY_MAP = {
    "cybersecurity": ["Information", "Professional, Scientific, and Technical Services"],
    "enterprise technology": ["Information", "Professional, Scientific, and Technical Services"],
    "consumer tech": ["Information", "Retail Trade"],
    "hr tech": ["Information", "Professional, Scientific, and Technical Services"],
    "marcomm": ["Information", "Professional, Scientific, and Technical Services"],
    "consulting": ["Professional, Scientific, and Technical Services"],
    "biotechnology": ["Professional, Scientific, and Technical Services", "Manufacturing"],
    "therapeutics": ["Health Care and Social Assistance", "Manufacturing"],
    "medical devices": ["Manufacturing", "Health Care and Social Assistance"],
    "medical device": ["Manufacturing", "Health Care and Social Assistance"],
    "healthcare services": ["Health Care and Social Assistance"],
    "healthcare it": ["Health Care and Social Assistance", "Information"],
    "digital health": ["Health Care and Social Assistance", "Information"],
    "diagnostics": ["Health Care and Social Assistance", "Manufacturing"],
    "supply chain & logistics": ["Transportation and Warehousing", "Wholesale Trade"],
    "quantum computing": ["Information", "Professional, Scientific, and Technical Services"],
    "creative technology": ["Arts, Entertainment, and Recreation", "Information"],
    "advanced manufacturing": ["Manufacturing"],
    "aerospace and defence": ["Manufacturing"],
    "aerospace and defense": ["Manufacturing"],
    "agtech": ["Agriculture, Forestry, Fishing and Hunting"],
    "apparel": ["Manufacturing", "Retail Trade"],
    "autonomous vehicles": ["Manufacturing", "Transportation and Warehousing"],
    "cannabis": ["Agriculture, Forestry, Fishing and Hunting", "Retail Trade"],
    "climate tech": ["Utilities", "Professional, Scientific, and Technical Services"],
    "construction / engineering": ["Professional, Scientific, and Technical Services"],
    "education / edtech": ["Educational Services"],
    "energy": ["Utilities"],
    "entertainment & recreation": ["Arts, Entertainment, and Recreation"],
    "financial services": ["Finance and Insurance"],
    "fintech": ["Finance and Insurance", "Information"],
    "food & beverage": ["Accommodation and Food Services", "Manufacturing"],
    "gaming": ["Arts, Entertainment, and Recreation", "Information"],
    "govtech": ["Public Administration", "Information"],
    "nanotechnology": ["Manufacturing", "Professional, Scientific, and Technical Services"],
    "personal care": ["Other Services (except Public Administration)", "Retail Trade"],
    "real estate": ["Real Estate and Rental and Leasing"],
    "retail": ["Retail Trade"],
    "robotics": ["Manufacturing", "Professional, Scientific, and Technical Services"],
    "semiconductors": ["Manufacturing"],
    "software development": ["Information", "Professional, Scientific, and Technical Services"],
    "sporting goods / recreation": ["Retail Trade", "Arts, Entertainment, and Recreation"],
    "strategic management services": ["Professional, Scientific, and Technical Services"],
    "telecommunications": ["Information"],
    "travel & tourism": ["Accommodation and Food Services", "Arts, Entertainment, and Recreation"],
    "venture studio": ["Professional, Scientific, and Technical Services"],
    "web3": ["Information", "Finance and Insurance"],
}

# Categories that describe generic small/local consumer businesses rather
# than startups. When a program's eligible industries are ONLY drawn from
# this set (no broader "Any" and no startup-relevant category), it's very
# unlikely to be a genuine match for an UpSurge-style startup and gets
# hard-excluded rather than left to soft AI judgment.
CONSUMER_SERVICE_CATEGORIES = {
    "Retail Trade",
    "Accommodation and Food Services",
    "Personal Care",
    "Other Services (except Public Administration)",
}


def map_industry_to_naics_buckets(industry: str):
    if not industry:
        return []
    key = industry.lower().split(" - ")[0].strip()
    return INDUSTRY_MAP.get(key, [])


def is_consumer_service_only(eligible_industries_raw: str) -> bool:
    """True if a program's eligible industries are ALL generic consumer-service
    categories (retail/food/personal-care/etc) with no broader qualifier."""
    if not eligible_industries_raw or not isinstance(eligible_industries_raw, str):
        return False
    if "any" in eligible_industries_raw.lower():
        return False
    categories = [c.strip() for c in eligible_industries_raw.split(";")]
    return len(categories) > 0 and all(c in CONSUMER_SERVICE_CATEGORIES for c in categories)