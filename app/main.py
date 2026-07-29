from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from typing import List, Optional
import pandas as pd
from app.data_loader import (
    load_all, get_incentives, search_companies, get_company_by_name,
    suggest_stage_from_rounds, get_industry_options, parse_employee_count,
)
from app.rules_engine import filter_eligible
from app.gemini_matcher import rank_shortlist
from app.sheets_logger import log_submission, update_feedback
app = FastAPI(title="Baltimore Incentives Matching Tool")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

COUNTIES = ["Baltimore City", "Baltimore County", "Anne Arundel", "Harford", "Howard", "Carroll", "Cecil"]
STAGES = ["pre-seed", "seed", "early", "growth", "established"]


@app.on_event("startup")
def startup():
    load_all()
    print("Data loaded: incentives, known companies, venture rounds.")


@app.get("/ping")
def ping():
    """Lightweight endpoint for an uptime monitor (e.g. UptimeRobot) to keep
    the Render free-tier instance from spinning down after 15 min idle.
    Returns instantly, no template rendering or data access."""
    return {"status": "ok"}


@app.get("/")
def home(request: Request):
    program_count = len(get_incentives())
    return templates.TemplateResponse("home.html", {
        "request": request,
        "program_count": program_count,
        "counties": COUNTIES,
        "stages": STAGES,
        "industries": get_industry_options(),
    })


@app.get("/how-it-works")
def how_it_works(request: Request):
    return templates.TemplateResponse("how_it_works.html", {"request": request})


def _company_prefill_data(company: dict) -> dict:
    """Shared prefill logic used by both the JSON API (for the single-page
    flow) and the old confirm page template."""
    suggested_stage = suggest_stage_from_rounds(company.get("Account ID"))
    default_county = None
    raw_county = company.get("County SoT")
    if raw_county:
        for c in COUNTIES:
            if c.lower() in str(raw_county).lower():
                default_county = c
                break

    default_industry = None
    raw_industry = company.get("Industry SoT")
    if raw_industry:
        default_industry = str(raw_industry).split(" - ")[0].strip()

    raw_address = company.get("Address SoT")
    default_address = raw_address if raw_address and str(raw_address) != "No Value" else None
    raw_zip = company.get("Derived_Zip")
    default_zip = raw_zip if raw_zip and not pd.isna(raw_zip) else None

    return {
        "account_name": company.get("Account Name"),
        "county": default_county,
        "industry": default_industry,
        "employee_count": parse_employee_count(company.get("Number of Employees SoT")),
        "stage": suggested_stage,
        "stage_suggested": suggested_stage is not None,
        "address": default_address,
        "zip_code": default_zip,
    }


@app.get("/api/companies/details")
def api_company_details(account: str):
    company = get_company_by_name(account)
    if company is None:
        return {"found": False}
    return {"found": True, **_company_prefill_data(company)}


@app.get("/match/known")
def match_known(request: Request):
    return templates.TemplateResponse("known_search.html", {"request": request})


@app.get("/api/companies/search")
def api_companies_search(q: str = ""):
    results = search_companies(q, limit=8)
    return [
        {
            "account_name": r.get("Account Name"),
            "county": r.get("County SoT"),
        }
        for r in results
    ]


@app.get("/match/known/confirm")
def match_known_confirm(request: Request, account: str):
    company = get_company_by_name(account)
    if company is None:
        return RedirectResponse("/match/known")

    prefill = _company_prefill_data(company)

    return templates.TemplateResponse("known_confirm.html", {
        "request": request,
        "company": company,
        "counties": COUNTIES,
        "stages": STAGES,
        "industries": get_industry_options(),
        "default_county": prefill["county"],
        "default_industry": prefill["industry"],
        "default_employees": prefill["employee_count"],
        "default_revenue": None,
        "suggested_stage": prefill["stage"],
        "default_address": prefill["address"],
        "default_zip": prefill["zip_code"],
    })


@app.get("/match/new")
def match_new(request: Request, name: str = ""):
    return templates.TemplateResponse("new_intake.html", {
        "request": request,
        "counties": COUNTIES,
        "stages": STAGES,
        "industries": get_industry_options(),
        "prefilled_name": name,
    })


@app.post("/match/results")
def match_results(
    request: Request,
    company_name: str = Form(...),
    county: str = Form(...),
    stage: str = Form(...),
    industry: str = Form(...),
    employee_count: Optional[str] = Form(None),
    annual_revenue: Optional[str] = Form(None),
    mwbe_groups: List[str] = Form([]),
    zip_code: Optional[str] = Form(None),
    street_address: Optional[str] = Form(None),
):
    def to_int_or_none(val):
        if val is None or val.strip() == "":
            return None
        try:
            return int(val)
        except ValueError:
            return None

    answers = {
        "county": county,
        "stage": stage,
        "employee_count": to_int_or_none(employee_count),
        "annual_revenue": to_int_or_none(annual_revenue),
        "industry": industry,
        "mwbe_groups": [g for g in mwbe_groups if g != "none"],
        "zip_code": zip_code.strip() if zip_code else None,  # used for real Enterprise Zone matching (see rules_engine._enterprise_zone_ok)
        "street_address": street_address.strip() if street_address else None,  # captured for the future admin/data-capture system, not used to filter yet
    }
    shortlist_df = filter_eligible(answers)
    ranked_shortlist, dropped_count, gemini_error = rank_shortlist(answers, shortlist_df)

# Log this submission to the Google Sheet -- every program that came
    # back gets logged, both "match" (90%+) and "possible" (75-89%)
    is_known_company = get_company_by_name(company_name) is not None
    submission_id = log_submission(
        flow_type="portfolio" if is_known_company else "intake",
        company_name=company_name,
        region=county,
        stage=stage,
        industry=industry,
        ownership="|".join(answers["mwbe_groups"]) if answers["mwbe_groups"] else "",
        zip_code=answers["zip_code"] or "",
        matched_programs=[p["Program Name"] for p in ranked_shortlist],
        match_tiers=[p["match_tier"] for p in ranked_shortlist],
        match_scores=[p.get("fit_score") for p in ranked_shortlist],
    )

    return templates.TemplateResponse("results.html", {
        "request": request,
        "company_name": company_name,
        "shortlist": ranked_shortlist,
        "total_programs": len(get_incentives()),
        "total_eligible": len(shortlist_df),
        "dropped_count": dropped_count,
        "gemini_enabled": gemini_error is None,
        "gemini_error": gemini_error,
        "submission_id": submission_id,
    })
