from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from typing import List, Optional
from app.data_loader import (
    load_all, get_incentives, search_companies, get_company_by_name,
    suggest_stage_from_rounds, get_industry_options, parse_employee_count,
)
from app.rules_engine import filter_eligible
from app.gemini_matcher import rank_shortlist

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
    return templates.TemplateResponse(
        "home.html", {"request": request, "program_count": program_count}
    )


@app.get("/how-it-works")
def how_it_works(request: Request):
    return templates.TemplateResponse("how_it_works.html", {"request": request})


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

    return templates.TemplateResponse("known_confirm.html", {
        "request": request,
        "company": company,
        "counties": COUNTIES,
        "stages": STAGES,
        "industries": get_industry_options(),
        "default_county": default_county,
        "default_industry": default_industry,
        "default_employees": parse_employee_count(company.get("Number of Employees SoT")),
        "default_revenue": None,
        "suggested_stage": suggested_stage,
    })


@app.get("/match/new")
def match_new(request: Request):
    return templates.TemplateResponse("new_intake.html", {
        "request": request,
        "counties": COUNTIES,
        "stages": STAGES,
        "industries": get_industry_options(),
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
        "zip_code": zip_code.strip() if zip_code else None,  # captured for future Enterprise Zone verification, not yet used to filter
    }
    shortlist_df = filter_eligible(answers)
    ranked_shortlist, dropped_count, gemini_error = rank_shortlist(answers, shortlist_df)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "company_name": company_name,
        "shortlist": ranked_shortlist,
        "total_programs": len(get_incentives()),
        "total_eligible": len(shortlist_df),
        "dropped_count": dropped_count,
        "gemini_enabled": gemini_error is None,
        "gemini_error": gemini_error,
    })
