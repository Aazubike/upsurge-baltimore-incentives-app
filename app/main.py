import threading
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import pandas as pd
from app.data_loader import (
    load_all, get_incentives, search_companies, get_company_by_name,
    suggest_stage_from_rounds, get_industry_options, parse_employee_count,
)
from app.rules_engine import filter_eligible, opportunity_zone_could_apply
from app.gemini_matcher import rank_shortlist
from app.opportunity_zones import check_opportunity_zone
from app.sheets_logger import log_submission, update_feedback

app = FastAPI(title="Baltimore Incentives Matching Tool")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

COUNTIES = ["Baltimore City", "Baltimore County", "Anne Arundel", "Harford", "Howard", "Carroll", "Cecil"]
STAGES = ["pre-seed", "seed", "early", "growth", "established"]

# In-memory job store for the background matching pipeline. Keyed by job_id.
# Each entry: {"status": "running"|"done"|"error", "message": str,
#              "completed": int, "total": int, "context": dict|None}
_jobs = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


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


@app.get("/privacy")
def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})


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


@app.get("/match/results")
def match_results_get_redirect():
    return RedirectResponse("/")


def _run_matching_job(
    job_id: str,
    company_name: str,
    county: str,
    stage: str,
    industry: str,
    cleaned_employee_count,
    cleaned_annual_revenue,
    cleaned_mwbe_groups: list,
    cleaned_zip,
    cleaned_address,
):
    """Runs the full matching pipeline on a background thread, updating the
    job's status/message/progress as it goes so the loading page has
    something real to poll. On success, the finished template context is
    stashed on the job for /match/results/{job_id} to render."""
    try:
        precheck_answers = {
            "county": county,
            "stage": stage,
            "employee_count": cleaned_employee_count,
            "industry": industry,
            "mwbe_groups": cleaned_mwbe_groups,
        }
        if cleaned_address and opportunity_zone_could_apply(precheck_answers):
            _update_job(job_id, message="Checking opportunity zone eligibility...")
            oz_eligible, oz_tract = check_opportunity_zone(cleaned_address)
        else:
            oz_eligible, oz_tract = False, None

        answers = {
            "county": county,
            "stage": stage,
            "employee_count": cleaned_employee_count,
            "annual_revenue": cleaned_annual_revenue,
            "industry": industry,
            "mwbe_groups": cleaned_mwbe_groups,
            "zip_code": cleaned_zip,
            "street_address": cleaned_address,
            "oz_eligible": oz_eligible,
            "oz_tract": oz_tract,
        }

        _update_job(job_id, message="Filtering eligible programs...")
        shortlist_df = filter_eligible(answers)

        def on_progress(event: str, **kwargs):
            if event == "total":
                total = kwargs["total"]
                _update_job(job_id, total=total, message=f"Scoring programs... 0 of {total} checked")
            elif event == "progress":
                with _jobs_lock:
                    job = _jobs.get(job_id)
                    if job is None:
                        return
                    job["completed"] = min(job["completed"] + kwargs["delta"], job["total"] or job["completed"] + kwargs["delta"])
                    job["message"] = f"Scoring programs... {job['completed']} of {job['total']} checked"

        _update_job(job_id, message="Scoring programs against eligibility rubric...")
        ranked_shortlist, dropped_count, gemini_error = rank_shortlist(
            answers, shortlist_df, progress_callback=on_progress
        )

        _update_job(job_id, message="Finalizing results...")

        is_known_company = get_company_by_name(company_name) is not None
        submission_id = str(uuid4())
        log_submission(
            submission_id=submission_id,
            flow_type="portfolio" if is_known_company else "intake",
            company_name=company_name,
            region=county,
            stage=stage,
            employee_count=cleaned_employee_count,
            annual_revenue=cleaned_annual_revenue,
            industry=industry,
            ownership="|".join(cleaned_mwbe_groups) if cleaned_mwbe_groups else "",
            zip_code=cleaned_zip or "",
            street_address=cleaned_address or "",
            oz_eligible=oz_eligible,
            oz_tract=oz_tract or "",
            matched_programs=[p["Program Name"] for p in ranked_shortlist],
            match_scores=[p.get("fit_score") for p in ranked_shortlist],
        )

        context = {
            "company_name": company_name,
            "shortlist": ranked_shortlist,
            "total_programs": len(get_incentives()),
            "total_eligible": len(shortlist_df),
            "dropped_count": dropped_count,
            "gemini_enabled": gemini_error is None,
            "gemini_error": gemini_error,
            "submission_id": submission_id,
        }
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["context"] = context
    except Exception as e:
        print(f"[match job {job_id}] failed: {e!r}")
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["message"] = "Something went wrong while matching. Please try again."


@app.post("/match/results")
def match_results(
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

    cleaned_address = street_address.strip() if street_address else None
    cleaned_employee_count = to_int_or_none(employee_count)
    cleaned_annual_revenue = to_int_or_none(annual_revenue)
    cleaned_mwbe_groups = [g for g in mwbe_groups if g != "none"]
    cleaned_zip = zip_code.strip() if zip_code else None

    job_id = str(uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "message": "Starting...",
            "completed": 0,
            "total": 0,
            "context": None,
        }

    thread = threading.Thread(
        target=_run_matching_job,
        args=(
            job_id, company_name, county, stage, industry,
            cleaned_employee_count, cleaned_annual_revenue,
            cleaned_mwbe_groups, cleaned_zip, cleaned_address,
        ),
        daemon=True,
    )
    thread.start()

    return RedirectResponse(f"/match/loading/{job_id}", status_code=303)


@app.get("/match/loading/{job_id}")
def match_loading(request: Request, job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return RedirectResponse("/")
    return templates.TemplateResponse("loading.html", {"request": request, "job_id": job_id})


@app.get("/api/match/status/{job_id}")
def match_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return {"status": "not_found"}
    return {
        "status": job["status"],
        "message": job["message"],
        "completed": job["completed"],
        "total": job["total"],
    }


@app.get("/match/results/{job_id}")
def match_results_view(request: Request, job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or job["status"] != "done" or job["context"] is None:
        return RedirectResponse("/")
    context = dict(job["context"])
    context["request"] = request
    return templates.TemplateResponse("results.html", context)


class FeedbackPayload(BaseModel):
    submission_id: str
    thumbs: str = ""
    comment: str = ""


@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload):
    """
    Fire-and-forget: runs the Sheets update on a background thread so the
    button click feels instant instead of waiting on a network round trip.
    """
    def _run():
        try:
            update_feedback(payload.submission_id, thumbs=payload.thumbs, comment=payload.comment)
        except Exception as e:
            print(f"[feedback] failed to log: {e!r}")

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "ok"}
