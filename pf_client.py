# pf_client.py
# Minimal live-ready PF client:
# - Works in Demo mode if no PF_API_KEY is set
# - Reads endpoint config if present; falls back to defaults if not

import os, io, json
import requests
import pandas as pd

PF_API_KEY = os.getenv("PF_API_KEY", "").strip()
PF_BASE_URL = os.getenv("PF_BASE_URL", "https://api.puntingform.com.au/v2").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

# Load endpoint config if available
_DEFAULT_ENDPOINTS = {
    "search": {"path": "/form/fields/csv"},
    "form": {"path": "/form/form"},
    "ratings": {"path": "/Ratings/MeetingRatings"},
    "speedmaps": {"path": "/User/Speedmaps"},
    "sectionals_csv": {"path": "/Ratings/MeetingSectionals/csv"},
    "benchmarks_csv": {"path": "/Ratings/MeetingBenchmarks/csv"},
}
ENDPOINTS = _DEFAULT_ENDPOINTS
try:
    with open("config/pf_endpoints.json", "r") as f:
        ENDPOINTS = json.load(f)
except Exception:
    ENDPOINTS = _DEFAULT_ENDPOINTS

HEAD = {"Authorization": f"Bearer {PF_API_KEY}"} if PF_API_KEY else {}

def is_live() -> bool:
    return bool(PF_API_KEY)

def _url(key: str) -> str:
    path = ENDPOINTS[key]["path"]
    return f"{PF_BASE_URL}{path}"

def _get_json(url: str, params: dict=None):
    r = requests.get(url, params=params or {}, headers=HEAD, timeout=REQUEST_TIMEOUT)
    if r.status_code == 401:
        raise RuntimeError("Unauthorized (401): Check PF_API_KEY or plan permissions.")
    r.raise_for_status()
    return r.json()

def _get_text(url: str, params: dict=None):
    r = requests.get(url, params=params or {}, headers=HEAD, timeout=REQUEST_TIMEOUT)
    if r.status_code == 401:
        raise RuntimeError("Unauthorized (401): Check PF_API_KEY or plan permissions.")
    r.raise_for_status()
    return r.text

# ---- Public functions used by app.py ----

def search_horse_by_name(name: str, yob: int=None, sire: str=None) -> dict:
    """Placeholder search; return minimal identity so app runs in Demo mode."""
    return {"found": True, "display_name": name, "horse_id": None, "yob": yob, "sire": sire}

def get_form(meeting_id: str=None, race_id: str=None) -> dict:
    url = _url("form")
    params = {}
    if meeting_id: params["meetingId"] = meeting_id
    if race_id: params["raceId"] = race_id
    return _get_json(url, params)

def get_ratings(meeting_id: str) -> dict:
    url = _url("ratings")
    return _get_json(url, {"meetingId": meeting_id})

def get_speedmap(race_id: str) -> dict:
    url = _url("speedmaps")
    return _get_json(url, {"raceId": race_id})

def get_sectionals_csv(meeting_id: str) -> pd.DataFrame:
    url = _url("sectionals_csv")
    text = _get_text(url, {"meetingId": meeting_id})
    return pd.read_csv(io.StringIO(text))

def get_benchmarks_csv(meeting_id: str) -> pd.DataFrame:
    url = _url("benchmarks_csv")
    text = _get_text(url, {"meetingId": meeting_id})
    return pd.read_csv(io.StringIO(text))
