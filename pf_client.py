# pf_client.py — minimal, stable client for Punting Form (V2)
import os
import requests
import streamlit as st

# ---- Read secrets ----
# In Streamlit Cloud: Settings → Secrets
# Required:
# PF_API_KEY = "..."
# Optional override:
# PF_BASE_URL = "https://api.puntingform.com.au/v2"
PF_API_KEY = st.secrets.get("PF_API_KEY", None)
PF_BASE_URL = st.secrets.get("PF_BASE_URL", "https://api.puntingform.com.au/v2")

if not PF_BASE_URL.endswith("/"):
    PF_BASE_URL = PF_BASE_URL  # keep as-is; we will prepend '/path'

def _auth_headers():
    if not PF_API_KEY:
        raise RuntimeError("PF_API_KEY missing from secrets.")
    return {"accept": "application/json", "Authorization": f"Bearer {PF_API_KEY}"}

def is_live() -> bool:
    # A very light ping — tries a harmless endpoint with a dummy param.
    try:
        r = requests.get(f"{PF_BASE_URL}/Ratings/MeetingBenchmarks", params={"meetingId": 0}, headers=_auth_headers(), timeout=10)
        # 401/403 still proves host reachable & key checked; treat as "reachable"
        return r.status_code in (200, 400, 401, 403, 404)
    except Exception:
        return False

# ---- Raw GET (for tester panel) ----
def pf_raw_get(path: str, params: dict | None = None) -> requests.Response:
    if not path.startswith("/"):
        path = "/" + path
    url = f"{PF_BASE_URL}{path}"
    r = requests.get(url, params=(params or {}), headers=_auth_headers(), timeout=25)
    r.raise_for_status()  # bubble up 4xx/5xx with clear error in UI
    return r

# ---- Common endpoints (adjust if your plan differs) ----
def get_meeting_benchmarks(meeting_id: int) -> dict:
    r = pf_raw_get("/Ratings/MeetingBenchmarks", params={"meetingId": int(meeting_id)})
    return r.json()

def get_meeting_sectionals(meeting_id: int) -> dict:
    r = pf_raw_get("/Ratings/MeetingSectionals", params={"meetingId": int(meeting_id)})
    return r.json()

def get_meeting_ratings(meeting_id: int) -> dict:
    r = pf_raw_get("/Ratings/MeetingRatings", params={"meetingId": int(meeting_id)})
    return r.json()

# ---- Search by horse name (ONLY if your plan exposes a search endpoint) ----
# If you *don't* have a search endpoint, this will throw a clear message.
def search_horse_by_name(name: str) -> dict:
    path = st.secrets.get("PF_PATH_SEARCH", None)  # e.g. "/Horses/Search" or "/Form/Search"
    if not path:
        raise RuntimeError("PF_PATH_SEARCH is not set in secrets for search. Remove calls or set this secret.")
    r = pf_raw_get(path, params={"q": name})
    return r.json()
