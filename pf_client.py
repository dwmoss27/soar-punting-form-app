# pf_client.py — Soar Bloodstock Punting Form API client (v2)
# Supports: Horses search (configurable), Form, MeetingRatings, MeetingSectionals,
# MeetingBenchmarks, Results, StrikeRate, SouthCoast Export.

import requests
import streamlit as st

# -------------------------------
# Config from Streamlit secrets
# -------------------------------
PF_BASE_URL = st.secrets.get("PF_BASE_URL", "https://api.puntingform.com.au/v2")
PF_API_KEY = st.secrets.get("PF_API_KEY", "")

# Paths are configurable because different plans expose different routes
PF_PATH_SEARCH       = st.secrets.get("PF_PATH_SEARCH", "/Horses")  # adjust if your plan uses another path
PF_PATH_FORM         = st.secrets.get("PF_PATH_FORM", "/form")
PF_PATH_RATINGS_ROOT = st.secrets.get("PF_PATH_RATINGS", "/Ratings")
PF_PATH_SECTIONALS   = st.secrets.get("PF_PATH_SECTIONALS", "/Ratings/MeetingSectionals")
PF_PATH_BENCHMARKS   = st.secrets.get("PF_PATH_BENCHMARKS", "/Ratings/MeetingBenchmarks")
PF_PATH_RESULTS      = st.secrets.get("PF_PATH_RESULTS", "/form/results")
PF_PATH_STRIKERATE   = st.secrets.get("PF_PATH_STRIKERATE", "/form/strikerate")
PF_PATH_EXPORT       = st.secrets.get("PF_PATH_EXPORT", "/Ratings/SouthCoastExport")

def _full_url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return PF_BASE_URL.rstrip("/") + path

def _headers() -> dict:
    headers = {"accept": "application/json"}
    # Two common auth patterns are supported:
    # 1) API key as query param (apiKey=...)
    # 2) Bearer token header (Authorization: Bearer ...)
    # We'll always pass apiKey in params; add Bearer if supplied.
    if PF_API_KEY:
        headers["Authorization"] = f"Bearer {PF_API_KEY}"
    return headers

def is_live() -> bool:
    return bool(PF_API_KEY and PF_API_KEY.strip())

# -------------------------------
# Endpoints
# -------------------------------
def search_horse_by_name(name: str):
    """
    Search for a horse by name.
    Many PF deployments expose /Horses?query=NAME&apiKey=... or similar.
    If your plan differs, set PF_PATH_SEARCH in secrets to the correct path.
    """
    url = _full_url(PF_PATH_SEARCH)
    params = {"query": name, "apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=15)
    if r.status_code == 404:
        raise ValueError("Search endpoint not found (404). Set PF_PATH_SEARCH in Secrets to your correct path.")
    r.raise_for_status()
    return r.json()

def get_form(horse_id: int):
    """
    Get horse form by horseId: /form/{horseId}?apiKey=...
    """
    url = _full_url(f"{PF_PATH_FORM}/{horse_id}")
    params = {"apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_ratings(meeting_id: int):
    """
    MeetingRatings: /Ratings/MeetingRatings?meetingId=...&apiKey=...
    """
    url = _full_url(PF_PATH_RATINGS_ROOT + "/MeetingRatings")
    params = {"meetingId": meeting_id, "apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_meeting_sectionals(meeting_id: int):
    """
    MeetingSectionals: /Ratings/MeetingSectionals?meetingId=...&apiKey=...
    """
    url = _full_url(PF_PATH_SECTIONALS)
    params = {"meetingId": meeting_id, "apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_meeting_benchmarks(meeting_id: int):
    """
    MeetingBenchmarks: /Ratings/MeetingBenchmarks?meetingId=...&apiKey=...
    """
    url = _full_url(PF_PATH_BENCHMARKS)
    params = {"meetingId": meeting_id, "apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_results():
    """
    Results feed: /form/results?apiKey=...
    """
    url = _full_url(PF_PATH_RESULTS)
    params = {"apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_strike_rate():
    """
    Strike rate feed: /form/strikerate?apiKey=...
    """
    url = _full_url(PF_PATH_STRIKERATE)
    params = {"apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_southcoast_export(meeting_id: int):
    """
    SouthCoast export (modeller/commercial): /Ratings/SouthCoastExport?meetingId=...&apiKey=...
    """
    url = _full_url(PF_PATH_EXPORT)
    params = {"meetingId": meeting_id, "apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# Optional quick diagnostic
def test_connection():
    try:
        url = _full_url(PF_PATH_BENCHMARKS)
        params = {"meetingId": 1, "apiKey": PF_API_KEY}
        r = requests.get(url, headers=_headers(), params=params, timeout=10)
        return {"ok": r.ok, "status": r.status_code, "url": url}
    except Exception as e:
        return {"ok": False, "error": str(e)}
