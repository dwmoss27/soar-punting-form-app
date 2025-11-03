# pf_client.py — Soar Bloodstock full Punting Form API client (v2)
# Supports: Form, Ratings, Benchmarks, Sectionals, Results, StrikeRate, SouthCoast Export

import os
import requests
import streamlit as st

# -------------------------------------------------------------
# Configuration from secrets
# -------------------------------------------------------------
PF_BASE_URL = st.secrets.get("PF_BASE_URL", "https://api.puntingform.com.au/v2")
PF_API_KEY = st.secrets.get("PF_API_KEY", None)

# Fallbacks for path configs
PF_PATH_SEARCH = st.secrets.get("PF_PATH_SEARCH", "/Horses")
PF_PATH_FORM = st.secrets.get("PF_PATH_FORM", "/form")
PF_PATH_RATINGS = st.secrets.get("PF_PATH_RATINGS", "/Ratings")
PF_PATH_SECTIONALS = st.secrets.get("PF_PATH_SECTIONALS", "/Ratings/MeetingSectionals")
PF_PATH_BENCHMARKS = st.secrets.get("PF_PATH_BENCHMARKS", "/Ratings/MeetingBenchmarks")
PF_PATH_RESULTS = st.secrets.get("PF_PATH_RESULTS", "/form/results")
PF_PATH_STRIKERATE = st.secrets.get("PF_PATH_STRIKERATE", "/form/strikerate")
PF_PATH_EXPORT = st.secrets.get("PF_PATH_EXPORT", "/Ratings/SouthCoastExport")

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _full_url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return PF_BASE_URL.rstrip("/") + path

def _headers() -> dict:
    headers = {"accept": "application/json"}
    if PF_API_KEY:
        headers["Authorization"] = f"Bearer {PF_API_KEY}"
    return headers

def is_live() -> bool:
    return bool(PF_API_KEY and PF_API_KEY.strip())

# -------------------------------------------------------------
# Core API functions
# -------------------------------------------------------------
def search_horse_by_name(name: str):
    """Search for horse by name (adjust path per PF subscription)."""
    url = _full_url(PF_PATH_SEARCH)
    params = {"query": name, "apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=15)
    if r.status_code == 404:
        raise ValueError("❌ Search endpoint not found (404). Check PF_PATH_SEARCH in secrets.")
    r.raise_for_status()
    return r.json()

def get_form(horse_id: int):
    url = _full_url(f"{PF_PATH_FORM}/{horse_id}")
    params = {"apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_ratings(meeting_id: int):
    """Returns Meeting Ratings (ratings by race/meeting)."""
    url = _full_url(PF_PATH_RATINGS + "/MeetingRatings")
    params = {"meetingId": meeting_id, "apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_meeting_sectionals(meeting_id: int):
    """Returns sectionals for a given meeting."""
    url = _full_url(PF_PATH_SECTIONALS)
    params = {"meetingId": meeting_id, "apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_meeting_benchmarks(meeting_id: int):
    """Returns benchmark data for a given meeting."""
    url = _full_url(PF_PATH_BENCHMARKS)
    params = {"meetingId": meeting_id, "apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_results():
    """Returns all available results."""
    url = _full_url(PF_PATH_RESULTS)
    params = {"apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_strike_rate():
    """Returns strike rate data."""
    url = _full_url(PF_PATH_STRIKERATE)
    params = {"apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_southcoast_export(meeting_id: int):
    """Exports SouthCoast data for modellers."""
    url = _full_url(PF_PATH_EXPORT)
    params = {"meetingId": meeting_id, "apiKey": PF_API_KEY}
    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# -------------------------------------------------------------
# Test connection
# -------------------------------------------------------------
def test_connection():
    try:
        url = _full_url("/Ratings/MeetingBenchmarks")
        params = {"meetingId": 1, "apiKey": PF_API_KEY}
        r = requests.get(url, headers=_headers(), params=params, timeout=10)
        return {"ok": r.ok, "status": r.status_code, "url": url}
    except Exception as e:
        return {"ok": False, "error": str(e)}
