# pf_client.py
# Soar Bloodstock - Punting Form API Client v2
# Handles ratings, sectionals, benchmarks, results, strike rate, and horse form data.
# Works with token-based authentication or apiKey query parameter.
# Includes graceful fallback for missing or invalid secrets.

import requests
import streamlit as st

# -------------------------------
# Configuration from Streamlit Secrets
# -------------------------------

# Base URL for API
PF_BASE_URL = st.secrets.get("PF_BASE_URL", "https://api.puntingform.com.au/v2")
PF_API_KEY = st.secrets.get("PF_API_KEY", "")

# Optional override paths (helpful if your plan uses different endpoints)
PF_PATH_SEARCH       = st.secrets.get("PF_PATH_SEARCH", "/Horses")
PF_PATH_FORM         = st.secrets.get("PF_PATH_FORM", "/form")
PF_PATH_RATINGS      = st.secrets.get("PF_PATH_RATINGS", "/Ratings/MeetingRatings")
PF_PATH_SECTIONALS   = st.secrets.get("PF_PATH_SECTIONALS", "/Ratings/MeetingSectionals")
PF_PATH_BENCHMARKS   = st.secrets.get("PF_PATH_BENCHMARKS", "/Ratings/MeetingBenchmarks")
PF_PATH_RESULTS      = st.secrets.get("PF_PATH_RESULTS", "/form/results")
PF_PATH_STRIKERATE   = st.secrets.get("PF_PATH_STRIKERATE", "/form/strikerate")
PF_PATH_EXPORT       = st.secrets.get("PF_PATH_EXPORT", "/Ratings/SouthCoastExport")

# -------------------------------
# Utility Functions
# -------------------------------

def _full_url(path: str) -> str:
    """Ensure correct API URL formation."""
    if not path.startswith("/"):
        path = "/" + path
    return PF_BASE_URL.rstrip("/") + path

def _headers() -> dict:
    """Standard headers with optional Authorization."""
    headers = {"accept": "application/json"}
    if PF_API_KEY:
        headers["Authorization"] = f"Bearer {PF_API_KEY}"
    return headers

def _params(extra: dict = None) -> dict:
    """Add API key to params automatically."""
    params = {"apiKey": PF_API_KEY} if PF_API_KEY else {}
    if extra:
        params.update(extra)
    return params

def _safe_request(path: str, extra_params: dict = None, timeout: int = 20):
    """Make a request with proper handling and user-friendly messages."""
    url = _full_url(path)
    try:
        r = requests.get(url, headers=_headers(), params=_params(extra_params), timeout=timeout)
        if r.status_code == 401:
            st.error("❌ Unauthorized (401): Check your PF_API_KEY in Streamlit Secrets.")
            return {"error": "Unauthorized (401)"}
        if r.status_code == 404:
            st.error(f"❌ Endpoint not found (404): {url}")
            return {"error": "Not Found (404)", "url": url}
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("⚠️ Request timed out. Try again later.")
        return {"error": "Timeout"}
    except Exception as e:
        st.error(f"⚠️ Could not retrieve data: {e}")
        return {"error": str(e)}

def is_live() -> bool:
    """Check if API key is set."""
    return bool(PF_API_KEY and PF_API_KEY.strip())

# -------------------------------
# API Endpoints
# -------------------------------

def search_horse_by_name(name: str):
    """Search for a horse by name."""
    if not is_live():
        return {"error": "Demo mode: no API key set", "display_name": name}
    return _safe_request(PF_PATH_SEARCH, {"query": name})

def get_form(horse_id: int):
    """Get horse form by horseId."""
    return _safe_request(f"{PF_PATH_FORM}/{horse_id}")

def get_ratings(meeting_id: int):
    """Get Meeting Ratings."""
    return _safe_request(PF_PATH_RATINGS, {"meetingId": meeting_id})

def get_meeting_sectionals(meeting_id: int):
    """Get Meeting Sectionals."""
    return _safe_request(PF_PATH_SECTIONALS, {"meetingId": meeting_id})

def get_meeting_benchmarks(meeting_id: int):
    """Get Meeting Benchmarks."""
    return _safe_request(PF_PATH_BENCHMARKS, {"meetingId": meeting_id})

def get_results():
    """Get Race Results."""
    return _safe_request(PF_PATH_RESULTS)

def get_strike_rate():
    """Get Strike Rate Data."""
    return _safe_request(PF_PATH_STRIKERATE)

def get_southcoast_export(meeting_id: int):
    """Get South Coast Export Data."""
    return _safe_request(PF_PATH_EXPORT, {"meetingId": meeting_id})

# -------------------------------
# Diagnostics
# -------------------------------

def test_connection():
    """Quick test to check connectivity and endpoint validity."""
    st.info("🔍 Testing Punting Form API connection...")
    result = _safe_request(PF_PATH_BENCHMARKS, {"meetingId": 1})
    if "error" in result:
        st.error("❌ Connection test failed.")
    else:
        st.success("✅ Connection test successful.")
    return result

# -------------------------------
# Demo Helpers (used in fallback mode)
# -------------------------------

def demo_stub():
    """Basic stub message when pf_client runs without key."""
    st.info("🔸 Punting Form API not configured — running in Demo Mode.")
    st.write("Set your PF_API_KEY and PF_BASE_URL in Streamlit → Settings → Secrets to enable live data.")
    st.write("Example endpoints:")
    st.code("""
PF_BASE_URL = "https://api.puntingform.com.au/v2"
PF_API_KEY = "YOUR_REAL_KEY"
PF_PATH_SEARCH = "/Horses"
PF_PATH_FORM = "/form"
PF_PATH_RATINGS = "/Ratings/MeetingRatings"
PF_PATH_SECTIONALS = "/Ratings/MeetingSectionals"
PF_PATH_BENCHMARKS = "/Ratings/MeetingBenchmarks"
PF_PATH_RESULTS = "/form/results"
PF_PATH_STRIKERATE = "/form/strikerate"
PF_PATH_EXPORT = "/Ratings/SouthCoastExport"
""")
