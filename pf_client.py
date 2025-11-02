import os
import requests

# Streamlit secrets are optional when running non-Streamlit code
try:
    import streamlit as st
    SECRETS = st.secrets
except Exception:
    SECRETS = {}

def _get(key, default=None):
    # Prefer Streamlit secrets, then env vars, then default
    if key in SECRETS:
        return SECRETS[key]
    return os.environ.get(key, default)

PF_BASE_URL   = _get("PF_BASE_URL",   "").rstrip("/")
PF_API_KEY    = _get("PF_API_KEY",    "")
PF_AUTH_HDR   = _get("PF_AUTH_HEADER","x-api-key")   # "x-api-key" or "Authorization"
PF_AUTH_PREF  = _get("PF_AUTH_PREFIX","")            # "", "Bearer ", "Token ", etc.

def _headers():
    if not PF_API_KEY:
        return {"Accept": "application/json"}
    val = f"{PF_AUTH_PREF}{PF_API_KEY}" if PF_AUTH_PREF else PF_API_KEY
    return {
        PF_AUTH_HDR: val,
        "Accept": "application/json",
    }

def _url(path: str) -> str:
    if not PF_BASE_URL:
        raise RuntimeError("PF_BASE_URL not set.")
    if not path.startswith("/"):
        path = "/" + path
    return PF_BASE_URL + path

def is_live() -> bool:
    # Consider “live” iff we have both base URL and API key present
    return bool(PF_BASE_URL and PF_API_KEY)

# ---------- Helpers ----------
def _raise_for_auth(r: requests.Response):
    if r.status_code == 401:
        raise RuntimeError("Unauthorized (401): Check PF_API_KEY or plan permissions.")
    if r.status_code == 403:
        raise RuntimeError("Forbidden (403): Your key is valid, but you lack access to this endpoint.")

# ---- Minimal wrappers (adjust endpoints to PF docs you have) ----
# Change these paths to the actual PF endpoints you were given.

def search_horse_by_name(name: str) -> dict:
    """
    Replace '/horses/search' with the real PF search endpoint you have.
    If PF uses ?q= or ?name= update params accordingly.
    """
    url = _url("/horses/search")
    r = requests.get(url, headers=_headers(), params={"q": name}, timeout=20)
    _raise_for_auth(r)
    r.raise_for_status()
    data = r.json()
    # Pick first hit or normalize dict
    if isinstance(data, list) and data:
        hit = data[0]
    elif isinstance(data, dict):
        hit = data
    else:
        hit = {"display_name": name, "found": False}
    hit.setdefault("display_name", hit.get("name", name))
    hit.setdefault("found", True)
    # Map id field if PF uses a different key
    if "horse_id" not in hit:
        for k in ("id", "horseId", "HorseId"):
            if k in hit:
                hit["horse_id"] = hit[k]
                break
    return hit

def get_form(horse_id: str | int) -> dict:
    url = _url(f"/horses/{horse_id}/form")
    r = requests.get(url, headers=_headers(), timeout=30)
    _raise_for_auth(r)
    r.raise_for_status()
    return r.json()

def get_ratings(horse_id: str | int) -> dict:
    url = _url(f"/horses/{horse_id}/ratings")
    r = requests.get(url, headers=_headers(), timeout=30)
    _raise_for_auth(r)
    r.raise_for_status()
    return r.json()

def get_speedmap(horse_id: str | int) -> dict:
    url = _url(f"/horses/{horse_id}/speedmap")
    r = requests.get(url, headers=_headers(), timeout=30)
    _raise_for_auth(r)
    r.raise_for_status()
    return r.json()

def get_sectionals_csv(meeting_id: str | int) -> str:
    url = _url(f"/meetings/{meeting_id}/sectionals.csv")
    r = requests.get(url, headers=_headers(), timeout=30)
    _raise_for_auth(r)
    r.raise_for_status()
    return r.text

def get_benchmarks_csv(meeting_id: str | int) -> str:
    url = _url(f"/meetings/{meeting_id}/benchmarks.csv")
    r = requests.get(url, headers=_headers(), timeout=30)
    _raise_for_auth(r)
    r.raise_for_status()
    return r.text
