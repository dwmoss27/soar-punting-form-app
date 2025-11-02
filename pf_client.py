import os
import requests

# Try Streamlit secrets; fall back to env vars
try:
    import streamlit as st
    SECRETS = st.secrets
except Exception:
    SECRETS = {}

def _get(key, default=None):
    if key in SECRETS:
        return SECRETS[key]
    return os.environ.get(key, default)

PF_BASE_URL    = _get("PF_BASE_URL", "").rstrip("/")
PF_API_KEY     = _get("PF_API_KEY", "")
PF_AUTH_HEADER = _get("PF_AUTH_HEADER", "x-api-key")   # or "Authorization"
PF_AUTH_PREFIX = _get("PF_AUTH_PREFIX", "")            # "", "Bearer ", "Token ", etc.

# Optional: allow overriding exact paths via secrets (most flexible)
PF_PATH_SEARCH      = _get("PF_PATH_SEARCH",      "")  # e.g. "/search/horses"
PF_PATH_FORM        = _get("PF_PATH_FORM",        "")  # e.g. "/horses/{horse_id}/form"
PF_PATH_RATINGS     = _get("PF_PATH_RATINGS",     "")
PF_PATH_SPEEDMAP    = _get("PF_PATH_SPEEDMAP",    "")
PF_PATH_SECTIONALS  = _get("PF_PATH_SECTIONALS",  "")  # e.g. "/meetings/{meeting_id}/sectionals.csv"
PF_PATH_BENCHMARKS  = _get("PF_PATH_BENCHMARKS",  "")

def _headers():
    if not PF_API_KEY:
        return {"Accept": "application/json"}
    token = f"{PF_AUTH_PREFIX}{PF_API_KEY}" if PF_AUTH_PREFIX else PF_API_KEY
    return {PF_AUTH_HEADER: token, "Accept": "application/json"}

def _url(path: str) -> str:
    if not PF_BASE_URL:
        raise RuntimeError("PF_BASE_URL not set.")
    if not path.startswith("/"):
        path = "/" + path
    return PF_BASE_URL + path

def is_live() -> bool:
    return bool(PF_BASE_URL and PF_API_KEY)

def _raise_for_auth(r: requests.Response):
    if r.status_code == 401:
        raise RuntimeError("Unauthorized (401): Check PF_API_KEY or auth header/prefix.")
    if r.status_code == 403:
        raise RuntimeError("Forbidden (403): Key valid, but your plan lacks this endpoint.")

# -------------------------------
# Endpoint autodiscovery helpers
# -------------------------------
def _first_ok_get(candidates: list[tuple[str, dict]]) -> requests.Response | None:
    """
    Try GET against multiple (path, params) pairs, return first 2xx response.
    """
    for path, params in candidates:
        try:
            r = requests.get(_url(path), headers=_headers(), params=params, timeout=25)
            if r.status_code == 200:
                return r
            # Auth errors should surface immediately
            _raise_for_auth(r)
            # Some APIs 204 on empty
            if r.status_code in (204,):
                return r
        except requests.RequestException:
            pass
    return None

def _extract_horse_hit(payload, typed_name: str) -> dict:
    """
    Normalize PF search response into a dict with at least:
    - display_name
    - horse_id (if discoverable)
    - found (bool)
    """
    if isinstance(payload, list) and payload:
        hit = payload[0]
    elif isinstance(payload, dict) and payload:
        hit = payload
    else:
        return {"display_name": typed_name, "found": False}

    hit.setdefault("display_name", hit.get("name", typed_name))
    # Map id field variants to horse_id
    if "horse_id" not in hit:
        for k in ("id", "horseId", "HorseId", "horse_id"):
            if k in hit:
                hit["horse_id"] = hit[k]
                break
    hit.setdefault("found", True)
    return hit

# -------------------------------
# Public API
# -------------------------------
def search_horse_by_name(name: str) -> dict:
    """
    Tries, in order:
    1) A user-provided PF_PATH_SEARCH (via secrets), using ?q= and ?name=
    2) A small set of common search endpoints
    """
    if not PF_BASE_URL:
        raise RuntimeError("PF_BASE_URL not set.")

    # 1) Secrets override
    if PF_PATH_SEARCH:
        r = _first_ok_get([
            (PF_PATH_SEARCH, {"q": name}),
            (PF_PATH_SEARCH, {"name": name}),
            (PF_PATH_SEARCH, {"query": name}),
        ])
        if r is not None:
            _raise_for_auth(r)
            r.raise_for_status()
            return _extract_horse_hit(_safe_json(r), name)

    # 2) Common guesses (adjust/add if needed once you know the real one)
    guesses = [
        ("/horses/search", {"q": name}),
        ("/horses/search", {"name": name}),
        ("/search/horses", {"q": name}),
        ("/search/horses", {"name": name}),
        ("/horses", {"q": name}),
        ("/horses", {"name": name}),
        ("/search", {"q": name}),
    ]
    r = _first_ok_get(guesses)
    if r is None:
        # bubble the last tried path for debugging
        raise RuntimeError(f"Search endpoint not found (404). Set PF_PATH_SEARCH in secrets to the correct path.")
    _raise_for_auth(r)
    r.raise_for_status()
    return _extract_horse_hit(_safe_json(r), name)

def get_form(horse_id: str | int) -> dict:
    path = PF_PATH_FORM or "/horses/{horse_id}/form"
    url = _url(path.format(horse_id=horse_id))
    r = requests.get(url, headers=_headers(), timeout=30)
    _raise_for_auth(r)
    r.raise_for_status()
    return _safe_json(r)

def get_ratings(horse_id: str | int) -> dict:
    path = PF_PATH_RATINGS or "/horses/{horse_id}/ratings"
    url = _url(path.format(horse_id=horse_id))
    r = requests.get(url, headers=_headers(), timeout=30)
    _raise_for_auth(r)
    r.raise_for_status()
    return _safe_json(r)

def get_speedmap(horse_id: str | int) -> dict:
    path = PF_PATH_SPEEDMAP or "/horses/{horse_id}/speedmap"
    url = _url(path.format(horse_id=horse_id))
    r = requests.get(url, headers=_headers(), timeout=30)
    _raise_for_auth(r)
    r.raise_for_status()
    return _safe_json(r)

def get_sectionals_csv(meeting_id: str | int) -> str:
    path = PF_PATH_SECTIONALS or "/meetings/{meeting_id}/sectionals.csv"
    url = _url(path.format(meeting_id=meeting_id))
    r = requests.get(url, headers=_headers(), timeout=30)
    _raise_for_auth(r)
    r.raise_for_status()
    return r.text

def get_benchmarks_csv(meeting_id: str | int) -> str:
    path = PF_PATH_BENCHMARKS or "/meetings/{meeting_id}/benchmarks.csv"
    url = _url(path.format(meeting_id=meeting_id))
    r = requests.get(url, headers=_headers(), timeout=30)
    _raise_for_auth(r)
    r.raise_for_status()
    return r.text

def _safe_json(r: requests.Response):
    try:
        return r.json()
    except Exception:
        return {}
