
import os, io, json, requests, pandas as pd
from dotenv import load_dotenv

load_dotenv()

PF_API_KEY = os.getenv("PF_API_KEY", "").strip()
PF_BASE_URL = os.getenv("PF_BASE_URL", "https://api.puntingform.com.au/v2").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

with open("config/pf_endpoints.json", "r") as f:
    ENDPOINTS = json.load(f)

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

def search_horse_by_name(name: str, yob: int=None, sire: str=None) -> dict:
    # Placeholder: point to a real horse lookup if available in your PF plan.
    # Return a dict with enough identity to fetch deeper data later.
    # You can also return meetingId/raceId if you resolve via Fields/Form first.
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
