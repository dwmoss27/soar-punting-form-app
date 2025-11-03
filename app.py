# ---- Soar Bloodstock Data - MoneyBall ----
# Streamlit App with pf_client fallback + horse data filters
# -----------------------------------------------------------

import os
import sys
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date
import re

# ==========================================================
# 1️⃣ SAFE PF CLIENT FALLBACK (no more import errors)
# ==========================================================
PF_IMPORT_ERROR = None
PF_AVAILABLE = False

# Make sure current directory is in path
try:
    HERE = os.path.dirname(__file__)
    if HERE and HERE not in sys.path:
        sys.path.append(HERE)
except Exception:
    pass

try:
    # Try to import the real client
    from pf_client import (
        is_live as _pf_is_live,
        search_horse_by_name as _pf_search_horse_by_name,
        get_form as _pf_get_form,
        get_ratings as _pf_get_ratings,
        get_meeting_sectionals as _pf_get_meeting_sectionals,
        get_meeting_benchmarks as _pf_get_meeting_benchmarks,
        get_results as _pf_get_results,
        get_strike_rate as _pf_get_strike_rate,
        get_southcoast_export as _pf_get_southcoast_export,
    )
    PF_AVAILABLE = True
except Exception as e:
    PF_IMPORT_ERROR = e
    PF_AVAILABLE = False


# --- Safe wrapper functions ---
def is_live():
    return PF_AVAILABLE and _pf_is_live() if PF_AVAILABLE else False

def search_horse_by_name(name: str):
    if PF_AVAILABLE:
        return _pf_search_horse_by_name(name)
    return {"found": False, "display_name": name, "horse_id": None}

def get_form(horse_id: int):
    if PF_AVAILABLE:
        return _pf_get_form(horse_id)
    return {"note": "pf_client not available; demo data only."}

def get_ratings(meeting_id: int):
    if PF_AVAILABLE:
        return _pf_get_ratings(meeting_id)
    return {"note": "pf_client not available; demo data only."}

def get_meeting_sectionals(meeting_id: int):
    if PF_AVAILABLE:
        return _pf_get_meeting_sectionals(meeting_id)
    return {"note": "pf_client not available; demo data only."}

def get_meeting_benchmarks(meeting_id: int):
    if PF_AVAILABLE:
        return _pf_get_meeting_benchmarks(meeting_id)
    return {"note": "pf_client not available; demo data only."}

def get_results():
    if PF_AVAILABLE:
        return _pf_get_results()
    return {"note": "pf_client not available; demo data only."}

def get_strike_rate():
    if PF_AVAILABLE:
        return _pf_get_strike_rate()
    return {"note": "pf_client not available; demo data only."}

def get_southcoast_export(meeting_id: int):
    if PF_AVAILABLE:
        return _pf_get_southcoast_export(meeting_id)
    return {"note": "pf_client not available; demo data only."}

# ==========================================================
# 2️⃣ APP HEADER & CONFIG
# ==========================================================
st.set_page_config(page_title="Soar Bloodstock Data - MoneyBall", layout="wide")
st.title("🏇 Soar Bloodstock Data — MoneyBall")

if not PF_AVAILABLE:
    with st.sidebar:
        st.warning(
            "⚠️ `pf_client` not found — running in Demo Mode.\n"
            "Upload data and explore features. Live API will work once pf_client.py is added."
        )

LIVE = is_live()
st.sidebar.success("✅ Live Mode (Punting Form API)" if LIVE else "🧩 Demo Mode")

# ==========================================================
# 3️⃣ HORSE DATA INPUT
# ==========================================================
st.sidebar.header("🧾 Horse Data Input")

pasted = st.sidebar.text_area(
    "Paste horse names (one per line):",
    height=180,
    placeholder="Little Spark\nSir Goldalot\nEleanor Nancy\n..."
)

uploaded_file = st.sidebar.file_uploader("…or upload CSV/Excel", type=["csv", "xlsx"])

def clean_headers(df: pd.DataFrame):
    clean = {}
    for c in df.columns:
        x = str(c).replace("\ufeff", "")
        x = re.sub(r"\s+", " ", x).strip()
        clean[c] = x
    return df.rename(columns=clean)

def detect_name_col(cols):
    norm = {re.sub(r"\s+", "", c).lower(): c for c in cols}
    for cand in ["name", "horse", "horse name", "lot name"]:
        if cand.replace(" ", "") in norm:
            return norm[cand.replace(" ", "")]
    for c in cols:
        if "name" in c.lower():
            return c
    return None

sale_df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".xlsx"):
            tmp = pd.read_excel(uploaded_file)
        else:
            tmp = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
        sale_df = clean_headers(tmp)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")

if sale_df is None:
    names = [n.strip() for n in pasted.splitlines() if n.strip()]
    sale_df = pd.DataFrame({"Name": names})

name_col = detect_name_col(sale_df.columns)
if not name_col:
    st.error("No valid name column found.")
    st.stop()

# ==========================================================
# 4️⃣ FILTERS
# ==========================================================
st.sidebar.header("🔍 Filters")

age_options = ["Any"] + [str(i) for i in range(1, 11)]
age_selected = st.sidebar.multiselect("Age", age_options, default=["Any"])

sex_options = ["Any", "Gelding", "Mare", "Horse", "Colt", "Filly"]
sex_selected = st.sidebar.multiselect("Sex", sex_options, default=["Any"])

state_options = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT"]
state_selected = st.sidebar.multiselect("State", state_options)

bm_cut = st.sidebar.number_input("Lowest Achieved All Avg Benchmark (≤)", value=5.0, step=0.1)
apply_filters = st.sidebar.button("✅ Apply Filters")

# ==========================================================
# 5️⃣ DISPLAY + FILTER RESULTS
# ==========================================================
st.header("🏇 Sale / Uploaded Horses")

filtered_df = sale_df.copy()
if apply_filters:
    if "Age" in filtered_df.columns and "Any" not in age_selected:
        filtered_df = filtered_df[filtered_df["Age"].astype(str).isin(age_selected)]
    if "Sex" in filtered_df.columns and "Any" not in sex_selected:
        filtered_df = filtered_df[filtered_df["Sex"].isin(sex_selected)]
    if "State" in filtered_df.columns and len(state_selected) > 0:
        filtered_df = filtered_df[filtered_df["State"].isin(state_selected)]

st.dataframe(filtered_df, use_container_width=True)

# ==========================================================
# 6️⃣ SELECT HORSE
# ==========================================================
st.header("🎯 Selected Horse")

horse_name = st.selectbox(
    "Select a horse:",
    sorted(filtered_df[name_col].dropna().astype(str).unique())
)

st.subheader(f"Selected: {horse_name}")

row = filtered_df[filtered_df[name_col].astype(str) == str(horse_name)]
if not row.empty:
    r = row.iloc[0].to_dict()
    for label in ["Lot", "Age", "Sex", "Sire", "Dam", "Vendor", "State"]:
        if label in r and pd.notnull(r[label]):
            st.write(f"**{label}:** {r[label]}")

# ==========================================================
# 7️⃣ VIEW PUNTING FORM DATA
# ==========================================================
st.markdown("---")
st.subheader("📊 Punting Form Data")

if st.button("🔍 View Punting Form Data"):
    with st.spinner(f"Fetching Punting Form data for {horse_name}..."):
        try:
            result = search_horse_by_name(horse_name)
            st.success(f"Found: {result.get('display_name', horse_name)}")
            st.json({
                "Form": get_form(result.get("horse_id")),
                "Ratings": get_ratings(result.get("horse_id")),
                "Sectionals": get_meeting_sectionals(result.get("horse_id")),
                "Benchmarks": get_meeting_benchmarks(result.get("horse_id")),
            })
        except Exception as e:
            st.error(f"Could not retrieve data: {e}")

st.caption("Tip: Add pf_client.py to enable live API access.")
