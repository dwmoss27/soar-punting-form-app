# app.py — Soar Bloodstock Data - MoneyBall (Final Stable Version)
import io
import re
from datetime import date
from typing import Optional

import streamlit as st
import pandas as pd

# ---- Punting Form client (pf_client.py must be in repo) ----
from pf_client import (
    is_live, search_horse_by_name, get_form,
    get_ratings, get_speedmap, get_sectionals_csv, get_benchmarks_csv
)

# ──────────────────────────────────────────────────────────────
# Page config & session keys
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Soar Bloodstock Data - MoneyBall", layout="wide")

SALE_BYTES_KEY     = "sale_uploaded_bytes"
SALE_NAME_KEY      = "sale_uploaded_name"
LOGO_BYTES_KEY     = "logo_bytes"
FILTERS_KEY        = "filters"
HIDE_INPUTS_KEY    = "hide_inputs"
SELECTED_HORSE_KEY = "selected_name"

# Defaults
st.session_state.setdefault(FILTERS_KEY, {
    "age_any": True,
    "age_selected": [],
    "sex_selected": [],
    "maiden_choice": "Any",
    "lowest_bm_max": 5.0,
    "state_selected": []
})
st.session_state.setdefault(HIDE_INPUTS_KEY, False)
LIVE = is_live()

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    clean = {}
    for c in df.columns:
        x = str(c).replace("\ufeff", "")
        x = re.sub(r"\s+", " ", x).strip()
        clean[c] = x
    return df.rename(columns=clean)

def detect_name_col(cols) -> Optional[str]:
    norm = {re.sub(r"\s+", "", c).lower(): c for c in cols}
    for cand in ["name", "horse", "horse name", "horsename", "lot name"]:
        key = re.sub(r"\s+", "", cand).lower()
        if key in norm:
            return norm[key]
    for c in cols:
        if "name" in c.lower():
            return c
    return None

def load_dataframe_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".xlsx"):
        return clean_headers(pd.read_excel(io.BytesIO(file_bytes)))
    bio = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(bio, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
        return clean_headers(df)
    except UnicodeDecodeError:
        bio.seek(0)
        df = pd.read_csv(bio, sep=None, engine="python", encoding="ISO-8859-1", on_bad_lines="skip")
        return clean_headers(df)

def normalize_sale_df(df: pd.DataFrame) -> pd.DataFrame:
    def to_int_age(val):
        if pd.isna(val): return None
        m = re.search(r"(\d+)", str(val))
        return int(m.group(1)) if m else None
    def normalize_sex(val):
        if pd.isna(val): return None
        s = str(val).strip().upper()
        mapping = {
            "G": "Gelding", "M": "Mare", "H": "Horse",
            "C": "Colt", "F": "Filly",
            "GELDING":"Gelding", "MARE":"Mare", "HORSE":"Horse",
            "COLT":"Colt", "FILLY":"Filly"
        }
        return mapping.get(s, s.title())
    out = df.copy()
    if "Age" in out.columns:
        out["_age_int"] = out["Age"].apply(to_int_age)
    if "Sex" in out.columns:
        out["_sex_norm"] = out["Sex"].apply(normalize_sex)
    return out

def show_kv(label, value):
    if value is not None and str(value).strip():
        st.write(f"**{label}:**", value)

# ──────────────────────────────────────────────────────────────
# Title + (optional) centered logo (from Settings tab)
# ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    logo_data = st.session_state.get(LOGO_BYTES_KEY, None)
    if isinstance(logo_data, (bytes, bytearray)) and len(logo_data) > 0:
        st.image(logo_data, use_container_width=False)
    st.title("Soar Bloodstock Data - MoneyBall")

# ──────────────────────────────────────────────────────────────
# Tabs: App  |  Settings
# ──────────────────────────────────────────────────────────────
tab_app, tab_settings = st.tabs(["App", "Settings"])

# =========================
# SETTINGS TAB
# =========================
with tab_settings:
    st.subheader("Page Settings")
    st.caption("Logo and persistent settings for this session.")

    logo_up = st.file_uploader("Upload a logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if logo_up is not None:
        st.write("Preview:")
        try:
            st.image(logo_up.getvalue(), use_container_width=False)
        except Exception:
            st.warning("⚠️ Could not preview this file. Make sure it’s a valid PNG or JPG image.")

    cols = st.columns([1,1,2])
    with cols[0]:
        if st.button("💾 Save logo", use_container_width=True):
            if logo_up and hasattr(logo_up, "getvalue"):
                st.session_state[LOGO_BYTES_KEY] = logo_up.getvalue()
                st.success("✅ Logo saved for this session. It will appear at the top.")
                st.rerun()
            else:
                st.warning("Please select an image file first.")
    with cols[1]:
        if st.button("🗑️ Clear saved logo", use_container_width=True):
            st.session_state.pop(LOGO_BYTES_KEY, None)
            st.success("🧹 Saved logo cleared.")
            st.rerun()

# =========================
# APP TAB
# =========================
with tab_app:
    st.subheader("Data Source")
    st.info("Upload or paste your horse list to begin analysis.")

    mode = st.radio("Choose input method", ["Paste list", "Upload file", "Use saved upload"], horizontal=True)

    pasted_text = ""
    upload_file = None

    if mode == "Paste list":
        pasted_text = st.text_area(
            "Paste horses (one per line):",
            height=140,
            placeholder="Hell Island\nInvincible Phantom\nIrish Bliss\n..."
        )
    elif mode == "Upload file":
        upload_file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
        if upload_file is not None and st.button("💾 Save this upload"):
            st.session_state[SALE_BYTES_KEY] = upload_file.getvalue()
            st.session_state[SALE_NAME_KEY] = upload_file.name
            st.success("Upload saved.")
    else:
        if (SALE_BYTES_KEY not in st.session_state) or (SALE_NAME_KEY not in st.session_state):
            st.info("No saved upload. Use Upload file and Save.")
        else:
            st.success(f"Using saved: {st.session_state[SALE_NAME_KEY]}")

    def build_sale_df():
        if mode == "Upload file" and upload_file is not None:
            return load_dataframe_from_bytes(upload_file.getvalue(), upload_file.name)
        if mode == "Use saved upload" and (SALE_BYTES_KEY in st.session_state):
            return load_dataframe_from_bytes(st.session_state[SALE_BYTES_KEY], st.session_state[SALE_NAME_KEY])
        names = [n.strip() for n in pasted_text.splitlines() if n.strip()]
        return pd.DataFrame({"Name": names})

    sale_df = build_sale_df()
    sale_df = clean_headers(sale_df)
    sale_df = normalize_sale_df(sale_df)
    name_col = detect_name_col(list(sale_df.columns)) or ("Name" if "Name" in sale_df.columns else None)
    if not name_col:
        st.error("No ‘Name’ column found. Provide data correctly.")
        st.stop()

    st.subheader("Filters")
    ages_all = list(range(1, 11))
    age_any = st.checkbox("Any age", value=st.session_state[FILTERS_KEY]["age_any"])
    age_selected = st.multiselect("Ages", ages_all, default=st.session_state[FILTERS_KEY]["age_selected"], disabled=age_any)
    sex_options = ["Gelding", "Mare", "Horse", "Colt", "Filly"]
    sex_selected = st.multiselect("Sex", sex_options, default=st.session_state[FILTERS_KEY]["sex_selected"])
    maiden_choice = st.selectbox("Maiden", ["Any", "Yes", "No"], index=["Any","Yes","No"].index(st.session_state[FILTERS_KEY]["maiden_choice"]))
    lowest_bm_max = st.number_input("Lowest achieved All Avg Benchmark ≤", value=float(st.session_state[FILTERS_KEY]["lowest_bm_max"]), step=0.1)
    state_options = sorted([s for s in sale_df["State"].dropna().astype(str).unique()]) if "State" in sale_df.columns else []
    state_selected = st.multiselect("State", state_options, default=[s for s in st.session_state[FILTERS_KEY]["state_selected"] if s in state_options])

    if st.button("Apply filters"):
        st.session_state[FILTERS_KEY] = {
            "age_any": age_any,
            "age_selected": age_selected,
            "sex_selected": sex_selected,
            "maiden_choice": maiden_choice,
            "lowest_bm_max": lowest_bm_max,
            "state_selected": state_selected,
        }
        st.success("Filters applied.")
        st.rerun()

    fs = st.session_state[FILTERS_KEY]
    st.caption(
        f"Applied: Age={'Any' if fs['age_any'] else fs['age_selected'] or '—'} | "
        f"Sex={fs['sex_selected'] or '—'} | Maiden={fs['maiden_choice']} | "
        f"Lowest BM≤{fs['lowest_bm_max']} | State={fs['state_selected'] or '—'}"
    )

    st.subheader("Filtered sale horses")
    filtered_df = sale_df
    if not filtered_df.empty:
        st.dataframe(filtered_df, use_container_width=True)
        st.download_button(
            "⬇️ Export shortlist (CSV)",
            data=filtered_df.to_csv(index=False),
            file_name="shortlist.csv",
            mime="text/csv",
        )

    all_names = sorted(sale_df[name_col].dropna().astype(str).unique())
    selected_name = st.selectbox("Selected horse", options=["—"] + all_names)
    if selected_name and selected_name != "—":
        st.write(f"### Selected Horse: {selected_name}")
        if st.button("🔎 View Punting Form Data"):
            with st.spinner(f"Searching Punting Form for “{selected_name}”…"):
                try:
                    ident = search_horse_by_name(selected_name)
                    if not ident:
                        st.error("No result found. Check name or PF config.")
                    else:
                        st.success(f"Found: {ident.get('display_name', selected_name)}")
                        horse_id = ident.get("horse_id") or ident.get("id")
                        form = get_form(horse_id)
                        ratings = get_ratings(horse_id)
                        speedmap = get_speedmap(horse_id)
                        tabs = st.tabs(["Form", "Ratings", "Speedmap"])
                        with tabs[0]: st.json(form)
                        with tabs[1]: st.json(ratings)
                        with tabs[2]: st.json(speedmap)
                except Exception as e:
                    st.error(f"Error retrieving data: {e}")

# ──────────────────────────────────────────────────────────────
# PF Diagnostics (sidebar)
# ──────────────────────────────────────────────────────────────
with st.sidebar.expander("🔧 PF Diagnostics"):
    try:
        from pf_client import PF_BASE_URL, PF_PATH_SEARCH, PF_API_KEY
        st.write("BASE_URL:", PF_BASE_URL)
        st.write("PATH_SEARCH:", PF_PATH_SEARCH)
        st.write("API KEY set?:", bool(PF_API_KEY))
    except Exception:
        st.write("pf_client.py not imported correctly.")

    test_name = st.text_input("Quick PF search:", "Little Spark")
    if st.button("Run search"):
        try:
            res = search_horse_by_name(test_name)
            st.success(res if res else "No result")
        except Exception as e:
            st.error(str(e))

st.sidebar.success("Live Mode (PF API)" if LIVE else "Demo Mode (no API key)")
