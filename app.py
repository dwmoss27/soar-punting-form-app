# =========================
# Soar Bloodstock Data - MoneyBall
# Full App (upload/paste/save list, filters, PF integration w/ fallback)
# =========================

import os, sys, io, base64, json, re, traceback
from datetime import date
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Soar Bloodstock Data - MoneyBall",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------
# SESSION KEYS
# --------------------------
DATA_KEY = "SBM_SALE_DF"                # stores DataFrame (sale horses)
DATA_SOURCE_KEY = "SBM_SALE_SOURCE"     # "uploaded" | "pasted"
SAVED_UPLOAD_BYTES_KEY = "SBM_SAVED_UPLOAD_BYTES"
LOGO_BYTES_KEY = "SBM_LOGO_BYTES"
FILTERS_KEY = "SBM_FILTERS"
SELECTED_HORSE_KEY = "SBM_SELECTED_HORSE"
SHORTLIST_KEY = "SBM_SHORTLIST"
EMBED_PF_PANEL_OPEN_KEY = "SBM_EMBED_OPEN"
PF_AVAILABLE_KEY = "SBM_PF_AVAILABLE"

# Initialize defaults
if FILTERS_KEY not in st.session_state:
    st.session_state[FILTERS_KEY] = {
        "ages": ["Any"],
        "sexes": ["Any"],
        "states": [],
        "maiden": "Any",                   # "Any" | "Yes" | "No"
        "lowest_all_avg_bm_max": None,    # float or None
        "apply_clicked": False,
    }

if SHORTLIST_KEY not in st.session_state:
    st.session_state[SHORTLIST_KEY] = []

if EMBED_PF_PANEL_OPEN_KEY not in st.session_state:
    st.session_state[EMBED_PF_PANEL_OPEN_KEY] = False

# ------------------------------------------------------
# pf_client FALLBACK SHIM (lets UI run even if missing)
# ------------------------------------------------------
PF_IMPORT_ERROR = None
PF_AVAILABLE = False

def _add_here_to_path():
    try:
        here = os.path.dirname(__file__)
        if here and here not in sys.path:
            sys.path.append(here)
    except Exception:
        pass

_add_here_to_path()

try:
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

def is_live() -> bool:
    return PF_AVAILABLE and _pf_is_live() if PF_AVAILABLE else False

def pf_search_horse_by_name(name: str) -> Dict[str, Any]:
    if PF_AVAILABLE:
        return _pf_search_horse_by_name(name)
    return {"found": False, "display_name": name, "horse_id": None}

def pf_get_form(horse_id: Any) -> Dict[str, Any]:
    if PF_AVAILABLE:
        return _pf_get_form(horse_id)
    return {"note": "pf_client not available; demo form."}

def pf_get_ratings(meeting_id: Any) -> Dict[str, Any]:
    if PF_AVAILABLE:
        return _pf_get_ratings(meeting_id)
    return {"note": "pf_client not available; demo ratings."}

def pf_get_meeting_sectionals(meeting_id: Any) -> Dict[str, Any]:
    if PF_AVAILABLE:
        return _pf_get_meeting_sectionals(meeting_id)
    return {"note": "pf_client not available; demo sectionals."}

def pf_get_meeting_benchmarks(meeting_id: Any) -> Dict[str, Any]:
    if PF_AVAILABLE:
        return _pf_get_meeting_benchmarks(meeting_id)
    return {"note": "pf_client not available; demo benchmarks."}

def pf_get_results() -> Dict[str, Any]:
    if PF_AVAILABLE:
        return _pf_get_results()
    return {"note": "pf_client not available; demo results."}

def pf_get_strike_rate() -> Dict[str, Any]:
    if PF_AVAILABLE:
        return _pf_get_strike_rate()
    return {"note": "pf_client not available; demo strikerate."}

def pf_get_southcoast_export(meeting_id: Any) -> Dict[str, Any]:
    if PF_AVAILABLE:
        return _pf_get_southcoast_export(meeting_id)
    return {"note": "pf_client not available; demo southcoast export."}

st.session_state[PF_AVAILABLE_KEY] = PF_AVAILABLE

# Helpful banner
if not PF_AVAILABLE:
    with st.sidebar:
        st.warning(
            "⚠️ `pf_client` not importable — running in **Demo Mode**.\n\n"
            "UI & filters work. PF calls will show demo placeholders.\n\n"
            "To enable live PF:\n"
            "1) Add `pf_client.py` next to `app.py` (committed to GitHub).\n"
            "2) Set secrets `PF_API_KEY`, and if needed `PF_BASE_URL` & paths.\n"
            "3) Confirm endpoints in `pf_client.py`.\n"
        )

# ------------------------------------------------------
# UTILS
# ------------------------------------------------------
def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        x = str(c).replace("\ufeff", "")
        x = re.sub(r"\s+", " ", x).strip()
        mapping[c] = x
    return df.rename(columns=mapping)

def detect_name_col(cols: List[str]) -> Optional[str]:
    norm = {re.sub(r"\s+", "", c).lower(): c for c in cols}
    candidates = ["name", "horse", "horse name", "horsename", "lot name"]
    for cand in candidates:
        key = cand.replace(" ", "").lower()
        if key in norm:
            return norm[key]
    for c in cols:
        if "name" in c.lower():
            return c
    return None

def compute_has_won(row: Dict[str, Any]) -> Optional[bool]:
    # if there is an explicit Wins column
    for k in ("Wins", "wins"):
        if k in row and pd.notnull(row[k]):
            try:
                return int(str(row[k]).split(".")[0]) > 0
            except Exception:
                pass
    return None

def derive_lowest_bm_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    # If no PF integration yet, create a placeholder column if missing
    if "Lowest All Avg Benchmark" not in df.columns:
        # fake metric: if we have "All Avg Benchmark" use that, else NaN
        if "All Avg Benchmark" in df.columns:
            df["Lowest All Avg Benchmark"] = df["All Avg Benchmark"]
        else:
            df["Lowest All Avg Benchmark"] = np.nan
    return df

def save_bytes_to_session(key: str, fbytes: bytes):
    st.session_state[key] = fbytes

def get_logo_html_center(img_bytes: Optional[bytes]) -> str:
    if not img_bytes:
        return ""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return (
        f'<div style="display:flex;justify-content:center;">'
        f'<img src="data:image/png;base64,{b64}" style="max-height:80px;">'
        f"</div>"
    )

def display_logo_top_center():
    html = get_logo_html_center(st.session_state.get(LOGO_BYTES_KEY))
    if html:
        st.markdown(html, unsafe_allow_html=True)

# ------------------------------------------------------
# TOP BAR (Title + Centered Logo)
# ------------------------------------------------------
display_logo_top_center()
st.title("Soar Bloodstock Data — MoneyBall")

# ------------------------------------------------------
# SIDEBAR: PAGE SETTINGS (Logo upload/save), PF settings help
# ------------------------------------------------------
with st.sidebar.expander("⚙️ Page Settings", expanded=False):
    st.write("Upload a logo and press **Save** to persist for this session.")
    logo_file = st.file_uploader("Upload logo (PNG/JPG)", type=["png", "jpg", "jpeg"], key="logo_uploader")
    col_lg1, col_lg2 = st.columns([1,1])
    with col_lg1:
        if st.button("Save logo"):
            if logo_file:
                try:
                    image = Image.open(logo_file).convert("RGBA")
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    save_bytes_to_session(LOGO_BYTES_KEY, buf.getvalue())
                    st.success("Logo saved for this session.")
                except Exception as e:
                    st.error(f"Could not process logo: {e}")
            else:
                st.info("Select a logo file first.")
    with col_lg2:
        if st.button("Clear logo"):
            st.session_state[LOGO_BYTES_KEY] = None
            st.success("Logo cleared.")

with st.sidebar.expander("🔑 Punting Form Settings", expanded=False):
    live_flag = is_live()
    st.write("**Mode:** " + ("Live (PF API reachable)" if live_flag else "Demo"))
    st.caption(
        "To enable live:\n"
        "- Ensure `pf_client.py` exists next to `app.py`\n"
        "- Add Streamlit secrets: `PF_API_KEY`, and optional `PF_BASE_URL` + path overrides.\n"
        "- Fix any 404s by matching the exact API docs for your plan."
    )

# ------------------------------------------------------
# SIDEBAR: HORSE LIST INPUT (hideable)
# ------------------------------------------------------
with st.sidebar.expander("🧾 Horse list", expanded=False):
    st.caption("Pick the source you want to use (you can save it).")
    input_mode = st.radio("Input mode", ["Paste", "Upload CSV/Excel", "Use Saved"], horizontal=False)

    pasted = None
    uploaded_df = None

    if input_mode == "Paste":
        pasted = st.text_area(
            "Paste horses (one per line):",
            height=180,
            placeholder="Hell Island\nInvincible Phantom\nIrish Bliss\n..."
        )
        if st.button("Save pasted list"):
            names = [n.strip() for n in (pasted or "").splitlines() if n.strip()]
            df = pd.DataFrame({"Name": names})
            st.session_state[DATA_KEY] = df
            st.session_state[DATA_SOURCE_KEY] = "pasted"
            st.success("Pasted list saved for this session.")

    elif input_mode == "Upload CSV/Excel":
        sale_file = st.file_uploader("Upload sales file", type=["csv", "xlsx"], key="sale_uploader")
        if sale_file:
            try:
                if sale_file.name.lower().endswith(".xlsx"):
                    tmp = pd.read_excel(sale_file)
                else:
                    try:
                        tmp = pd.read_csv(sale_file, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
                    except UnicodeDecodeError:
                        tmp = pd.read_csv(sale_file, sep=None, engine="python", encoding="ISO-8859-1", on_bad_lines="skip")
                tmp = clean_headers(tmp)
                uploaded_df = tmp.copy()
                st.success(f"Loaded file with {len(uploaded_df)} rows.")
            except Exception as e:
                st.error(f"Could not read uploaded file: {e}")
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Save uploaded file"):
                if uploaded_df is None:
                    st.info("Upload a file first.")
                else:
                    st.session_state[DATA_KEY] = uploaded_df
                    st.session_state[DATA_SOURCE_KEY] = "uploaded"
                    # keep raw bytes for re-use across reruns in cloud
                    try:
                        sale_file.seek(0)
                        st.session_state[SAVED_UPLOAD_BYTES_KEY] = sale_file.read()
                    except Exception:
                        pass
                    st.success("Uploaded data saved for this session.")
        with c2:
            if st.button("Clear saved list"):
                st.session_state[DATA_KEY] = None
                st.session_state[SAVED_UPLOAD_BYTES_KEY] = None
                st.session_state[DATA_SOURCE_KEY] = None
                st.success("Cleared saved list.")

    else:  # Use Saved
        if st.session_state.get(DATA_KEY) is not None:
            st.success("Using previously saved list.")
        else:
            st.info("No saved list yet. Use Paste or Upload, then Save.")

# ------------------------------------------------------
# BUILD sale_df
# ------------------------------------------------------
sale_df = st.session_state.get(DATA_KEY)

# If no saved data and no new upload/paste in this run, still allow paste quick start
if sale_df is None:
    # Minimal quick start: allow manual paste in main (hidden by default)
    st.info("No horses loaded. Open **🧾 Horse list** in the sidebar to paste or upload, then Save.")
    sale_df = pd.DataFrame(columns=["Name"])

# Clean headers & add helpful derived fields
if not sale_df.empty:
    sale_df = clean_headers(sale_df)
    # Standardize typical columns if present
    # (Lot, Age, Sex, Sire, Dam, Vendor, State, Bid, Name)
    if "Name" not in sale_df.columns:
        name_col_guess = detect_name_col(list(sale_df.columns))
        if name_col_guess:
            sale_df = sale_df.rename(columns={name_col_guess: "Name"})
    sale_df = derive_lowest_bm_placeholder(sale_df)

# ------------------------------------------------------
# FILTERS (multi + Apply)
# ------------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar:
    st.header("🔎 Filters")

    # Age multi-select (Any + 1..10)
    age_options = ["Any"] + [str(i) for i in range(1, 11)]
    ages = st.multiselect(
        "Age",
        options=age_options,
        default=st.session_state[FILTERS_KEY]["ages"]
    )

    # Sex multi
    sex_options = ["Any", "Gelding", "Mare", "Horse", "Colt", "Filly"]
    sexes = st.multiselect(
        "Sex",
        options=sex_options,
        default=st.session_state[FILTERS_KEY]["sexes"]
    )

    # State multi (only show if present in data)
    state_opts = []
    if "State" in sale_df.columns:
        state_opts = sorted(
            list({s for s in sale_df["State"].dropna().astype(str).str.strip().tolist() if s})
        )
    states = st.multiselect("State", options=state_opts, default=st.session_state[FILTERS_KEY]["states"])

    maiden = st.selectbox("Maiden", ["Any", "Yes", "No"], index=["Any", "Yes", "No"].index(
        st.session_state[FILTERS_KEY]["maiden"]
    ))

    bm_max = st.number_input("Lowest achieved All Avg Benchmark (max)", value=0.0, step=0.1, help="Filters horses whose *lowest achieved* All Avg Benchmark is ≤ value. (Uses PF data if available, fallback to a placeholder if not.)")
    bm_use = bm_max

    if st.button("✅ Apply Filters"):
        st.session_state[FILTERS_KEY]["ages"] = ages if ages else ["Any"]
        st.session_state[FILTERS_KEY]["sexes"] = sexes if sexes else ["Any"]
        st.session_state[FILTERS_KEY]["states"] = states
        st.session_state[FILTERS_KEY]["maiden"] = maiden
        st.session_state[FILTERS_KEY]["lowest_all_avg_bm_max"] = bm_use
        st.session_state[FILTERS_KEY]["apply_clicked"] = True

# ------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------
def normalize_age_col(df: pd.DataFrame) -> pd.DataFrame:
    if "Age" not in df.columns:
        # Try Year of Birth -> age
        if "yob" in df.columns:
            df["Age"] = date.today().year - pd.to_numeric(df["yob"], errors="coerce")
        else:
            df["Age"] = np.nan
    return df

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    f = df.copy()
    f = normalize_age_col(f)

    # Age filter
    chosen_ages = st.session_state[FILTERS_KEY]["ages"]
    if "Any" not in chosen_ages:
        # ensure df age as string to compare multi
        f = f[f["Age"].astype(str).isin(chosen_ages)]

    # Sex filter
    chosen_sexes = st.session_state[FILTERS_KEY]["sexes"]
    if "Any" not in chosen_sexes and "Sex" in f.columns:
        f = f[f["Sex"].astype(str).str.capitalize().isin([s.capitalize() for s in chosen_sexes])]

    # Maiden
    maiden_sel = st.session_state[FILTERS_KEY]["maiden"]
    if maiden_sel != "Any":
        # if we have a Maiden boolean or text
        if "Maiden" in f.columns:
            if maiden_sel == "Yes":
                # Accept typical strings: "Yes", True, "True", "Y"
                mask = f["Maiden"].astype(str).str.strip().str.lower().isin(["yes", "y", "true", "1"])
            else:
                mask = f["Maiden"].astype(str).str.strip().str.lower().isin(["no", "n", "false", "0"])
            f = f[mask]

    # State
    chosen_states = st.session_state[FILTERS_KEY]["states"]
    if chosen_states and "State" in f.columns:
        f = f[f["State"].astype(str).isin(chosen_states)]

    # Lowest achieved All Avg Benchmark (max)
    bm_cut = st.session_state[FILTERS_KEY]["lowest_all_avg_bm_max"]
    if bm_cut is not None and "Lowest All Avg Benchmark" in f.columns:
        with pd.option_context('mode.use_inf_as_na', True):
            f = f[pd.to_numeric(f["Lowest All Avg Benchmark"], errors="coerce") <= bm_cut]

    return f

filtered_df = apply_filters(sale_df) if st.session_state[FILTERS_KEY]["apply_clicked"] else sale_df.copy()

# ------------------------------------------------------
# MAIN LAYOUT
# ------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("📋 Filtered Sale Horses")
    if filtered_df.empty or "Name" not in filtered_df.columns:
        st.info("No rows to show. Make sure you have a **Name** column and applied filters.")
    else:
        # augment with helper columns
        view_df = filtered_df.copy()
        # Has Won?
        if "Wins" in view_df.columns:
            try:
                view_df["Has Won?"] = view_df["Wins"].apply(lambda x: int(str(x).split(".")[0]) > 0 if pd.notnull(x) else None)
            except Exception:
                view_df["Has Won?"] = None
        elif "wins" in view_df.columns:
            try:
                view_df["Has Won?"] = view_df["wins"].apply(lambda x: int(str(x).split(".")[0]) > 0 if pd.notnull(x) else None)
            except Exception:
                view_df["Has Won?"] = None
        else:
            view_df["Has Won?"] = None

        # present
        show_cols = [c for c in ["Lot", "Name", "Age", "Sex", "State", "Bid", "Lowest All Avg Benchmark", "Wins", "Has Won?"] if c in view_df.columns]
        st.dataframe(view_df[show_cols].reset_index(drop=True), use_container_width=True)

        # Select → top "Selected Horse"
        selected_name = st.selectbox(
            "Select a horse to view:",
            sorted(view_df["Name"].dropna().astype(str).unique())
        )
        if selected_name:
            st.session_state[SELECTED_HORSE_KEY] = selected_name

        # Save shortlist button (append current filtered list)
        col_sl1, col_sl2 = st.columns([1,1])
        with col_sl1:
            if st.button("➕ Add filtered to shortlist"):
                add_names = view_df["Name"].dropna().astype(str).tolist()
                # keep unique, preserve existing
                current = st.session_state[SHORTLIST_KEY]
                st.session_state[SHORTLIST_KEY] = list(dict.fromkeys(current + add_names))
                st.success(f"Added {len(add_names)} horses to shortlist.")
        with col_sl2:
            if st.button("⬇️ Download shortlist CSV"):
                sd = pd.DataFrame({"Name": st.session_state[SHORTLIST_KEY]})
                st.download_button(
                    "Download now",
                    data=sd.to_csv(index=False),
                    file_name="shortlist.csv",
                    mime="text/csv",
                    use_container_width=True
                )

with right:
    st.subheader("🎯 Selected Horse")
    selected_horse = st.session_state.get(SELECTED_HORSE_KEY)

    if not selected_horse:
        st.info("Pick a horse from the left table to view details.")
    else:
        # Matching row if available
        row = filtered_df[filtered_df["Name"].astype(str) == str(selected_horse)]
        if row.empty:
            row = sale_df[sale_df["Name"].astype(str) == str(selected_horse)]
        detail = row.iloc[0].to_dict() if not row.empty else {"Name": selected_horse}

        # Show useful details from the sale list
        st.markdown(f"### {detail.get('Name', selected_horse)}")
        cols_info = [("Lot","Lot"),("Age","Age"),("Sex","Sex"),("Sire","Sire"),("Dam","Dam"),
                     ("Vendor","Vendor"),("State","State"),("Bid","Bid"),
                     ("Lowest All Avg Benchmark","Lowest All Avg Benchmark")]
        info_pairs = []
        for label, key in cols_info:
            if key in detail and pd.notnull(detail[key]) and str(detail[key]).strip():
                info_pairs.append((label, detail[key]))
        if info_pairs:
            st.table(pd.DataFrame(info_pairs, columns=["Field","Value"]))

        # PF button ON TOP (as requested)
        st.markdown("#### 📡 Punting Form")
        with st.container():
            btn_cols = st.columns([1,1,1])
            with btn_cols[0]:
                get_pf = st.button("🔍 View Punting Form Data", use_container_width=True)
            with btn_cols[1]:
                st.session_state[EMBED_PF_PANEL_OPEN_KEY] = st.toggle("Show embedded PF panel", value=st.session_state[EMBED_PF_PANEL_OPEN_KEY])
            with btn_cols[2]:
                pass

        # When clicked, fetch basic PF hooks
        if get_pf:
            with st.spinner("Searching Punting Form…"):
                try:
                    ident = pf_search_horse_by_name(selected_horse)
                    if not ident.get("found"):
                        st.warning("Horse not found in PF (or pf_client in Demo Mode).")
                    else:
                        st.success(f"Found on PF: {ident.get('display_name', selected_horse)} (id: {ident.get('horse_id')})")

                        # If you had meeting/race ids, fetch more:
                        # form = pf_get_form(ident["horse_id"])
                        # ratings = pf_get_ratings(<meeting_id>)
                        # sectionals = pf_get_meeting_sectionals(<meeting_id>)
                        # benchmarks = pf_get_meeting_benchmarks(<meeting_id>)
                        # Here we just show the search result for now:
                        with st.expander("🔎 PF Search Result", expanded=True):
                            st.json(ident)
                except Exception as e:
                    st.error(f"Could not retrieve data: {e}")

        # Embedded PF page (if they want to see the real PF page right inside)
        if st.session_state[EMBED_PF_PANEL_OPEN_KEY]:
            st.markdown("##### 🌐 Embedded Punting Form Page")
            st.caption(
                "Paste a Punting Form page URL here (e.g., horse profile or meeting page). "
                "Note: some sites block embedding; if the frame fails, click the external link."
            )
            pf_url = st.text_input("PF Page URL", value="", placeholder="https://api.puntingform.com.au/...")
            if pf_url:
                # Show an external link
                st.markdown(f"[Open in new tab]({pf_url})")
                # Try to embed
                st.components.v1.iframe(pf_url, height=680, scrolling=True)

# ------------------------------------------------------
# QUALITY-OF-LIFE: QUICK STATS + HOW-TO
# ------------------------------------------------------
st.markdown("---")
with st.expander("📊 Quick Summary & How to Make This Powerful", expanded=False):
    total = len(sale_df) if not sale_df.empty else 0
    filtered_n = len(filtered_df) if not filtered_df.empty else 0
    shortlist_n = len(st.session_state[SHORTLIST_KEY])
    st.write(f"**Loaded horses:** {total} | **Filtered:** {filtered_n} | **In shortlist:** {shortlist_n}")

    st.markdown(
        """
**Tips to go next-level:**
- Add a **Meeting/Race mapping** to your sale list (columns `MeetingId`, `RaceId`), then call:
  - `pf_get_ratings(meeting_id)`, `pf_get_meeting_sectionals(meeting_id)`, `pf_get_meeting_benchmarks(meeting_id)`.
- Use Punting Form's "lowest achieved All Avg Benchmark" into your dataset. Put it in a column named **`Lowest All Avg Benchmark`**, and the filter here will be exact.
- Make your own **scoring** column (e.g. *ModelScore*) and add a sort control to surface best value quickly.
- Use the **shortlist** to export a CSV and share with partners/investors.
"""
    )

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.caption("© Soar Bloodstock — MoneyBall")
