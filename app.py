# ===============================
# Soar Bloodstock Data - MoneyBall
# Stable build: working filters + PF connectivity test
# ===============================

import os, sys, io, base64, json, re, traceback
from datetime import date
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# ---------- Page config ----------
st.set_page_config(
    page_title="Soar Bloodstock Data - MoneyBall",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Session keys ----------
DATA_KEY = "SALE_DF"
NAME_COL_KEY = "NAME_COL"
LOGO_BYTES_KEY = "LOGO_BYTES"
FILTERS_KEY = "FILTERS"
SELECTED_HORSE_KEY = "SELECTED_HORSE"
SHORTLIST_KEY = "SHORTLIST"

if FILTERS_KEY not in st.session_state:
    st.session_state[FILTERS_KEY] = {
        "age_opts": ["Any"],          # ["Any"] or list of str ages
        "sex_opts": ["Any"],          # ["Any"] or chosen sexes
        "state_opts": [],             # list of chosen states
        "maiden": "Any",              # "Any" | "Yes" | "No"
        "bm_max": None,               # float / None
        "applied": False,
    }

if SHORTLIST_KEY not in st.session_state:
    st.session_state[SHORTLIST_KEY] = []

# ---------- Helpers ----------
def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        x = str(c).replace("\ufeff","")
        x = re.sub(r"\s+"," ", x).strip()
        mapping[c] = x
    return df.rename(columns=mapping)

def detect_name_col(cols: List[str]) -> Optional[str]:
    # smart guess for name column
    norm = {re.sub(r"\s+","", c).lower(): c for c in cols}
    for cand in ["name","horse","horse name","horsename","lot name"]:
        k = cand.replace(" ", "").lower()
        if k in norm:
            return norm[k]
    for c in cols:
        if "name" in c.lower():
            return c
    return None

def center_logo_html(img_bytes: Optional[bytes]) -> str:
    if not img_bytes:
        return ""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"""
<div style="display:flex;justify-content:center;margin-top:-8px;">
  <img src="data:image/png;base64,{b64}" style="max-height:80px;" />
</div>
"""

def show_logo_top():
    if LOGO_BYTES_KEY in st.session_state and st.session_state[LOGO_BYTES_KEY]:
        st.markdown(center_logo_html(st.session_state[LOGO_BYTES_KEY]), unsafe_allow_html=True)

def ensure_age_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Age" not in df.columns:
        if "yob" in df.columns:
            df = df.copy()
            df["Age"] = date.today().year - pd.to_numeric(df["yob"], errors="coerce")
        else:
            df = df.copy()
            df["Age"] = np.nan
    return df

def coerce_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

# ---------- PF client import with safe fallback ----------
PF_AVAILABLE = False
PF_IMPORT_ERROR = None

try:
    from pf_client import (
        is_live as pf_is_live,
        search_horse_by_name,
        get_meeting_benchmarks,
        get_meeting_sectionals,
        get_meeting_ratings,
        pf_raw_get,           # for tester
    )
    PF_AVAILABLE = True
except Exception as e:
    PF_IMPORT_ERROR = e
    PF_AVAILABLE = False

# ---------- Top: Logo + Title ----------
show_logo_top()
st.title("Soar Bloodstock Data — MoneyBall")

# ---------- Sidebar: Page Settings ----------
with st.sidebar.expander("⚙️ Page settings", expanded=False):
    logo = st.file_uploader("Upload logo (png/jpg)", type=["png","jpg","jpeg"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save logo"):
            if logo:
                try:
                    img = Image.open(logo).convert("RGBA")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.session_state[LOGO_BYTES_KEY] = buf.getvalue()
                    st.success("Logo saved.")
                except Exception as e:
                    st.error(f"Logo error: {e}")
            else:
                st.info("Choose a file first.")
    with c2:
        if st.button("Clear logo"):
            st.session_state[LOGO_BYTES_KEY] = None
            st.success("Logo cleared.")

with st.sidebar.expander("🔌 Punting Form status", expanded=False):
    if PF_AVAILABLE and pf_is_live():
        st.success("PF: Live")
    elif PF_AVAILABLE:
        st.warning("PF: Client imported, but not live (check API key/secrets).")
    else:
        st.error("PF: Client not importable. UI will still work.")
        if PF_IMPORT_ERROR:
            st.caption(f"{PF_IMPORT_ERROR}")

# ---------- Sidebar: Data Input ----------
with st.sidebar.expander("🧾 Horse list", expanded=True):
    input_mode = st.radio("Source", ["Upload CSV/Excel", "Paste", "Keep current"], index=0)

    new_df = None
    if input_mode == "Upload CSV/Excel":
        f = st.file_uploader("Upload", type=["csv","xlsx"])
        if f:
            try:
                if f.name.lower().endswith(".xlsx"):
                    tmp = pd.read_excel(f)
                else:
                    try:
                        tmp = pd.read_csv(f, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
                    except UnicodeDecodeError:
                        tmp = pd.read_csv(f, sep=None, engine="python", encoding="ISO-8859-1", on_bad_lines="skip")
                tmp = clean_headers(tmp)
                new_df = tmp
                st.success(f"Loaded {len(new_df)} rows.")
            except Exception as e:
                st.error(f"Read error: {e}")
        if st.button("Save data"):
            if new_df is not None:
                st.session_state[DATA_KEY] = new_df
                st.success("Saved. Scroll down to filters and results.")
            else:
                st.info("Upload a file first.")

    elif input_mode == "Paste":
        pasted = st.text_area("One horse per line", height=180, placeholder="Hell Island\nInvincible Phantom\nIrish Bliss")
        if st.button("Save pasted"):
            names = [n.strip() for n in pasted.splitlines() if n.strip()]
            st.session_state[DATA_KEY] = pd.DataFrame({"Name": names})
            st.success(f"Saved {len(names)} names.")

    else:
        if st.session_state.get(DATA_KEY) is not None:
            st.success("Using current data in memory.")
        else:
            st.info("No current data. Upload or paste, then Save.")

# ---------- Build base df ----------
sale_df = st.session_state.get(DATA_KEY, pd.DataFrame(columns=["Name"])).copy()
if sale_df.empty:
    st.info("No data loaded yet. Use **Horse list** in the sidebar to upload/paste and save.")
else:
    # let user confirm the Name column
    guessed = detect_name_col(list(sale_df.columns))
    current_choice = st.session_state.get(NAME_COL_KEY, guessed if guessed else "Name")
    st.markdown("#### Name Column")
    name_col = st.selectbox(
        "Pick the column that contains the horse name.",
        options=list(sale_df.columns),
        index=(list(sale_df.columns).index(current_choice)
               if current_choice in sale_df.columns else 0),
        help="This must contain the horse names."
    )
    st.session_state[NAME_COL_KEY] = name_col
    if name_col != "Name":
        sale_df = sale_df.rename(columns={name_col: "Name"})

# Add/ensure standard columns where possible
if not sale_df.empty:
    sale_df = ensure_age_column(sale_df)
    if "Lowest All Avg Benchmark" not in sale_df.columns:
        # placeholder so filter UI works; you can add real values in your CSV
        sale_df["Lowest All Avg Benchmark"] = np.nan

# ---------- Sidebar: Filters ----------
with st.sidebar.expander("🔎 Filters", expanded=True):
    ages = ["Any"] + [str(i) for i in range(1, 11)]
    sel_age = st.multiselect("Age", options=ages, default=st.session_state[FILTERS_KEY]["age_opts"])
    sexes = ["Any", "Gelding", "Mare", "Horse", "Colt", "Filly"]
    sel_sex = st.multiselect("Sex", options=sexes, default=st.session_state[FILTERS_KEY]["sex_opts"])

    state_options = sorted(sale_df["State"].dropna().astype(str).unique().tolist()) if "State" in sale_df.columns else []
    sel_state = st.multiselect("State", options=state_options, default=st.session_state[FILTERS_KEY]["state_opts"])

    maiden = st.selectbox("Maiden", ["Any", "Yes", "No"], index=["Any","Yes","No"].index(st.session_state[FILTERS_KEY]["maiden"]))

    bm_max = st.text_input("Lowest achieved All Avg Benchmark (max)", value=str(st.session_state[FILTERS_KEY]["bm_max"]) if st.session_state[FILTERS_KEY]["bm_max"] is not None else "")

    if st.button("✅ Apply"):
        st.session_state[FILTERS_KEY]["age_opts"] = sel_age if sel_age else ["Any"]
        st.session_state[FILTERS_KEY]["sex_opts"] = sel_sex if sel_sex else ["Any"]
        st.session_state[FILTERS_KEY]["state_opts"] = sel_state
        st.session_state[FILTERS_KEY]["maiden"] = maiden
        st.session_state[FILTERS_KEY]["bm_max"] = (coerce_float(bm_max) if bm_max.strip() != "" else None)
        st.session_state[FILTERS_KEY]["applied"] = True

# ---------- Apply filters ----------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    f = df.copy()

    # age
    age_sel = st.session_state[FILTERS_KEY]["age_opts"]
    if "Any" not in age_sel:
        f = f[f["Age"].astype(str).isin(age_sel)]

    # sex
    sex_sel = st.session_state[FILTERS_KEY]["sex_opts"]
    if "Any" not in sex_sel and "Sex" in f.columns:
        f = f[f["Sex"].astype(str).str.capitalize().isin([s.capitalize() for s in sex_sel])]

    # state
    st_sel = st.session_state[FILTERS_KEY]["state_opts"]
    if st_sel and "State" in f.columns:
        f = f[f["State"].astype(str).isin(st_sel)]

    # maiden
    msel = st.session_state[FILTERS_KEY]["maiden"]
    if msel != "Any" and "Maiden" in f.columns:
        # interpret typical truthy/falsey strings
        truthy = ["yes","y","true","1"]
        falsy = ["no","n","false","0"]
        col = f["Maiden"].astype(str).str.strip().str.lower()
        if msel == "Yes":
            f = f[col.isin(truthy)]
        else:
            f = f[col.isin(falsy)]

    # bm max
    bm = st.session_state[FILTERS_KEY]["bm_max"]
    if bm is not None and "Lowest All Avg Benchmark" in f.columns:
        f = f[pd.to_numeric(f["Lowest All Avg Benchmark"], errors="coerce") <= float(bm)]

    return f

filtered_df = apply_filters(sale_df) if st.session_state[FILTERS_KEY]["applied"] else sale_df

# ---------- Layout ----------
left, right = st.columns([1,1])

with left:
    st.subheader("📋 Filtered Sale Horses")
    if filtered_df.empty:
        st.info("No rows to show. Adjust filters or load data.")
    else:
        # Attach a "Has Won?" column if Wins exists
        view = filtered_df.copy()
        if "Wins" in view.columns:
            def _has_won(val):
                try:
                    return int(str(val).split(".")[0]) > 0
                except Exception:
                    return None
            view["Has Won?"] = view["Wins"].apply(_has_won)
        show_cols = [c for c in ["Lot","Name","Age","Sex","State","Bid","Wins","Has Won?","Lowest All Avg Benchmark"] if c in view.columns]
        st.dataframe(view[show_cols].reset_index(drop=True), use_container_width=True)

        selected = st.selectbox(
            "Select a horse to view",
            options=sorted(view["Name"].dropna().astype(str).unique())
        )
        if selected:
            st.session_state[SELECTED_HORSE_KEY] = selected

        # shortlist controls
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Add filtered to shortlist"):
                add = view["Name"].dropna().astype(str).tolist()
                sl = st.session_state[SHORTLIST_KEY]
                st.session_state[SHORTLIST_KEY] = list(dict.fromkeys(sl + add))
                st.success(f"Added {len(add)} names to shortlist.")
        with c2:
            if st.button("⬇️ Download shortlist CSV"):
                out = pd.DataFrame({"Name": st.session_state[SHORTLIST_KEY]})
                st.download_button("Download now", data=out.to_csv(index=False), file_name="shortlist.csv", mime="text/csv")

with right:
    st.subheader("🎯 Selected Horse")
    selected = st.session_state.get(SELECTED_HORSE_KEY)
    if not selected:
        st.info("Pick a horse on the left.")
    else:
        # find detail row (prefer filtered; fallback to base)
        row = filtered_df[filtered_df["Name"].astype(str) == str(selected)]
        if row.empty:
            row = sale_df[sale_df["Name"].astype(str) == str(selected)]
        d = row.iloc[0].to_dict() if not row.empty else {"Name": selected}

        st.markdown(f"### {d.get('Name', selected)}")
        pairs = []
        for label in ["Lot","Age","Sex","Sire","Dam","Vendor","State","Bid","Wins","Lowest All Avg Benchmark"]:
            if label in d and pd.notnull(d[label]) and str(d[label]).strip():
                pairs.append((label, d[label]))
        if pairs:
            st.table(pd.DataFrame(pairs, columns=["Field","Value"]))

        # ---- PF panel (button + connectivity tester) ----
        st.markdown("#### 📡 Punting Form")
        col_a, col_b = st.columns([1,1])
        with col_a:
            if st.button("🔍 View Punting Form Data"):
                if not PF_AVAILABLE:
                    st.error("pf_client not importable. Add pf_client.py & secrets.")
                else:
                    try:
                        res = search_horse_by_name(selected)
                        st.json(res)
                    except Exception as e:
                        st.error(f"PF search failed: {e}")

        with col_b:
            with st.expander("Connectivity test (direct API)", expanded=False):
                st.caption("Use this to verify your **API key & endpoint** are correct.")
                endpoint = st.selectbox(
                    "Endpoint",
                    ["Ratings/MeetingBenchmarks", "Ratings/MeetingSectionals", "Ratings/MeetingRatings"],
                    index=0
                )
                meeting_id = st.text_input("meetingId (int)", value="")
                if st.button("Test call"):
                    if not PF_AVAILABLE:
                        st.error("pf_client not importable.")
                    elif not meeting_id.strip().isdigit():
                        st.warning("Enter a numeric meetingId.")
                    else:
                        try:
                            path = f"/{endpoint}"
                            r = pf_raw_get(path, params={"meetingId": int(meeting_id)})
                            st.write(f"HTTP {r.status_code}")
                            if r.headers.get("content-type","").startswith("application/json"):
                                st.json(r.json())
                            else:
                                st.text(r.text)
                        except Exception as e:
                            st.error(f"Test failed: {e}")

st.markdown("---")
st.caption("© Soar Bloodstock — MoneyBall")
