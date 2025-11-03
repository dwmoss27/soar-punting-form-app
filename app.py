# =========================================================
# Soar Bloodstock Data - MoneyBall (Stable Filter Build)
# =========================================================
import os, io, re, json, base64, traceback
from datetime import date
from typing import Optional, Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(page_title="Soar Bloodstock Data - MoneyBall", layout="wide")

# ---- Session keys
DATA_KEY = "SALE_DF"
CANON_DF_KEY = "CANON_DF"          # normalized dataframe used for filters
NAME_COL_KEY = "NAME_COL"
FILTERS_KEY = "FILTERS"
SELECTED_HORSE_KEY = "SELECTED_HORSE"
LOGO_BYTES_KEY = "LOGO_BYTES"

# ---- Safe initialization helpers
DEFAULT_FILTERS = {
    "age": ["Any"],      # multiselect: ["Any","1","2",...,"10"]
    "sex": ["Any"],      # multiselect: ["Any","Gelding","Mare","Horse","Colt","Filly"]
    "state": [],         # multiselect of states
    "maiden": "Any",     # "Any" | "Yes" | "No"
    "bm_max": None,      # float or None
    "applied": False,
}

def ensure_filters():
    """Ensure FILTERS_KEY exists and has all required keys."""
    f = st.session_state.get(FILTERS_KEY)
    if not isinstance(f, dict):
        st.session_state[FILTERS_KEY] = DEFAULT_FILTERS.copy()
        return
    # backfill any missing keys
    changed = False
    for k, v in DEFAULT_FILTERS.items():
        if k not in f:
            f[k] = v
            changed = True
    if changed:
        st.session_state[FILTERS_KEY] = f

ensure_filters()

# =========================================================
# Optional PF client (won't break UI if missing)
# =========================================================
PF_AVAILABLE = False
try:
    from pf_client import (
        is_live as pf_is_live,
        search_horse_by_name,
        get_meeting_benchmarks,
        get_meeting_sectionals,
        get_meeting_ratings,
        pf_raw_get,
    )
    PF_AVAILABLE = True
except Exception:
    PF_AVAILABLE = False

# =========================================================
# UI helpers
# =========================================================
def center_logo_html(img_bytes: Optional[bytes]) -> str:
    if not img_bytes:
        return ""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"""
<div style="display:flex;justify-content:center;margin-top:-8px;margin-bottom:8px;">
  <img src="data:image/png;base64,{b64}" style="max-height:80px;" />
</div>
"""

def show_logo_top():
    if st.session_state.get(LOGO_BYTES_KEY):
        st.markdown(center_logo_html(st.session_state[LOGO_BYTES_KEY]), unsafe_allow_html=True)

# =========================================================
# Data normalization
# =========================================================
HEADER_ALIASES = {
    "Name": [r"^name$", r"^horse\s*name$", r"^horse$", r"^lot\s*name$"],
    "Age": [r"^age$", r"^yob$", r"^year\s*of\s*birth$"],
    "Sex": [r"^sex$", r"^gender$"],
    "State": [r"^state$", r"^location$", r"^jurisdiction$", r"^st$"],
    "Maiden": [r"^maiden$", r"^is\s*maiden$", r"^maiden\s*status$"],
    "Wins": [r"^wins?$", r"^win\s*count$", r"^starts?\s*wins?$", r"^\s*wins\s*$"],
    "Benchmark_any": [
        r"lowest.*all.*avg.*benchmark",
        r"min.*all.*avg.*benchmark",
        r"all.*avg.*benchmark",
        r"avg.*benchmark",
        r"benchmark(?!.*price)",
        r"^bm$",
    ],
    "Bid": [r"^bid$", r"^current\s*bid$"],
    "Lot": [r"^lot$"],
    "Sire": [r"^sire$"],
    "Dam": [r"^dam$"],
    "Vendor": [r"^vendor$", r"^consignor$", r"^trainer$"],
}

SEX_MAP = {
    "g": "Gelding", "geld": "Gelding", "gelding": "Gelding",
    "m": "Mare", "mare": "Mare",
    "h": "Horse", "horse": "Horse", "entire": "Horse",
    "c": "Colt", "colt": "Colt",
    "f": "Filly", "filly": "Filly",
}

TRUTHY = {"yes", "y", "true", "1", "t"}
FALSY  = {"no", "n", "false", "0", "f"}

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        x = str(c).replace("\ufeff","")
        x = re.sub(r"\s+", " ", x).strip()
        mapping[c] = x
    return df.rename(columns=mapping)

def find_first_match(cols: List[str], patterns: List[str]) -> Optional[str]:
    for p in patterns:
        rgx = re.compile(p, re.I)
        for c in cols:
            if rgx.search(c):
                return c
    return None

def normalize_sex(val: Any) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    s = re.sub(r"[^a-z]", "", s)
    return SEX_MAP.get(s, None)

def detect_benchmark_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    out = []
    for pattern in HEADER_ALIASES["Benchmark_any"]:
        rgx = re.compile(pattern, re.I)
        out += [c for c in cols if rgx.search(c)]
    return list(dict.fromkeys(out))  # unique, preserve order

def derive_age(col_age: Optional[pd.Series], col_yob: Optional[pd.Series]) -> pd.Series:
    if col_age is not None:
        age = pd.to_numeric(col_age, errors="coerce")
        if age.notna().any() or col_yob is None:
            return age
    if col_yob is not None:
        yob = pd.to_numeric(col_yob, errors="coerce")
        return date.today().year - yob
    return pd.Series(np.nan, index=(col_age.index if col_age is not None else col_yob.index if col_yob is not None else []))

def canonize(df_raw: pd.DataFrame, debug: Dict[str, Any]) -> pd.DataFrame:
    df = clean_headers(df_raw)
    cols = list(df.columns)
    debug["clean_columns"] = cols

    # Name
    nm_col = find_first_match(cols, HEADER_ALIASES["Name"]) or ("Name" if "Name" in cols else None)
    if not nm_col:
        df["Name"] = df.index.astype(str)
        nm_col = "Name"

    # Age vs YOB
    age_col = find_first_match(cols, HEADER_ALIASES["Age"])
    yob_col = None
    if age_col and age_col.lower() == "yob":
        yob_col = age_col
        age_col = None
    elif "yob" in [c.lower() for c in cols]:
        yob_col = [c for c in cols if c.lower()=="yob"][0]

    sex_col    = find_first_match(cols, HEADER_ALIASES["Sex"])
    state_col  = find_first_match(cols, HEADER_ALIASES["State"])
    maiden_col = find_first_match(cols, HEADER_ALIASES["Maiden"])
    wins_col   = find_first_match(cols, HEADER_ALIASES["Wins"])

    bm_candidates = detect_benchmark_columns(df)
    bid_col   = find_first_match(cols, HEADER_ALIASES["Bid"])
    lot_col   = find_first_match(cols, HEADER_ALIASES["Lot"])
    sire_col  = find_first_match(cols, HEADER_ALIASES["Sire"])
    dam_col   = find_first_match(cols, HEADER_ALIASES["Dam"])
    vendor_col= find_first_match(cols, HEADER_ALIASES["Vendor"])

    debug["mapped"] = {
        "Name": nm_col, "Age_col": age_col, "YOB_col": yob_col,
        "Sex": sex_col, "State": state_col, "Maiden": maiden_col, "Wins": wins_col,
        "Benchmark_candidates": bm_candidates,
        "Bid": bid_col, "Lot": lot_col, "Sire": sire_col, "Dam": dam_col, "Vendor": vendor_col,
    }

    out = pd.DataFrame()
    out["Name"] = df[nm_col].astype(str)

    col_age_series = df[age_col] if (age_col in df.columns) else None
    col_yob_series = df[yob_col] if (yob_col in df.columns) else None
    out["Age"] = derive_age(col_age_series, col_yob_series).round(0)

    if sex_col:
        out["Sex"] = df[sex_col].map(normalize_sex)
    else:
        out["Sex"] = None

    if state_col:
        out["State"] = df[state_col].astype(str).str.strip().str.upper()
    else:
        out["State"] = None

    if maiden_col:
        m = df[maiden_col].astype(str).str.strip().str.lower()
        out["Maiden"] = np.where(m.isin(TRUTHY), True,
                           np.where(m.isin(FALSY), False, None))
    elif wins_col:
        w = pd.to_numeric(df[wins_col], errors="coerce")
        out["Maiden"] = np.where(w==0, True, np.where(w>0, False, None))
    else:
        out["Maiden"] = None

    if wins_col:
        out["Wins"] = pd.to_numeric(df[wins_col], errors="coerce")
    else:
        out["Wins"] = np.nan

    # Lowest All Avg Benchmark (min across candidates)
    bm_vals = []
    for c in bm_candidates:
        v = pd.to_numeric(df[c], errors="coerce")
        bm_vals.append(v)
    if bm_vals:
        bm_stack = pd.concat(bm_vals, axis=1)
        out["Lowest All Avg Benchmark"] = bm_stack.min(axis=1)
    else:
        out["Lowest All Avg Benchmark"] = np.nan

    for label, col in [("Bid", bid_col), ("Lot", lot_col), ("Sire", sire_col), ("Dam", dam_col), ("Vendor", vendor_col)]:
        if col:
            out[label] = df[col]
        else:
            out[label] = None

    debug["samples"] = out.head(8).to_dict(orient="records")
    return out

# =========================================================
# Sidebar: settings + data input
# =========================================================
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

with st.sidebar.expander("🧾 Horse list", expanded=True):
    mode = st.radio("Source", ["Upload CSV/Excel", "Paste", "Keep current"], index=0)

    if mode == "Upload CSV/Excel":
        up = st.file_uploader("Upload", type=["csv","xlsx"])
        if up:
            try:
                if up.name.lower().endswith(".xlsx"):
                    df = pd.read_excel(up)
                else:
                    try:
                        df = pd.read_csv(up, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
                    except UnicodeDecodeError:
                        df = pd.read_csv(up, sep=None, engine="python", encoding="ISO-8859-1", on_bad_lines="skip")
                st.session_state[DATA_KEY] = df
                st.success(f"Loaded {len(df)} rows")
            except Exception as e:
                st.error(f"Read error: {e}")

    elif mode == "Paste":
        pasted = st.text_area("One horse per line", height=150, placeholder="Hell Island\nInvincible Phantom\nIrish Bliss")
        if st.button("Save pasted"):
            names = [n.strip() for n in pasted.splitlines() if n.strip()]
            st.session_state[DATA_KEY] = pd.DataFrame({"Name": names})
            st.success(f"Saved {len(names)} names.")

    else:
        if st.session_state.get(DATA_KEY) is not None:
            st.success("Using current data in memory.")
        else:
            st.info("No current data yet.")

# =========================================================
# Build canonical dataframe
# =========================================================
show_logo_top()
st.title("Soar Bloodstock Data — MoneyBall")

base_df = st.session_state.get(DATA_KEY, pd.DataFrame())
debug_map: Dict[str, Any] = {}

if base_df.empty:
    st.info("Upload or paste your list in the sidebar to begin.")
    st.stop()

# Choose which column is the horse name (defensive)
clean = clean_headers(base_df)
auto_name = None
for pat in HEADER_ALIASES["Name"]:
    found = find_first_match(list(clean.columns), [pat])
    if found:
        auto_name = found
        break

name_col = st.selectbox(
    "Pick the column that contains horse names:",
    options=list(clean.columns),
    index=(list(clean.columns).index(auto_name) if (auto_name in clean.columns) else 0),
)
st.session_state[NAME_COL_KEY] = name_col

work_df = clean.rename(columns={name_col: "Name"}) if name_col != "Name" else clean
canon_df = canonize(work_df, debug_map)
st.session_state[CANON_DF_KEY] = canon_df

# =========================================================
# Filters (safe defaults)
# =========================================================
fstate = st.session_state.get(FILTERS_KEY, DEFAULT_FILTERS.copy())

with st.sidebar.expander("🔎 Filters", expanded=True):
    ages = ["Any"] + [str(i) for i in range(1, 11)]
    sel_age = st.multiselect("Age", ages, default=fstate.get("age", ["Any"]))

    sexes = ["Any", "Gelding", "Mare", "Horse", "Colt", "Filly"]
    sel_sex = st.multiselect("Sex", sexes, default=fstate.get("sex", ["Any"]))

    state_options = sorted(canon_df["State"].dropna().astype(str).unique().tolist())
    # keep only still-valid defaults
    prev_states = [s for s in fstate.get("state", []) if s in state_options]
    sel_state = st.multiselect("State", state_options, default=prev_states)

    maiden = st.selectbox("Maiden", ["Any", "Yes", "No"], index=["Any","Yes","No"].index(fstate.get("maiden","Any")))

    bm_default = "" if fstate.get("bm_max") is None else str(fstate["bm_max"])
    bm_input = st.text_input("Lowest achieved All Avg Benchmark (max)", value=bm_default)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Apply"):
            st.session_state[FILTERS_KEY] = {
                "age": sel_age if sel_age else ["Any"],
                "sex": sel_sex if sel_sex else ["Any"],
                "state": sel_state,
                "maiden": maiden,
                "bm_max": (float(bm_input) if bm_input.strip() != "" else None),
                "applied": True,
            }
            ensure_filters()
    with c2:
        if st.button("♻️ Reset"):
            st.session_state[FILTERS_KEY] = DEFAULT_FILTERS.copy()
            ensure_filters()

# Use a fresh copy of filter state now that it may have been updated
fstate = st.session_state.get(FILTERS_KEY, DEFAULT_FILTERS.copy())

def apply_filters(df: pd.DataFrame, fs: Dict[str, Any]) -> pd.DataFrame:
    f = df.copy()

    # Age
    age_sel = fs.get("age", ["Any"])
    if "Any" not in age_sel:
        f = f[f["Age"].astype("Int64").astype(str).isin(age_sel)]

    # Sex
    sex_sel = fs.get("sex", ["Any"])
    if "Any" not in sex_sel:
        f = f[f["Sex"].isin(sex_sel)]

    # State
    st_sel = fs.get("state", [])
    if st_sel:
        f = f[f["State"].isin(st_sel)]

    # Maiden
    msel = fs.get("maiden", "Any")
    if msel != "Any":
        if msel == "Yes":
            f = f[f["Maiden"] == True]
        else:
            f = f[f["Maiden"] == False]

    # Benchmark max
    bm_max = fs.get("bm_max", None)
    if bm_max is not None:
        f = f[pd.to_numeric(f["Lowest All Avg Benchmark"], errors="coerce") <= float(bm_max)]

    return f

filtered = apply_filters(canon_df, fstate) if fstate.get("applied", False) else canon_df

# =========================================================
# Debug panel (helps diagnose filter issues)
# =========================================================
with st.expander("🧪 Debug (what the app sees)"):
    st.write("**Mapped columns & samples:**")
    st.json(debug_map)
    st.write("**Filter state:**")
    st.json(st.session_state.get(FILTERS_KEY, DEFAULT_FILTERS.copy()))

# =========================================================
# Main layout
# =========================================================
left, right = st.columns([1,1])

with left:
    st.subheader("📋 Filtered Sale Horses")
    if filtered.empty:
        st.warning("No rows match the current filters.")
    else:
        view_cols = [c for c in ["Lot","Name","Age","Sex","State","Wins","Maiden","Bid","Lowest All Avg Benchmark"] if c in filtered.columns]
        st.dataframe(filtered[view_cols].reset_index(drop=True), use_container_width=True)

        chosen = st.selectbox(
            "Select a horse",
            options=sorted(filtered["Name"].dropna().astype(str).unique())
        )
        if chosen:
            st.session_state[SELECTED_HORSE_KEY] = chosen

with right:
    st.subheader("🎯 Selected Horse")
    sel = st.session_state.get(SELECTED_HORSE_KEY)
    if not sel:
        st.info("Pick a horse on the left.")
    else:
        row = filtered[filtered["Name"].astype(str) == str(sel)]
        if row.empty:
            row = canon_df[canon_df["Name"].astype(str) == str(sel)]
        d = row.iloc[0].to_dict() if not row.empty else {"Name": sel}

        st.markdown(f"### {d.get('Name', sel)}")
        fields = ["Lot","Age","Sex","State","Wins","Maiden","Bid","Lowest All Avg Benchmark","Sire","Dam","Vendor"]
        pairs = [(k, d[k]) for k in fields if k in d and (pd.notna(d[k]) and str(d[k]).strip() != "")]
        if pairs:
            st.table(pd.DataFrame(pairs, columns=["Field","Value"]))
        else:
            st.caption("No extra fields present in the file.")

        # ---- Optional PF tester ----
        st.markdown("#### 📡 Punting Form (tester)")
        if not PF_AVAILABLE:
            st.info("pf_client not importable (filters still work). Add pf_client.py and secrets to enable.")
        else:
            if pf_is_live():
                st.success("PF client reachable.")
            else:
                st.warning("PF client imported, but not confirmed live. Check your secrets.")

            with st.expander("Connectivity test", expanded=False):
                endpoint = st.selectbox("Endpoint", ["Ratings/MeetingBenchmarks","Ratings/MeetingSectionals","Ratings/MeetingRatings"])
                meeting_id = st.text_input("meetingId (integer)", value="")
                if st.button("Test PF call"):
                    try:
                        if not meeting_id.strip().isdigit():
                            st.warning("Enter a numeric meetingId.")
                        else:
                            r = pf_raw_get("/"+endpoint, params={"meetingId": int(meeting_id)})
                            st.write(f"HTTP {r.status_code}")
                            if r.headers.get("content-type","").startswith("application/json"):
                                st.json(r.json())
                            else:
                                st.text(r.text)
                    except Exception as e:
                        st.error(f"PF test failed: {e}")

st.markdown("---")
st.caption("© Soar Bloodstock — MoneyBall")
