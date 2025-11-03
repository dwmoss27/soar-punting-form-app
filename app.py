# ==============================================
# Soar Bloodstock Data - MoneyBall (Live Filter)
# ==============================================
import os, io, re, base64
from datetime import date
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Soar Bloodstock Data - MoneyBall", layout="wide")

# ---------------- Optional PF client (safe import) ----------------
PF_AVAILABLE = False
try:
    from pf_client import (
        is_live as pf_is_live,
        pf_raw_get,                   # simple GET helper (see pf_client template I gave you)
    )
    PF_AVAILABLE = True
except Exception:
    PF_AVAILABLE = False

# ---------------- Helpers ----------------
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
        x = str(c).replace("\ufeff", "")
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
    if val is None or (isinstance(val, float) and np.isnan(val)): return None
    s = str(val).strip().lower()
    s = re.sub(r"[^a-z]", "", s)
    return SEX_MAP.get(s, None)

def detect_benchmark_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    out = []
    for pattern in HEADER_ALIASES["Benchmark_any"]:
        rgx = re.compile(pattern, re.I)
        out += [c for c in cols if rgx.search(c)]
    # unique, keep order
    return list(dict.fromkeys(out))

def derive_age(col_age: Optional[pd.Series], col_yob: Optional[pd.Series]) -> pd.Series:
    if col_age is not None:
        age = pd.to_numeric(col_age, errors="coerce")
        if age.notna().any() or col_yob is None:
            return age
    if col_yob is not None:
        yob = pd.to_numeric(col_yob, errors="coerce")
        return date.today().year - yob
    idx = col_age.index if col_age is not None else (col_yob.index if col_yob is not None else pd.RangeIndex(0))
    return pd.Series(np.nan, index=idx)

def canonize(df_raw: pd.DataFrame, name_col: str) -> pd.DataFrame:
    df = clean_headers(df_raw)
    if name_col != "Name" and name_col in df.columns:
        df = df.rename(columns={name_col: "Name"})
    cols = list(df.columns)

    # Age / YOB
    age_col = find_first_match(cols, HEADER_ALIASES["Age"])
    yob_col = None
    if age_col and age_col.lower() == "yob": yob_col, age_col = age_col, None
    elif "yob" in [c.lower() for c in cols]:
        yob_col = [c for c in cols if c.lower()=="yob"][0]

    sex_col    = find_first_match(cols, HEADER_ALIASES["Sex"])
    state_col  = find_first_match(cols, HEADER_ALIASES["State"])
    maiden_col = find_first_match(cols, HEADER_ALIASES["Maiden"])
    wins_col   = find_first_match(cols, HEADER_ALIASES["Wins"])
    bm_cols    = detect_benchmark_columns(df)
    bid_col    = find_first_match(cols, HEADER_ALIASES["Bid"])
    lot_col    = find_first_match(cols, HEADER_ALIASES["Lot"])
    sire_col   = find_first_match(cols, HEADER_ALIASES["Sire"])
    dam_col    = find_first_match(cols, HEADER_ALIASES["Dam"])
    vendor_col = find_first_match(cols, HEADER_ALIASES["Vendor"])

    out = pd.DataFrame()
    out["Name"] = df["Name"].astype(str)

    col_age_series = df[age_col] if (age_col in df.columns) else None
    col_yob_series = df[yob_col] if (yob_col in df.columns) else None
    out["Age"] = derive_age(col_age_series, col_yob_series).round(0)

    out["Sex"]   = df[sex_col].map(normalize_sex) if sex_col else None
    out["State"] = df[state_col].astype(str).str.strip().str.upper() if state_col else None

    if maiden_col:
        m = df[maiden_col].astype(str).str.strip().str.lower()
        out["Maiden"] = np.where(m.isin(TRUTHY), True,
                           np.where(m.isin(FALSY), False, None))
    elif wins_col:
        w = pd.to_numeric(df[wins_col], errors="coerce")
        out["Maiden"] = np.where(w==0, True, np.where(w>0, False, None))
    else:
        out["Maiden"] = None

    out["Wins"] = pd.to_numeric(df[wins_col], errors="coerce") if wins_col else np.nan

    if bm_cols:
        bm_stack = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in bm_cols], axis=1)
        out["Lowest All Avg Benchmark"] = bm_stack.min(axis=1)
    else:
        out["Lowest All Avg Benchmark"] = np.nan

    for label, col in [("Bid",bid_col),("Lot",lot_col),("Sire",sire_col),("Dam",dam_col),("Vendor",vendor_col)]:
        out[label] = df[col] if col else None

    return out

def apply_filters(df: pd.DataFrame,
                  ages: List[str], sexes: List[str], states: List[str],
                  maiden: str, bm_max_text: str) -> pd.DataFrame:
    f = df.copy()

    # Age
    if ages and "Any" not in ages:
        f = f[f["Age"].astype("Int64").astype(str).isin(ages)]

    # Sex
    if sexes and "Any" not in sexes:
        f = f[f["Sex"].isin(sexes)]

    # State
    if states:
        f = f[f["State"].isin(states)]

    # Maiden
    if maiden != "Any":
        f = f[f["Maiden"] == (maiden == "Yes")]

    # Benchmark max
    bm_max = None
    if bm_max_text.strip() != "":
        try:
            bm_max = float(bm_max_text.strip())
        except:
            pass
    if bm_max is not None:
        f = f[pd.to_numeric(f["Lowest All Avg Benchmark"], errors="coerce") <= bm_max]

    return f

# ---------------- Input: upload or paste ----------------
st.sidebar.header("🧾 Load horses")
src = st.sidebar.radio("Source", ["Upload CSV/Excel", "Paste"], index=0, key="src_mode")

raw_df = None
if src == "Upload CSV/Excel":
    up = st.sidebar.file_uploader("Upload file", type=["csv","xlsx"], key="file_uploader")
    if up is not None:
        try:
            if up.name.lower().endswith(".xlsx"):
                raw_df = pd.read_excel(up)
            else:
                try:
                    raw_df = pd.read_csv(up, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
                except UnicodeDecodeError:
                    raw_df = pd.read_csv(up, sep=None, engine="python", encoding="ISO-8859-1", on_bad_lines="skip")
        except Exception as e:
            st.sidebar.error(f"Read error: {e}")
else:
    pasted = st.sidebar.text_area("One horse per line", height=160, placeholder="Hell Island\nInvincible Phantom\nIrish Bliss")
    if pasted.strip():
        names = [n.strip() for n in pasted.splitlines() if n.strip()]
        raw_df = pd.DataFrame({"Name": names})

if raw_df is None or raw_df.empty:
    st.info("Upload a file or paste names in the left sidebar to begin.")
    st.stop()

# ---------------- Map the name column ----------------
clean = clean_headers(raw_df)
auto_name = None
for pat in HEADER_ALIASES["Name"]:
    found = find_first_match(list(clean.columns), [pat])
    if found:
        auto_name = found
        break

name_col = st.selectbox(
    "Select the column that contains the horse name:",
    options=list(clean.columns),
    index=(list(clean.columns).index(auto_name) if (auto_name in clean.columns) else 0),
)

canon = canonize(clean, name_col)

# ---------------- Filters (ALWAYS LIVE) ----------------
st.sidebar.header("🔎 Filters (live)")
age_opts = ["Any"] + [str(i) for i in range(1, 11)]
f_age  = st.sidebar.multiselect("Age", age_opts, default=["Any"])
sexes  = ["Any", "Gelding", "Mare", "Horse", "Colt", "Filly"]
f_sex  = st.sidebar.multiselect("Sex", sexes, default=["Any"])
states = sorted(canon["State"].dropna().astype(str).unique().tolist())
f_state = st.sidebar.multiselect("State", states, default=[])
f_maiden = st.sidebar.selectbox("Maiden", ["Any","Yes","No"], index=0)
f_bmmax = st.sidebar.text_input("Lowest achieved All Avg Benchmark (max)", value="", placeholder="e.g. -2.0")

filtered = apply_filters(canon, f_age, f_sex, f_state, f_maiden, f_bmmax)

# ---------------- Filtered table (ALWAYS SHOWN) ----------------
st.subheader("📋 Filtered Sale Horses")
if filtered.empty:
    st.warning("No rows match the current filters.")
else:
    to_show = [c for c in ["Lot","Name","Age","Sex","State","Wins","Maiden","Bid","Lowest All Avg Benchmark","Sire","Dam","Vendor"] if c in filtered.columns]
    st.dataframe(filtered[to_show].reset_index(drop=True), use_container_width=True)

# ---------------- Choose horse ----------------
names_list = ["(none)"] + sorted(filtered["Name"].dropna().astype(str).unique().tolist())
chosen = st.selectbox("Pick a horse from the filtered list", names_list, index=0)
st.markdown("---")

# ---------------- Selected Horse panel ----------------
st.subheader("🎯 Selected Horse")
if chosen == "(none)":
    st.info("Select a horse from the filtered list above.")
else:
    row = filtered[filtered["Name"].astype(str)==chosen]
    d = row.iloc[0].to_dict() if not row.empty else {"Name": chosen}

    # PF tester (simple)
    st.markdown("##### 📡 Punting Form (quick test)")
    if not PF_AVAILABLE:
        st.info("pf_client.py not found — PF calls disabled here. UI still works.")
    else:
        if pf_is_live():
            st.success("PF client connected.")
        else:
            st.warning("PF client imported, but not live (check API key/paths).")

    with st.expander("Run a PF GET (optional)"):
        endpoint = st.selectbox("Endpoint", ["Ratings/MeetingBenchmarks","Ratings/MeetingSectionals","Ratings/MeetingRatings"])
        meeting_id = st.text_input("meetingId (integer)", value="")
        if st.button("Run PF GET"):
            if not PF_AVAILABLE:
                st.error("PF client disabled.")
            else:
                try:
                    if not meeting_id.strip().isdigit():
                        st.warning("Enter a numeric meetingId.")
                    else:
                        r = pf_raw_get("/"+endpoint, params={"meetingId": int(meeting_id)})
                        st.write(f"HTTP {r.status_code}")
                        ct = r.headers.get("content-type","")
                        if ct.startswith("application/json"):
                            st.json(r.json())
                        else:
                            st.text(r.text)
                except Exception as e:
                    st.error(f"PF test failed: {e}")

    st.markdown(f"### {d.get('Name', chosen)}")
    fields = ["Lot","Age","Sex","State","Wins","Maiden","Bid","Lowest All Avg Benchmark","Sire","Dam","Vendor"]
    pairs = [(k, d[k]) for k in fields if k in d and pd.notna(d[k]) and str(d[k]).strip() != ""]
    if pairs:
        st.table(pd.DataFrame(pairs, columns=["Field","Value"]))
    else:
        st.caption("No extra fields present in the file.")

st.caption("© Soar Bloodstock — MoneyBall")
