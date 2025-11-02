import streamlit as st
import pandas as pd
import re
from datetime import date
from fuzzywuzzy import fuzz, process

from pf_client import (
    is_live, search_horse_by_name, get_form,
    get_ratings, get_speedmap, get_sectionals_csv, get_benchmarks_csv
)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Soar — PF Click-to-Load", layout="wide")
st.title("Soar — Punting Form Click-to-Load")

LIVE = is_live()
st.sidebar.success("✅ Live Mode (PF API)" if LIVE else "💤 Demo Mode (no API key)")

# =========================================================
# 1️⃣ INPUT — Paste or Upload
# =========================================================
with st.sidebar:
    st.header("🧾 Horse list input")
    pasted = st.text_area(
        "Paste horses (one per line):",
        height=180,
        placeholder="Hell Island\nInvincible Phantom\nIrish Bliss\nLittle Spark"
    )
    file = st.file_uploader("…or upload CSV/Excel (optional)", type=["csv", "xlsx"])

# =========================================================
# 2️⃣ LOAD DATA
# =========================================================
def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    clean = {}
    for c in df.columns:
        x = str(c).replace("\ufeff", "").strip()
        x = re.sub(r"\s+", " ", x)
        clean[c] = x
    return df.rename(columns=clean)

def detect_name_col(cols) -> str | None:
    norm = {re.sub(r"\s+", "", c).lower(): c for c in cols}
    for cand in ["name", "horse", "horse name", "horsename", "lot name"]:
        key = re.sub(r"\s+", "", cand).lower()
        if key in norm:
            return norm[key]
    for c in cols:
        if "name" in c.lower():
            return c
    return None

sale_df = None

if file is not None:
    try:
        if file.name.lower().endswith(".xlsx"):
            tmp = pd.read_excel(file)
        else:
            try:
                tmp = pd.read_csv(file, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                tmp = pd.read_csv(file, sep=None, engine="python", encoding="ISO-8859-1", on_bad_lines="skip")
        sale_df = clean_headers(tmp)
    except Exception as e:
        st.error(f"❌ Could not read uploaded file: {e}")

# fallback to pasted names
if sale_df is None:
    names = [n.strip() for n in pasted.splitlines() if n.strip()]
    sale_df = pd.DataFrame({"Name": names})

# =========================================================
# 3️⃣ SIDEBAR FILTERS + HORSE SELECT
# =========================================================
name_col = detect_name_col(list(sale_df.columns))
if not name_col:
    st.error("No 'Name' column found and no pasted names. Paste names (one per line) or upload a file.")
    st.stop()

with st.sidebar:
    st.header("🔍 Filters")
    age = st.number_input("Age (years)", min_value=2, max_value=12, value=3)
    sex = st.selectbox("Sex", ["Any", "Gelding", "Mare", "Horse", "Colt", "Filly"])
    maiden = st.selectbox("Maiden", ["Any", "Yes", "No"])
    bm_cut = st.number_input("Max All Avg Benchmark", value=5.0, step=0.1)

    st.header("🐎 Select a horse")
    horse_name = st.selectbox(
        "Horse",
        sorted(sale_df[name_col].dropna().astype(str).unique())
    )

st.write(f"### Selected Horse: {horse_name}")

row = sale_df[sale_df[name_col].astype(str) == str(horse_name)]
if not row.empty:
    r = row.iloc[0].to_dict()
    def show(label, key):
        if key in r and pd.notnull(r[key]) and str(r[key]).strip():
            st.write(f"**{label}:**", r[key])
    for label, key in [
        ("Lot", "Lot"), ("Age", "Age"), ("Sex", "Sex"),
        ("Sire", "Sire"), ("Dam", "Dam"), ("Vendor", "Vendor"), ("Bid", "Bid")
    ]:
        show(label, key)

# =========================================================
# 4️⃣ CONNECT TO PUNTING FORM
# =========================================================
if st.button("🔍 View Punting Form Data"):
    with st.spinner(f"Fetching Punting Form data for {horse_name}..."):
        try:
            result = search_horse_by_name(horse_name)
            st.success(f"Found: {result.get('display_name', horse_name)}")

            form_data = get_form(result.get("horse_id"))
            ratings = get_ratings(result.get("horse_id"))
            speedmap = get_speedmap(result.get("horse_id"))

            with st.expander("📄 Form Summary"):
                st.json(form_data)
            with st.expander("📊 Ratings"):
                st.json(ratings)
            with st.expander("🏃 Speedmap"):
                st.json(speedmap)

        except Exception as e:
            st.error(f"Could not retrieve data: {e}")

# =========================================================
# 5️⃣ DEMO DATA + FILTERING
# =========================================================
names_text = st.text_area(
    "Horse list (optional, for demo mode):",
    height=220,
    placeholder="Eleanor Nancy\nFast Intentions\nSir Goldalot\nLittle Spark"
)

names = [n.strip() for n in names_text.splitlines() if n.strip()]
unique_names = sorted(set(names)) if names else []

@st.cache_data
def load_demo_db():
    df = pd.read_csv("data/puntingform_demo.csv")
    df["name_std"] = (
        df["horse_name"].str.upper()
        .str.replace(r"[^A-Z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df

DEMO = None if LIVE else load_demo_db()

def std_name(x: str) -> str:
    return (x or "").upper().strip()

def demo_fuzzy_lookup(name: str):
    target = std_name(name)
    choices = DEMO["name_std"].tolist()
    best = process.extractOne(target, choices, scorer=fuzz.WRatio, score_cutoff=70)
    if not best:
        return None, 0
    row = DEMO[DEMO["name_std"] == best[0]].iloc[0].to_dict()
    return row, best[1]

@st.cache_data(show_spinner=False)
def prefetch_summary(names_tuple):
    rows = []
    for n in names_tuple:
        if not n:
            continue
        if LIVE:
            ident = search_horse_by_name(n)
            out = {"display_name": n, "_found": ident.get("found", False), "_match_score": 100}
            out.update({
                "avg_benchmark_all": None,
                "last3_L600": None,
                "starts": None,
                "wins": None,
                "sex": None,
                "maiden": None,
                "yob": None
            })
        else:
            d, score = demo_fuzzy_lookup(n)
            if d:
                out = {
                    "display_name": n,
                    "_found": True,
                    "_match_score": score,
                    "horse_name": d.get("horse_name"),
                    "yob": d.get("yob"),
                    "sex": d.get("sex"),
                    "maiden": d.get("maiden"),
                    "avg_benchmar_
