import os
import re
import streamlit as st
import pandas as pd
from datetime import date

# Try to use rapidfuzz if available; fall back gracefully if not
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

from pf_client import (
    is_live,
    search_horse_by_name,
    get_form,
    get_ratings,
    get_speedmap,
    get_sectionals_csv,
    get_benchmarks_csv,
)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Soar Bloodstock Data - MoneyBall", layout="wide")
st.title("Soar Bloodstock Data - MoneyBall")

LIVE = is_live()
st.sidebar.success("✅ Live Mode (PF API)" if LIVE else "💤 Demo Mode (no API key)")

# =========================================================
# 🔐 Quick connection test (optional but handy)
# =========================================================
st.sidebar.markdown("---")
if st.sidebar.button("🔐 Test PF connection"):
    base_url = os.getenv("PF_BASE_URL", st.secrets.get("PF_BASE_URL", "(missing)"))
    key_present = bool(os.getenv("PF_API_KEY", st.secrets.get("PF_API_KEY", "")))
    st.write("PF_BASE_URL:", base_url)
    st.write("PF_API_KEY present:", key_present)
    st.info("If present is True but you still get 401, adjust PF_AUTH_HEADER / PF_AUTH_PREFIX in Secrets.")

# =========================================================
# 1) INPUT — Paste or Upload
# =========================================================
with st.sidebar:
    st.header("🧾 Horse list input")
    pasted = st.text_area(
        "Paste horses (one per line):",
        height=180,
        placeholder="Hell Island\nInvincible Phantom\nIrish Bliss\nLittle Spark",
    )
    file = st.file_uploader("…or upload CSV/Excel (optional)", type=["csv", "xlsx"])

# =========================================================
# 2) LOAD DATA (robust)
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
    for c in cols:  # fallback: anything containing "name"
        if "name" in c.lower():
            return c
    return None

sale_df = None

if file is not None:
    try:
        if file.name.lower().endswith(".xlsx"):
            tmp = pd.read_excel(file)
        else:
            # tolerate messy CSVs and encodings
            try:
                tmp = pd.read_csv(file, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                tmp = pd.read_csv(file, sep=None, engine="python", encoding="ISO-8859-1", on_bad_lines="skip")
        sale_df = clean_headers(tmp)
    except Exception as e:
        st.error(f"❌ Could not read uploaded file: {e}")

# Fallback to pasted names if no file or read failed
if sale_df is None:
    names = [n.strip() for n in pasted.splitlines() if n.strip()]
    sale_df = pd.DataFrame({"Name": names})

# =========================================================
# 3) SIDEBAR FILTERS + HORSE SELECT
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

# Show any auction fields if present
row = sale_df[sale_df[name_col].astype(str) == str(horse_name)]
if not row.empty:
    r = row.iloc[0].to_dict()

    def show(label, key):
        if key in r and pd.notnull(r[key]) and str(r[key]).strip():
            st.write(f"**{label}:**", r[key])

    for label, key in [
        ("Lot", "Lot"),
        ("Age", "Age"),
        ("Sex", "Sex"),
        ("Sire", "Sire"),
        ("Dam", "Dam"),
        ("Vendor", "Vendor"),
        ("Bid", "Bid"),
    ]:
        show(label, key)

# =========================================================
# 4) CONNECT TO PUNTING FORM (button triggers)
# =========================================================
if st.button("🔍 View Punting Form Data"):
    with st.spinner(f"Fetching Punting Form data for {horse_name}..."):
        try:
            ident = search_horse_by_name(horse_name)
            st.success(f"Found: {ident.get('display_name', horse_name)}")

            # NOTE: Depending on PF's API, you may need meeting_id/race_id instead of horse_id
            horse_id = ident.get("horse_id")

            form_data = get_form(horse_id)
            ratings = get_ratings(horse_id)
            speedmap = get_speedmap(horse_id)

            with st.expander("📄 Form Summary"):
                st.json(form_data)
            with st.expander("📊 Ratings"):
                st.json(ratings)
            with st.expander("🏃 Speedmap"):
                st.json(speedmap)

        except Exception as e:
            st.error(f"Could not retrieve data: {e}")

# =========================================================
# 5) DEMO DATA + FILTERING (only used in Demo Mode)
# =========================================================
st.markdown("---")
st.subheader("Optional: shortlist builder (Demo Mode)")

names_text = st.text_area(
    "Horse list (optional, for demo mode):",
    height=220,
    placeholder="Eleanor Nancy\nFast Intentions\nSir Goldalot\nLittle Spark",
)
names = [n.strip() for n in names_text.splitlines() if n.strip()]
unique_names = sorted(set(names)) if names else []

@st.cache_data
def load_demo_db():
    # Make demo optional so the app doesn't crash if file is missing
    try:
        df = pd.read_csv("data/puntingform_demo.csv")
        df["name_std"] = (
            df["horse_name"].str.upper()
            .str.replace(r"[^A-Z0-9 ]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return df
    except Exception:
        return None

DEMO = None if LIVE else load_demo_db()

def std_name(x: str) -> str:
    return (x or "").upper().strip()

def demo_fuzzy_lookup(name: str):
    if DEMO is None:
        return None, 0
    target = std_name(name)
    choices = DEMO["name_std"].tolist()
    if HAS_RAPIDFUZZ:
        best = process.extractOne(target, choices, scorer=fuzz.WRatio, score_cutoff=70)
        if not best:
            return None, 0
        row = DEMO[DEMO["name_std"] == best[0]].iloc[0].to_dict()
        return row, int(best[1])
    else:
        # Simple fallback: exact/startswith matching
        m = DEMO[DEMO["name_std"].str.startswith(target)]
        if m.empty:
            m = DEMO[DEMO["name_std"] == target]
        if m.empty:
            return None, 0
        return m.iloc[0].to_dict(), 100

@st.cache_data(show_spinner=False)
def prefetch_summary(names_tuple):
    rows = []
    for n in names_tuple:
        if not n:
            continue
        if LIVE:
            ident = search_horse_by_name(n)
            out = {
                "display_name": n,
                "_found": bool(ident.get("found", False)),
                "_match_score": 100,
                "avg_benchmark_all": None,
                "last3_L600": None,
                "starts": None,
                "wins": None,
                "sex": None,
                "maiden": None,
                "yob": None,
            }
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
                    "avg_benchmark_all": d.get("avg_benchmark_all"),
                    "last3_L600": d.get("last3_L600"),
                    "last3_L400": d.get("last3_L400"),
                    "last3_L200": d.get("last3_L200"),
                    "starts": d.get("starts"),
                    "wins": d.get("wins"),
                    "trainer": d.get("trainer"),
                    "sp_trend": d.get("sp_trend"),
                }
            else:
                out = {"display_name": n, "_found": False, "_match_score": 0}
        rows.append(out)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

df = prefetch_summary(tuple(unique_names))

def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return df_in
    out = df_in.copy()
    if "yob" in out.columns and out["yob"].notnull().any():
        this_year = date.today().year
        out["age"] = this_year - out["yob"]
        out = out[out["age"] == age]
    if sex != "Any" and "sex" in out.columns:
        out = out[out["sex"].str.capitalize() == sex]
    if maiden != "Any" and "maiden" in out.columns:
        want = (maiden == "Yes")
        out = out[out["maiden"] == want]
    if "avg_benchmark_all" in out.columns and out["avg_benchmark_all"].notnull().any():
        out = out[out["avg_benchmark_all"] <= bm_cut]
    return out

left, right = st.columns([1, 1])

with left:
    st.subheader("Filtered horses")
    filtered = apply_filters(df) if not df.empty else df
    if filtered is None or filtered.empty:
        st.info("No horses match filters yet. Paste names or relax filters.")
        choice = None
    else:
        show_cols = [
            "display_name",
            "avg_benchmark_all",
            "last3_L600",
            "starts",
            "wins",
            "sex",
            "maiden",
        ]
        show_cols = [c for c in show_cols if c in filtered.columns]

        st.dataframe(
            filtered[show_cols].reset_index(drop=True),
            use_container_width=True,
        )

        choice = st.selectbox(
            "Select a horse for full Punting Form data",
            filtered["display_name"].tolist(),
        )

        st.download_button(
            "Export shortlist (CSV)",
            data=filtered.to_csv(index=False),
            file_name="shortlist.csv",
            mime="text/csv",
        )

with right:
    st.subheader("Punting Form data")
    if df is None or df.empty or filtered is None or filtered.empty or not choice:
        st.caption("Paste some demo names on the left and apply filters to view details here.")
    else:
        if LIVE:
            st.warning("Live mode needs meeting/race mapping to fetch full data via PF endpoints.")
            st.code(
                "# Example usage once you have IDs\n"
                "form = get_form(meeting_id='MEETING_ID')\n"
                "ratings = get_ratings(meeting_id='MEETING_ID')\n"
                "speedmap = get_speedmap(race_id='RACE_ID')\n"
                "sectionals = get_sectionals_csv(meeting_id='MEETING_ID')\n"
                "benchmarks = get_benchmarks_csv(meeting_id='MEETING_ID')\n"
            )
        else:
            row = filtered[filtered["display_name"] == choice].iloc[0].to_dict()
            st.markdown(
                f"**{choice}** — Avg Benchmark (All): {row.get('avg_benchmark_all')} "
                f"| L600 (Last 3): {row.get('last3_L600')}"
            )
            tabs = st.tabs(["Overview", "Form", "Sectionals", "Benchmarks", "Ratings", "Speedmap"])
            with tabs[0]:
                k = {
                    "Horse": row.get("horse_name", choice),
                    "Age": (date.today().year - row["yob"]) if row.get("yob") else None,
                    "Sex": row.get("sex"),
                    "Maiden": row.get("maiden"),
                    "Starts": row.get("starts"),
                    "Wins": row.get("wins"),
                    "Trainer": row.get("trainer"),
                    "SP Trend": row.get("sp_trend"),
                    "All Avg Benchmark": row.get("avg_benchmark_all"),
                    "L600 (Last 3)": row.get("last3_L600"),
                    "L400 (Last 3)": row.get("last3_L400"),
                    "L200 (Last 3)": row.get("last3_L200"),
                }
                st.table(pd.DataFrame(k.items(), columns=["Metric", "Value"]))
            with tabs[1]:
                st.caption("Demo: add recent runs here when wired to PF Form.")
            with tabs[2]:
                st.caption("Demo: show sectionals table here when wired to MeetingSectionals CSV.")
            with tabs[3]:
                st.caption("Demo: show benchmarks table here when wired to MeetingBenchmarks CSV.")
            with tabs[4]:
                st.caption("Demo: show ratings table here when wired to MeetingRatings.")
            with tabs[5]:
                st.caption("Demo: render speedmap data here when wired to User/Speedmaps.")
