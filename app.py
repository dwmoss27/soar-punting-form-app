
import os, json
import streamlit as st
import pandas as pd
# Load Inglis sale data
SALE_DATA_PATH = "inglis_sale.csv"

# Try UTF-8 first, fallback to ISO-8859-1 if needed
try:
    sale_df = pd.read_csv(SALE_DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    sale_df = pd.read_csv(SALE_DATA_PATH, encoding="ISO-8859-1")


from datetime import date
from rapidfuzz import process, fuzz

from pf_client import is_live, search_horse_by_name, get_form, get_ratings, get_speedmap, get_sectionals_csv, get_benchmarks_csv

st.set_page_config(page_title="Soar — PF Click-to-Load (Live-ready)", layout="wide")
st.title("Soar — Punting Form Click-to-Load")

LIVE = is_live()
st.sidebar.success("Live Mode (PF API)" if LIVE else "Demo Mode (no API key)")

with st.sidebar:
    st.header("Filters")
    age = st.number_input("Age (years)", min_value=2, max_value=12, value=3)
    sex = st.selectbox("Sex", ["Any", "Gelding", "Mare", "Horse", "Colt", "Filly"])
    maiden = st.selectbox("Maiden", ["Any", "Yes", "No"])
    bm_cut = st.number_input("Max All Avg Benchmark", value=5.0, step=0.1)
st.sidebar.header("🧾 Inglis Sale Horses")

horse_name = st.sidebar.selectbox(
    "Select a horse",
    sorted(sale_df["Name"].dropna().unique())
)

st.write(f"### Selected Horse: {horse_name}")
horse_row = sale_df[sale_df["Name"] == horse_name].iloc[0]

st.write("**Lot:**", horse_row["Lot"])
st.write("**Age:**", horse_row["Age"])
st.write("**Sex:**", horse_row["Sex"])
st.write("**Sire:**", horse_row["Sire"])
st.write("**Dam:**", horse_row["Dam"])
st.write("**Vendor:**", horse_row["Vendor"])
st.write("**Current Bid:**", horse_row["Bid"])

st.write("Paste **one horse per line** (copied from Inglis list):")
horse_name = st.sidebar.selectbox(
    "Select a horse",
    sorted(sale_df["Name"].dropna().unique())
)

st.write(f"### Selected Horse: {horse_name}")
# 🔍 Connect to Punting Form API
from pf_client import search_horse_by_name, get_form, get_ratings, get_speedmap

if st.button("🔍 View Punting Form Data"):
    with st.spinner(f"Fetching Punting Form data for {horse_name}..."):
        try:
            # Step 1: Search the horse on Punting Form
            result = search_horse_by_name(horse_name)
            st.success(f"Found: {result.get('display_name', horse_name)}")

            # Step 2: Retrieve detailed form and ratings (if available)
            form_data = get_form(result.get("horse_id"))
            ratings = get_ratings(result.get("horse_id"))
            speedmap = get_speedmap(result.get("horse_id"))

            # Step 3: Display in expandable sections
            with st.expander("📄 Form Summary"):
                st.json(form_data)
            with st.expander("📊 Ratings"):
                st.json(ratings)
            with st.expander("🏃 Speedmap"):
                st.json(speedmap)

        except Exception as e:
            st.error(f"Could not retrieve data: {e}")

names_text = st.text_area("Horse list", height=220, placeholder="Eleanor Nancy\nFast Intentions\nSir Goldalot\nLittle Spark")

names = [n.strip() for n in names_text.splitlines() if n.strip()]
unique_names = sorted(set(names)) if names else []

@st.cache_data
def load_demo_db():
    df = pd.read_csv("data/puntingform_demo.csv")
    df["name_std"] = (df["horse_name"].str.upper()
                      .str.replace(r"[^A-Z0-9 ]","", regex=True)
                      .str.replace(r"\s+"," ", regex=True).str.strip())
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
    return pd.DataFrame(rows)

df = prefetch_summary(tuple(unique_names))

def apply_filters(df):
    if df.empty: return df
    out = df.copy()
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

left, right = st.columns([1,1])

with left:
    st.subheader("Filtered horses")
    filtered = apply_filters(df) if not df.empty else df
    if filtered.empty:
        st.info("No horses match filters yet. Paste names and/or relax filters.")
    else:
        show_cols = ["display_name","avg_benchmark_all","last3_L600","starts","wins","sex","maiden"]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[show_cols].reset_index(drop=True), use_container_width=True)
        choice = st.selectbox("Select a horse for full Punting Form data", filtered["display_name"].tolist())
        st.download_button("Export shortlist (CSV)",
            data=filtered.to_csv(index=False), file_name="shortlist.csv", mime="text/csv")

with right:
    st.subheader("Punting Form data")
    if df.empty or filtered.empty:
        st.stop()
    if not choice:
        st.stop()

    if LIVE:
        st.warning("Live mode needs mapping from horse → meeting/race to fetch Form/Ratings/Sectionals/Benchmarks. Configure in pf_client & your workflow.")
        st.caption("Once you have meetingId/raceId for the selected horse, call the functions below and render the tabs.")
        st.code(
            "# Example usage (once you have ids):\n"
            "form = get_form(meeting_id='MEETING_ID')\n"
            "ratings = get_ratings(meeting_id='MEETING_ID')\n"
            "speedmap = get_speedmap(race_id='RACE_ID')\n"
            "sectionals = get_sectionals_csv(meeting_id='MEETING_ID')\n"
            "benchmarks = get_benchmarks_csv(meeting_id='MEETING_ID')\n"
        )
    else:
        row = filtered[filtered["display_name"] == choice].iloc[0].to_dict()
        st.markdown(f"**{choice}** — Avg Benchmark (All): {row.get('avg_benchmark_all')} | L600 (Last 3): {row.get('last3_L600')}")
        tabs = st.tabs(["Overview","Form","Sectionals","Benchmarks","Ratings","Speedmap"])
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
            st.table(pd.DataFrame(k.items(), columns=["Metric","Value"]))
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
