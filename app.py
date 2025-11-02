import os
import io
import re
import json
from datetime import date

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# Optional (better fuzzy): rapidfuzz
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

# ────────────────────────────────────────────────────────────────────────────────
# PuntingForm client (you already have this file)
# ────────────────────────────────────────────────────────────────────────────────
from pf_client import (
    is_live,
    search_horse_by_name,
    get_form,
    get_ratings,
    get_speedmap,
    get_sectionals_csv,
    get_benchmarks_csv,
)

# ────────────────────────────────────────────────────────────────────────────────
# Page + Logo
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Soar Bloodstock Data - MoneyBall", layout="wide")

def save_logo(file):
    os.makedirs("assets", exist_ok=True)
    with open("assets/logo.png", "wb") as f:
        f.write(file.getbuffer())

def render_logo():
    # Priority: secrets URL → saved file → none
    try:
        logo_url = st.secrets.get("LOGO_URL", None)
    except Exception:
        logo_url = None

    if logo_url:
        st.markdown(
            f"<div style='text-align:center;margin-top:6px'><img src='{logo_url}' width='240'></div>",
            unsafe_allow_html=True,
        )
    elif os.path.exists("assets/logo.png"):
        # Streamlit can serve local file paths
        st.markdown(
            "<div style='text-align:center;margin-top:6px'>"
            "<img src='assets/logo.png' width='240'>"
            "</div>",
            unsafe_allow_html=True,
        )

render_logo()
st.title("Soar Bloodstock Data - MoneyBall")
LIVE = is_live()
st.sidebar.success("✅ Live Mode (PF API)" if LIVE else "💤 Demo Mode (no API key)")

# ────────────────────────────────────────────────────────────────────────────────
# Simple persistence for horse lists (JSON file)
# ────────────────────────────────────────────────────────────────────────────────
CACHE_FILE = "saved_horses.json"

def save_to_cache(df: pd.DataFrame, filename: str = CACHE_FILE):
    try:
        # Store a minimal subset for portability
        df.to_json(filename, orient="records", force_ascii=False)
    except Exception as e:
        st.sidebar.warning(f"Could not save cache: {e}")

def load_from_cache(filename: str = CACHE_FILE) -> pd.DataFrame | None:
    try:
        if os.path.exists(filename):
            return pd.read_json(filename)
    except Exception:
        return None
    return None

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar — inputs and controls
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧾 Horse list input")
    pasted = st.text_area(
        "Paste horses (one per line):",
        height=140,
        placeholder="Hell Island\nInvincible Phantom\nIrish Bliss\nLittle Spark",
        key="paste_box",
    )
    file = st.file_uploader("…or upload CSV/Excel", type=["csv", "xlsx"], key="sale_upload")

    st.markdown("**Or fetch directly from an Inglis sale page**")
    inglis_url = st.text_input("Inglis Page URL (optional)")
    fetch_btn = st.button("🌐 Fetch from page", use_container_width=True, key="fetch_inglis")

    st.header("🖼️ Logo Settings")
    uploaded_logo = st.file_uploader("Upload Logo (PNG/JPG)", type=["png", "jpg", "jpeg"], key="logo_up")
    if uploaded_logo is not None:
        st.image(uploaded_logo, width=200)
        if st.button("💾 Save Logo", use_container_width=True):
            save_logo(uploaded_logo)
            st.success("✅ Logo saved — it will appear at the top center.")

# ────────────────────────────────────────────────────────────────────────────────
# Helpers — header cleaning, name detection
# ────────────────────────────────────────────────────────────────────────────────
def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    clean = {}
    for c in df.columns:
        x = str(c).replace("\ufeff", "")
        x = re.sub(r"\s+", " ", x).strip()
        clean[c] = x
    return df.rename(columns=clean)

def detect_name_col(cols) -> str | None:
    norm = {re.sub(r"\s+", "", str(c)).lower(): c for c in cols}
    for cand in ["name", "horse", "horse name", "horsename", "lot name"]:
        key = re.sub(r"\s+", "", cand).lower()
        if key in norm:
            return norm[key]
    for c in cols:
        if "name" in str(c).lower():
            return c
    return None

def _looks_like_sale_table(df: pd.DataFrame) -> bool:
    cols = [str(c).lower() for c in df.columns]
    must_have_any = ["name", "horse"]
    return any(any(m in c for m in must_have_any) for c in cols) and len(cols) >= 3

# ────────────────────────────────────────────────────────────────────────────────
# Static Inglis table scraper (fast path)
# ────────────────────────────────────────────────────────────────────────────────
def fetch_inglis_table(url: str) -> pd.DataFrame | None:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    # 1) Try read_html directly
    try:
        tables = pd.read_html(url, flavor="lxml")
        candidates = [clean_headers(t) for t in tables if _looks_like_sale_table(clean_headers(t))]
        if candidates:
            return max(candidates, key=lambda d: len(d))
    except Exception:
        pass

    # 2) requests + bs4 + read_html on HTML
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return None
        html = resp.text
        try:
            tables = pd.read_html(html)
            candidates = [clean_headers(t) for t in tables if _looks_like_sale_table(clean_headers(t))]
            if candidates:
                return max(candidates, key=lambda d: len(d))
        except Exception:
            pass

        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if table:
            df = pd.read_html(str(table))[0]
            df = clean_headers(df)
            if _looks_like_sale_table(df):
                return df
    except Exception:
        pass

    return None

# ────────────────────────────────────────────────────────────────────────────────
# Optional Playwright fallback (dynamic JS pages) — only if installed
# ────────────────────────────────────────────────────────────────────────────────
def fetch_inglis_dynamic(url: str) -> pd.DataFrame | None:
    """
    Uses Playwright (headless Chromium) to render pages that build tables via JS.
    Works locally or on private servers where Playwright is installed.
    """
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        # Not installed: fail quietly
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=60000)
            html = page.content()
            browser.close()
        tables = pd.read_html(html)
        if not tables:
            return None
        df = max(tables, key=lambda t: len(t))
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.warning(f"Playwright scraper failed: {e}")
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Build sale_df (load cache → upload → fetch URL → paste)
# ────────────────────────────────────────────────────────────────────────────────
sale_df = None
file_error = None

# Try to load previous saved horses
if "horse_cache_loaded" not in st.session_state:
    cached = load_from_cache()
    if cached is not None and not cached.empty:
        sale_df = cached.copy()
        st.sidebar.success(f"✅ Loaded {len(sale_df)} horses from previous session.")
    st.session_state["horse_cache_loaded"] = True

# 1) Uploaded file takes priority
if file is not None:
    try:
        if file.name.lower().endswith(".xlsx"):
            try:
                tmp = pd.read_excel(file)
            except Exception as e:
                raise RuntimeError(f"Excel requires 'openpyxl': {e}")
        else:
            try:
                tmp = pd.read_csv(file, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                tmp = pd.read_csv(file, sep=None, engine="python", encoding="ISO-8859-1", on_bad_lines="skip")
        sale_df = clean_headers(tmp)
        save_to_cache(sale_df)
        st.sidebar.success("✅ Uploaded list saved for next session.")
    except Exception as e:
        file_error = f"❌ Could not read uploaded file: {e}"

# 2) Fetch from Inglis page (merges with any existing)
if fetch_btn and inglis_url:
    with st.spinner("Fetching Inglis page…"):
        url_df = fetch_inglis_table(inglis_url)
        if url_df is None:
            st.warning("Static scrape failed — trying Playwright (headless browser)…")
            url_df = fetch_inglis_dynamic(inglis_url)
        if url_df is None:
            st.error("Couldn’t parse that page. If it’s JavaScript-only, use the page’s CSV export or paste names.")
        else:
            st.success(f"Fetched {len(url_df)} rows from Inglis page.")
            url_df = clean_headers(url_df)
            if sale_df is None:
                sale_df = url_df.copy()
            else:
                sale_df = pd.concat([sale_df, url_df], ignore_index=True).drop_duplicates()
            save_to_cache(sale_df)

# 3) Fallback: pasted names
if sale_df is None:
    names = [n.strip() for n in pasted.splitlines() if n.strip()]
    sale_df = pd.DataFrame({"Name": names})

if file_error:
    st.error(file_error)

# Manual Save button for the current list
with st.sidebar:
    if st.button("💾 Save Horses", use_container_width=True):
        save_to_cache(sale_df)
        st.success("Saved current horses list to cache.")

# ────────────────────────────────────────────────────────────────────────────────
# Filters (Age multi, Sex multi, Benchmark threshold)
# ────────────────────────────────────────────────────────────────────────────────
name_col = detect_name_col(list(sale_df.columns))
if not name_col:
    st.error("No 'Name' column found and no pasted names. Paste names, upload a file, or fetch a page.")
    st.stop()

with st.sidebar:
    st.header("🔍 Filters")

    # Age: Any + 1..10
    age_options = ["Any"] + list(range(1, 11))
    sel_ages = st.multiselect(
        "Age (select one or many)",
        age_options,
        default=["Any"],
        help="Choose 'Any' or one/more exact ages.",
    )

    # Sex: multi
    sex_options = ["Gelding", "Mare", "Horse", "Colt", "Filly"]
    sel_sex = st.multiselect(
        "Sex (multi-select)",
        sex_options,
        default=[],
        help="Leave empty for any sex, or select one/more.",
    )

    # Benchmark threshold: max of “lowest achieved”
    bm_cut = st.number_input(
        "Max 'Lowest All Avg Benchmark'",
        value=5.0,
        step=0.1,
        help="Filters on each horse’s lowest achieved All Avg Benchmark (computed via PF).",
    )

# ────────────────────────────────────────────────────────────────────────────────
# Show selected horse basics (if present in sale data)
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
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
        ("Lot", "Lot"),
        ("Age", "Age"),
        ("Sex", "Sex"),
        ("Sire", "Sire"),
        ("Dam", "Dam"),
        ("Vendor", "Vendor"),
        ("Bid", "Bid"),
        ("State", "State"),
        ("Purchaser", "Purchaser"),
    ]:
        show(label, key)

# ────────────────────────────────────────────────────────────────────────────────
# Demo DB (optional) for matching
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_demo_db():
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
        m = DEMO[DEMO["name_std"].str.startswith(target)]
        if m.empty:
            m = DEMO[DEMO["name_std"] == target]
        if m.empty:
            return None, 0
        return m.iloc[0].to_dict(), 100

# ────────────────────────────────────────────────────────────────────────────────
# PF Enrichment: Lowest All Avg Benchmark per horse (best-effort)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_lowest_benchmarks(names: list[str]) -> pd.DataFrame:
    """
    Returns DataFrame(display_name, lowest_all_avg_benchmark).
    LIVE: tries PF endpoints (best-effort; depends on plan/endpoints).
    DEMO: falls back to demo CSV avg_benchmark_all.
    """
    rows = []
    for n in names:
        if not n:
            continue

        out_row = {"display_name": n, "lowest_all_avg_benchmark": None}

        if LIVE:
            try:
                ident = search_horse_by_name(n)
                horse_id = ident.get("horse_id")
                form_payload = get_form(horse_id)

                meetings = []
                if isinstance(form_payload, dict):
                    for k in ("meetings", "data", "runs", "recentRuns", "items"):
                        if k in form_payload and isinstance(form_payload[k], list):
                            for it in form_payload[k]:
                                for mk in ("meeting_id", "meetingId", "MeetingId", "meeting"):
                                    if mk in it:
                                        mid = it[mk] if isinstance(it[mk], (int, str)) else it[mk].get("id")
                                        if mid:
                                            meetings.append(str(mid))

                meetings = list(dict.fromkeys(meetings))  # unique
                vals = []
                for mid in meetings[:10]:  # avoid spamming
                    try:
                        csvtext = get_benchmarks_csv(mid)
                        try:
                            bmdf = pd.read_csv(io.StringIO(csvtext))
                        except Exception:
                            bmdf = pd.read_csv(io.StringIO(csvtext), sep=None, engine="python")
                        name_cand = detect_name_col(list(bmdf.columns)) or "Horse"
                        mask = bmdf[name_cand].astype(str).str.strip().str.upper() == n.strip().upper()
                        sub = bmdf[mask]
                        col_candidates = [c for c in bmdf.columns if "all" in str(c).lower() and "bench" in str(c).lower()]
                        if not sub.empty and col_candidates:
                            for c in col_candidates:
                                vals.extend(pd.to_numeric(sub[c], errors="coerce").dropna().tolist())
                    except Exception:
                        pass

                if vals:
                    out_row["lowest_all_avg_benchmark"] = min(vals)

            except Exception:
                pass

        if not LIVE and DEMO is not None and out_row["lowest_all_avg_benchmark"] is None:
            d, _ = demo_fuzzy_lookup(n)
            if d and "avg_benchmark_all" in d and pd.notnull(d["avg_benchmark_all"]):
                out_row["lowest_all_avg_benchmark"] = float(d["avg_benchmark_all"])

        rows.append(out_row)

    if not rows:
        return pd.DataFrame(columns=["display_name", "lowest_all_avg_benchmark"])
    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────────────
# Shortlist builder + PF enrichment + filters + save shortlist
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Shortlist Builder")

names_text = st.text_area(
    "Horse list (optional — you can paste another list here for filtering):",
    height=160,
    placeholder="Eleanor Nancy\nFast Intentions\nSir Goldalot\nLittle Spark",
    key="shortlist_box",
)
names = [n.strip() for n in names_text.splitlines() if n.strip()]
unique_names = sorted(set(names)) if names else []

colA, colB = st.columns([1, 1])
with colA:
    enrich_clicked = st.button("⚙️ Enrich via PF (compute 'Lowest All Avg Benchmark')", use_container_width=True)
with colB:
    save_clicked = st.button("💾 Save current filtered list", use_container_width=True)

@st.cache_data(show_spinner=False)
def build_filter_table(base_df: pd.DataFrame, pasted_names: list[str]) -> pd.DataFrame:
    # Start from pasted names if present, else from sale_df’s name column
    if pasted_names:
        df = pd.DataFrame({"display_name": pasted_names})
    else:
        df = pd.DataFrame({"display_name": base_df[detect_name_col(list(base_df.columns))].dropna().astype(str).unique().tolist()})

    # Try to bring across Age/Sex from sale_df if present
    base_norm = base_df.copy()
    name_c = detect_name_col(list(base_norm.columns))
    base_norm["__key"] = base_norm[name_c].astype(str).str.upper().str.strip()
    df["__key"] = df["display_name"].astype(str).str.upper().str.strip()
    for col_src, col_out in [("Age", "_age_src"), ("Sex", "_sex_src")]:
        if col_src in base_norm.columns:
            df = df.merge(base_norm[["__key", col_src]].rename(columns={col_src: col_out}), on="__key", how="left")
    return df.drop(columns=["__key"])

working = build_filter_table(sale_df, unique_names)

# Enrich (best effort)
if enrich_clicked and not working.empty:
    with st.spinner("Querying Punting Form for lowest 'All Avg Benchmark'…"):
        bench_df = compute_lowest_benchmarks(working["display_name"].tolist())
        if not bench_df.empty:
            working = working.merge(bench_df, on="display_name", how="left")
        else:
            st.warning("No benchmark data could be computed. Check PF credentials/endpoints or try again.")
else:
    # Try to reuse any cached results silently
    try:
        bench_df = compute_lowest_benchmarks(working["display_name"].tolist())
        working = working.merge(bench_df, on="display_name", how="left")
    except Exception:
        pass

# Apply filters
def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return df_in
    out = df_in.copy()

    # Age filtering (only if not "Any")
    if "Any" not in sel_ages:
        if "_age_src" in out.columns:
            try:
                out["_age_num"] = pd.to_numeric(out["_age_src"], errors="coerce")
                wanted = [a for a in sel_ages if isinstance(a, int)]
                out = out[out["_age_num"].isin(wanted)]
            except Exception:
                pass

    # Sex filtering
    if sel_sex:
        if "_sex_src" in out.columns:
            out["_sex_norm"] = out["_sex_src"].astype(str).str.strip().str.capitalize()
            out = out[out["_sex_norm"].isin(sel_sex)]

    # Benchmark filtering
    if "lowest_all_avg_benchmark" in out.columns and out["lowest_all_avg_benchmark"].notnull().any():
        out = out[pd.to_numeric(out["lowest_all_avg_benchmark"], errors="coerce") <= bm_cut]

    return out

filtered = apply_filters(working)

# UI: filtered table + selection + saved shortlist
left, right = st.columns([1, 1])

with left:
    st.subheader("Filtered horses")
    if filtered is None or filtered.empty:
        st.info("No horses match filters yet. Paste names, Enrich via PF, and/or relax filters.")
        choice = None
    else:
        show_cols = ["display_name", "_age_src", "_sex_src", "lowest_all_avg_benchmark"]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(
            filtered[show_cols]
            .rename(columns={
                "display_name": "Horse",
                "_age_src": "Age",
                "_sex_src": "Sex",
                "lowest_all_avg_benchmark": "Lowest All Avg Benchmark",
            })
            .reset_index(drop=True),
            use_container_width=True,
        )

        choice = st.selectbox(
            "Select a horse for full Punting Form data",
            filtered["display_name"].tolist(),
            key="detail_select",
        )

        if save_clicked and not filtered.empty:
            st.session_state["saved_shortlist"] = filtered["display_name"].tolist()
            st.success(f"Saved {len(st.session_state['saved_shortlist'])} horses.")

        if "saved_shortlist" in st.session_state and st.session_state["saved_shortlist"]:
            st.markdown("#### 📌 Saved shortlist")
            st.write(", ".join(st.session_state["saved_shortlist"]))
            st.download_button(
                "Download saved shortlist (CSV)",
                data=pd.DataFrame({"Horse": st.session_state["saved_shortlist"]}).to_csv(index=False),
                file_name="saved_shortlist.csv",
                mime="text/csv",
            )

with right:
    st.subheader("Punting Form data")
    if filtered is None or filtered.empty or not choice:
        st.caption("Paste some names, click 'Enrich via PF', apply filters, then pick a horse.")
    else:
        if st.button("🔍 View Punting Form Data for selection"):
            with st.spinner(f"Fetching Punting Form data for {choice}..."):
                try:
                    ident = search_horse_by_name(choice)
                    st.success(f"Found: {ident.get('display_name', choice)}")

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

st.caption("Tip: For dynamic Inglis pages, install Playwright locally and the app will fall back automatically.")
