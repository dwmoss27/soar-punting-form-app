# app.py — Soar Bloodstock Data - MoneyBall (Filters + PF Metrics + Value Score + Watchlist)
import io
import re
from datetime import date
from typing import Optional, Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np

# Optional: for logo preview fallback
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Optional embed support
import streamlit.components.v1 as components
# ---- pf_client import guard (keep at TOP of app.py) ----
import os, sys, traceback
import streamlit as st

# Ensure current directory is on path (helps some cloud runners)
if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

def _diagnose_missing_pf_client(e):
    st.error("❌ Couldn't import `pf_client`. See details below.")
    st.caption("What I can see in your app folder:")
    try:
        st.code("\n".join(sorted(os.listdir("."))))
    except Exception:
        st.code("(could not list directory)")

    st.caption("Import error:")
    st.code("".join(traceback.format_exception_only(type(e), e)))
    st.stop()

try:
    # try a cheap probe before the real import
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "pf_client.py")):
        raise ImportError("pf_client.py is missing next to app.py")
except Exception as _e:
    _diagnose_missing_pf_client(_e)
# ---- end guard ----
from pf_client import (
    is_live,
    search_horse_by_name,
    get_form,
    get_ratings,
    get_meeting_sectionals,
    get_meeting_benchmarks,
    get_results,
    get_strike_rate,
    get_southcoast_export,
)

# ------------ Punting Form client (must exist in repo) ------------
# Your pf_client must read Streamlit secrets for BASE_URL/paths/API key.
# It should expose: is_live, search_horse_by_name, get_form, get_ratings,
# get_speedmap, get_sectionals_csv, get_benchmarks_csv

from pf_client import (
    is_live, search_horse_by_name, get_form,
    get_ratings, get_speedmap, get_sectionals_csv, get_benchmarks_csv
)

# ──────────────────────────────────────────────────────────────
# Page config & session keys
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Soar Bloodstock Data - MoneyBall", layout="wide")

LIVE = is_live()

SALE_BYTES_KEY      = "sale_uploaded_bytes"
SALE_NAME_KEY       = "sale_uploaded_name"
LOGO_BYTES_KEY      = "logo_bytes"
FILTERS_KEY         = "filters"
PF_METRICS_KEY      = "pf_metrics_cache"      # { horse_name: {...} }
SELECTED_HORSE_KEY  = "selected_horse"
WATCHLIST_KEY       = "watchlist"              # List[str]
WEIGHTS_KEY         = "value_score_weights"    # dict of weights
LAST_SELECTED_FROM_TABLE = "last_selected_from_table"

# defaults
st.session_state.setdefault(FILTERS_KEY, {
    "age_any": True,
    "age_selected": [],
    "sex_selected": [],
    "maiden_choice": "Any",
    "lowest_bm_max": 5.0,
    "state_selected": []
})
st.session_state.setdefault(PF_METRICS_KEY, {})   # name -> dict
st.session_state.setdefault(SELECTED_HORSE_KEY, None)
st.session_state.setdefault(WATCHLIST_KEY, [])
st.session_state.setdefault(WEIGHTS_KEY, {
    "w_lowest_bm": 1.0,
    # (room to add more inputs later: w_recent_l600, w_win_sr, etc.)
})

# read optional embed config from secrets (only works if PF allows iframes)
PF_WEB_BASE_URL     = st.secrets.get("PF_WEB_BASE_URL", None)         # e.g. "https://puntingform.com.au"
PF_HORSE_ROUTE_TMPL = st.secrets.get("PF_HORSE_ROUTE_TEMPLATE", None) # e.g. "/horses/{horse_id}"

# ──────────────────────────────────────────────────────────────
# Helpers: IO / cleaning
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

def render_logo_center():
    logo_data = st.session_state.get(LOGO_BYTES_KEY, None)
    if not logo_data:
        return
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        try:
            st.image(io.BytesIO(logo_data), use_container_width=False)
        except Exception:
            if PIL_AVAILABLE:
                try:
                    img = Image.open(io.BytesIO(logo_data))
                    st.image(img, use_container_width=False)
                except Exception:
                    st.warning("⚠️ Logo could not be displayed and has been cleared. Please re-upload under Settings.")
                    st.session_state.pop(LOGO_BYTES_KEY, None)
            else:
                st.warning("⚠️ Logo could not be displayed and has been cleared. Please re-upload under Settings.")
                st.session_state.pop(LOGO_BYTES_KEY, None)

# ──────────────────────────────────────────────────────────────
# PF metrics
# ──────────────────────────────────────────────────────────────
def compute_lowest_all_avg_bm_from_df(df: pd.DataFrame) -> Optional[float]:
    """
    Try to find the right column(s) in a PF Benchmarks/Sectionals CSV/DataFrame
    and compute the minimum 'All Avg Benchmark' achieved by the horse.
    """
    if df is None or df.empty:
        return None

    candidates = [
        "All Avg Benchmark", "All Bmark Var L", "all_avg_benchmark",
        "all_bmark_var_l", "All_Benchmark_Var_L"
    ]
    for col in df.columns:
        norm = re.sub(r"[^a-z]", "", col.lower())
        for c in candidates:
            if norm == re.sub(r"[^a-z]", "", c.lower()):
                try:
                    return float(pd.to_numeric(df[col], errors="coerce").min())
                except Exception:
                    pass

    # fallback: any column containing 'bmark' or 'benchmark'
    for col in df.columns:
        if "bmark" in col.lower() or "benchmark" in col.lower():
            try:
                return float(pd.to_numeric(df[col], errors="coerce").min())
            except Exception:
                continue
    return None

def cache_pf_metrics_for_names(names: List[str]) -> None:
    """
    For each horse in names, fetch PF ids, then get benchmarks (or sectionals)
    and compute 'lowest achieved All Avg Benchmark'. Cache to session.
    """
    if not LIVE:
        st.info("Demo mode: PF metrics won’t be fetched (no API key).")
        return

    pf_cache: Dict[str, Any] = st.session_state[PF_METRICS_KEY]
    prog = st.progress(0.0)
    total = len(names) if names else 0

    for i, nm in enumerate(names, start=1):
        nm_key = str(nm).strip()
        if not nm_key:
            prog.progress(min(i/total, 1.0) if total else 1.0)
            continue
        if nm_key in pf_cache and pf_cache[nm_key].get("lowest_all_avg_bm") is not None:
            prog.progress(min(i/total, 1.0) if total else 1.0)
            continue

        try:
            ident = search_horse_by_name(nm_key)
            if not ident:
                pf_cache[nm_key] = {"error": "not_found"}
                prog.progress(min(i/total, 1.0) if total else 1.0)
                continue

            horse_id = ident.get("horse_id") or ident.get("id")

            # Prefer Benchmarks CSV
            df_b = None
            try:
                b_csv = get_benchmarks_csv(horse_id)  # str | bytes | DataFrame
                if isinstance(b_csv, (bytes, bytearray)):
                    df_b = pd.read_csv(io.BytesIO(b_csv))
                elif isinstance(b_csv, str):
                    df_b = pd.read_csv(io.StringIO(b_csv))
                elif isinstance(b_csv, pd.DataFrame):
                    df_b = b_csv
            except Exception:
                df_b = None

            lowest_bm = compute_lowest_all_avg_bm_from_df(df_b)

            # Fallback: Sectionals
            if lowest_bm is None:
                try:
                    s_csv = get_sectionals_csv(horse_id)
                    if isinstance(s_csv, (bytes, bytearray)):
                        df_s = pd.read_csv(io.BytesIO(s_csv))
                    elif isinstance(s_csv, str):
                        df_s = pd.read_csv(io.StringIO(s_csv))
                    elif isinstance(s_csv, pd.DataFrame):
                        df_s = s_csv
                    else:
                        df_s = None
                    lowest_bm = compute_lowest_all_avg_bm_from_df(df_s)
                except Exception:
                    pass

            pf_cache[nm_key] = {
                "horse_id": horse_id,
                "display_name": ident.get("display_name", nm_key),
                "lowest_all_avg_bm": lowest_bm
            }
        except Exception as e:
            pf_cache[nm_key] = {"error": str(e)}

        prog.progress(min(i/total, 1.0) if total else 1.0)

    st.session_state[PF_METRICS_KEY] = pf_cache
    st.success("PF metrics updated.")

# ──────────────────────────────────────────────────────────────
# Filters / apply
# ──────────────────────────────────────────────────────────────
def apply_filters_df(sale_df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """
    Apply filters using:
      - Age (from _age_int)
      - Sex (_sex_norm)
      - Maiden (if 'Maiden' col)
      - State (if 'State' col)
      - Lowest achieved All Avg Benchmark (from PF cache)
    """
    out = sale_df.copy()
    fs = st.session_state[FILTERS_KEY]

    # Age
    if not fs["age_any"] and fs["age_selected"]:
        if "_age_int" in out.columns:
            out = out[out["_age_int"].isin(fs["age_selected"])]

    # Sex
    if fs["sex_selected"]:
        if "_sex_norm" in out.columns:
            out = out[out["_sex_norm"].isin(fs["sex_selected"])]

    # Maiden
    if fs["maiden_choice"] != "Any" and "Maiden" in out.columns:
        want = True if fs["maiden_choice"] == "Yes" else False
        def as_bool(v):
            if pd.isna(v): return None
            s = str(v).strip().lower()
            if s in ["true", "yes", "y", "1"]: return True
            if s in ["false", "no", "n", "0"]: return False
            return None
        out = out[out["Maiden"].apply(as_bool) == want]

    # State
    if fs["state_selected"] and "State" in out.columns:
        out = out[out["State"].astype(str).isin(fs["state_selected"])]

    # Lowest achieved All Avg Benchmark (PF cache)
    if fs["lowest_bm_max"] is not None:
        cache = st.session_state[PF_METRICS_KEY]
        def pass_bm(name_val):
            nm = str(name_val).strip()
            m = cache.get(nm, {})
            lbm = m.get("lowest_all_avg_bm", None)
            if lbm is None:
                return True  # not fetched → don’t exclude
            try:
                return float(lbm) <= float(fs["lowest_bm_max"])
            except Exception:
                return True
        out = out[out[name_col].apply(pass_bm)]

    return out

# ──────────────────────────────────────────────────────────────
# Value Score
# ──────────────────────────────────────────────────────────────
def compute_value_score_for_names(names: List[str]) -> pd.DataFrame:
    """
    Build a DataFrame with columns:
      name, lowest_all_avg_bm, value_score
    Currently score = -z(lowest_bm) * weight  (more negative BM = better)
    """
    cache = st.session_state[PF_METRICS_KEY]
    rows = []
    for nm in names:
        info = cache.get(nm, {})
        lbm = info.get("lowest_all_avg_bm", None)
        rows.append({"name": nm, "lowest_all_avg_bm": lbm})
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # z-normalize where available
    lbm_series = pd.to_numeric(df["lowest_all_avg_bm"], errors="coerce")
    mean = lbm_series.mean(skipna=True)
    std = lbm_series.std(skipna=True)
    if std and std > 0:
        z = (lbm_series - mean) / std
    else:
        z = pd.Series([0.0]*len(df))

    w = float(st.session_state[WEIGHTS_KEY]["w_lowest_bm"])

    # Lower (more negative) BM is better → negate z
    df["value_score"] = (-z.fillna(0.0)) * w
    return df

# ──────────────────────────────────────────────────────────────
# Embed PF page (optional)
# ──────────────────────────────────────────────────────────────
def maybe_embed_pf_page(horse_id: Optional[str]):
    """
    If secrets PF_WEB_BASE_URL + PF_HORSE_ROUTE_TEMPLATE are set AND the site allows iframes,
    embed the page. Otherwise show a link.
    """
    if not horse_id:
        return

    if PF_WEB_BASE_URL and PF_HORSE_ROUTE_TMPL:
        url = PF_WEB_BASE_URL.rstrip("/") + PF_HORSE_ROUTE_TMPL.format(horse_id=str(horse_id))
        st.markdown(f"[Open on Punting Form]({url})")
        try:
            components.iframe(src=url, height=900, scrolling=True)
        except Exception:
            st.info("This site may block embedding. Use the link above.")
    else:
        st.caption("Add PF_WEB_BASE_URL and PF_HORSE_ROUTE_TEMPLATE in your Streamlit Secrets to enable in-app embedding.")

# ──────────────────────────────────────────────────────────────
# UI — Title + logo
# ──────────────────────────────────────────────────────────────
render_logo_center()
st.title("Soar Bloodstock Data - MoneyBall")
st.sidebar.success("Live Mode (PF API)" if LIVE else "Demo Mode (no API key)")

# Tabs
tab_app, tab_settings, tab_watch = st.tabs(["App", "Settings", "Watchlist"])

# =========================
# SETTINGS TAB
# =========================
with tab_settings:
    st.subheader("Page Settings")

    logo_up = st.file_uploader("Upload a logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if logo_up is not None:
        st.write("Preview:")
        try:
            st.image(logo_up.getvalue(), use_container_width=False)
        except Exception:
            st.warning("Could not preview this file.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save logo"):
            if logo_up and hasattr(logo_up, "getvalue"):
                st.session_state[LOGO_BYTES_KEY] = logo_up.getvalue()
                st.success("Logo saved. It will show at the top.")
                st.rerun()
            else:
                st.warning("Please select a file first.")
    with c2:
        if st.button("🗑️ Clear logo"):
            st.session_state.pop(LOGO_BYTES_KEY, None)
            st.success("Logo cleared.")
            st.rerun()

    st.markdown("---")
    st.subheader("Value Score Weights")
    w = st.number_input("Weight: Lowest achieved All Avg Benchmark", value=float(st.session_state[WEIGHTS_KEY]["w_lowest_bm"]), step=0.1)
    if st.button("Save weights"):
        st.session_state[WEIGHTS_KEY]["w_lowest_bm"] = float(w)
        st.success("Weights saved.")
        st.rerun()

# =========================
# WATCHLIST TAB
# =========================
with tab_watch:
    st.subheader("⭐ Watchlist")
    wl = st.session_state[WATCHLIST_KEY]

    if wl:
        # Show scores if present
        scores_df = compute_value_score_for_names(wl)
        if not scores_df.empty:
            st.dataframe(scores_df.sort_values("value_score", ascending=False), use_container_width=True)
        else:
            st.write(pd.DataFrame({"name": wl}))
        st.download_button("Download watchlist CSV", data=pd.DataFrame({"name": wl}).to_csv(index=False),
                           file_name="watchlist.csv", mime="text/csv")
    else:
        st.info("Your watchlist is empty. Add horses from the App tab.")

# =========================
# APP TAB
# =========================
with tab_app:
    st.subheader("Data Source")
    st.info("Upload or paste your sale list, then compute PF metrics and filter.")

    mode = st.radio("Choose input", ["Paste list", "Upload file", "Use saved upload"], horizontal=True)
    pasted_text = ""
    upload_file = None

    if mode == "Paste list":
        pasted_text = st.text_area("Horses (one per line)", height=140,
                                   placeholder="Hell Island\nInvincible Phantom\nIrish Bliss\n...")
    elif mode == "Upload file":
        upload_file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
        if upload_file is not None and st.button("💾 Save this upload"):
            st.session_state[SALE_BYTES_KEY] = upload_file.getvalue()
            st.session_state[SALE_NAME_KEY] = upload_file.name
            st.success("Upload saved for reuse.")
    else:
        if SALE_BYTES_KEY in st.session_state and SALE_NAME_KEY in st.session_state:
            st.success(f"Using saved file: {st.session_state[SALE_NAME_KEY]}")
        else:
            st.info("No saved upload available.")

    # Build sale_df
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
        st.error("No ‘Name’ column found. Paste a name list or upload a CSV/Excel that has a Name column.")
        st.stop()

    # Controls for PF metrics
    st.markdown("#### Punting Form Metrics")
    colm = st.columns([1,1,2])
    with colm[0]:
        if st.button("🔄 Refresh PF metrics for all listed horses"):
            all_names = sorted(sale_df[name_col].dropna().astype(str).unique())
            if not all_names:
                st.warning("No names to fetch.")
            else:
                cache_pf_metrics_for_names(all_names)
                st.rerun()
    with colm[1]:
        if st.button("🧹 Clear PF metrics cache"):
            st.session_state[PF_METRICS_KEY] = {}
            st.success("PF metrics cache cleared.")
            st.rerun()
    with colm[2]:
        st.caption("This fills the cache with 'lowest achieved All Avg Benchmark'. Filtering will use it when available.")

    # Filters
    st.subheader("Filters")
    fs_prev = st.session_state[FILTERS_KEY]
    ages_all = list(range(1, 11))

    age_any = st.checkbox("Any age", value=fs_prev["age_any"])
    age_selected = st.multiselect("Ages (choose one or more)", ages_all,
                                  default=[a for a in fs_prev["age_selected"] if a in ages_all],
                                  disabled=age_any)

    sex_options = ["Gelding", "Mare", "Horse", "Colt", "Filly"]
    sex_selected = st.multiselect("Sex (multi-select)", sex_options,
                                  default=[s for s in fs_prev["sex_selected"] if s in sex_options])

    maiden_choice = st.selectbox("Maiden", ["Any", "Yes", "No"],
                                 index=["Any","Yes","No"].index(fs_prev["maiden_choice"]))

    lowest_bm_max = st.number_input("Lowest achieved All Avg Benchmark ≤",
                                    value=float(fs_prev["lowest_bm_max"]), step=0.1)

    state_options = sorted([s for s in sale_df["State"].dropna().astype(str).unique()]) if "State" in sale_df.columns else []
    state_selected = st.multiselect("State (multi-select)", state_options,
                                    default=[s for s in fs_prev["state_selected"] if s in state_options])

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
        f"Applied → Age={'Any' if fs['age_any'] else (fs['age_selected'] or '—')} | "
        f"Sex={fs['sex_selected'] or '—'} | Maiden={fs['maiden_choice']} | "
        f"Lowest BM≤{fs['lowest_bm_max']} | State={fs['state_selected'] or '—'}"
    )

    # Apply filters for real
    filtered_df = apply_filters_df(sale_df, name_col)

    # Add Value Score column for filtered list where we have PF cache
    if not filtered_df.empty:
        vals = compute_value_score_for_names(filtered_df[name_col].astype(str).tolist())
        if not vals.empty:
            filtered_df = filtered_df.merge(vals, left_on=name_col, right_on="name", how="left")
            filtered_df.drop(columns=["name"], inplace=True, errors="ignore")

    st.subheader("Filtered sale horses")
    if not filtered_df.empty:
        # show and let user pick → selecting here updates "Selected horse"
        show_cols = [name_col, "State", "Age", "Sex", "Bid", "Vendor", "Sire", "Dam", "lowest_all_avg_bm", "value_score"]
        show_cols = [c for c in show_cols if c in filtered_df.columns or c in ["lowest_all_avg_bm", "value_score"]]
        # ensure columns exist even if null
        for c in ["lowest_all_avg_bm", "value_score"]:
            if c not in filtered_df.columns:
                filtered_df[c] = np.nan

        st.dataframe(filtered_df[show_cols], use_container_width=True)
        sel = st.selectbox("Pick horse from filtered list",
                           ["—"] + sorted(filtered_df[name_col].astype(str).unique()))
        if sel and sel != "—":
            st.session_state[SELECTED_HORSE_KEY] = sel
            st.success(f"Selected: {sel}")
    else:
        st.info("No horses match filters yet. Refresh PF metrics and/or relax filters.")

    st.markdown("---")
    st.subheader("Selected horse")

    # prefer the last selected via picker; else keep manual selection
    all_names = sorted(sale_df[name_col].dropna().astype(str).unique())
    default_idx = 0
    if st.session_state[SELECTED_HORSE_KEY] in all_names:
        default_idx = 1 + all_names.index(st.session_state[SELECTED_HORSE_KEY])
    selected_name = st.selectbox("Choose a horse", options=["—"] + all_names, index=default_idx)

    if selected_name and selected_name != "—":
        st.session_state[SELECTED_HORSE_KEY] = selected_name

        # Show summary from sale list
        row = sale_df[sale_df[name_col].astype(str) == str(selected_name)]
        if not row.empty:
            r = row.iloc[0].to_dict()
            st.write(f"### {selected_name}")
            cols_top = st.columns(3)
            with cols_top[0]:
                show_kv("Lot", r.get("Lot"))
                show_kv("Age", r.get("Age"))
                show_kv("Sex", r.get("Sex"))
            with cols_top[1]:
                show_kv("Sire", r.get("Sire"))
                show_kv("Dam", r.get("Dam"))
                show_kv("State", r.get("State"))
            with cols_top[2]:
                show_kv("Vendor", r.get("Vendor"))
                show_kv("Bid", r.get("Bid"))

        # PF quick metric under the summary (if fetched)
        m = st.session_state[PF_METRICS_KEY].get(selected_name, {})
        if m.get("lowest_all_avg_bm") is not None:
            st.info(f"Lowest achieved All Avg Benchmark: **{m['lowest_all_avg_bm']}**")
        elif LIVE:
            st.caption("No PF metric in cache yet. Click “Refresh PF metrics for all listed horses” above.")

        # Action bar: PF view + embed + watchlist
        cA, cB, cC = st.columns([1,1,2])
        with cA:
            if st.button("🔎 View Punting Form Data"):
                with st.spinner(f"Fetching Punting Form data for “{selected_name}”…"):
                    try:
                        ident = search_horse_by_name(selected_name)
                        if not ident:
                            st.error("No PF result. Check the name or your PF config.")
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

                            st.markdown("##### Punting Form page (embedded)")
                            maybe_embed_pf_page(horse_id)
                    except Exception as e:
                        st.error(f"Error retrieving PF data: {e}")
        with cB:
            if selected_name in st.session_state[WATCHLIST_KEY]:
                if st.button("➖ Remove from Watchlist"):
                    st.session_state[WATCHLIST_KEY] = [x for x in st.session_state[WATCHLIST_KEY] if x != selected_name]
                    st.success("Removed from watchlist.")
            else:
                if st.button("⭐ Add to Watchlist"):
                    st.session_state[WATCHLIST_KEY].append(selected_name)
                    st.success("Added to watchlist.")
        with cC:
            st.caption("Embed shows only if PF permits iframes. If it’s blank, use the link above the frame.")

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
        st.write("pf_client not fully configured.")

    test_name = st.text_input("Quick PF search:", "Little Spark")
    if st.button("Run search"):
        try:
            res = search_horse_by_name(test_name)
            st.success(res if res else "No result")
        except Exception as e:
            st.error(str(e))

st.sidebar.success("Live Mode (PF API)" if LIVE else "Demo Mode (no API key)")
