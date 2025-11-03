# app.py — Soar Bloodstock Data - MoneyBall
import io
import re
from datetime import date
from typing import Optional

import streamlit as st
import pandas as pd
from PIL import Image  # validate/preview uploaded logo images

# ---- Punting Form client (pf_client.py must be in repo) ----
# pf_client should read secrets for:
# PF_BASE_URL, PF_PATH_SEARCH, PF_PATH_FORM, PF_PATH_RATINGS, PF_PATH_SPEEDMAP,
# PF_PATH_SECTIONALS, PF_PATH_BENCHMARKS, PF_API_KEY
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
    """Remove BOMs, strip, compress spaces in column headers."""
    clean = {}
    for c in df.columns:
        x = str(c).replace("\ufeff", "")
        x = re.sub(r"\s+", " ", x).strip()
        clean[c] = x
    return df.rename(columns=clean)

def detect_name_col(cols) -> Optional[str]:
    """Find a column that looks like a horse name column."""
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
    """Load CSV or XLSX from bytes with robust parsing."""
    if filename.lower().endswith(".xlsx"):
        try:
            return clean_headers(pd.read_excel(io.BytesIO(file_bytes)))
        except Exception as e:
            raise RuntimeError("Excel reading requires 'openpyxl'. Add it to requirements.txt.") from e
    # CSV: try smart separator & encodings
    bio = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(bio, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
        return clean_headers(df)
    except UnicodeDecodeError:
        bio.seek(0)
        df = pd.read_csv(bio, sep=None, engine="python", encoding="ISO-8859-1", on_bad_lines="skip")
        return clean_headers(df)

def normalize_sale_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add convenience normalized columns (_age_int, _sex_norm)."""
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
    # Render a valid image only if we have true image bytes
    logo_data = st.session_state.get(LOGO_BYTES_KEY, None)
    if logo_data:
        try:
            if isinstance(logo_data, (bytes, bytearray)):
                img = Image.open(io.BytesIO(logo_data))
                st.image(img, use_container_width=False)
            else:
                st.session_state.pop(LOGO_BYTES_KEY, None)
                st.warning("⚠️ Saved logo was invalid and has been cleared. Please re-upload under Settings.")
        except Exception:
            st.session_state.pop(LOGO_BYTES_KEY, None)
            st.warning("⚠️ Logo could not be displayed and has been cleared. Please re-upload under Settings.")
    st.title("Soar Bloodstock Data - MoneyBall")

# ──────────────────────────────────────────────────────────────
# Tabs: App  |  Settings
# ──────────────────────────────────────────────────────────────
tab_app, tab_settings = st.tabs(["App", "Settings"])

# =========================
# SETTINGS TAB (logo etc.)
# =========================
with tab_settings:
    st.subheader("Page Settings")
    st.caption("Logo and other persistent settings for this session.")

    logo_up = st.file_uploader("Upload a logo (PNG/JPG)", type=["png", "jpg", "jpeg"])

    cols = st.columns([1,1,2])
    with cols[0]:
        if st.button("💾 Save logo", use_container_width=True):
            if logo_up and hasattr(logo_up, "getvalue"):
                raw = logo_up.getvalue()
                try:
                    _ = Image.open(io.BytesIO(raw))  # validate
                    st.session_state[LOGO_BYTES_KEY] = raw
                    st.success("Logo saved for this session. It will appear at the top.")
                    st.rerun()
                except Exception:
                    st.error("That file is not a valid image. Please choose a PNG/JPG.")
            else:
                st.warning("Please select an image file first.")

    with cols[1]:
        if st.button("🗑️ Clear saved logo", use_container_width=True):
            st.session_state.pop(LOGO_BYTES_KEY, None)
            st.success("Saved logo cleared.")
            st.rerun()

# =========================
# APP TAB
# =========================
with tab_app:

    # ──────────────────────────────────────────────────────────
    # DATA INPUT PANEL (can be hidden once saved)
    # ──────────────────────────────────────────────────────────
    st.subheader("Data Source")
    if not st.session_state[HIDE_INPUTS_KEY]:
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
                st.success("Upload saved for this session.")
        else:  # Use saved upload
            if (SALE_BYTES_KEY not in st.session_state) or (SALE_NAME_KEY not in st.session_state):
                st.info("No saved upload yet. Switch to 'Upload file' and save it.")
            else:
                st.success(f"Using saved: {st.session_state[SALE_NAME_KEY]}")

        def build_sale_df_from_mode():
            if mode == "Upload file" and upload_file is not None:
                return load_dataframe_from_bytes(upload_file.getvalue(), upload_file.name)
            if mode == "Use saved upload" and (SALE_BYTES_KEY in st.session_state):
                return load_dataframe_from_bytes(st.session_state[SALE_BYTES_KEY], st.session_state[SALE_NAME_KEY])
            # Paste fallback
            names = [n.strip() for n in pasted_text.splitlines() if n.strip()]
            return pd.DataFrame({"Name": names})

        sale_df = build_sale_df_from_mode()

        if st.button("✅ Save & hide data inputs"):
            st.session_state[HIDE_INPUTS_KEY] = True
            st.success("Inputs hidden. Use the Settings tab for logo; unhide inputs with the button below.")
            st.rerun()

        st.caption("Need to switch inputs later? Unhide below.")
        if st.button("👀 Unhide data inputs"):
            st.session_state[HIDE_INPUTS_KEY] = False
            st.rerun()

    else:
        # Inputs hidden: reconstruct sale_df from saved upload if present
        sale_df = None
        if (SALE_BYTES_KEY in st.session_state) and (SALE_NAME_KEY in st.session_state):
            try:
                sale_df = load_dataframe_from_bytes(st.session_state[SALE_BYTES_KEY], st.session_state[SALE_NAME_KEY])
            except Exception as e:
                st.error(f"Saved upload failed to load: {e}")
        if sale_df is None:
            sale_df = pd.DataFrame({"Name": []})
        if st.button("👀 Unhide data inputs"):
            st.session_state[HIDE_INPUTS_KEY] = False
            st.rerun()

    # Normalize + detect name column
    sale_df = clean_headers(sale_df)
    sale_df = normalize_sale_df(sale_df)
    name_col = detect_name_col(list(sale_df.columns)) or ("Name" if "Name" in sale_df.columns else None)
    if not name_col:
        st.error("No 'Name' or similar column found. Provide a pasted list or upload a file that contains horse names.")
        st.stop()

    # ──────────────────────────────────────────────────────────
    # Filters (multi-select + Apply button + State)
    # ──────────────────────────────────────────────────────────
    st.subheader("Filters")

    ages_all = list(range(1, 11))  # 1..10
    age_any = st.checkbox("Any age", value=st.session_state[FILTERS_KEY]["age_any"])
    age_selected = st.multiselect("Ages", ages_all, default=st.session_state[FILTERS_KEY]["age_selected"], disabled=age_any)

    sex_options = ["Gelding", "Mare", "Horse", "Colt", "Filly"]
    sex_selected = st.multiselect("Sex (choose one or more)", sex_options, default=st.session_state[FILTERS_KEY]["sex_selected"])

    maiden_choice = st.selectbox("Maiden", ["Any", "Yes", "No"], index=["Any","Yes","No"].index(st.session_state[FILTERS_KEY]["maiden_choice"]))

    lowest_bm_max = st.number_input("Lowest achieved All Avg Benchmark ≤", value=float(st.session_state[FILTERS_KEY]["lowest_bm_max"]), step=0.1)

    # State filter (multi)
    state_options = []
    if "State" in sale_df.columns:
        state_options = sorted([s for s in sale_df["State"].dropna().astype(str).unique() if s.strip()])
    state_selected = st.multiselect("State (choose one or more)", state_options, default=st.session_state[FILTERS_KEY]["state_selected"])

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

    # Filter summary
    fs = st.session_state[FILTERS_KEY]
    st.caption(
        f"Applied: Age={'Any' if fs['age_any'] else fs['age_selected'] or '—'} | "
        f"Sex={fs['sex_selected'] or '—'} | Maiden={fs['maiden_choice']} | "
        f"Lowest All Avg BM≤{fs['lowest_bm_max']} | State={fs['state_selected'] or '—'}"
    )

    # Apply filters
    def apply_filters_to(sdf: pd.DataFrame) -> pd.DataFrame:
        f = st.session_state[FILTERS_KEY]
        out = sdf.copy()

        # Age
        if not f["age_any"] and ("_age_int" in out.columns):
            out = out[out["_age_int"].isin(f["age_selected"])]

        # Sex
        if f["sex_selected"] and ("_sex_norm" in out.columns):
            out = out[out["_sex_norm"].isin(f["sex_selected"])]

        # Maiden (best-effort if present)
        if f["maiden_choice"] != "Any":
            maiden_cols = [c for c in out.columns if "maiden" in c.lower()]
            if maiden_cols:
                col = maiden_cols[0]
                want = (f["maiden_choice"] == "Yes")
                if out[col].dropna().isin([True, False]).any():
                    out = out[out[col] == want]
                else:
                    out = out[out[col].astype(str).str.lower().isin(["yes" if want else "no"])]

        # Lowest achieved All Avg Benchmark (try to find a column)
        bench_cols_priority = [
            "lowest_achieved_all_avg_benchmark",
            "lowest_all_avg_benchmark",
            "min_all_avg_benchmark",
            "all avg benchmark (min)",
            "all_avg_benchmark_min",
            "avg_benchmark_all_min",
            "avg_benchmark_all",  # fallback
        ]
        bench_col = None
        for c in out.columns:
            norm = re.sub(r"[^a-z0-9]", "", c.lower())
            for probe in bench_cols_priority:
                if re.sub(r"[^a-z0-9]", "", probe) == norm:
                    bench_col = c
                    break
            if bench_col:
                break
        if bench_col:
            with pd.option_context('mode.chained_assignment', None):
                out[bench_col] = pd.to_numeric(out[bench_col], errors="coerce")
            out = out[(out[bench_col].notna()) & (out[bench_col] <= f["lowest_bm_max"])]

        # State
        if f["state_selected"] and ("State" in out.columns):
            out = out[out["State"].astype(str).isin(f["state_selected"])]

        return out

    filtered_df = apply_filters_to(sale_df)

    # ──────────────────────────────────────────────────────────
    # Filtered list + shortlist download + selection control
    # ──────────────────────────────────────────────────────────
    st.subheader("Filtered sale horses")
    if filtered_df.empty:
        st.info("No horses match filters yet. Upload/paste data and/or relax filters.")
    else:
        show_cols = []
        for c in ["Lot", name_col, "Age", "_age_int", "Sex", "_sex_norm", "State", "Vendor", "Bid"]:
            if c in filtered_df.columns:
                show_cols.append(c)
        st.dataframe(filtered_df[show_cols].reset_index(drop=True), use_container_width=True)

        st.download_button(
            "⬇️ Export shortlist (CSV)",
            data=filtered_df.to_csv(index=False),
            file_name="shortlist.csv",
            mime="text/csv",
        )

    # Selection area (prefers filtered list if available)
    candidate_source = filtered_df if not filtered_df.empty else sale_df
    all_names = sorted(candidate_source[name_col].dropna().astype(str).unique())

    # Preserve previously selected horse if possible
    default_index = 0
    if st.session_state.get(SELECTED_HORSE_KEY) in all_names:
        default_index = all_names.index(st.session_state[SELECTED_HORSE_KEY]) + 1

    selected_name = st.selectbox("Selected horse", options=["—"] + all_names, index=default_index)

    if selected_name and selected_name != "—":
        st.session_state[SELECTED_HORSE_KEY] = selected_name

    # Quick-pick from filtered list (syncs selected)
    st.caption("Tip: picking a name here updates the ‘Selected horse’ above.")
    quick_pick = st.selectbox("Pick from filtered list", options=["—"] + all_names, key="quickpick")
    if quick_pick and quick_pick != "—" and quick_pick != st.session_state.get(SELECTED_HORSE_KEY):
        st.session_state[SELECTED_HORSE_KEY] = quick_pick
        st.rerun()

    # ──────────────────────────────────────────────────────────
    # Selected horse details + PF data button
    # ──────────────────────────────────────────────────────────
    if st.session_state.get(SELECTED_HORSE_KEY):
        sel = st.session_state[SELECTED_HORSE_KEY]
        st.write(f"### Selected Horse: {sel}")

        # View PF data button placed above details
        if st.button("🔎 View Punting Form Data"):
            with st.spinner(f"Searching Punting Form for “{sel}”…"):
                try:
                    ident = search_horse_by_name(sel)
                    if not ident:
                        st.error("No result returned by search. Check name spelling or PF configuration.")
                    else:
                        st.success(f"Found: {ident.get('display_name', sel)}")
                        horse_id = ident.get("horse_id") or ident.get("id")
                        # Try per-horse endpoints (your PF plan may need meeting/race ids)
                        try:
                            form = get_form(horse_id)
                        except Exception as e:
                            form = {"info": f"Form not available for id={horse_id}", "error": str(e)}
                        try:
                            ratings = get_ratings(horse_id)
                        except Exception as e:
                            ratings = {"info": f"Ratings not available for id={horse_id}", "error": str(e)}
                        try:
                            speedmap = get_speedmap(horse_id)
                        except Exception as e:
                            speedmap = {"info": f"Speedmap not available for id={horse_id}", "error": str(e)}

                        tabs = st.tabs(["Form", "Ratings", "Speedmap"])
                        with tabs[0]: st.json(form)
                        with tabs[1]: st.json(ratings)
                        with tabs[2]: st.json(speedmap)
                except Exception as e:
                    st.error(f"Could not retrieve data: {e}")

        st.divider()

        # Show sale fields for selected horse
        hr = candidate_source[candidate_source[name_col].astype(str) == str(sel)]
        if not hr.empty:
            row = hr.iloc[0].to_dict()
            show_kv("Lot", row.get("Lot"))
            show_kv("Age", row.get("Age"))
            show_kv("Sex", row.get("Sex"))
            show_kv("Sire", row.get("Sire"))
            show_kv("Dam", row.get("Dam"))
            show_kv("State", row.get("State"))
            show_kv("Vendor", row.get("Vendor"))
            show_kv("Bid", row.get("Bid"))

    # ──────────────────────────────────────────────────────────
    # Inglis Page Import (optional)
    # ──────────────────────────────────────────────────────────
    st.subheader("Inglis Sale Page Import (optional)")
    st.caption("If the table is JavaScript-rendered, use the page’s CSV export or copy/paste.")
    inglis_url = st.text_input("Paste Inglis sale page URL (optional):", placeholder="https://inglis.com.au/sales/online/...")
    if st.button("Fetch table from URL"):
        if not inglis_url.strip():
            st.warning("Please paste a URL first.")
        else:
            try:
                tables = pd.read_html(inglis_url)
                if not tables:
                    st.error("Couldn’t find HTML tables. If it’s JS-rendered, copy/paste or upload CSV instead.")
                else:
                    st.success(f"Found {len(tables)} table(s). Showing the first:")
                    st.dataframe(clean_headers(tables[0]), use_container_width=True)
            except Exception as e:
                st.error(f"Couldn’t parse that page. If it’s JS-only, use CSV export or copy/paste. Details: {e}")

# ──────────────────────────────────────────────────────────────
# PF Diagnostics (sidebar)
# ──────────────────────────────────────────────────────────────
with st.sidebar.expander("🔧 PF Diagnostics"):
    try:
        from pf_client import PF_BASE_URL, PF_PATH_SEARCH, PF_API_KEY
        st.write("BASE_URL:", PF_BASE_URL)
        st.write("PATH_SEARCH:", PF_PATH_SEARCH)
        st.write("API KEY set?:", bool(PF_API_KEY))
        try:
            base = PF_BASE_URL.rstrip("/")
            path = PF_PATH_SEARCH if PF_PATH_SEARCH.startswith("/") else "/" + PF_PATH_SEARCH
            st.write("Full search URL:", base + path)
        except Exception:
            pass
    except Exception:
        st.write("pf_client.py not imported correctly.")

    test_name = st.text_input("Quick PF search:", "Little Spark")
    if st.button("Run search"):
        try:
            res = search_horse_by_name(test_name)
            st.success(res if res else "No result")
        except Exception as e:
            st.error(str(e))

# Sidebar live/demo indicator at bottom
st.sidebar.success("Live Mode (PF API)" if LIVE else "Demo Mode (no API key)")
