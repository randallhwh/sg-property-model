"""
app.py — Streamlit fair value estimator for Singapore private residential property.

Run:  streamlit run app.py
      (from the sg_property_model/ directory)

Tabs:
  1. Estimate   — input property spec → fair value + CI + comps
  2. Browse     — filter & explore historical transactions
  3. Model      — train / retrain model, view walk-forward CV performance
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SG Property Fair Value",
    page_icon="🏙️",
    layout="wide",
)

# ── Password protection ───────────────────────────────────────────────────────
def _check_password() -> bool:
    correct = st.secrets.get("APP_PASSWORD", "") if hasattr(st, "secrets") else ""
    if not correct:
        return True  # no password set — allow access
    if st.session_state.get("authenticated"):
        return True
    st.markdown("### 🔒 SG Property Fair Value")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pwd == correct:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not _check_password():
    st.stop()

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main, .stApp { background-color: #060c18; color: #f1f5f9; }
    h1, h2, h3    { color: #f1f5f9; }
    p, label, .stSlider label,
    div[data-testid="stWidgetLabel"] p { color: #e2e8f0 !important; font-size: 14px !important; }
    div[data-testid="stTabs"] button  { color: #94a3b8; font-size: 14px; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #60a5fa; border-bottom-color: #3b82f6;
    }
    div[data-testid="stCheckbox"] span,
    div[data-testid="stCheckbox"] p   { color: #e2e8f0 !important; }
    div[data-testid="stSlider"] span  { color: #e2e8f0 !important; }
    span[data-baseweb="tag"] span     { color: #f1f5f9 !important; }
    .metric-card {
        background: #0d1a2d; border: 1px solid #1e2d45; border-radius: 10px;
        padding: 16px 20px; text-align: center;
    }
    .metric-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 26px; font-weight: 700; font-family: monospace; }
    .metric-sub   { font-size: 12px; color: #64748b; margin-top: 4px; }
    .result-card {
        background: #0a1423; border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 24px 28px; margin-bottom: 16px;
    }
    .comps-table { font-size: 13px; font-family: monospace; }
    .info-box {
        background: #0a1423; border-left: 3px solid #3b82f6;
        border-radius: 8px; padding: 12px 16px; font-size: 13px;
        color: #94a3b8; margin-top: 12px;
    }
    .warn-box {
        background: #1a1400; border-left: 3px solid #fbbf24;
        border-radius: 8px; padding: 12px 16px; font-size: 13px;
        color: #94a3b8; margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH  = "data/processed/transactions.parquet"
MODEL_DIR  = "models"
RAW_DIR    = "data/raw"

DISTRICTS  = list(range(1, 29))
PROP_TYPES = ["condo", "ec", "semi_d", "detached", "terrace", "strata_landed"]
TENURES    = ["Freehold", "99 yrs from 2010", "99 yrs from 2015", "99 yrs from 2020",
              "999 yrs from 1885", "Custom…"]
SALE_TYPES = ["Resale", "New Sale", "Sub Sale"]

SEG_LABEL  = {"CCR": "Core Central (CCR)", "RCR": "Rest of Central (RCR)", "OCR": "Outside Central (OCR)"}
TYPE_LABEL = {
    "condo":         "Condominium / Apartment",
    "ec":            "Executive Condominium",
    "semi_d":        "Semi-Detached House",
    "detached":      "Detached House",
    "terrace":       "Terraced House",
    "strata_landed": "Strata Landed",
}

# ── Data / model loaders (cached) ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_history():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_parquet(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(os.path.join(MODEL_DIR, "lgbm_psf_model.joblib")):
        return None
    try:
        from src.valuation import FairValueModel
        return FairValueModel.load(MODEL_DIR, DATA_PATH)
    except Exception as e:
        st.warning(f"Model load error: {e}")
        return None

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_sgd(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"${v:,.0f}"

def fmt_psf(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"${v:,.0f} psf"

def pct_color(v):
    if v > 5:  return "#22c55e"
    if v > 0:  return "#86efac"
    if v > -5: return "#f87171"
    return "#ef4444"

def verdict_html(pct, label_pos="above", label_neg="below"):
    color = pct_color(pct)
    sign = "+" if pct >= 0 else ""
    direction = label_pos if pct >= 0 else label_neg
    return (f'<span style="color:{color};font-weight:700">{sign}{pct:.1f}% {direction} '
            f'district median</span>')

# ── Chart helpers ─────────────────────────────────────────────────────────────

def _build_gauge(psf_est, ci_lo, ci_hi, listing_psf=None, district_med=None):
    """Horizontal bar chart showing estimate, CI, listing, and district median."""
    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=ci_lo, x1=ci_hi, y0=-0.4, y1=0.4,
        fillcolor="rgba(59,130,246,0.12)",
        line=dict(color="rgba(59,130,246,0.3)", width=1),
    )
    for x, label in [(ci_lo, f"${ci_lo:,.0f}"), (ci_hi, f"${ci_hi:,.0f}")]:
        fig.add_annotation(x=x, y=0.55, text=label,
                           showarrow=False, font=dict(size=11, color="#3b82f6"))
    fig.add_shape(type="line", x0=psf_est, x1=psf_est, y0=-0.5, y1=0.5,
                  line=dict(color="#60a5fa", width=3))
    fig.add_annotation(x=psf_est, y=0.65, text=f"<b>${psf_est:,.0f}</b>",
                       showarrow=False, font=dict(size=14, color="#60a5fa"))
    if district_med:
        fig.add_shape(type="line", x0=district_med, x1=district_med, y0=-0.5, y1=0.5,
                      line=dict(color="#94a3b8", width=1.5, dash="dash"))
        fig.add_annotation(x=district_med, y=-0.65, text=f"D median ${district_med:,.0f}",
                           showarrow=False, font=dict(size=11, color="#64748b"))
    if listing_psf:
        color = "#22c55e" if listing_psf < psf_est * 0.95 else "#ef4444" if listing_psf > psf_est * 1.05 else "#fbbf24"
        fig.add_shape(type="line", x0=listing_psf, x1=listing_psf, y0=-0.5, y1=0.5,
                      line=dict(color=color, width=2.5))
        fig.add_annotation(x=listing_psf, y=-0.82, text=f"Asking ${listing_psf:,.0f}",
                           showarrow=False, font=dict(size=11, color=color))
    pad = (ci_hi - ci_lo) * 0.5
    fig.update_layout(
        paper_bgcolor="#060c18", plot_bgcolor="#0a1120",
        height=160, margin=dict(l=20, r=20, t=40, b=50),
        xaxis=dict(
            range=[max(0, ci_lo - pad), ci_hi + pad],
            tickformat="$,.0f", gridcolor="#0f1e35",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(visible=False, range=[-1, 1]),
        showlegend=False,
    )
    return fig


def _render_shap(shap_series: pd.Series, top_n: int = 10):
    """Render a horizontal bar chart of SHAP contributions."""
    if shap_series is None:
        return
    top = shap_series.abs().nlargest(top_n)
    vals  = shap_series[top.index]
    names = [f.replace("_", " ").title() for f in top.index]
    colors = ["#22c55e" if v > 0 else "#ef4444" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals.values, y=names, orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#060c18", plot_bgcolor="#0a1120",
        height=max(200, top_n * 28),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor="#0f1e35", tickfont=dict(color="#94a3b8"),
                   title=dict(text="SHAP contribution (log PSF)", font=dict(color="#64748b"))),
        yaxis=dict(tickfont=dict(color="#e2e8f0")),
        font=dict(color="#cbd5e1"),
    )
    st.plotly_chart(fig, width='stretch')


def _render_comps(comps: pd.DataFrame):
    """Render a styled comparable transactions table."""
    headers = list(comps.columns)
    header_html = "".join(
        f'<th style="padding:8px 10px;text-align:left;font-size:10px;letter-spacing:1px;'
        f'text-transform:uppercase;color:#64748b;background:#07111f;white-space:nowrap">{h}</th>'
        for h in headers
    )
    rows_html = ""
    for i, row in comps.iterrows():
        bg = "#0a1423" if i % 2 == 0 else "#060c18"
        cells = ""
        for col, val in row.items():
            if col == "PSF ($)":
                cells += f'<td style="padding:7px 10px;font-family:monospace;color:#60a5fa;font-weight:700">${val:,}</td>'
            elif col == "Price ($)" and val:
                cells += f'<td style="padding:7px 10px;font-family:monospace;color:#fbbf24">${val:,}</td>'
            elif col == "Project":
                cells += f'<td style="padding:7px 10px;font-size:12px;color:#e2e8f0;max-width:160px">{val}</td>'
            else:
                cells += f'<td style="padding:7px 10px;font-size:12px;color:#94a3b8">{val}</td>'
        rows_html += f'<tr style="background:{bg}">{cells}</tr>'
    table = f"""
    <div style="overflow-x:auto;border:1px solid #1e2d45;border-radius:8px;overflow:hidden">
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>"""
    st.markdown(table, unsafe_allow_html=True)


def _build_district_bar(df: pd.DataFrame):
    """Median PSF by postal district, coloured by market segment."""
    seg_colors = {"CCR": "#22c55e", "RCR": "#60a5fa", "OCR": "#a78bfa"}
    grp = (
        df.groupby(["postal_district", "market_segment"])["psf"]
        .median()
        .reset_index()
        .sort_values("postal_district")
    )
    fig = go.Figure()
    for seg, color in seg_colors.items():
        sub = grp[grp["market_segment"] == seg]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Bar(
            x=sub["postal_district"].astype(str).apply(lambda d: f"D{int(d):02d}"),
            y=sub["psf"],
            name=seg,
            marker_color=color,
            hovertemplate="D%{x}<br>Median PSF: $%{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="#060c18", plot_bgcolor="#0a1120",
        height=300, barmode="group",
        margin=dict(l=20, r=20, t=20, b=40),
        legend=dict(bgcolor="#0d1a2d", bordercolor="#1e2d45", font=dict(color="#e2e8f0")),
        xaxis=dict(gridcolor="#0f1e35", tickfont=dict(color="#e2e8f0"), title=""),
        yaxis=dict(gridcolor="#0f1e35", tickfont=dict(color="#e2e8f0"),
                   tickprefix="$", tickformat=",.0f"),
        font=dict(color="#cbd5e1"),
    )
    return fig


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("### 🏙️ Singapore Property Fair Value Estimator")
st.markdown("Private Residential — Condo & Landed · Powered by URA transaction data + LightGBM")
st.markdown("---")

# ── District Reference ────────────────────────────────────────────────────────
with st.expander("📍 District Reference", expanded=False):
    st.markdown("### District Reference")
    st.markdown(
        '<p style="font-size:11px;color:#64748b;margin-top:-8px">Singapore Postal Districts</p>',
        unsafe_allow_html=True,
    )

    _DISTRICT_REF = [
        ("D01", "Raffles Place, Cecil, Marina, People's Park",   "CCR"),
        ("D02", "Anson, Tanjong Pagar",                          "CCR"),
        ("D03", "Queenstown, Tiong Bahru",                       "CCR"),
        ("D04", "Telok Blangah, Harbourfront",                   "CCR"),
        ("D05", "Pasir Panjang, Clementi",                       "RCR"),
        ("D06", "High Street, Beach Road",                       "CCR"),
        ("D07", "Middle Road, Golden Mile",                      "RCR"),
        ("D08", "Little India",                                  "RCR"),
        ("D09", "Orchard, Cairnhill, River Valley",              "CCR"),
        ("D10", "Ardmore, Buona Vista, Holland, Tanglin",        "CCR"),
        ("D11", "Watten, Novena, Thomson",                       "CCR"),
        ("D12", "Balestier, Toa Payoh, Serangoon",               "RCR"),
        ("D13", "Macpherson, Braddell",                          "RCR"),
        ("D14", "Geylang, Eunos",                                "RCR"),
        ("D15", "Katong, Joo Chiat, Amber Road",                 "RCR"),
        ("D16", "Bedok, Upper East Coast, Eastwood",             "OCR"),
        ("D17", "Loyang, Changi",                                "OCR"),
        ("D18", "Tampines, Pasir Ris",                           "OCR"),
        ("D19", "Serangoon Gardens, Hougang, Punggol",           "OCR"),
        ("D20", "Bishan, Ang Mo Kio",                            "RCR"),
        ("D21", "Upper Bukit Timah, Ulu Pandan",                 "OCR"),
        ("D22", "Jurong",                                        "OCR"),
        ("D23", "Bukit Batok, Bukit Panjang, Choa Chu Kang",     "OCR"),
        ("D24", "Lim Chu Kang, Tengah",                          "OCR"),
        ("D25", "Kranji, Woodgrove, Woodlands",                  "OCR"),
        ("D26", "Upper Thomson, Springleaf",                     "OCR"),
        ("D27", "Sembawang, Yishun",                             "OCR"),
        ("D28", "Seletar, Yio Chu Kang",                         "OCR"),
    ]

    _SEG_COLORS = {"CCR": "#22c55e", "RCR": "#60a5fa", "OCR": "#a78bfa"}

    rows_html = ""
    for dist, area, seg in _DISTRICT_REF:
        color = _SEG_COLORS[seg]
        rows_html += (
            f'<tr>'
            f'<td style="padding:4px 6px;font-family:monospace;font-size:12px;'
            f'color:#60a5fa;white-space:nowrap">{dist}</td>'
            f'<td style="padding:4px 6px;font-size:11px;color:#cbd5e1">{area}</td>'
            f'<td style="padding:4px 6px;font-size:11px;font-weight:700;'
            f'color:{color};white-space:nowrap">{seg}</td>'
            f'</tr>'
        )

    st.markdown(f"""
    <div style="overflow-y:auto;max-height:72vh;border:1px solid #1e2d45;
                border-radius:8px;background:#07111f">
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr style="background:#0d1a2d;position:sticky;top:0">
            <th style="padding:5px 6px;font-size:10px;color:#64748b;text-align:left">Dist</th>
            <th style="padding:5px 6px;font-size:10px;color:#64748b;text-align:left">Area</th>
            <th style="padding:5px 6px;font-size:10px;color:#64748b;text-align:left">Seg</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:11px;color:#475569;margin-top:6px">'
        '<span style="color:#22c55e">■</span> CCR — Core Central Region &nbsp;|&nbsp;'
        '<span style="color:#60a5fa">■</span> RCR — Rest of Central Region &nbsp;|&nbsp;'
        '<span style="color:#a78bfa">■</span> OCR — Outside Central Region'
        '</p>',
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_estimate, tab_browse, tab_model = st.tabs(
    ["🎯 Fair Value Estimate", "🔍 Browse Transactions", "⚙️ Model"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: ESTIMATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_estimate:

    df_hist = load_history()
    fvm     = load_model()

    if df_hist is None:
        st.markdown("""
        <div class="warn-box">
            <b style="color:#fbbf24">No data loaded yet.</b><br>
            Go to the <b>Model</b> tab to ingest your URA CSVs and train the model.
            Or drop CSVs into <code>data/raw/</code> and click "Run Pipeline" below.
        </div>
        """, unsafe_allow_html=True)

    # ── PropertyGuru URL autofill ─────────────────────────────────────────────
    # Initialise session state defaults
    for _k, _v in [("pg_type","condo"),("pg_dist",10),("pg_area",None),
                   ("pg_tenure","Freehold"),("pg_floor",10),("pg_sale","Resale"),
                   ("pg_price",0),("pg_project",""),("pg_loaded_url",""),("pg_raw",None)]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    pg_url = st.text_input(
        "Paste a PropertyGuru listing URL to auto-fill the form",
        placeholder="https://www.propertyguru.com.sg/property-listing/...",
    )
    if pg_url and pg_url.startswith("http") and pg_url != st.session_state["pg_loaded_url"]:
        with st.spinner("Fetching listing…"):
            try:
                from src.propertyguru import scrape
                _pg = scrape(pg_url)
                st.session_state["pg_type"]       = _pg.get("prop_type", "condo")
                st.session_state["pg_dist"]       = _pg.get("district") or 10
                st.session_state["pg_area"]       = int(_pg["area_sqft"]) if _pg.get("area_sqft") else None
                st.session_state["pg_tenure"]     = _pg.get("tenure_raw", "Freehold")
                st.session_state["pg_floor"]      = _pg.get("floor_level", 10)
                st.session_state["pg_sale"]       = _pg.get("sale_type", "Resale")
                st.session_state["pg_price"]      = int(_pg["listing_price"]) if _pg.get("listing_price") else 0
                st.session_state["pg_project"]    = _pg.get("project_name", "")
                st.session_state["pg_loaded_url"] = pg_url
                st.session_state["pg_raw"]        = _pg
                _price_str = f" · ${st.session_state['pg_price']:,}" if st.session_state["pg_price"] else ""
                st.success(f"Loaded: **{st.session_state['pg_project']}**{_price_str} — check fields below.")
                st.rerun()
            except Exception as e:
                st.warning(f"Could not fetch listing: {e}")

    if st.session_state.get("pg_raw") and st.session_state.get("pg_loaded_url") == pg_url:
        with st.expander("🔍 Scraper debug — what was extracted", expanded=False):
            _r = st.session_state["pg_raw"]
            st.json({k: v for k, v in _r.items() if k != "raw"})

    col_form, col_result = st.columns([1, 1.6], gap="large")

    # ── Input form ────────────────────────────────────────────────────────────
    with col_form:
        st.markdown("**Property Specification**")

        _def_type    = st.session_state["pg_type"]
        _def_dist    = st.session_state["pg_dist"]
        _def_area    = st.session_state["pg_area"]
        _def_tenure  = st.session_state["pg_tenure"]
        _def_floor   = st.session_state["pg_floor"]
        _def_sale    = st.session_state["pg_sale"]
        _def_price   = st.session_state["pg_price"]
        _def_project = st.session_state["pg_project"]

        prop_type = st.selectbox(
            "Property type",
            options=PROP_TYPES,
            index=PROP_TYPES.index(_def_type) if _def_type in PROP_TYPES else 0,
            format_func=lambda x: TYPE_LABEL.get(x, x),
        )
        district = st.selectbox(
            "Postal district (D1–D28)",
            options=DISTRICTS,
            index=DISTRICTS.index(_def_dist) if _def_dist in DISTRICTS else 9,
            format_func=lambda d: f"D{d:02d}",
        )
        _landed_types = {"semi_d", "detached", "terrace"}
        _area_label = (
            "Land area (sqft)  — URA uses plot size for landed"
            if prop_type in _landed_types
            else "Strata floor area (sqft)  — URA built-up area"
        )
        _area_default = _def_area or (1_800 if prop_type in _landed_types else 1_000)
        area_sqft = st.number_input(
            _area_label, min_value=200, max_value=20_000,
            value=_area_default, step=50,
        )

        _tenure_options = TENURES if _def_tenure in TENURES else TENURES[:-1] + [_def_tenure, "Custom…"]
        tenure_choice = st.selectbox(
            "Tenure", _tenure_options,
            index=_tenure_options.index(_def_tenure) if _def_tenure in _tenure_options else 0,
        )
        if tenure_choice == "Custom…":
            tenure_raw = st.text_input(
                "Enter tenure (e.g. '99 yrs from 2018')", value=_def_tenure
            )
        else:
            tenure_raw = tenure_choice

        if prop_type in _landed_types:
            floor_level = 1
            st.caption("Floor level: not applicable for landed — defaulting to ground (1).")
        else:
            floor_level = st.slider(
                "Floor level (approximate)", min_value=1, max_value=60,
                value=min(_def_floor, 60),
            )
        sale_type = st.selectbox(
            "Type of sale", SALE_TYPES,
            index=SALE_TYPES.index(_def_sale) if _def_sale in SALE_TYPES else 0,
        )

        # Optional: listing price for comparison
        st.markdown("---")
        st.markdown("**Optional: Compare vs listing price**")
        listing_price = st.number_input(
            "Listing / asking price ($)", min_value=0, value=_def_price, step=50_000
        )
        listing_psf = listing_price / area_sqft if listing_price > 0 else None

        project_name = st.text_input(
            "Project name (optional — improves accuracy)",
            value=_def_project,
            placeholder="e.g. THE SAIL @ MARINA BAY",
        )

        top_year = st.number_input(
            "TOP year (optional — year project received Temporary Occupation Permit)",
            min_value=1960, max_value=2030, value=None,
            placeholder="e.g. 2010",
            help="Helps model estimate property age. Leave blank if unknown.",
        )

        estimate_btn = st.button("Estimate Fair Value", type="primary", width='stretch')

    # ── Results panel ─────────────────────────────────────────────────────────
    with col_result:
        if not estimate_btn:
            st.markdown("""
            <div class="info-box">
                Fill in the property spec on the left and click
                <b>Estimate Fair Value</b> to see the model's assessment.
                <br><br>
                The model uses:
                <ul style="margin:8px 0; padding-left:18px; color:#94a3b8">
                    <li>District / CCR / RCR / OCR tier</li>
                    <li>Tenure type & remaining lease</li>
                    <li>Floor level premium</li>
                    <li>Area (log-scaled)</li>
                    <li>Trailing district & project PSF</li>
                    <li>Time seasonality</li>
                    <li>Distance to nearest MRT</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        elif fvm is None and df_hist is None:
            st.error("No model or data available. Please train the model first.")

        else:
            spec = {
                "property_type":  prop_type,
                "postal_district": district,
                "area_sqft":       area_sqft,
                "tenure_raw":      tenure_raw,
                "floor_midpoint":  float(floor_level),
                "type_of_sale":    sale_type,
                "project_name":    project_name.strip().upper() or None,
                "top_year":        int(top_year) if top_year else None,
            }

            # ── With trained model ────────────────────────────────────────────
            if fvm is not None:
                with st.spinner("Estimating…"):
                    try:
                        res = fvm.estimate(spec)
                    except Exception as e:
                        st.error(f"Estimation error: {e}")
                        st.stop()

                psf_est    = res["psf_estimate"]
                price_est  = res["price_estimate"]
                ci_lo_psf  = res["ci_low_psf"]
                ci_hi_psf  = res["ci_high_psf"]
                d_med      = res["district_median_psf"]
                pct_dist   = res["pct_vs_district"]

                # ── Key metrics ───────────────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Fair Value PSF</div>
                        <div class="metric-value" style="color:#60a5fa">{fmt_psf(psf_est)}</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Fair Value Price</div>
                        <div class="metric-value" style="color:#fbbf24">{fmt_sgd(price_est)}</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    d_color = pct_color(pct_dist)
                    sign = "+" if pct_dist >= 0 else ""
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">vs District D{district:02d} median</div>
                        <div class="metric-value" style="color:{d_color}">{sign}{pct_dist:.1f}%</div>
                        <div class="metric-sub">District median: {fmt_psf(d_med)}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Gauge chart ───────────────────────────────────────────────
                fig_gauge = _build_gauge(
                    psf_est, ci_lo_psf, ci_hi_psf, listing_psf, d_med
                )
                st.plotly_chart(fig_gauge, width='stretch')

                # ── Listing comparison ────────────────────────────────────────
                if listing_psf:
                    diff_pct = (listing_psf / psf_est - 1) * 100
                    diff_color = "#22c55e" if diff_pct < -5 else "#fbbf24" if abs(diff_pct) <= 5 else "#ef4444"
                    verdict = (
                        "Good value — listing is below fair value estimate"
                        if diff_pct < -5 else
                        "Fair — listing is roughly at model estimate"
                        if abs(diff_pct) <= 5 else
                        "Caution — listing is above fair value estimate"
                    )
                    st.markdown(f"""
                    <div style="background:#0d1a2d; border:1px solid {diff_color}44;
                                border-left:4px solid {diff_color}; border-radius:10px;
                                padding:16px 20px; margin-bottom:12px">
                        <div style="display:flex; justify-content:space-between; align-items:center">
                            <div>
                                <div style="font-size:12px; color:#94a3b8; margin-bottom:4px">Asking price analysis</div>
                                <div style="font-size:15px; font-weight:600; color:{diff_color}">{verdict}</div>
                            </div>
                            <div style="text-align:right">
                                <div style="font-size:11px; color:#64748b">Asking PSF</div>
                                <div style="font-size:20px; font-family:monospace; color:#f1f5f9">{fmt_psf(listing_psf)}</div>
                                <div style="font-size:13px; color:{diff_color}; font-weight:600">
                                    {"+" if diff_pct >= 0 else ""}{diff_pct:.1f}% vs estimate
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Confidence interval note ──────────────────────────────────
                st.markdown(f"""
                <div class="info-box">
                    <b>80% confidence interval:</b>
                    {fmt_psf(ci_lo_psf)} — {fmt_psf(ci_hi_psf)}
                    &nbsp;·&nbsp; {fmt_sgd(res["ci_low_price"])} — {fmt_sgd(res["ci_high_price"])}
                </div>
                """, unsafe_allow_html=True)

                # ── SHAP explanation ──────────────────────────────────────────
                if res.get("shap_values") is not None:
                    st.markdown("<br>**Value drivers (SHAP)**", unsafe_allow_html=True)
                    _render_shap(res["shap_values"])

                # ── Comparables ───────────────────────────────────────────────
                st.markdown("<br>**Comparable transactions**", unsafe_allow_html=True)
                comps = res["comps"]
                if len(comps) > 0:
                    _render_comps(comps)
                else:
                    st.info("No comparable transactions found in dataset.")

            # ── Comps-only fallback (data but no model) ───────────────────────
            else:
                st.markdown("""
                <div class="warn-box">
                    Model not trained yet — showing comparable transactions only.
                </div>
                """, unsafe_allow_html=True)
                from src.valuation import get_comps, SPEC_DEFAULTS
                comps = get_comps(spec, df_hist, n=15)
                if len(comps) > 0:
                    st.markdown("**Comparable transactions (trailing 18 months)**")
                    _render_comps(comps)
                    med_psf = comps["PSF ($)"].median()
                    st.markdown(f"""
                    <div class="info-box">
                        Comps median PSF: <b style="color:#60a5fa">{fmt_psf(med_psf)}</b>
                        &nbsp;·&nbsp; Implied price: <b style="color:#fbbf24">{fmt_sgd(med_psf * area_sqft)}</b>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No comparable transactions found.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BROWSE TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_browse:
    df_hist2 = load_history()
    if df_hist2 is None:
        st.info("Load data first via the Model tab.")
    else:
        st.markdown(f"**{len(df_hist2):,} transactions** · {df_hist2['date_of_sale'].min().date()} → {df_hist2['date_of_sale'].max().date()}")

        # Filters
        f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
        with f1:
            _avail_districts = sorted(df_hist2["postal_district"].dropna().astype(int).unique().tolist())
            dist_filter = st.multiselect(
                "District", _avail_districts,
                format_func=lambda d: f"D{int(d):02d}",
                default=[]
            )
        with f2:
            type_filter = st.multiselect(
                "Property type", PROP_TYPES,
                format_func=lambda x: TYPE_LABEL.get(x, x),
                default=["condo", "ec"]
            )
        with f3:
            year_range = st.slider(
                "Year", int(df_hist2["year"].min()), int(df_hist2["year"].max()),
                (int(df_hist2["year"].max()) - 3, int(df_hist2["year"].max()))
            ) if "year" in df_hist2.columns else (2020, 2025)
        with f4:
            psf_range = st.slider(
                "PSF range ($)",
                int(df_hist2["psf"].quantile(0.01)),
                int(df_hist2["psf"].quantile(0.99)),
                (int(df_hist2["psf"].quantile(0.05)), int(df_hist2["psf"].quantile(0.95)))
            )

        mask = (
            (df_hist2["property_type"].isin(type_filter)) &
            (df_hist2["psf"].between(*psf_range))
        )
        if dist_filter:
            mask &= df_hist2["postal_district"].isin(dist_filter)
        if "year" in df_hist2.columns:
            mask &= df_hist2["year"].between(*year_range)
        dff = df_hist2[mask]

        st.markdown(f"Showing **{len(dff):,}** transactions")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Median PSF</div>
                <div class="metric-value" style="color:#60a5fa">{fmt_psf(dff['psf'].median())}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Median Price</div>
                <div class="metric-value" style="color:#fbbf24">{fmt_sgd(dff['price'].median())}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Projects</div>
                <div class="metric-value" style="color:#a78bfa">{dff['project_name'].nunique():,}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Transactions</div>
                <div class="metric-value" style="color:#94a3b8">{len(dff):,}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # PSF by district chart
        if "postal_district" in dff.columns and len(dff) > 0:
            fig_bar = _build_district_bar(dff)
            st.plotly_chart(fig_bar, width='stretch')

        # Transaction table (sample)
        show_cols = [c for c in ["project_name", "property_type", "postal_district",
                                  "area_sqft", "psf", "price", "floor_range",
                                  "tenure_raw", "date_of_sale", "type_of_sale"]
                     if c in dff.columns]
        sample = dff[show_cols].sort_values("date_of_sale", ascending=False).head(200)
        st.dataframe(
            sample.rename(columns={
                "project_name": "Project", "property_type": "Type",
                "postal_district": "D", "area_sqft": "Area",
                "psf": "PSF", "price": "Price",
                "floor_range": "Floor", "tenure_raw": "Tenure",
                "date_of_sale": "Date", "type_of_sale": "Sale Type"
            }),
            width='stretch',
            height=350,
        )
        csv_data = dff[show_cols].to_csv(index=False)
        st.download_button(
            "⬇ Download filtered data as CSV",
            csv_data, "sg_transactions_filtered.csv", "text/csv"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("### Data Pipeline & Model Training")

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    st.markdown("#### Step 1 — Ingest URA data")
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")] if os.path.exists(RAW_DIR) else []

    if raw_files:
        st.markdown(f"Found **{len(raw_files)}** CSV file(s) in `data/raw/`:")
        for f in raw_files:
            st.markdown(f"  - `{f}`")
    else:
        st.markdown("""
        <div class="warn-box">
            No CSV files found in <code>data/raw/</code>.<br>
            Download URA private residential transaction data from
            <a href="https://www.ura.gov.sg/reis/dataDL" target="_blank" style="color:#60a5fa">
            URA REIS</a> and place the CSV files there.
        </div>
        """, unsafe_allow_html=True)

    geocode_flag = st.checkbox(
        "Geocode projects via OneMap API (adds MRT distance features; ~5 min for large datasets)",
        value=False,
    )

    if st.button("Run Data Pipeline", type="primary", disabled=len(raw_files) == 0):
        with st.spinner("Ingesting and cleaning URA data…"):
            try:
                from src.pipeline import load_ura_folder, clean_transactions, save_processed
                from src.features import build_features
                df_raw = load_ura_folder(RAW_DIR)
                df_clean = clean_transactions(df_raw)
                df_feats = build_features(df_clean, geocode=geocode_flag)
                save_processed(df_feats, DATA_PATH)
                st.success(f"Pipeline complete. {len(df_feats):,} transactions saved.")
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback; st.code(traceback.format_exc())

    st.markdown("---")

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    st.markdown("#### Step 2 — Train fair value model")

    df_for_train = load_history()
    if df_for_train is None:
        st.info("Run the pipeline above first to generate processed data.")
    else:
        st.markdown(
            f"Training data: **{len(df_for_train):,}** transactions · "
            f"{df_for_train['date_of_sale'].min().date()} → {df_for_train['date_of_sale'].max().date()}"
        )

        run_cv = st.checkbox("Run walk-forward CV before training (slower)", value=False)

        if st.button("Train Model", type="primary"):
            with st.spinner("Training LightGBM… (this may take 1–3 minutes)"):
                try:
                    from src.features import get_feature_matrix
                    from src import model as M
                    from datetime import datetime

                    X, feat_cols = get_feature_matrix(df_for_train)
                    y = df_for_train["psf"].values

                    if run_cv:
                        st.markdown("**Walk-forward CV results:**")
                        cv_results = M.walk_forward_cv(df_for_train, X, pd.Series(y))
                        st.dataframe(cv_results, width='stretch')

                    # Split: last 6 months as hold-out
                    cutoff = df_for_train["date_of_sale"].max() - pd.DateOffset(months=6)
                    train_mask = df_for_train["date_of_sale"] < cutoff
                    X_tr, y_tr = X[train_mask], pd.Series(y)[train_mask]
                    X_va, y_va = X[~train_mask], pd.Series(y)[~train_mask]

                    model = M.train(X_tr, y_tr, eval_set=(X_va, y_va))

                    # Quantile models for CI
                    q_models = M.train_quantile_models(X_tr, y_tr)

                    # Evaluate on hold-out
                    metrics = M.evaluate(model, X_va, y_va)
                    st.markdown("**Hold-out evaluation (last 6 months):**")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("MAE (PSF)", f"${metrics['mae']:.0f}")
                    mc2.metric("MAPE", f"{metrics['mape']:.1f}%")
                    mc3.metric("Within 10%", f"{metrics['within_10pct']:.0f}%")
                    mc4.metric("Within 20%", f"{metrics['within_20pct']:.0f}%")

                    metadata = {
                        "trained_at":       datetime.now().isoformat(),
                        "n_train":          int(train_mask.sum()),
                        "n_holdout":        int((~train_mask).sum()),
                        "hold_out_mae":     metrics["mae"],
                        "hold_out_mape":    metrics["mape"],
                        "within_10pct":     metrics["within_10pct"],
                        "within_20pct":     metrics["within_20pct"],
                    }
                    M.save(model, feat_cols, metadata, q_models, MODEL_DIR)
                    st.success("Model trained and saved.")
                    st.cache_resource.clear()
                    st.rerun()

                except Exception as e:
                    st.error(f"Training error: {e}")
                    import traceback; st.code(traceback.format_exc())

    # ── Model info ────────────────────────────────────────────────────────────
    fvm2 = load_model()
    if fvm2 is not None:
        st.markdown("---")
        st.markdown("#### Current model")
        info = fvm2.model_info()
        ci1, ci2, ci3, ci4 = st.columns(4)
        _mape = info.get('hold_out_mape')
        _w10  = info.get('within_10pct')
        _ntr  = info.get('n_train')
        _proj = info.get('projects')
        ci1.metric("Hold-out MAPE",    f"{_mape:.1f}%" if isinstance(_mape, (int, float)) else "—")
        ci2.metric("Within 10%",       f"{_w10:.0f}%"  if isinstance(_w10,  (int, float)) else "—")
        ci3.metric("Training rows",    f"{_ntr:,}"     if isinstance(_ntr,  (int, float)) else "—")
        ci4.metric("Projects in data", f"{_proj:,}"    if isinstance(_proj, (int, float)) else "—")
        st.caption(f"Trained: {info.get('trained_at', 'unknown')} · Features: {info.get('feature_count', '?')}")

