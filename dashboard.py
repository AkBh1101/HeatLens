"""
HeatLens — Streamlit Interface    Pro Dark Edition
"""

import json, math, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, BEST_MODEL_PATH,
    METRICS_PATH, SCALER_PATH, FEATURES_PATH, COLORS, CITIES, MODEL_DIR,
)

OUTLIER_REPORT_PATH = MODEL_DIR / "outlier_stats.json"

# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeatLens",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Stylesheet ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

@keyframes gradientFlow {{
    0%   {{ background-position: 0%   50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0%   50%; }}
}}
@keyframes pulseGlow {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(88,166,255,0); }}
    50%       {{ box-shadow: 0 0 20px 4px rgba(88,166,255,0.18); }}
}}
@keyframes slideUp {{
    from {{ opacity:0; transform:translateY(16px); }}
    to   {{ opacity:1; transform:translateY(0);    }}
}}
@keyframes dotBlink {{
    0%, 100% {{ transform:scale(1);   opacity:1;   }}
    50%       {{ transform:scale(1.4); opacity:0.6; }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -200% center; }}
    100% {{ background-position:  200% center; }}
}}
@keyframes scrollTicker {{
    from{{ transform:translateX(0)  }}
    to  {{ transform:translateX(-50%) }}
}}

html, body, [class*="css"],
.stApp, .main, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: {COLORS['background']} !important;
    color: {COLORS['text']} !important;
}}
.main .block-container {{
    padding: 0 2.5rem 3rem !important;
    max-width: 1500px;
    background-color: {COLORS['background']} !important;
}}
[data-testid="stHeader"] {{ background: {COLORS['background']} !important; border-bottom:1px solid #21262d; }}
[data-testid="stToolbar"] {{ display:none; }}
footer {{ display:none !important; }}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a0e18 0%, #0d1117 60%, #0a0e18 100%) !important;
    border-right: 1px solid #1c2230 !important;
}}
section[data-testid="stSidebar"] .block-container {{ padding: 1.2rem 1rem !important; }}
[data-testid="stSidebar"] [data-baseweb="radio"] label {{
    color: {COLORS['subtext']} !important; font-weight: 500; font-size: 0.85rem;
    padding: 0.5rem 0.9rem; border-radius: 9px; transition: all 0.15s;
    display: flex; align-items: center;
}}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {{
    background: rgba(88,166,255,0.08) !important; color: {COLORS['primary']} !important; padding-left: 1.1rem;
}}
[data-testid="stSidebar"] hr {{ border-color: #1c2230 !important; margin:0.7rem 0; }}

.banner {{
    background: linear-gradient(135deg, #0d1117 0%, #0e1520 40%, #12101c 70%, #0d1117 100%);
    border-bottom: 1px solid #21262d; padding: 2.4rem 2.5rem 2rem;
    margin: 0 -2.5rem 2rem; position: relative; overflow: hidden;
}}
.banner::before {{
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(88,166,255,0.07) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 30%, rgba(188,140,255,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 50% 90%, rgba(63,185,80,0.04)  0%, transparent 50%);
    pointer-events: none;
}}
.banner-title {{
    font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700;
    letter-spacing: -0.03em; line-height: 1.1;
    background: linear-gradient(135deg, #e6edf3, {COLORS['primary']}, {COLORS['secondary']}, #e6edf3);
    background-size: 300% 300%; animation: gradientFlow 6s ease infinite;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.5rem;
}}
.banner-desc {{
    font-size: 0.88rem; color: {COLORS['subtext']}; font-weight: 400;
    letter-spacing: 0.01em; max-width: 600px; line-height: 1.6;
}}
.banner-chips {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem; }}
.banner-chip {{
    background: rgba(255,255,255,0.04); border: 1px solid #2d333b; border-radius: 999px;
    padding: 0.25rem 0.75rem; font-size: 0.72rem; font-weight: 600;
    color: {COLORS['subtext']}; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.04em;
}}

.stat-card {{
    background: linear-gradient(145deg, {COLORS['card']} 0%, {COLORS['card2']} 100%);
    border: 1px solid {COLORS['border']}; border-radius: 18px; padding: 1.4rem 1.5rem;
    text-align: center; position: relative; overflow: hidden; cursor: default;
    transition: transform 0.25s cubic-bezier(.34,1.56,.64,1), border-color 0.2s, box-shadow 0.2s;
    animation: slideUp 0.5s ease both;
}}
.stat-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--sc, {COLORS['primary']}), transparent); opacity: 0.8;
}}
.stat-card::after {{
    content: ''; position: absolute; inset: 0; border-radius: 18px;
    background: radial-gradient(circle at 50% -20%, var(--sc, {COLORS['primary']})0f 0%, transparent 70%);
    pointer-events: none;
}}
.stat-card:hover {{
    transform: translateY(-5px) scale(1.02); border-color: var(--sc, {COLORS['primary']});
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 0 1px var(--sc, {COLORS['primary']})22;
}}
.stat-icon  {{ font-size: 1.4rem; margin-bottom: 0.5rem; }}
.stat-value {{
    font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700;
    letter-spacing: -0.02em; line-height: 1; color: var(--sc, {COLORS['primary']});
}}
.stat-label {{
    font-size: 0.68rem; color: {COLORS['subtext']}; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.4rem;
}}
.stat-note {{
    font-size: 0.7rem; color: #484f58; margin-top: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}}

.section-title {{
    font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700;
    color: {COLORS['text']}; letter-spacing: -0.02em; margin: 1.8rem 0 0.25rem;
    display: flex; align-items: center; gap: 0.6rem;
}}
.section-sub {{ font-size: 0.82rem; color: {COLORS['subtext']}; margin-bottom: 1.2rem; line-height: 1.55; }}

.frosted {{
    background: rgba(22,27,39,0.7); backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.2rem 1.4rem;
}}

.stage-card {{
    background: {COLORS['card']}; border: 1px solid {COLORS['border']}; border-radius: 14px;
    padding: 1.1rem 1rem; text-align: center;
    transition: transform 0.2s, border-color 0.2s; animation: slideUp 0.5s ease both;
}}
.stage-card:hover {{ transform:translateY(-3px); }}
.stage-label {{
    font-size: 0.6rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace; padding: 0.15rem 0.55rem;
    border-radius: 5px; display: inline-block; margin-bottom: 0.6rem;
}}

.feat-tag {{
    background: {COLORS['card2']}; border: 1px solid {COLORS['border']};
    border-left: 3px solid {COLORS['primary']}; border-radius: 7px;
    padding: 0.4rem 0.7rem; margin-bottom: 0.35rem; font-size: 0.76rem;
    font-family: 'JetBrains Mono', monospace; color: {COLORS['text']};
    display: block; transition: border-left-color 0.15s, background 0.15s;
}}
.feat-tag:hover {{ border-left-color: {COLORS['secondary']}; background: rgba(188,140,255,0.06); }}

.top-badge {{
    display: inline-block;
    background: linear-gradient(135deg, {COLORS['accent']}, #1cb845);
    color: #071a0f; font-size: 0.63rem; font-weight: 800;
    padding: 0.15rem 0.6rem; border-radius: 999px;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-left: 0.4rem; vertical-align: middle;
}}

.result-box {{
    background: linear-gradient(135deg,
        rgba(88,166,255,0.07) 0%, rgba(188,140,255,0.07) 50%, rgba(63,185,80,0.05) 100%);
    border: 1px solid rgba(88,166,255,0.25); border-radius: 22px;
    padding: 2rem 1.8rem; text-align: center; position: relative; overflow: hidden;
}}
.result-box::before {{
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(88,166,255,0.08) 0%, transparent 65%);
    pointer-events: none;
}}
.result-num {{
    font-family: 'Space Grotesk', sans-serif; font-size: 4rem; font-weight: 800;
    background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
    background-size: 200% 200%; animation: gradientFlow 4s ease infinite;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.04em; line-height: 1;
}}
.level-tag {{
    display: inline-block; padding: 0.4rem 1.8rem; border-radius: 999px;
    font-weight: 800; font-size: 0.88rem; letter-spacing: 0.05em;
    margin-top: 0.8rem; border: 1px solid; transition: box-shadow 0.3s;
}}

.mono-tag {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    color: {COLORS['primary']}; background: rgba(88,166,255,0.08);
    padding: 0.1rem 0.4rem; border-radius: 4px;
}}

[data-testid="stAlert"] {{
    background: {COLORS['card']} !important; border: 1px solid {COLORS['border']} !important;
    border-radius: 10px !important;
}}
[data-testid="stDataFrame"] {{
    background: {COLORS['card']} !important; border: 1px solid {COLORS['border']} !important;
    border-radius: 12px !important; overflow: hidden;
}}
[data-baseweb="select"] > div {{
    background: {COLORS['card2']} !important; border-color: {COLORS['border']} !important;
    color: {COLORS['text']} !important; border-radius: 9px !important;
}}
[data-baseweb="menu"] {{
    background: {COLORS['card2']} !important; border: 1px solid {COLORS['border']} !important;
    border-radius: 10px !important; box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}}
[data-baseweb="option"]:hover {{ background: rgba(88,166,255,0.1) !important; }}
[data-testid="stSlider"] [role="slider"] {{
    background: {COLORS['primary']} !important; border: 2px solid {COLORS['background']} !important;
    box-shadow: 0 0 0 3px {COLORS['primary']}44 !important;
}}
[data-baseweb="tab-list"] {{
    background: {COLORS['card']} !important; border-radius: 12px !important;
    border: 1px solid {COLORS['border']} !important; padding: 5px !important; gap: 3px;
}}
[data-baseweb="tab"] {{
    font-weight: 600 !important; font-size: 0.82rem !important;
    color: {COLORS['subtext']} !important; border-radius: 8px !important;
    padding: 0.45rem 1.1rem !important; background: transparent !important;
    border: none !important; transition: all 0.15s !important;
}}
[data-baseweb="tab"]:hover {{ color: {COLORS['text']} !important; background: rgba(255,255,255,0.05) !important; }}
[aria-selected="true"][data-baseweb="tab"] {{
    background: linear-gradient(135deg, {COLORS['primary']}, #3d8ef8) !important;
    color: #0d1117 !important; box-shadow: 0 2px 12px rgba(88,166,255,0.35) !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display:none !important; }}
.stButton > button {{
    background: linear-gradient(135deg, {COLORS['primary']}, #3d8ef8) !important;
    color: #0d1117 !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 0.88rem !important; padding: 0.6rem 2rem !important;
    box-shadow: 0 4px 16px rgba(88,166,255,0.3) !important; transition: all 0.2s !important;
}}
.stButton > button:hover {{ transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(88,166,255,0.45) !important; }}
[data-testid="stDownloadButton"] > button {{
    background: {COLORS['card2']} !important; color: {COLORS['primary']} !important;
    border: 1px solid {COLORS['primary']}33 !important; border-radius: 9px !important;
    font-size: 0.82rem !important; font-weight: 600 !important; transition: all 0.15s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: rgba(88,166,255,0.1) !important; border-color: {COLORS['primary']} !important;
    transform: translateY(-1px) !important;
}}
[data-testid="metric-container"] {{
    background: {COLORS['card']} !important; border: 1px solid {COLORS['border']} !important;
    border-radius: 12px !important; padding: 0.8rem 1rem !important;
}}
div[data-testid="stMetricValue"] {{
    font-size:1.7rem !important; font-weight:800 !important; color:{COLORS['text']} !important;
    font-family:'Space Grotesk',sans-serif !important;
}}
div[data-testid="stMetricLabel"] {{
    color:{COLORS['subtext']} !important; font-size:0.7rem !important;
    font-weight:700 !important; text-transform:uppercase !important; letter-spacing:0.08em !important;
}}
[data-testid="stExpander"] {{
    background: {COLORS['card']} !important; border: 1px solid {COLORS['border']} !important;
    border-radius: 12px !important;
}}
[data-testid="stExpander"] summary {{ color: {COLORS['text']} !important; font-weight: 600 !important; }}
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{COLORS['background']}; }}
::-webkit-scrollbar-thumb {{ background: #30363d; border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {COLORS['primary']}66; }}
hr {{ border-color: {COLORS['border']} !important; margin:1.2rem 0 !important; }}

.ticker-strip {{
    width:100%; overflow:hidden;
    background:linear-gradient(90deg,#080c14 0%,#0a0e18 50%,#080c14 100%);
    border-top:1px solid #141a26; border-bottom:1px solid #141a26;
    padding:0.42rem 0; margin-bottom:1.2rem; position:relative;
}}
.ticker-strip::before,.ticker-strip::after {{
    content:''; position:absolute; top:0; bottom:0; width:80px; z-index:2; pointer-events:none;
}}
.ticker-strip::before {{ left:0;  background:linear-gradient(90deg,#0d1117,transparent); }}
.ticker-strip::after  {{ right:0; background:linear-gradient(270deg,#0d1117,transparent); }}
.ticker-inner {{ display:flex; animation:scrollTicker 36s linear infinite; white-space:nowrap; }}
.ticker-cell {{
    font-family:'JetBrains Mono',monospace; font-size:0.7rem; font-weight:600;
    color:#484f58; padding:0 2.5rem; display:inline-flex; align-items:center; gap:0.45rem;
}}
.ticker-cell .tval {{ color:{COLORS['primary']}; font-weight:800; }}
.ticker-sep {{ color:{COLORS['accent']}; font-size:0.5rem; margin:0 1rem; }}

.rank-circle {{
    display:inline-flex; align-items:center; justify-content:center;
    width:22px; height:22px; border-radius:50%;
    font-size:0.6rem; font-weight:900; margin-right:0.35rem; vertical-align:middle;
}}
.rank-1 {{ background:linear-gradient(135deg,#ffd700,#d4a017);color:#1a0e00;box-shadow:0 0 8px #ffd70066; }}
.rank-2 {{ background:linear-gradient(135deg,#c0c0c0,#9e9e9e);color:#1a1a1a; }}
.rank-3 {{ background:linear-gradient(135deg,#cd7f32,#a0522d);color:#1a0a00; }}

.heat-table {{ width:100%;border-collapse:collapse;font-size:0.78rem; }}
.heat-table th {{
    font-size:0.6rem;font-weight:800;text-transform:uppercase;letter-spacing:0.1em;
    color:{COLORS['subtext']};padding:0.5rem 0.9rem;border-bottom:1px solid #1c2230;text-align:left;
}}
.heat-table td {{ padding:0.5rem 0.9rem;border-bottom:1px solid #111520; }}
.heat-row:hover {{ background:rgba(255,255,255,0.025); }}
.heat-dot {{ display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:0.4rem;vertical-align:middle; }}

.glow-card {{
    background:linear-gradient(145deg,#0f1520,{COLORS['card']});
    border:1px solid rgba(88,166,255,0.18); border-radius:18px; padding:1.4rem 1.5rem;
    position:relative; overflow:hidden;
    box-shadow:0 0 40px rgba(88,166,255,0.04),inset 0 1px 0 rgba(255,255,255,0.04);
    transition:box-shadow 0.3s,border-color 0.3s;
}}
.glow-card:hover {{
    border-color:rgba(88,166,255,0.42);
    box-shadow:0 0 50px rgba(88,166,255,0.12),inset 0 1px 0 rgba(255,255,255,0.07);
}}

.chart-wrap {{
    border:1px solid rgba(88,166,255,0.14);border-radius:16px;overflow:hidden;
    box-shadow:0 8px 40px rgba(0,0,0,0.35),0 0 60px rgba(88,166,255,0.04);
}}

.bar-row {{ display:flex;align-items:center;gap:0.5rem;margin:0.22rem 0;padding:0.18rem 0;font-size:0.77rem; }}
.bar-label {{
    width:155px;color:{COLORS['subtext']};font-family:'JetBrains Mono',monospace;
    font-size:0.7rem;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}}
.bar-track {{ flex:1;background:{COLORS['card2']};border-radius:4px;overflow:hidden;height:13px;position:relative; }}
.bar-fill  {{ height:100%;border-radius:4px;min-width:2px; }}
.bar-val   {{ width:60px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:0.72rem;font-weight:700;flex-shrink:0; }}

.note-card {{
    background:{COLORS['card']};border:1px solid {COLORS['border']};border-radius:12px;
    padding:0.9rem 1rem;margin-bottom:0.5rem;font-size:0.78rem;line-height:1.6;
    border-left:3px solid var(--nc,{COLORS['primary']});
}}
.note-card strong {{ color:{COLORS['text']}; }}

.city-chip {{
    display:inline-flex;align-items:center;gap:0.3rem;background:{COLORS['card2']};
    border:1px solid {COLORS['border']};border-radius:999px;padding:0.2rem 0.65rem;
    font-size:0.7rem;font-weight:600;color:{COLORS['text']};margin:0.15rem;white-space:nowrap;
    transition:background 0.15s,border-color 0.15s;
}}
.city-chip:hover {{ background:rgba(88,166,255,0.1);border-color:{COLORS['primary']}44; }}

.res-footer {{
    font-size:0.72rem;color:{COLORS['subtext']};margin-top:0.4rem;
    font-family:'JetBrains Mono',monospace;
}}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_raw():
    return pd.read_csv(RAW_DATA_PATH) if RAW_DATA_PATH.exists() else None

@st.cache_data(ttl=300)
def fetch_processed():
    return pd.read_csv(PROCESSED_DATA_PATH) if PROCESSED_DATA_PATH.exists() else None

@st.cache_resource
def fetch_model():
    if BEST_MODEL_PATH.exists():
        with open(BEST_MODEL_PATH, "rb") as fh: return pickle.load(fh)
    return None

@st.cache_resource
def fetch_scaler():
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as fh: return pickle.load(fh)
    return None

@st.cache_data
def fetch_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as fh: return json.load(fh)
    return None

@st.cache_data
def fetch_feature_names():
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH) as fh: return json.load(fh)
    return None

@st.cache_data(ttl=300)
def fetch_outlier_report():
    if OUTLIER_REPORT_PATH.exists():
        with open(OUTLIER_REPORT_PATH) as fh: return json.load(fh)
    return None


# ── Plotly theme ───────────────────────────────────────────────────────────────
_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Space Grotesk, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=24, r=24, t=44, b=24),
    xaxis=dict(gridcolor="#1c2230", zerolinecolor="#1c2230",
               linecolor="#30363d", tickcolor=COLORS["subtext"]),
    yaxis=dict(gridcolor="#1c2230", zerolinecolor="#1c2230",
               linecolor="#30363d", tickcolor=COLORS["subtext"]),
    legend=dict(bgcolor="rgba(13,17,23,0.85)", bordercolor="#30363d",
                borderwidth=1, font_size=11),
    hoverlabel=dict(bgcolor=COLORS["card2"], bordercolor=COLORS["border"],
                    font_color=COLORS["text"], font_size=12),
)

PALETTE = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
           COLORS["warning"], COLORS["danger"],
           "#38bdf8","#f472b6","#34d399","#fb923c","#a78bfa","#fbbf24","#60a5fa"]

def apply_theme(fig): fig.update_layout(**_THEME); return fig

def rgba_from_hex(hex_val: str, alpha: float = 1.0) -> str:
    h = hex_val.lstrip("#")
    if len(h) == 3: h = "".join(c*2 for c in h)
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


# ── UI helpers ─────────────────────────────────────────────────────────────────
def stat_card(col, icon, value, label, note="", accent=None):
    c = accent or COLORS["primary"]
    col.markdown(f"""
    <div class="stat-card" style="--sc:{c}">
        <div class="stat-icon">{icon}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
        <div class="stat-note">{note}</div>
    </div>""", unsafe_allow_html=True)


def section(heading, caption=""):
    st.markdown(f'<div class="section-title">{heading}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-sub">{caption}</div>', unsafe_allow_html=True)


def compute_input_vector(t, rh, ws, ndvi_v, uf, lat, lon,
                         hr, mo, pres, cld, fnames):
    daytime   = int(6 <= hr <= 18)
    nighttime = int(hr < 6 or hr > 18)
    t_rh      = t * rh / 100
    w_cooling = ws * max(0.0, t - 20)
    ndvi_c    = max(-1.0, min(1.0, ndvi_v))
    vclass    = 0 if ndvi_c < 0.2 else (2 if ndvi_c >= 0.5 else 1)
    h_ret     = uf * t / (ws + 1)
    h_sin = math.sin(2*math.pi*hr/24);  h_cos = math.cos(2*math.pi*hr/24)
    m_sin = math.sin(2*math.pi*mo/12); m_cos = math.cos(2*math.pi*mo/12)
    hi    = (-8.78 + 1.61*t + 2.34*rh/100 - 0.15*t*rh/100)
    lookup = {
        "temperature": t, "humidity": rh, "wind_speed": ws,
        "pressure": pres, "clouds": float(cld),
        "ndvi": ndvi_v, "urban_fraction": uf, "veg_class": float(vclass),
        "lat": lat, "lon": lon, "distance_from_equator": abs(lat),
        "hour": float(hr), "month": float(mo),
        "is_daytime": float(daytime), "is_night": float(nighttime),
        "temp_humidity_interaction": t_rh,
        "wind_cooling_effect": w_cooling,
        "temp_anomaly": 0.0,
        "heat_retention": h_ret,
        "hour_sin": h_sin, "hour_cos": h_cos,
        "month_sin": m_sin, "month_cos": m_cos,
        "lat_abs": abs(lat),
        "lon_sin": math.sin(math.radians(lon)),
        "lon_cos": math.cos(math.radians(lon)),
        "heat_index": hi,
    }
    return [lookup.get(f, 0.0) for f in fnames]


MO_LABELS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

INTENSITY_LEVELS = [
    ("< 1 °C",  "Negligible", "#3fb950", "Urban area barely warmer than surroundings. Dense vegetation or coastal breeze likely."),
    ("1 – 2 °C","Low",        "#58a6ff", "Mild warming. Common in cities with parks & low building density."),
    ("2 – 3 °C","Moderate",   "#d29922", "Noticeable heat island. Higher energy demand; vulnerable populations at risk during heatwaves."),
    ("3 – 5 °C","High",       "#f85149", "Strong UHI. Significant health risk. Typical of dense megacities in summer."),
    ("> 5 °C",  "Extreme",    "#ff6b6b", "Extreme heat island. Life-threatening during heatwaves. Urgent green-infrastructure needed."),
]

CITY_FLAGS = {
    "Delhi":"🇮🇳","Mumbai":"🇮🇳","New York":"🇺🇸","Los Angeles":"🇺🇸",
    "London":"🇬🇧","Tokyo":"🇯🇵","Shanghai":"🇨🇳","São Paulo":"🇧🇷",
    "Cairo":"🇪🇬","Lagos":"🇳🇬","Jakarta":"🇮🇩","Mexico City":"🇲🇽",
    "Karachi":"🇵🇰","Beijing":"🇨🇳","Dhaka":"🇧🇩","Bangkok":"🇹🇭",
    "Kolkata":"🇮🇳","Chicago":"🇺🇸","Paris":"🇫🇷","Istanbul":"🇹🇷",
    "Sydney":"🇦🇺","Toronto":"🇨🇦","Singapore":"🇸🇬","Berlin":"🇩🇪","Seoul":"🇰🇷",
}


def render_live_ticker(p_df, m_data, r_df):
    entries = []
    if p_df is not None and "uhi_intensity" in p_df.columns:
        avg = p_df["uhi_intensity"].mean()
        pk  = p_df["uhi_intensity"].max()
        entries += [("🌡 Global Mean UHI", f"{avg:.2f} °C"), ("🔥 Peak UHI", f"{pk:.2f} °C")]
        if "city_name" in p_df.columns:
            hot_city = p_df.groupby("city_name")["uhi_intensity"].mean().idxmax()
            hot_val  = p_df.groupby("city_name")["uhi_intensity"].mean().max()
            entries.append(("🏙 Hottest City", f"{hot_city}  {hot_val:.2f} °C"))
        entries.append(("📊 Processed Rows", f"{len(p_df):,}"))
    if r_df is not None:
        entries.append(("🛰 Raw Samples", f"{len(r_df):,}"))
    if m_data:
        entries += [
            ("🤖 Best Model", m_data.get("best_model","—")),
            ("📉 Best RMSE",  f"{m_data.get('best_rmse','—')} °C"),
        ]
        sk = m_data["models"].get(m_data.get("best_model",""),{}).get("skill_vs_baseline")
        if sk is not None:
            entries.append(("⚡ Skill Score", f"+{sk*100:.1f}%"))
    if not entries: return
    cells = "".join(
        f'<span class="ticker-cell">{lbl} <span class="tval">{val}</span></span>'
        f'<span class="ticker-sep">◆</span>'
        for lbl, val in entries
    ) * 2
    st.markdown(
        f'<div class="ticker-strip"><div class="ticker-inner">{cells}</div></div>',
        unsafe_allow_html=True
    )


def render_intensity_guide():
    rows = "".join(
        f'<tr class="heat-row">'
        f'<td><span class="heat-dot" style="background:{c}"></span>'
        f'<b style="color:{c}">{rng}</b></td>'
        f'<td><b style="color:{c}">{lvl}</b></td>'
        f'<td style="color:#8b949e;font-size:0.74rem">{desc}</td>'
        f'</tr>'
        for rng, lvl, c, desc in INTENSITY_LEVELS
    )
    st.markdown(f"""
    <div class="glow-card" style="padding:1rem 1.2rem">
        <div style="font-size:0.62rem;font-weight:800;text-transform:uppercase;
                    letter-spacing:0.12em;color:{COLORS['subtext']};margin-bottom:0.8rem">
            🌡️ UHI Severity Reference Guide
        </div>
        <table class="heat-table">
            <thead><tr><th>Range</th><th>Severity</th><th>Implications</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)


def render_scaled_bars(scaled_vals, fnames):
    if not fnames: return
    fv = dict(zip(fnames, scaled_vals))
    top_fv = sorted(fv.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    max_abs = max(abs(v) for _, v in top_fv) or 1.0
    rows_html = ""
    for fname, fval in top_fv:
        pct  = abs(fval) / max_abs * 100
        col  = COLORS["primary"] if fval >= 0 else COLORS["danger"]
        bg   = rgba_from_hex(col, 0.2)
        sign = "+" if fval >= 0 else "−"
        rows_html += (
            f'<div class="bar-row">'
            f'<div class="bar-label" title="{fname}">{fname}</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct:.1f}%;background:{bg};border-right:2px solid {col}"></div>'
            f'</div>'
            f'<div class="bar-val" style="color:{col}">{sign}{abs(fval):.2f}</div>'
            f'</div>'
        )
    hdr = (f'<div style="font-size:0.62rem;font-weight:800;text-transform:uppercase;'
           f'letter-spacing:0.12em;color:{COLORS["subtext"]};margin-bottom:0.7rem">'
           f'&#128208; Scaled Feature Values (top 12 by magnitude)</div>')
    ftr = (f'<div style="font-size:0.7rem;color:{COLORS["subtext"]};'
           f'font-family:\'JetBrains Mono\',monospace;margin-top:0.6rem">'
           f'Values are post-StandardScaler. Positive&nbsp;=&nbsp;above&nbsp;mean.</div>')
    st.markdown(f'<div class="glow-card" style="padding:1rem 1.2rem">{hdr}{rows_html}{ftr}</div>',
                unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem">
        <div style="font-size:3rem;line-height:1;filter:drop-shadow(0 0 16px #58a6ff88)">🌆</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                    font-size:1.1rem;color:#e6edf3;margin-top:0.6rem;letter-spacing:-0.01em">
            Heat Island Analyser</div>
        <div style="font-size:0.65rem;color:#484f58;margin-top:0.2rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.12em">
            Urban Heat Island System
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("page", [
        "📊  Overview", "📁  Data Explorer", "🔬  Preprocessing",
        "🤖  Models",   "🗺️  Heatmap",       "🔮  Prediction",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)

    result_data = fetch_metrics()
    fnames      = fetch_feature_names()

    if result_data:
        top_algo   = result_data.get("best_model","—")
        top_rmse   = result_data.get("best_rmse","—")
        bl_rmse    = result_data.get("baseline_rmse")
        skill_pct  = result_data["models"].get(top_algo,{}).get("skill_vs_baseline")
        mae_score  = result_data["models"].get(top_algo,{}).get("mae","—")
        r2_score   = result_data["models"].get(top_algo,{}).get("r2","—")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{COLORS['card']},{COLORS['card2']});
                    border:1px solid {COLORS['border']};border-radius:14px;padding:1rem;
                    font-size:0.8rem;border-left:3px solid {COLORS['accent']};margin-bottom:0.6rem">
            <div style="color:{COLORS['subtext']};font-size:0.62rem;font-weight:800;
                        text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem">
                🏆 Best Model
            </div>
            <div style="color:{COLORS['primary']};font-weight:800;font-size:1rem;
                        font-family:'Space Grotesk',sans-serif">{top_algo}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-top:0.6rem">
                <div style="background:rgba(255,255,255,0.03);border-radius:7px;padding:0.4rem;text-align:center">
                    <div style="color:{COLORS['subtext']};font-size:0.6rem;text-transform:uppercase">RMSE</div>
                    <div style="color:{COLORS['text']};font-weight:700;font-family:'JetBrains Mono',monospace;font-size:0.82rem">{top_rmse}°C</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border-radius:7px;padding:0.4rem;text-align:center">
                    <div style="color:{COLORS['subtext']};font-size:0.6rem;text-transform:uppercase">MAE</div>
                    <div style="color:{COLORS['text']};font-weight:700;font-family:'JetBrains Mono',monospace;font-size:0.82rem">{mae_score}°C</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border-radius:7px;padding:0.4rem;text-align:center">
                    <div style="color:{COLORS['subtext']};font-size:0.6rem;text-transform:uppercase">R²</div>
                    <div style="color:{COLORS['text']};font-weight:700;font-family:'JetBrains Mono',monospace;font-size:0.82rem">{r2_score}</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);border-radius:7px;padding:0.4rem;text-align:center">
                    <div style="color:{COLORS['subtext']};font-size:0.6rem;text-transform:uppercase">Skill</div>
                    <div style="color:{COLORS['accent']};font-weight:700;font-family:'JetBrains Mono',monospace;font-size:0.82rem">+{skill_pct*100:.1f}%</div>
                </div>
            </div>
            {"<div style='margin-top:0.5rem;padding-top:0.5rem;border-top:1px solid " + COLORS['border'] + ";font-size:0.7rem;color:" + COLORS['subtext'] + "'>Baseline RMSE <b style=color:" + COLORS['text'] + ">" + str(bl_rmse) + "°C</b></div>" if bl_rmse else ""}
        </div>""", unsafe_allow_html=True)

    pipeline_checks = [
        ("Data collected",  RAW_DATA_PATH.exists(),       COLORS["primary"]),
        ("Data processed",  PROCESSED_DATA_PATH.exists(), COLORS["secondary"]),
        ("Model trained",   BEST_MODEL_PATH.exists(),     COLORS["accent"]),
    ]
    for lbl, ready, dc in pipeline_checks:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.55rem;padding:0.22rem 0.3rem;
                    font-size:0.76rem;color:{COLORS['subtext']}">
            <span style="color:{dc if ready else COLORS['border']};font-size:0.75rem;
                         {'animation:dotBlink 2s ease infinite' if ready else ''}">●</span>
            {lbl}
        </div>""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
raw_df         = fetch_raw()
proc_df        = fetch_processed()
mdl            = fetch_model()
scl            = fetch_scaler()
result_data    = fetch_metrics()
fnames         = fetch_feature_names()
outlier_report = fetch_outlier_report()


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "📊  Overview":
    chips = ["ERA5 reanalysis","MODIS 8-day LST","25 global cities",
             "GroupKFold evaluation","GridSearchCV","Zero synthetic data"]
    chip_html = "".join(f'<span class="banner-chip">{t}</span>' for t in chips)
    st.markdown(f"""
    <div class="banner">
        <div class="banner-title">Urban Heat Island Estimation System</div>
        <div class="banner-desc">
            End-to-end machine learning pipeline estimating UHI intensity from
            satellite-derived Land Surface Temperature and ERA5 reanalysis weather —
            fully automated, no synthetic data.
        </div>
        <div class="banner-chips">{chip_html}</div>
    </div>""", unsafe_allow_html=True)

    render_live_ticker(proc_df, result_data, raw_df)

    v_raw   = f"{len(raw_df):,}"  if raw_df  is not None else "—"
    v_proc  = f"{len(proc_df):,}" if proc_df is not None else "—"
    v_feat  = len(fnames) if fnames else "—"
    v_mdl   = len(result_data["models"]) if result_data else "—"
    v_rmse  = f"{result_data['best_rmse']} °C" if result_data else "—"
    v_skill = (f"+{result_data['models'].get(result_data['best_model'],{}).get('skill_vs_baseline',0)*100:.0f}%"
               if result_data else "—")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    stat_card(c1,"🛰️", v_raw,  "Raw Samples",    "ERA5 + GEE rows",   COLORS["primary"])
    stat_card(c2,"⚙️", v_proc, "Processed Rows", "after pipeline",    COLORS["secondary"])
    stat_card(c3,"🔬", v_feat, "Features",        "engineered",        COLORS["accent"])
    stat_card(c4,"🤖", v_mdl,  "Models",          "GridSearch-tuned",  COLORS["warning"])
    stat_card(c5,"📉", v_rmse, "Best RMSE",       result_data.get("best_model","") if result_data else "", COLORS["danger"])
    stat_card(c6,"⚡", v_skill,"Skill Score",     "vs mean baseline",  COLORS["accent"])

    st.markdown("<br>", unsafe_allow_html=True)
    section("⚙️ Pipeline Status")
    stages = [
        ("01","Data Collection",  raw_df  is not None,"🛰️","ERA5 + MODIS via GEE", COLORS["primary"]),
        ("02","Preprocessing",    proc_df is not None,"⚙️","IQR cap + 26 features",COLORS["secondary"]),
        ("03","Model Training",   result_data is not None,"🤖","GroupKFold + GridSearch",COLORS["accent"]),
        ("04","Dashboard",        True,               "📊","Streamlit on :8505",   COLORS["warning"]),
    ]
    stage_cols = st.columns(4)
    for col,(num,name,done,ico,desc,c) in zip(stage_cols,stages):
        col.markdown(f"""
        <div class="stage-card" style="border-left:3px solid {'c' if not done else c}">
            <div class="stage-label" style="background:{c}22;color:{c}">STEP {num} {'✓' if done else '…'}</div>
            <div style="font-size:1.6rem;margin:0.3rem 0">{ico}</div>
            <div style="font-weight:700;font-size:0.88rem;color:{'var(--text)' if done else COLORS['subtext']}">{name}</div>
            <div style="font-size:0.7rem;color:{COLORS['subtext']};margin-top:0.25rem">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if proc_df is not None and "uhi_intensity" in proc_df.columns:
        ca, cb = st.columns([3,2])
        with ca:
            section("📈 UHI Intensity Distribution")
            mu_u  = proc_df["uhi_intensity"].mean()
            med_u = proc_df["uhi_intensity"].median()
            fig = px.histogram(proc_df, x="uhi_intensity", nbins=55,
                               color_discrete_sequence=[COLORS["primary"]],
                               labels={"uhi_intensity":"UHI Intensity (°C)"})
            fig.update_traces(marker_line_color=COLORS["card"], marker_line_width=0.5, opacity=0.85)
            fig.add_vline(x=mu_u,  line_dash="dash", line_color=COLORS["warning"],
                          annotation_text=f"Mean {mu_u:.2f}°C",   annotation_font_color=COLORS["warning"])
            fig.add_vline(x=med_u, line_dash="dot",  line_color=COLORS["secondary"],
                          annotation_text=f"Median {med_u:.2f}°C", annotation_font_color=COLORS["secondary"],
                          annotation_position="top left")
            fig.update_layout(title="All cities · 180-day window", **_THEME)
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            section("🏙️ City Rankings")
            if "city_name" in proc_df.columns:
                city_s = (proc_df.groupby("city_name")["uhi_intensity"]
                          .agg(["mean","std","min","max"])
                          .sort_values("mean",ascending=True).reset_index())
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=city_s["mean"], y=city_s["city_name"], orientation="h",
                    marker=dict(color=city_s["mean"],
                                colorscale=[[0,COLORS["accent"]],[0.45,COLORS["warning"]],[1,COLORS["danger"]]],
                                showscale=False),
                    error_x=dict(array=city_s["std"],color=COLORS["subtext"],thickness=1.2),
                    text=city_s["mean"].round(2), textposition="outside",
                    textfont_color=COLORS["subtext"],
                    hovertemplate="<b>%{y}</b><br>Mean UHI: %{x:.2f}°C<br>Std: %{error_x.array:.2f}°C<extra></extra>",
                ))
                fig2.update_layout(title="Mean UHI ± std", xaxis_title="°C", height=480, **_THEME)
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    sv_col, vio_col = st.columns([1.15,1])
    with sv_col:
        render_intensity_guide()
    with vio_col:
        if proc_df is not None and "uhi_intensity" in proc_df.columns:
            section("🎻 UHI Distribution (Violin)")
            if "city_name" in proc_df.columns:
                top8 = (proc_df.groupby("city_name")["uhi_intensity"]
                        .mean().sort_values(ascending=False).head(8).index.tolist())
                vio_sub = proc_df[proc_df["city_name"].isin(top8)]
                fig_v = go.Figure()
                for i, city in enumerate(top8):
                    vals = vio_sub[vio_sub["city_name"]==city]["uhi_intensity"]
                    fig_v.add_trace(go.Violin(
                        y=vals, name=CITY_FLAGS.get(city,"🏙") + " " + city,
                        line_color=PALETTE[i%len(PALETTE)],
                        fillcolor=rgba_from_hex(PALETTE[i%len(PALETTE)], 0.13),
                        meanline_visible=True, box_visible=True, points=False,
                    ))
                fig_v.update_layout(title="Top-8 hottest cities — distribution",
                                    yaxis_title="UHI (°C)", showlegend=False,
                                    violinmode="overlay", height=310, **_THEME)
                st.plotly_chart(fig_v, use_container_width=True)

    if proc_df is not None:
        section("🌍 Global UHI Map")
        if "city_name" in proc_df.columns and "uhi_intensity" in proc_df.columns:
            has_ll = "lat" in proc_df.columns and "lon" in proc_df.columns
            if has_ll:
                agg = (proc_df.groupby("city_name")
                       .agg(lat=("lat","first"), lon=("lon","first"),
                            uhi_intensity=("uhi_intensity","mean"))
                       .reset_index())
            elif raw_df is not None and "name" in raw_df.columns:
                coords = raw_df[["name","lat","lon"]].drop_duplicates("name").rename(columns={"name":"city_name"})
                agg = (proc_df.groupby("city_name")["uhi_intensity"]
                       .mean().reset_index().merge(coords, on="city_name", how="left"))
            else:
                agg = None
            if agg is not None and not agg.empty:
                fig3 = px.scatter_mapbox(
                    agg.dropna(subset=["lat","lon"]), lat="lat", lon="lon",
                    color="uhi_intensity", size="uhi_intensity",
                    hover_name="city_name",
                    hover_data={"uhi_intensity":":.2f","lat":False,"lon":False},
                    color_continuous_scale=[[0,COLORS["accent"]],[0.4,COLORS["warning"]],[1,COLORS["danger"]]],
                    size_max=28, zoom=1.1, mapbox_style="carto-darkmatter",
                )
                fig3.update_layout(
                    height=420, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor=COLORS["card"],
                    coloraxis_colorbar=dict(
                        title=dict(text="°C",font_color=COLORS["subtext"]),
                        tickcolor=COLORS["subtext"], tickfont_color=COLORS["subtext"],
                        bgcolor=COLORS["card"], outlinecolor=COLORS["border"]),
                    font_color=COLORS["text"])
                st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📁  Data Explorer":
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    section("📁 Data Explorer", "Inspect raw ERA5 + GEE data and the processed feature matrix")

    t_raw, t_proc, t_ts, t_corr = st.tabs([
        "📄  Raw Data","✨  Processed","📈  Time Series","🔗  Correlations"])

    with t_raw:
        if raw_df is None:
            st.warning("Run `python main.py` first.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Rows",    f"{len(raw_df):,}")
            c2.metric("Columns", len(raw_df.columns))
            c3.metric("Cities",  raw_df["name"].nunique() if "name" in raw_df.columns else "—")
            if "timestamp" in raw_df.columns:
                ts = pd.to_datetime(raw_df["timestamp"],errors="coerce")
                c4.metric("Span", f"{int((ts.max()-ts.min()).days)}d" if not ts.isna().all() else "—")
            st.markdown("<br>", unsafe_allow_html=True)
            ca, cb = st.columns([1,2])
            with ca:
                if "source" in raw_df.columns:
                    fp = px.pie(raw_df.groupby("source").size().reset_index(name="count"),
                                values="count",names="source",hole=0.5,
                                color_discrete_sequence=PALETTE,title="Data Sources")
                    fp.update_traces(textfont_color=COLORS["text"])
                    apply_theme(fp); st.plotly_chart(fp,use_container_width=True)
            with cb:
                nc = [c for c in ["temperature","humidity","wind_speed","ndvi","urban_fraction"] if c in raw_df.columns]
                if nc:
                    fb = go.Figure()
                    for i,col in enumerate(nc):
                        fb.add_trace(go.Box(y=raw_df[col],name=col.replace("_"," ").title(),
                                            marker_color=PALETTE[i%len(PALETTE)],boxmean="sd",
                                            boxpoints="outliers",marker=dict(outliercolor=COLORS["warning"],size=3)))
                    fb.update_layout(title="Key feature distributions",showlegend=False,**_THEME)
                    st.plotly_chart(fb,use_container_width=True)
            st.dataframe(raw_df.head(300),use_container_width=True,height=280)
            st.download_button("⬇ Download Raw CSV",raw_df.to_csv(index=False),"raw_data.csv","text/csv")

    with t_proc:
        if proc_df is None:
            st.warning("Run `python main.py` first.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("Rows",     f"{len(proc_df):,}")
            c2.metric("Features", len(proc_df.columns)-2)
            if "uhi_intensity" in proc_df.columns:
                c3.metric("UHI Range",
                          f"{proc_df['uhi_intensity'].min():.1f}–{proc_df['uhi_intensity'].max():.1f}°C")
            st.markdown("<br>", unsafe_allow_html=True)
            key_cols = [c for c in proc_df.select_dtypes(include=np.number).columns
                        if c in (fnames or []) + ["uhi_intensity"]][:22]
            if len(key_cols)>2:
                corr = proc_df[key_cols].corr()
                fh = go.Figure(go.Heatmap(
                    z=corr.values,x=corr.columns,y=corr.index,
                    colorscale=[[0,COLORS["danger"]],[0.5,"#21262d"],[1,COLORS["primary"]]],
                    zmid=0,text=corr.values.round(2),texttemplate="%{text}",textfont_size=8,
                    colorbar=dict(tickcolor=COLORS["subtext"],tickfont_color=COLORS["subtext"],
                                  bgcolor=COLORS["card"],outlinecolor=COLORS["border"])))
                fh.update_layout(title="Pearson correlation matrix",height=520,**_THEME)
                st.plotly_chart(fh,use_container_width=True)
            st.dataframe(proc_df.head(300),use_container_width=True,height=280)
            st.download_button("⬇ Download Processed CSV",
                               proc_df.to_csv(index=False),"processed_data.csv","text/csv")

    with t_ts:
        if proc_df is None or raw_df is None:
            st.warning("Run `python main.py` first.")
        else:
            section("📈 UHI Intensity — Time Series per City")
            if "timestamp" in raw_df.columns and "city_name" in proc_df.columns:
                ts_df = proc_df.copy()
                ts_df["timestamp"] = pd.to_datetime(raw_df["timestamp"].values[:len(ts_df)],errors="coerce")
                ts_df["date"] = ts_df["timestamp"].dt.date
                city_opts = sorted(ts_df["city_name"].dropna().unique().tolist())
                chosen = st.multiselect("Select cities",city_opts,default=city_opts[:6])
                if chosen:
                    ft = go.Figure()
                    for i,city in enumerate(chosen):
                        sub=(ts_df[ts_df["city_name"]==city].groupby("date")["uhi_intensity"]
                             .mean().reset_index())
                        ft.add_trace(go.Scatter(x=sub["date"],y=sub["uhi_intensity"],mode="lines",
                                                name=city,line=dict(color=PALETTE[i%len(PALETTE)],width=2),
                                                hovertemplate=f"<b>{city}</b><br>%{{x}}<br>UHI: %{{y:.2f}}°C<extra></extra>"))
                    ft.update_layout(title="Daily mean UHI intensity (180-day window)",
                                     xaxis_title="Date",yaxis_title="UHI (°C)",
                                     hovermode="x unified",**_THEME)
                    st.plotly_chart(ft,use_container_width=True)
                    section("📅 Monthly Average UHI")
                    if "month" in ts_df.columns:
                        sub2=(ts_df[ts_df["city_name"].isin(chosen)]
                              .groupby(["city_name","month"])["uhi_intensity"]
                              .mean().reset_index())
                        sub2["month_name"]=sub2["month"].map(MO_LABELS)
                        fm=px.line(sub2,x="month_name",y="uhi_intensity",color="city_name",
                                   color_discrete_sequence=PALETTE,markers=True,
                                   labels={"uhi_intensity":"UHI (°C)","month_name":"Month","city_name":"City"})
                        fm.update_layout(title="Seasonal UHI pattern",**_THEME)
                        st.plotly_chart(fm,use_container_width=True)
            else:
                st.info("Timestamp data not available in this dataset snapshot.")

    with t_corr:
        if proc_df is None:
            st.warning("Run `python main.py` first.")
        elif "uhi_intensity" in proc_df.columns and fnames:
            section("🔗 Feature Correlation with UHI Intensity",
                    "Pearson r — how strongly each feature co-varies with the target")
            num_f = [f for f in fnames if f in proc_df.columns]
            corr_uhi = (proc_df[num_f+["uhi_intensity"]].corr()["uhi_intensity"]
                        .drop("uhi_intensity").sort_values())
            bar_c = [COLORS["danger"] if v<0 else COLORS["accent"] for v in corr_uhi.values]
            fc = go.Figure(go.Bar(x=corr_uhi.values,y=corr_uhi.index,orientation="h",
                                  marker_color=bar_c,text=corr_uhi.values.round(3),
                                  textposition="outside",textfont_color=COLORS["subtext"],
                                  hovertemplate="<b>%{y}</b><br>r = %{x:.4f}<extra></extra>"))
            fc.add_vline(x=0,line_color=COLORS["border"],line_width=1.5)
            fc.update_layout(title="Pearson r with uhi_intensity  (red=negative, green=positive)",
                             xaxis_title="Pearson r",height=520,**_THEME)
            st.plotly_chart(fc,use_container_width=True)
            top_f = corr_uhi.abs().idxmax()
            st.markdown(f'<div class="section-sub">Top correlator: <span class="mono-tag">{top_f}</span></div>',
                        unsafe_allow_html=True)
            if top_f in proc_df.columns:
                color_by = "city_name" if "city_name" in proc_df.columns else None
                fs = px.scatter(proc_df.sample(min(2000,len(proc_df)),random_state=42),
                                x=top_f,y="uhi_intensity",color=color_by,
                                color_discrete_sequence=PALETTE,opacity=0.5,
                                trendline="ols",trendline_scope="overall",
                                trendline_color_override=COLORS["warning"],
                                labels={"uhi_intensity":"UHI (°C)"})
                fs.update_traces(marker_size=4)
                fs.update_layout(title=f"{top_f} vs UHI Intensity",showlegend=color_by is not None,**_THEME)
                st.plotly_chart(fs,use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════
elif page == "🔬  Preprocessing":
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    section("🔬 Preprocessing & Feature Engineering",
            "Data cleaning · IQR outlier capping · 26-feature engineering · LST-based UHI target")

    if raw_df is None or proc_df is None:
        st.warning("Run the pipeline first: `python main.py`")
    else:
        pa, pb = st.columns(2)
        with pa:
            section("🚨 Missing Values — Raw Data")
            missing = raw_df.isnull().sum(); missing = missing[missing>0]
            if missing.empty:
                st.markdown(f'<div class="frosted" style="text-align:center;color:{COLORS["accent"]};font-weight:700;font-size:0.9rem;padding:1.2rem">✓ No missing values in raw data</div>',unsafe_allow_html=True)
            else:
                fm=px.bar(x=missing.values,y=missing.index,orientation="h",color=missing.values,
                          color_continuous_scale=[[0,COLORS["warning"]],[1,COLORS["danger"]]],
                          labels={"x":"Missing Count","y":"Column"})
                fm.update_layout(showlegend=False,coloraxis_showscale=False,title="Columns with nulls",**_THEME)
                st.plotly_chart(fm,use_container_width=True)
        with pb:
            section("📐 Shape: Before → After")
            cmp=pd.DataFrame({"Stage":["Raw","Processed"],"Rows":[len(raw_df),len(proc_df)],
                              "Columns":[len(raw_df.columns),len(proc_df.columns)]})
            fs=go.Figure()
            fs.add_bar(x=cmp["Stage"],y=cmp["Rows"],name="Rows",marker_color=COLORS["primary"],
                       text=cmp["Rows"],textposition="outside")
            fs.add_bar(x=cmp["Stage"],y=cmp["Columns"],name="Columns",marker_color=COLORS["secondary"],
                       text=cmp["Columns"],textposition="outside")
            fs.update_layout(barmode="group",title="Rows & Columns",**_THEME)
            st.plotly_chart(fs,use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        section("📦 Outlier Detection & IQR Capping",
                "Winsorisation: values outside [Q1−1.5·IQR, Q3+1.5·IQR] are capped, not dropped")

        if outlier_report:
            cap_list=[c for c,s in outlier_report.items() if s["n_capped"]>0]
            if cap_list:
                cap_cols=st.columns(min(len(cap_list),5))
                for cw,feat in zip(cap_cols,cap_list):
                    s=outlier_report[feat]
                    cw.markdown(f"""
                    <div class="stat-card" style="--sc:{COLORS['warning']}">
                        <div class="stat-icon">📌</div>
                        <div class="stat-value" style="color:{COLORS['warning']}">{s['n_capped']}</div>
                        <div class="stat-label">{feat.replace('_',' ')}</div>
                        <div class="stat-note">{s['pct']}% · [{s['lower_fence']}, {s['upper_fence']}]</div>
                    </div>""",unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="frosted" style="text-align:center;color:{COLORS["accent"]};font-weight:700;padding:1rem">✓ No outliers found — all within 1.5 × IQR</div>',unsafe_allow_html=True)

            with st.expander("📋 Full IQR Statistics Table"):
                iqr_rows=[{"Feature":k,"Q1":v["q1"],"Q3":v["q3"],"IQR":v["iqr"],
                           "Lower Fence":v["lower_fence"],"Upper Fence":v["upper_fence"],
                           "Capped Low":v["n_low"],"Capped High":v["n_high"],
                           "Total":v["n_capped"],"% Rows":v["pct"]}
                          for k,v in outlier_report.items()]
                iqr_df=pd.DataFrame(iqr_rows)
                st.dataframe(iqr_df.style
                    .map(lambda v:"color:#d29922;font-weight:700" if isinstance(v,(int,float)) and v>0 else "",
                         subset=["Total","Capped Low","Capped High"])
                    .format({"Q1":"{:.3f}","Q3":"{:.3f}","IQR":"{:.3f}",
                             "Lower Fence":"{:.3f}","Upper Fence":"{:.3f}","% Rows":"{:.2f}%"}),
                    use_container_width=True,hide_index=True)

            avail=[c for c in outlier_report if c in raw_df.columns and c in proc_df.columns]
            if avail:
                section("📊 Box Plots: Raw vs Capped")
                n_bc=min(len(avail),3); box_cols=st.columns(n_bc)
                for i,feat in enumerate(avail):
                    s=outlier_report[feat]
                    with box_cols[i%n_bc]:
                        fb=go.Figure()
                        fb.add_trace(go.Box(y=raw_df[feat].dropna(),name="Raw",
                                            marker_color=COLORS["primary"],boxmean="sd",
                                            marker=dict(outliercolor=COLORS["warning"],symbol="circle-open",size=4),
                                            boxpoints="outliers",line_width=1.5))
                        fb.add_trace(go.Box(y=proc_df[feat].dropna(),name="Capped",
                                            marker_color=COLORS["accent"],boxmean="sd",
                                            boxpoints=False,line_width=1.5))
                        for fence,txt,anc in [(s["upper_fence"],"↑","bottom right"),(s["lower_fence"],"↓","top right")]:
                            fb.add_hline(y=fence,line_dash="dot",line_color=COLORS["danger"],line_width=1,
                                         annotation_text=f"{txt}{fence}",annotation_font_color=COLORS["danger"],
                                         annotation_font_size=9,annotation_position=anc)
                        fb.update_layout(**{**_THEME,"title":feat.replace("_"," ").title(),
                                           "height":310,"showlegend":True,
                                           "legend":dict(orientation="h",y=1.12,font_size=9,bgcolor="rgba(0,0,0,0)")})
                        st.plotly_chart(fb,use_container_width=True)

            if any(s["n_capped"]>0 for s in outlier_report.values()):
                oc=pd.DataFrame([{"Feature":k,"Below":v["n_low"],"Above":v["n_high"]}
                                  for k,v in outlier_report.items()])
                fo=go.Figure()
                fo.add_trace(go.Bar(name="Below lower fence",x=oc["Feature"],y=oc["Below"],
                                    marker_color=COLORS["primary"],text=oc["Below"],textposition="auto"))
                fo.add_trace(go.Bar(name="Above upper fence",x=oc["Feature"],y=oc["Above"],
                                    marker_color=COLORS["warning"],text=oc["Above"],textposition="auto"))
                fo.update_layout(barmode="stack",title="Outlier counts by column",
                                 xaxis_title="Feature",yaxis_title="Count",**_THEME)
                st.plotly_chart(fo,use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        section("🏗️ Engineered Features")
        if fnames:
            FEAT_GROUPS={
                "🌡️ Weather":    ["temperature","humidity","wind_speed","pressure","clouds"],
                "🛰️ Satellite":  ["ndvi","urban_fraction","veg_class"],
                "📍 Geographic": ["lat","lon","distance_from_equator","lat_abs","lon_sin","lon_cos"],
                "⏰ Temporal":   ["hour","month","is_daytime","is_night","hour_sin","hour_cos","month_sin","month_cos"],
                "🔬 Derived":    ["temp_humidity_interaction","wind_cooling_effect","temp_anomaly","heat_retention","heat_index"],
            }
            gcols=st.columns(len(FEAT_GROUPS))
            for gc,(gn,gf) in zip(gcols,FEAT_GROUPS.items()):
                gc.markdown(f'<div style="font-size:0.7rem;font-weight:800;color:{COLORS["subtext"]};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">{gn}</div>',unsafe_allow_html=True)
                for f in gf:
                    if f in fnames:
                        gc.markdown(f'<span class="feat-tag">{f}</span>',unsafe_allow_html=True)

        common=[c for c in["temperature","humidity","wind_speed","ndvi","uhi_intensity"]
                if c in raw_df.columns or c in proc_df.columns]
        if common:
            st.markdown("<br>", unsafe_allow_html=True)
            section("📉 Feature Distribution: Raw vs Processed")
            dc,_=st.columns([1,2])
            with dc: sel=st.selectbox("Feature",common)
            fd=make_subplots(rows=1,cols=2,subplot_titles=["Raw","Processed"])
            if sel in raw_df.columns:
                fd.add_trace(go.Histogram(x=raw_df[sel],marker_color=COLORS["primary"],opacity=0.8,
                                          name="Raw",nbinsx=40,marker_line_color=COLORS["card"],marker_line_width=0.5),
                             row=1,col=1)
            if sel in proc_df.columns:
                fd.add_trace(go.Histogram(x=proc_df[sel],marker_color=COLORS["secondary"],opacity=0.8,
                                          name="Processed",nbinsx=40,marker_line_color=COLORS["card"],marker_line_width=0.5),
                             row=1,col=2)
            fd.update_layout(showlegend=False,**_THEME)
            st.plotly_chart(fd,use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — MODELS
# ══════════════════════════════════════════════════════════════
elif page == "🤖  Models":
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    section("🤖 Model Comparison",
            "GroupShuffleSplit evaluation — entire cities held out · GridSearchCV tuning · 5-fold GroupKFold")

    if result_data is None:
        st.warning("No trained models found. Run `python main.py` first.")
    else:
        top_algo  = result_data["best_model"]
        algo_data = result_data["models"]
        bl_rmse   = result_data.get("baseline_rmse")
        bl_mae    = result_data.get("baseline_mae")
        tr_cities = result_data.get("train_cities",[])
        te_cities = result_data.get("test_cities",[])

        if tr_cities or te_cities:
            ca,cb=st.columns(2)
            ca.markdown(f"""<div class="frosted" style="font-size:0.8rem">
                <div style="color:{COLORS['subtext']};font-size:0.65rem;font-weight:800;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem">
                    🏋️ Train Cities ({len(tr_cities)})</div>
                <div style="color:{COLORS['primary']};font-weight:600;line-height:1.8">
                    {' · '.join(tr_cities)}</div></div>""",unsafe_allow_html=True)
            cb.markdown(f"""<div class="frosted" style="font-size:0.8rem;border-left:3px solid {COLORS['accent']}">
                <div style="color:{COLORS['subtext']};font-size:0.65rem;font-weight:800;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem">
                    🧪 Test Cities ({len(te_cities)}) — unseen during training</div>
                <div style="color:{COLORS['accent']};font-weight:600;line-height:1.8">
                    {' · '.join(te_cities)}</div></div>""",unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        ranked = sorted(algo_data.items(),key=lambda x:x[1]["rmse"])
        medals = {1:"🥇",2:"🥈",3:"🥉"}
        rcols  = st.columns(min(len(ranked),4))
        for ri,(rn,rm) in enumerate(ranked[:4]):
            medal = medals.get(ri+1,f"#{ri+1}")
            rbg   = [COLORS["accent"],COLORS["primary"],COLORS["secondary"],COLORS["warning"]][ri%4]
            rcols[ri].markdown(f"""
            <div class="stat-card" style="--sc:{rbg};animation-delay:{ri*0.08}s">
                <div style="font-size:1.6rem">{medal}</div>
                <div style="font-weight:800;font-size:0.85rem;color:{rbg};font-family:'Space Grotesk',sans-serif;margin:0.3rem 0 0.1rem">{rn}</div>
                <div style="font-size:0.7rem;color:{COLORS['subtext']};font-family:'JetBrains Mono',monospace">RMSE {rm['rmse']:.4f}°C</div>
                <div style="font-size:0.65rem;color:{COLORS['subtext']};font-family:'JetBrains Mono',monospace">R² {rm['r2']:.4f} · Skill {rm.get('skill_vs_baseline',0):.3f}</div>
            </div>""",unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        tbl_rows=[]
        for n,m in algo_data.items():
            tbl_rows.append({"Model":n+(" 🏆" if n==top_algo else ""),
                             "RMSE":m["rmse"],"MAE":m.get("mae","—"),
                             "R²":m["r2"],"CV RMSE":m["cv_rmse"],"CV Std":m["cv_std"],
                             "Skill":m.get("skill_vs_baseline","—")})
        df_tbl=pd.DataFrame(tbl_rows).sort_values("RMSE")
        st.dataframe(
            df_tbl.style
            .apply(lambda r:["color:#3fb950;font-weight:700"]*len(r) if "🏆" in str(r["Model"]) else [""]*len(r),axis=1)
            .format({"RMSE":"{:.4f}","R²":"{:.4f}","CV RMSE":"{:.4f}","CV Std":"{:.4f}",
                     "MAE":lambda v:f"{v:.4f}" if isinstance(v,float) else v,
                     "Skill":lambda v:f"{v:.4f}" if isinstance(v,float) else v}),
            use_container_width=True,hide_index=True)

        if bl_rmse:
            st.markdown(f"""<div style="font-size:0.77rem;color:{COLORS['subtext']};margin:0.3rem 0 1rem">
                📏 Baseline (always predict mean): RMSE = <b style="color:{COLORS['warning']}">{bl_rmse}°C</b>
                {f'  ·  MAE = <b style="color:{COLORS["warning"]}">{bl_mae}°C</b>' if bl_mae else ''}
                &nbsp;·&nbsp; Skill = <b>1 − model_RMSE / baseline_RMSE</b></div>""",unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cc1,cc2=st.columns(2)
        with cc1:
            bc=[COLORS["accent"] if "🏆" in str(n) else COLORS["primary"] for n in df_tbl["Model"]]
            fR=go.Figure(go.Bar(x=df_tbl["RMSE"],y=df_tbl["Model"],orientation="h",marker_color=bc,
                                text=df_tbl["RMSE"].round(4),textposition="outside",textfont_color=COLORS["subtext"]))
            if bl_rmse:
                fR.add_vline(x=bl_rmse,line_dash="dash",line_color=COLORS["warning"],line_width=2,
                             annotation_text=f"Baseline {bl_rmse}",annotation_font_color=COLORS["warning"])
            fR.update_layout(title="RMSE  (lower = better)",xaxis_title="°C",**_THEME)
            st.plotly_chart(fR,use_container_width=True)
        with cc2:
            mae_vals=df_tbl["MAE"].apply(lambda v:v if isinstance(v,float) else 0)
            bc2=[COLORS["accent"] if "🏆" in str(n) else COLORS["secondary"] for n in df_tbl["Model"]]
            fM=go.Figure(go.Bar(x=mae_vals,y=df_tbl["Model"],orientation="h",marker_color=bc2,
                                text=mae_vals.round(4),textposition="outside",textfont_color=COLORS["subtext"]))
            if bl_mae:
                fM.add_vline(x=bl_mae,line_dash="dash",line_color=COLORS["warning"],line_width=2,
                             annotation_text=f"Baseline {bl_mae}",annotation_font_color=COLORS["warning"])
            fM.update_layout(title="MAE  (lower = better)",xaxis_title="°C",**_THEME)
            st.plotly_chart(fM,use_container_width=True)

        skill_map={n:m.get("skill_vs_baseline",0) for n,m in algo_data.items()
                   if isinstance(m.get("skill_vs_baseline"),(int,float))}
        if skill_map:
            section("⚡ Skill Score vs Baseline")
            sk_df=pd.DataFrame(sorted(skill_map.items(),key=lambda x:x[1],reverse=True),columns=["Model","Skill"])
            fsk=go.Figure(go.Bar(x=sk_df["Skill"],y=sk_df["Model"],orientation="h",
                                 marker_color=[COLORS["accent"] if s>0 else COLORS["danger"] for s in sk_df["Skill"]],
                                 text=sk_df["Skill"].round(4),textposition="outside",textfont_color=COLORS["subtext"],
                                 hovertemplate="<b>%{y}</b><br>Skill: %{x:.4f}<extra></extra>"))
            fsk.add_vline(x=0,line_color=COLORS["border"],line_width=1.5)
            fsk.update_layout(title="1 − RMSE/Baseline_RMSE  · green = beats baseline",xaxis_title="Skill Score",**_THEME)
            st.plotly_chart(fsk,use_container_width=True)

        section("🕸️ Model Comparison Radar","Each axis normalised 0→1 (higher = better on all axes)")
        r_a,r_b=st.columns([3,2])
        with r_a:
            mnames=list(algo_data.keys())
            r_rmse=np.array([algo_data[n]["rmse"] for n in mnames])
            r_mae =np.array([algo_data[n].get("mae",algo_data[n]["rmse"]) for n in mnames])
            r_r2  =np.array([max(0,algo_data[n]["r2"]) for n in mnames])
            r_sk  =np.array([max(0,algo_data[n].get("skill_vs_baseline",0)) for n in mnames])
            r_stab=np.array([1/(algo_data[n]["cv_std"]+0.01) for n in mnames])
            def n01(a):
                mn,mx=a.min(),a.max()
                return (a-mn)/(mx-mn+1e-9) if mx>mn else np.ones_like(a)*0.5
            axes_scores={"RMSE (inv)":1-n01(r_rmse),"MAE (inv)":1-n01(r_mae),
                         "R²":n01(r_r2),"Skill":n01(r_sk),"CV Stability":n01(r_stab)}
            ax_list=list(axes_scores.keys())
            fr=go.Figure()
            for i,nm in enumerate(sorted(mnames,key=lambda n:algo_data[n]["rmse"])[:6]):
                idx=mnames.index(nm)
                rv=[axes_scores[a][idx] for a in ax_list]+[axes_scores[ax_list[0]][idx]]
                fr.add_trace(go.Scatterpolar(r=rv,theta=ax_list+[ax_list[0]],fill="toself",name=nm,
                                             line=dict(color=PALETTE[i%len(PALETTE)],width=2),
                                             fillcolor=rgba_from_hex(PALETTE[i%len(PALETTE)],0.13),
                                             marker=dict(color=PALETTE[i%len(PALETTE)],size=6)))
            fr.update_layout(polar=dict(bgcolor=COLORS["card2"],
                                        radialaxis=dict(visible=True,range=[0,1],tickcolor=COLORS["subtext"],gridcolor="#1c2230",linecolor="#1c2230"),
                                        angularaxis=dict(tickcolor=COLORS["subtext"],gridcolor="#1c2230",linecolor="#1c2230")),
                             showlegend=True,height=380,paper_bgcolor="rgba(0,0,0,0)",
                             font=dict(color=COLORS["text"],family="Inter",size=11),
                             margin=dict(l=30,r=30,t=30,b=30),
                             legend=dict(bgcolor="rgba(13,17,23,0.8)",bordercolor=COLORS["border"],borderwidth=1,font_size=11))
            st.plotly_chart(fr,use_container_width=True)
        with r_b:
            r2_list=[(n,algo_data[n]["r2"],algo_data[n]["rmse"]) for n in mnames]
            r2_df=pd.DataFrame(r2_list,columns=["Model","R²","RMSE"])
            fb2=px.scatter(r2_df,x="RMSE",y="R²",text="Model",size=[20]*len(r2_df),color="R²",
                           color_continuous_scale=[[0,COLORS["danger"]],[0.5,COLORS["warning"]],[1,COLORS["accent"]]],
                           labels={"RMSE":"RMSE (°C)","R²":"R²"})
            fb2.update_traces(textposition="top center",textfont_size=9,marker_sizemin=12)
            if bl_rmse:
                fb2.add_vline(x=bl_rmse,line_dash="dash",line_color=COLORS["warning"],line_width=1.5)
            fb2.update_layout(title="R² vs RMSE",coloraxis_showscale=False,**_THEME)
            st.plotly_chart(fb2,use_container_width=True)

        if mdl and scl and proc_df is not None and fnames:
            st.markdown("<hr>", unsafe_allow_html=True)
            section("🎯 Predicted vs Actual  (full dataset)",
                    "OLS trendline shown in amber · dashed = perfect prediction line")
            try:
                Xa  =proc_df[[f for f in fnames if f in proc_df.columns]].values
                Xsc =scl.transform(Xa)
                yp  =mdl.predict(Xsc)
                ya  =proc_df["uhi_intensity"].values
                cc  =proc_df.get("city_name",pd.Series(["all"]*len(ya)))
                pva =pd.DataFrame({"Actual":ya,"Predicted":np.clip(yp,0,None),
                                   "City":cc.values if hasattr(cc,"values") else cc})
                samp=pva.sample(min(3000,len(pva)),random_state=42)
                fpv=go.Figure()
                for i,city in enumerate(samp["City"].unique()):
                    sub=samp[samp["City"]==city]
                    fpv.add_trace(go.Scatter(x=sub["Actual"],y=sub["Predicted"],mode="markers",name=city,
                                             marker=dict(color=PALETTE[i%len(PALETTE)],size=4,opacity=0.6,line=dict(width=0)),
                                             hovertemplate=f"<b>{city}</b><br>Actual: %{{x:.2f}}°C<br>Predicted: %{{y:.2f}}°C<extra></extra>"))
                lim=max(ya.max(),yp.max())*1.05
                fpv.add_trace(go.Scatter(x=[0,lim],y=[0,lim],mode="lines",name="Perfect",
                                         line=dict(dash="dash",color=COLORS["warning"],width=2),showlegend=True))
                sl,ic,rv,_,_=sp_stats.linregress(ya,np.clip(yp,0,None))
                xr=np.linspace(0,lim,100)
                fpv.add_trace(go.Scatter(x=xr,y=sl*xr+ic,mode="lines",name=f"OLS  r={rv:.3f}",
                                         line=dict(color=COLORS["accent"],width=2.5)))
                fpv.update_layout(title=f"Predicted vs Actual UHI — {top_algo}  (all {len(pva):,} rows)",
                                  xaxis_title="Actual UHI (°C)",yaxis_title="Predicted UHI (°C)",
                                  xaxis=dict(**_THEME["xaxis"],range=[0,lim]),
                                  yaxis=dict(**_THEME["yaxis"],range=[0,lim]),**_THEME)
                st.plotly_chart(fpv,use_container_width=True)
            except Exception as e:
                st.info(f"Predicted vs Actual not available: {e}")

        if mdl and scl and proc_df is not None and fnames:
            st.markdown("<hr>", unsafe_allow_html=True)
            section("📐 Residual Analysis",
                    "Error distribution · per-city breakdown · ideal residuals cluster around zero")
            try:
                Xr  =proc_df[[f for f in fnames if f in proc_df.columns]].values
                ypr =mdl.predict(scl.transform(Xr))
                yar =proc_df["uhi_intensity"].values
                res =yar-np.clip(ypr,0,None)
                ra,rb,rc=st.columns(3)
                with ra:
                    xh=np.linspace(res.min(),res.max(),200)
                    mu_r,sd_r=res.mean(),res.std()
                    pdf_r=sp_stats.norm.pdf(xh,mu_r,sd_r)*len(res)*(res.max()-res.min())/40
                    frh=go.Figure()
                    frh.add_trace(go.Histogram(x=res,nbinsx=40,name="Residuals",
                                              marker_color=COLORS["primary"],opacity=0.75,
                                              marker_line_color=COLORS["card"],marker_line_width=0.5))
                    frh.add_trace(go.Scatter(x=xh,y=pdf_r,mode="lines",name="Normal fit",
                                             line=dict(color=COLORS["secondary"],width=2.5)))
                    frh.add_vline(x=0,line_dash="dash",line_color=COLORS["warning"],line_width=1.5)
                    frh.update_layout(title=f"Residuals  μ={mu_r:.3f}  σ={sd_r:.3f}",
                                      xaxis_title="Actual − Predicted (°C)",showlegend=True,height=300,**_THEME)
                    st.plotly_chart(frh,use_container_width=True)
                with rb:
                    frp=go.Figure(go.Scatter(x=np.clip(ypr,0,None),y=res,mode="markers",
                                             marker=dict(color=res,colorscale=[[0,COLORS["danger"]],[0.5,COLORS["subtext"]],[1,COLORS["accent"]]],
                                                         size=3,opacity=0.5,showscale=False),
                                             hovertemplate="Pred: %{x:.2f}°C<br>Residual: %{y:.2f}°C<extra></extra>"))
                    frp.add_hline(y=0,line_dash="dash",line_color=COLORS["warning"],line_width=1.5)
                    frp.update_layout(title="Residuals vs Predicted",xaxis_title="Predicted UHI (°C)",
                                      yaxis_title="Residual (°C)",height=300,**_THEME)
                    st.plotly_chart(frp,use_container_width=True)
                with rc:
                    if "city_name" in proc_df.columns:
                        ce=pd.DataFrame({"city":proc_df["city_name"].values,"abs_err":np.abs(res)})
                        ce=ce.groupby("city")["abs_err"].mean().sort_values(ascending=True)
                        bce=[COLORS["accent"] if v<ce.median() else COLORS["danger"] for v in ce.values]
                        fce=go.Figure(go.Bar(x=ce.values,y=ce.index,orientation="h",marker_color=bce,
                                             text=ce.values.round(3),textposition="outside",textfont_color=COLORS["subtext"],
                                             hovertemplate="<b>%{y}</b><br>MAE: %{x:.3f}°C<extra></extra>"))
                        fce.update_layout(title="Per-city MAE (green=below median)",xaxis_title="MAE (°C)",height=300,**_THEME)
                        st.plotly_chart(fce,use_container_width=True)
                mae_r=np.mean(np.abs(res)); rmse_r=np.sqrt(np.mean(res**2))
                st.markdown(f"""<div class="res-footer">
                    Test residuals — MAE <b style="color:{COLORS['primary']}">{mae_r:.4f} °C</b>
                    · RMSE <b style="color:{COLORS['primary']}">{rmse_r:.4f} °C</b>
                    · Skewness <b style="color:{COLORS['subtext']}">{float(pd.Series(res).skew()):.3f}</b>
                    · Kurtosis <b style="color:{COLORS['subtext']}">{float(pd.Series(res).kurtosis()):.3f}</b>
                </div>""",unsafe_allow_html=True)
            except Exception as e:
                st.info(f"Residual analysis unavailable: {e}")

        st.markdown("<hr>", unsafe_allow_html=True)
        section("🔍 Feature Importance")
        fi_c1,_=st.columns([1,3])
        with fi_c1:
            fi_sel=st.selectbox("Model",list(algo_data.keys()),
                                index=list(algo_data.keys()).index(top_algo) if top_algo in algo_data else 0)
        fi=algo_data[fi_sel].get("feature_importance",{})
        if fi:
            fi_df=(pd.DataFrame(list(fi.items()),columns=["Feature","Importance"])
                   .sort_values("Importance",ascending=True).tail(15))
            med_imp=fi_df["Importance"].median()
            bfi=[COLORS["secondary"] if v>med_imp else COLORS["primary"] for v in fi_df["Importance"]]
            ffi=go.Figure(go.Bar(x=fi_df["Importance"],y=fi_df["Feature"],orientation="h",marker_color=bfi,
                                 text=fi_df["Importance"].map(lambda v:f"{v:.4f}"),
                                 textposition="outside",textfont_color=COLORS["subtext"],
                                 hovertemplate="<b>%{y}</b><br>Importance: %{x:.5f}<extra></extra>"))
            ffi.update_layout(title=f"Top-15 Features — {fi_sel}",xaxis_title="Normalised Importance",**_THEME)
            st.plotly_chart(ffi,use_container_width=True)
        else:
            st.info("Feature importance not available for this model type (e.g., KNN).")

        section("📊 Cross-Validation RMSE ± Std  (GroupKFold · 5 folds)")
        sorted_algos=sorted(algo_data.items(),key=lambda x:x[1]["cv_rmse"])
        fcv=go.Figure()
        for nm,m in sorted_algos:
            fcv.add_trace(go.Bar(name=nm,x=[nm.replace(" ","<br>")],y=[m["cv_rmse"]],
                                 error_y=dict(type="data",array=[m["cv_std"]],color=COLORS["subtext"],thickness=1.5,width=6),
                                 marker_color=COLORS["accent"] if nm==top_algo else COLORS["primary"],
                                 text=[f"{m['cv_rmse']:.3f}"],textposition="outside",textfont_color=COLORS["subtext"]))
        if bl_rmse:
            fcv.add_hline(y=bl_rmse,line_dash="dash",line_color=COLORS["warning"],line_width=2,
                          annotation_text=f"Baseline {bl_rmse}",annotation_font_color=COLORS["warning"])
        fcv.update_layout(barmode="group",showlegend=False,yaxis_title="RMSE (°C)",
                          title="CV RMSE ± Std — lower + narrower = better",**_THEME)
        st.plotly_chart(fcv,use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — HEATMAP
# ══════════════════════════════════════════════════════════════
elif page == "🗺️  Heatmap":
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    section("🗺️ Global UHI Heatmap","Mean UHI intensity per city across the 180-day observation window")

    if proc_df is None:
        st.warning("Run `python main.py` first.")
    else:
        df_map=proc_df.copy()
        if raw_df is not None and "name" in raw_df.columns:
            ci=raw_df[["lat","lon","name"]].drop_duplicates(subset=["lat","lon"])
            df_map=df_map.merge(ci,on=["lat","lon"],how="left")
        elif "city_name" in df_map.columns:
            df_map=df_map.rename(columns={"city_name":"name"})

        if "uhi_intensity" in df_map.columns:
            gcols=["name","lat","lon"] if "name" in df_map.columns else ["lat","lon"]
            agg=(df_map[gcols+["uhi_intensity"]].groupby(gcols).mean().reset_index())
            ctrl,map_area=st.columns([1,4])
            with ctrl:
                mn,mx=float(agg["uhi_intensity"].min()),float(agg["uhi_intensity"].max())
                rng=st.slider("UHI range (°C)",mn,mx,(mn,mx),step=0.1)
                mtype=st.selectbox("Style",["Scatter","Density"])
                mtheme=st.selectbox("Base map",["carto-darkmatter","carto-positron","open-street-map"])
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""<div class="frosted" style="font-size:0.78rem">
                    <div style="color:{COLORS['subtext']};font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">Summary</div>
                    <div style="line-height:2">
                        <span style="color:{COLORS['subtext']}">Cities: </span><b style="color:{COLORS['text']}">{len(agg)}</b><br>
                        <span style="color:{COLORS['subtext']}">Min: </span><b style="color:{COLORS['accent']}">{mn:.2f}°C</b><br>
                        <span style="color:{COLORS['subtext']}">Max: </span><b style="color:{COLORS['danger']}">{mx:.2f}°C</b><br>
                        <span style="color:{COLORS['subtext']}">Mean: </span><b style="color:{COLORS['warning']}">{agg['uhi_intensity'].mean():.2f}°C</b>
                    </div></div>""",unsafe_allow_html=True)
            df_filt=agg[agg["uhi_intensity"].between(rng[0],rng[1])]
            with map_area:
                cs=[[0,COLORS["accent"]],[0.4,COLORS["warning"]],[1,COLORS["danger"]]]
                if mtype=="Scatter":
                    fm=px.scatter_mapbox(df_filt,lat="lat",lon="lon",color="uhi_intensity",size="uhi_intensity",
                                         hover_name="name" if "name" in df_filt.columns else None,
                                         hover_data={"uhi_intensity":":.2f","lat":False,"lon":False},
                                         color_continuous_scale=cs,size_max=32,zoom=1.1,mapbox_style=mtheme)
                else:
                    fm=px.density_mapbox(df_filt,lat="lat",lon="lon",z="uhi_intensity",radius=55,zoom=1.1,
                                          mapbox_style=mtheme,
                                          color_continuous_scale=[[0,"rgba(63,185,80,0)"],[0.5,COLORS["warning"]],[1,COLORS["danger"]]])
                fm.update_layout(height=500,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor=COLORS["card"],
                                 coloraxis_colorbar=dict(title=dict(text="°C",font_color=COLORS["subtext"]),
                                                         tickcolor=COLORS["subtext"],tickfont_color=COLORS["subtext"],
                                                         bgcolor=COLORS["card"],outlinecolor=COLORS["border"]),
                                 font_color=COLORS["text"])
                st.plotly_chart(fm,use_container_width=True)

            if "name" in agg.columns:
                ct,cb2=st.columns(2)
                for cw,asc,ttl,cs2 in [(ct,False,"🔴 Hottest 10",[[0,COLORS["warning"]],[1,COLORS["danger"]]]),
                                        (cb2,True,"🟢 Coolest 10",[[0,COLORS["accent"]],[1,COLORS["primary"]]])]:
                    sub=agg.sort_values("uhi_intensity",ascending=asc).head(10)
                    fbar=go.Figure(go.Bar(x=sub["uhi_intensity"],y=sub["name"],orientation="h",
                                         marker=dict(color=sub["uhi_intensity"],colorscale=cs2,showscale=False),
                                         text=sub["uhi_intensity"].round(2),textposition="outside",textfont_color=COLORS["subtext"]))
                    fbar.update_layout(title=ttl,**_THEME)
                    cw.plotly_chart(fbar,use_container_width=True)

            if "city_name" in proc_df.columns and "month" in proc_df.columns:
                st.markdown("<hr>", unsafe_allow_html=True)
                section("📅 Seasonal UHI Heatmap  (City × Month)",
                        "Mean UHI per city for each month of the year")
                pivot=(proc_df.groupby(["city_name","month"])["uhi_intensity"].mean().unstack(fill_value=0))
                pivot.columns=[MO_LABELS.get(int(c),c) for c in pivot.columns]
                fsh=go.Figure(go.Heatmap(z=pivot.values,x=pivot.columns,y=pivot.index,
                                          colorscale=[[0,"#0d2b1a"],[0.3,COLORS["accent"]],[0.6,COLORS["warning"]],[1,COLORS["danger"]]],
                                          text=np.round(pivot.values,1),texttemplate="%{text}",textfont_size=8,
                                          hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}°C<extra></extra>",
                                          colorbar=dict(title=dict(text="°C",font_color=COLORS["subtext"]),
                                                        tickcolor=COLORS["subtext"],tickfont_color=COLORS["subtext"],
                                                        bgcolor=COLORS["card"],outlinecolor=COLORS["border"])))
                fsh.update_layout(title="Mean UHI (°C) — city × month",height=max(320,len(pivot)*20+80),**_THEME)
                st.plotly_chart(fsh,use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 6 — PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "🔮  Prediction":
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    section("🔮 Live UHI Prediction",
            "Sliders update instantly · all 26 engineered features computed in real time")

    if mdl is None or scl is None or fnames is None:
        st.error("No trained model found. Run `python main.py` first.")
    else:
        inp_col, out_col = st.columns([1.6,1], gap="large")

        with inp_col:
            def grp(icon,title):
                st.markdown(f'<div style="font-size:0.68rem;font-weight:800;color:{COLORS["subtext"]};text-transform:uppercase;letter-spacing:0.12em;margin:0.8rem 0 0.5rem">{icon} {title}</div>',unsafe_allow_html=True)

            grp("🌡️","Weather Conditions")
            wa,wb=st.columns(2)
            with wa:
                temperature=st.slider("Temperature (°C)",-10.0,50.0,32.0,0.5)
                wind_speed =st.slider("Wind Speed (m/s)",0.0,20.0,2.5,0.1)
                pressure   =st.slider("Pressure (hPa)",950.0,1050.0,1013.0,0.5)
            with wb:
                humidity=st.slider("Humidity (%)",0,100,65,1)
                clouds  =st.slider("Cloud Cover (%)",0,100,30,1)

            grp("🌿","Land Cover")
            la,lb=st.columns(2)
            with la: ndvi      =st.slider("NDVI",0.0,1.0,0.25,0.01)
            with lb: urban_frac=st.slider("Urban Fraction",0.0,1.0,0.70,0.01)
            vc="Sparse (0)" if ndvi<0.2 else ("Dense (2)" if ndvi>=0.5 else "Moderate (1)")
            st.markdown(f'<div style="font-size:0.72rem;color:{COLORS["subtext"]};margin-top:-0.3rem">veg_class → <span class="mono-tag">{vc}</span> &nbsp;·&nbsp; heat_retention → <span class="mono-tag">{urban_frac*temperature/(wind_speed+1):.2f}</span></div>',unsafe_allow_html=True)

            grp("📍","Location & Time")
            lc,ld=st.columns(2)
            with lc:
                lat  =st.slider("Latitude",-90.0,90.0,28.6,0.1)
                hour =st.slider("Hour",0,23,14,1)
            with ld:
                lon  =st.slider("Longitude",-180.0,180.0,77.2,0.1)
                month=st.slider("Month",1,12,6,1)

            st.markdown("<br>", unsafe_allow_html=True)
            preset=st.selectbox("⚡ Quick-load city",["Custom"]+[c["name"] for c in CITIES])
            if preset!="Custom":
                cd=next(c for c in CITIES if c["name"]==preset)
                lat=cd["lat"]; lon=cd["lon"]
                st.markdown(f'<div style="font-size:0.72rem;color:{COLORS["accent"]}">✓ Loaded <b>{preset}</b>: lat={lat}, lon={lon}</div>',unsafe_allow_html=True)

        with out_col:
            try:
                vec =compute_input_vector(temperature,humidity,wind_speed,ndvi,urban_frac,
                                          lat,lon,hour,month,pressure,clouds,fnames)
                pred=float(max(0.0,mdl.predict(scl.transform([vec]))[0]))

                if pred<1:   lv,lc2,lbg="Low",     COLORS["accent"], "rgba(63,185,80,0.1)"
                elif pred<2: lv,lc2,lbg="Moderate",COLORS["warning"],"rgba(210,153,34,0.1)"
                elif pred<4: lv,lc2,lbg="High",    COLORS["danger"], "rgba(248,81,73,0.1)"
                else:        lv,lc2,lbg="Extreme", "#ff6b6b",        "rgba(255,107,107,0.12)"

                st.markdown(f"""
                <div class="result-box">
                    <div style="font-size:0.68rem;color:{COLORS['subtext']};font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem">Predicted UHI Intensity</div>
                    <div class="result-num">{pred:.2f} °C</div>
                    <div><span class="level-tag" style="background:{lbg};border-color:{lc2}44;color:{lc2}">{lv} UHI</span></div>
                    <div style="margin-top:1rem;font-size:0.68rem;color:{COLORS['subtext']}">
                        Model: <b style="color:{COLORS['primary']}">{result_data['best_model'] if result_data else '—'}</b>
                        &nbsp;·&nbsp; RMSE: <b style="color:{COLORS['primary']}">{result_data['best_rmse'] if result_data else '—'}°C</b>
                    </div>
                </div>""",unsafe_allow_html=True)

                fg=go.Figure(go.Indicator(
                    mode="gauge+number+delta",value=pred,
                    number={"suffix":" °C","font":{"size":32,"color":COLORS["text"]}},
                    delta={"reference":result_data.get("baseline_rmse",2.0) if result_data else 2.0,
                           "valueformat":".2f","font":{"size":13}},
                    gauge={"axis":{"range":[0,8],"tickwidth":1,"tickcolor":COLORS["subtext"],"nticks":9},
                           "bar":{"color":lc2,"thickness":0.25},
                           "bgcolor":COLORS["card2"],"bordercolor":COLORS["border"],"borderwidth":1,
                           "steps":[{"range":[0,1],"color":"rgba(63,185,80,0.18)"},
                                    {"range":[1,2],"color":"rgba(210,153,34,0.18)"},
                                    {"range":[2,4],"color":"rgba(248,81,73,0.18)"},
                                    {"range":[4,8],"color":"rgba(255,107,107,0.22)"}],
                           "threshold":{"line":{"color":COLORS["text"],"width":2},"thickness":0.8,"value":pred}},
                    title={"text":"UHI Severity","font":{"color":COLORS["subtext"],"size":12}},
                ))
                fg.update_layout(height=255,paper_bgcolor="rgba(0,0,0,0)",
                                 font=dict(color=COLORS["text"],family="Inter"),
                                 margin=dict(l=18,r=18,t=28,b=8))
                st.plotly_chart(fg,use_container_width=True)

                st.markdown(f'<div class="section-title" style="font-size:1rem;margin-top:0.5rem">Key Drivers</div>',unsafe_allow_html=True)
                drv={"Urban Fraction":urban_frac,"Temperature":min(1.0,max(0.0,(temperature+10)/60)),
                     "Low Vegetation":1-ndvi,"High Humidity":humidity/100,
                     "Low Wind":1-min(1.0,wind_speed/15),"No Clouds":1-clouds/100}
                fr2=go.Figure(go.Scatterpolar(
                    r=list(drv.values())+[list(drv.values())[0]],
                    theta=list(drv.keys())+[list(drv.keys())[0]],
                    fill="toself",fillcolor="rgba(88,166,255,0.1)",
                    line=dict(color=COLORS["primary"],width=2.5),
                    marker=dict(color=COLORS["primary"],size=7)))
                fr2.update_layout(polar=dict(bgcolor=COLORS["card2"],
                                             radialaxis=dict(visible=True,range=[0,1],tickcolor=COLORS["subtext"],gridcolor="#1c2230",linecolor="#1c2230"),
                                             angularaxis=dict(tickcolor=COLORS["subtext"],gridcolor="#1c2230",linecolor="#1c2230")),
                                  showlegend=False,height=290,paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color=COLORS["text"],family="Inter",size=10),
                                  margin=dict(l=28,r=28,t=16,b=16))
                st.plotly_chart(fr2,use_container_width=True)

                st.markdown(f'<div class="section-title" style="font-size:1rem;margin-top:0.6rem">Scaled Feature Inputs</div>',unsafe_allow_html=True)
                render_scaled_bars(scl.transform([vec])[0], fnames)

            except Exception as e:
                st.error(f"Prediction error: {e}")

        st.markdown("<hr>", unsafe_allow_html=True)
        section("📐 Sensitivity Analysis",
                "How predicted UHI changes as one parameter varies — all others fixed at slider values")
        PMAP={"temperature":"t","humidity":"rh","wind_speed":"ws","ndvi":"ndvi_v",
              "urban_fraction":"uf","pressure":"pres","clouds":"cld"}
        PRANGES={"temperature":(-10.0,50.0),"urban_fraction":(0.0,1.0),"ndvi":(0.0,1.0),
                 "humidity":(10.0,100.0),"wind_speed":(0.0,20.0),
                 "pressure":(950.0,1050.0),"clouds":(0.0,100.0)}
        sc1,sc2=st.columns([1,4])
        with sc1: vary=st.selectbox("Vary",list(PRANGES.keys()))
        if fnames:
            lo,hi=PRANGES[vary]; pr=np.linspace(lo,hi,60)
            base=dict(t=temperature,rh=humidity,ws=wind_speed,ndvi_v=ndvi,uf=urban_frac,
                      lat=lat,lon=lon,hr=hour,mo=month,pres=pressure,cld=clouds)
            bk=PMAP[vary]; psa=[]
            for v in pr:
                vs=compute_input_vector(**{**base,bk:v},fnames=fnames)
                psa.append(max(0.0,float(mdl.predict(scl.transform([vs]))[0])))
            cur=base.get(bk)
            fsa=go.Figure()
            fsa.add_trace(go.Scatter(x=pr,y=psa,mode="lines",fill="tozeroy",
                                     line=dict(color=COLORS["primary"],width=2.5),
                                     fillcolor="rgba(88,166,255,0.07)",
                                     hovertemplate=f"<b>{vary}</b>: %{{x:.2f}}<br>UHI: %{{y:.3f}}°C<extra></extra>"))
            if cur is not None and lo<=cur<=hi:
                cp=max(0.0,float(mdl.predict(scl.transform([compute_input_vector(**{**base,bk:cur},fnames=fnames)]))[0]))
                fsa.add_vline(x=cur,line_dash="dash",line_color=COLORS["secondary"],line_width=2,
                              annotation_text=f"current {cur:.2f}",annotation_font_color=COLORS["secondary"])
                fsa.add_trace(go.Scatter(x=[cur],y=[cp],mode="markers",showlegend=False,
                                         marker=dict(color=COLORS["secondary"],size=11,line=dict(color=COLORS["background"],width=2)),
                                         hovertemplate=f"Current: {cur:.2f}<br>UHI: {cp:.3f}°C<extra></extra>"))
            fsa.update_layout(xaxis_title=vary.replace("_"," ").title(),yaxis_title="Predicted UHI (°C)",
                              title=f"UHI Sensitivity — {vary.replace('_',' ').title()}",**_THEME)
            with sc2: st.plotly_chart(fsa,use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        section("🌐 3D Response Surface",
                "How UHI changes with Temperature × Urban Fraction — all other parameters fixed at slider values")
        if fnames:
            try:
                t_ax=np.linspace(-5.0,50.0,35); uf_ax=np.linspace(0.0,1.0,35)
                TG,UFG=np.meshgrid(t_ax,uf_ax); ZG=np.zeros_like(TG)
                b3=dict(rh=humidity,ws=wind_speed,ndvi_v=ndvi,lat=lat,lon=lon,hr=hour,mo=month,pres=pressure,cld=clouds)
                for i in range(len(uf_ax)):
                    for j in range(len(t_ax)):
                        v3=compute_input_vector(t=float(TG[i,j]),uf=float(UFG[i,j]),**b3,fnames=fnames)
                        ZG[i,j]=max(0.0,float(mdl.predict(scl.transform([v3]))[0]))
                f3d=go.Figure(go.Surface(x=TG,y=UFG,z=ZG,
                                          colorscale=[[0,"#0d2b1a"],[0.25,COLORS["accent"]],[0.55,COLORS["warning"]],[1,COLORS["danger"]]],
                                          opacity=0.92,
                                          contours=dict(z=dict(show=True,usecolormap=True,highlightcolor="#e6edf3",project_z=True,width=1)),
                                          hovertemplate="Temp: %{x:.1f}°C<br>Urban Frac: %{y:.2f}<br>UHI: %{z:.2f}°C<extra></extra>"))
                cz=max(0.0,float(mdl.predict(scl.transform([compute_input_vector(t=temperature,uf=urban_frac,**b3,fnames=fnames)]))[0]))
                f3d.add_trace(go.Scatter3d(x=[temperature],y=[urban_frac],z=[cz+0.05],mode="markers+text",
                                            marker=dict(color=COLORS["secondary"],size=8,symbol="circle",line=dict(color=COLORS["background"],width=2)),
                                            text=["← You"],textfont=dict(color=COLORS["secondary"],size=11),showlegend=False,
                                            hovertemplate=f"Your point<br>Temp: {temperature:.1f}°C<br>Urban: {urban_frac:.2f}<br>UHI: {cz:.2f}°C<extra></extra>"))
                f3d.update_layout(
                    scene=dict(xaxis=dict(title="Temperature (°C)",gridcolor="#1c2230",backgroundcolor=COLORS["card"],color=COLORS["subtext"]),
                               yaxis=dict(title="Urban Fraction",gridcolor="#1c2230",backgroundcolor=COLORS["card"],color=COLORS["subtext"]),
                               zaxis=dict(title="UHI (°C)",gridcolor="#1c2230",backgroundcolor=COLORS["card"],color=COLORS["subtext"]),
                               bgcolor=COLORS["card"]),
                    paper_bgcolor="rgba(0,0,0,0)",font=dict(color=COLORS["text"],family="Inter"),
                    margin=dict(l=0,r=0,t=36,b=0),height=500,
                    title=dict(text="UHI = f(Temperature, Urban Fraction)",font=dict(color=COLORS["text"],size=13)))
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.plotly_chart(f3d,use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.info(f"3D surface unavailable: {e}")

        st.markdown("<hr>", unsafe_allow_html=True)
        sim_col,mit_col=st.columns(2)

        with sim_col:
            section("🔍 City Similarity","Which training city does your input climate profile most resemble?")
            if proc_df is not None and "city_name" in proc_df.columns and fnames:
                try:
                    profiles=(proc_df.groupby("city_name")[[f for f in fnames if f in proc_df.columns]].mean())
                    u_sc=np.array(scl.transform([vec])[0])
                    p_sc=scl.transform(profiles.values)
                    dists=np.linalg.norm(p_sc-u_sc,axis=1)
                    top5i=dists.argsort()[:5]
                    top5c=profiles.index[top5i]; top5d=dists[top5i]; mx_d=top5d.max() or 1.0
                    pills="".join(
                        f'<span class="city-chip">{CITY_FLAGS.get(c,"🏙")} {c}'
                        f'<span style="color:{COLORS["subtext"]};font-size:0.62rem;margin-left:0.3rem">{100*(1-d/mx_d):.0f}%</span>'
                        f'</span>'
                        for c,d in zip(top5c,top5d))
                    st.markdown(
                        f'<div class="glow-card" style="padding:1rem 1.2rem">'
                        f'<div style="font-size:0.62rem;font-weight:800;text-transform:uppercase;letter-spacing:0.12em;color:{COLORS["subtext"]};margin-bottom:0.7rem">Most similar cities (feature-space distance)</div>'
                        f'<div style="display:flex;flex-wrap:wrap">{pills}</div></div>',
                        unsafe_allow_html=True)
                    cl=top5c[0]
                    cl_uhi=proc_df[proc_df["city_name"]==cl]["uhi_intensity"].mean()
                    dv=pred-cl_uhi
                    ar="▲" if dv>0 else "▼"
                    dc=COLORS["danger"] if dv>0 else COLORS["accent"]
                    st.markdown(f"""
                    <div class="note-card" style="--nc:{COLORS['primary']};margin-top:0.6rem">
                        Closest match: <strong>{CITY_FLAGS.get(cl,"🏙")} {cl}</strong><br>
                        Mean UHI there: <strong>{cl_uhi:.2f}°C</strong> · Your prediction: <strong>{pred:.2f}°C</strong>
                        <span style="color:{dc};font-weight:700"> {ar} {abs(dv):.2f}°C {"hotter" if dv>0 else "cooler"}</span>
                    </div>""",unsafe_allow_html=True)
                except Exception as e:
                    st.info(f"City similarity unavailable: {e}")

        with mit_col:
            section("💡 Mitigation Insights","Evidence-based strategies to reduce UHI at the predicted intensity level")
            hints=[]
            if urban_frac>0.6:
                hints.append(("🌳","Increase urban greenery",
                    f"Urban fraction is {urban_frac:.0%}. Adding green roofs or parks could reduce NDVI gap and lower surface temperature by 1–3°C."))
            if ndvi<0.25:
                hints.append(("🌿","Vegetation cover",
                    f"NDVI of {ndvi:.2f} indicates sparse vegetation. Street trees and green corridors significantly reduce LST."))
            if wind_speed<2.0:
                hints.append(("💨","Improve ventilation corridors",
                    f"Wind speed {wind_speed:.1f} m/s is low. Urban canyon orientation and reduced building density improve air circulation."))
            if temperature>35:
                hints.append(("🔆","Cool pavements & roofs",
                    f"At {temperature:.0f}°C, high-albedo surfaces can reflect solar radiation and reduce ambient temperature by 0.5–2°C."))
            if clouds<20:
                hints.append(("☁️","Shading structures",
                    f"Cloud cover only {clouds}%. Cool corridors, pergolas, and reflective materials provide relief when natural shading is absent."))
            if humidity>80:
                hints.append(("💧","Reduce anthropogenic heat",
                    f"High humidity ({humidity}%) amplifies heat stress. Reducing AC exhaust, vehicle emissions, and industrial heat helps."))
            if not hints:
                hints.append(("✅","Conditions are relatively favourable",
                    "Your parameter combination shows moderate UHI risk. Maintain current green coverage and ventilation."))
            for ico,title,desc in hints[:4]:
                nc=COLORS["accent"] if ico in ("🌳","🌿","✅") else COLORS["warning"]
                st.markdown(f"""
                <div class="note-card" style="--nc:{nc}">
                    <strong>{ico} {title}</strong><br>
                    <span style="font-size:0.73rem;color:{COLORS['subtext']}">{desc}</span>
                </div>""",unsafe_allow_html=True)
