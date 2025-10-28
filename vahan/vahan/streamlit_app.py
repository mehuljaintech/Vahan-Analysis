# =============================
# 📚 Cleaned & Consolidated Imports
# =============================
# Standard library
import os
import sys
import time
import traceback
import io
import json
import random
from datetime import date, timedelta

# Third-party
import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st  # re-import is safe; already imported above
from dotenv import load_dotenv

# Excel / Openpyxl
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference

# Local vahan package modules (keep unchanged)
from vahan.api import build_params, get_json
from vahan.parsing import (
    to_df, normalize_trend, parse_duration_table,
    parse_top5_revenue, parse_revenue_trend, parse_makers
)
from vahan.metrics import compute_yoy, compute_qoq
from vahan.charts import (
    bar_from_df, pie_from_df, line_from_trend,
    show_metrics, show_tables
)

# Optional advanced libraries (import gracefully)
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# Load environment variables
load_dotenv()

# NOTE:
# - If you want to trigger a programmatic restart from anywhere in the file,
#   call: auto_restart(delay=3)
# - Keep this top block intact. It ensures a self-restart behaves like an app reboot
#   without adding external scripts or OS-specific services.


# ================================
# ⚡ UI: Streamlit Page Setup
# ================================
# --- Page Config ---
st.set_page_config(
    page_title="🚀 Vahan Registrations | Parivahan Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Dynamic Styling ---
st.markdown("""
<style>
/* 🌀 Global gradient background */
.stApp {
    background: linear-gradient(135deg, #141E30 0%, #243B55 100%);
    color: #FFFFFF;
}

/* ✨ Card & section enhancements */
.block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 95%;
}

/* 🌈 Title styling */
h1 {
    text-align: center;
    color: #00E0FF;
    font-weight: 900;
    text-shadow: 0px 0px 10px rgba(0,224,255,0.6);
}

/* 🔹 Sidebar enhancements */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    color: #E2E8F0;
    border-right: 2px solid #00E0FF33;
}

[data-testid="stSidebar"] * {
    font-size: 15px !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #00E0FF !important;
}

/* 🧩 Buttons */
div.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #00E0FF 0%, #0077FF 100%);
    color: white;
    font-weight: bold;
    transition: 0.3s ease-in-out;
}
div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #00E0FFAA;
}

/* 🧠 AI warning or success message styling */
.stAlert {
    border-radius: 10px;
    background-color: #1E293B;
    color: #E2E8F0;
    border-left: 5px solid #00E0FF;
}
</style>
""", unsafe_allow_html=True)

# --- Header with dynamic effects ---
st.title("🚀 **Vahan Registrations Dashboard**")
st.markdown("""
<center>
<h3>🌍 <span style='color:#00E0FF;'>Parivahan Analytics</span> — KPIs, AI Narratives (DeepInfra), Forecasting, Clustering, Anomaly Detection & Smart Exports</h3>
</center>
""", unsafe_allow_html=True)

st.markdown("---")

# ================================
# 🧭 Sidebar — Dynamic Filter Panel
# ================================
import requests
from datetime import date
import streamlit as st

today = date.today()
default_from_year = max(2017, today.year - 1)

st.sidebar.header("⚙️ Filters & Dynamic Options")
st.sidebar.markdown("Use these options to customize data queries and analytics behavior.")

from_year = st.sidebar.number_input("📅 From Year", min_value=2012, max_value=today.year, value=default_from_year)
to_year = st.sidebar.number_input("📆 To Year", min_value=from_year, max_value=today.year, value=today.year)
state_code = st.sidebar.text_input("🏙️ State Code (blank = All-India)", value="")
rto_code = st.sidebar.text_input("🏢 RTO Code (0 = aggregate)", value="0")
vehicle_classes = st.sidebar.text_input("🚘 Vehicle Classes (e.g., 2W,3W,4W)", value="")
vehicle_makers = st.sidebar.text_input("🏭 Vehicle Makers (comma-separated or IDs)", value="")
time_period = st.sidebar.selectbox("⏱️ Time Period", options=[0, 1, 2], index=0)
fitness_check = st.sidebar.selectbox("🧾 Fitness Check", options=[0, 1], index=0)
vehicle_type = st.sidebar.text_input("🛻 Vehicle Type (optional)", value="")

st.sidebar.markdown("---")

# --- Advanced Analytics Toggles ---
st.sidebar.subheader("🧠 Smart Analytics Modes")
enable_forecast = st.sidebar.checkbox("📈 Enable Forecasting", value=True)
enable_anomaly = st.sidebar.checkbox("⚠️ Enable Anomaly Detection", value=True)
enable_clustering = st.sidebar.checkbox("🔍 Enable Clustering", value=True)
enable_ai = st.sidebar.checkbox("🧠 Enable DeepInfra AI Narratives", value=True)
forecast_periods = st.sidebar.number_input("⏳ Forecast Horizon (months)", min_value=1, max_value=36, value=3)

# ================================
# 🔐 DeepInfra Connection via Streamlit Secrets
# ================================
def load_deepinfra_config():
    try:
        key = st.secrets["DEEPINFRA_API_KEY"]
        model = st.secrets.get("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        return key, model
    except Exception:
        st.sidebar.error("🚫 Missing DeepInfra secrets — please add DEEPINFRA_API_KEY in Streamlit Secrets.")
        return None, None

DEEPINFRA_API_KEY, DEEPINFRA_MODEL = load_deepinfra_config()

# --- Verify connection ---
if enable_ai and DEEPINFRA_API_KEY:
    try:
        # ✅ DeepInfra expects GET /v1/openai/models, not POST
        resp = requests.get(
            "https://api.deepinfra.com/v1/openai/models",
            headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
            timeout=8
        )

        if resp.status_code == 200:
            st.sidebar.success(f"✅ DeepInfra Connected — Model: {DEEPINFRA_MODEL}")
        elif resp.status_code == 401:
            st.sidebar.error("🚫 Unauthorized — invalid or expired DEEPINFRA_API_KEY.")
        elif resp.status_code == 405:
            st.sidebar.warning("⚠️ DeepInfra returned 405 — check endpoint or API format.")
        else:
            st.sidebar.warning(f"⚠️ DeepInfra status: {resp.status_code}")

    except requests.exceptions.Timeout:
        st.sidebar.error("⏱️ DeepInfra request timed out.")
    except Exception as e:
        st.sidebar.error(f"❌ DeepInfra connection error: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Toggle features dynamically — the dashboard adapts instantly.")

# ================================
# ⚙️ Build & Display Vahan Parameters
# ================================

# --- Smart Animated Info Section ---
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    padding: 15px 25px;
    border-radius: 12px;
    color: white;
    font-size: 18px;
    font-weight: 600;
    box-shadow: 0 0 20px rgba(0,114,255,0.5);
    display: flex; justify-content: space-between; align-items: center;">
    <span>🧩 Building Dynamic API Parameters for Vahan Analytics...</span>
    <span style="font-size:14px;opacity:0.8;">Auto-synced with selected filters 🔁</span>
</div>
""", unsafe_allow_html=True)

# --- Build API Params Dynamically ---
with st.spinner("🚀 Generating request parameters... please wait"):
    try:
        params_common = build_params(
            from_year, to_year,
            state_code=state_code,
            rto_code=rto_code,
            vehicle_classes=vehicle_classes,
            vehicle_makers=vehicle_makers,
            time_period=time_period,
            fitness_check=fitness_check,
            vehicle_type=vehicle_type
        )

        # --- Pretty JSON preview ---
        st.markdown("### 🔧 Generated Vahan Request Parameters")
        st.code(json.dumps(params_common, indent=2), language="json")

        # --- Dynamic success message ---
        st.success(f"✅ Parameters built successfully for {to_year}. Ready for data fetch!")

    except Exception as e:
        # --- Error handling ---
        st.error(f"❌ Error while building Vahan parameters: {str(e)}")

        if st.button("🔄 Auto-Retry Build"):
            st.rerun()

# --- Optional auto-refresh animation ---
if st.button("♻️ Refresh Parameters"):
    st.toast("Refreshing filters & rebuilding params...", icon="🔁")
    st.rerun()

# ================================
# ⚙️ Dynamic Safe API Fetch Layer
# ================================

# Utility: dynamic color tag
def _tag(text, color):
    return f"<span style='background:{color};padding:3px 8px;border-radius:8px;color:white;font-size:12px;margin-right:4px;'>{text}</span>"

# ---------------- Helper: Safe API Call (MAXED)
def fetch_json(endpoint, params=params_common, desc=""):
    """Smart safe API fetch with UI logging, retries, and dynamic feedback."""
    max_retries = 3
    delay = 1 + random.random()  # random backoff to avoid hitting rate limits

    st.markdown(f"""
    <div style="padding:10px 15px;margin:8px 0;border-radius:10px;
        background:rgba(0, 150, 255, 0.1);border-left:4px solid #0096ff;">
        <b>{_tag("API", "#007bff")} {_tag(desc or "Fetching...", "#00b894")}</b> 
        <span style="font-size:13px;color:#444;">→ <code>{endpoint}</code></span>
    </div>
    """, unsafe_allow_html=True)

    for attempt in range(1, max_retries + 1):
        try:
            with st.spinner(f"🔄 Fetching {desc or endpoint} (attempt {attempt}/{max_retries})..."):
                json_data, _ = get_json(endpoint, params)
                if json_data:
                    st.toast(f"✅ {desc or endpoint} fetched successfully!", icon="🚀")
                    return json_data
                else:
                    st.warning(f"⚠️ Empty response for {desc or endpoint}. Retrying...")
            time.sleep(delay)
        except Exception as e:
            st.error(f"❌ Error fetching {desc or endpoint}: {e}")
            time.sleep(delay * attempt)

    st.error(f"⛔ Failed to fetch {desc or endpoint} after {max_retries} attempts.")
    st.markdown(
        f"<div style='margin-top:5px;'>💡 You can try manually reloading this API below.</div>",
        unsafe_allow_html=True
    )

    if st.button(f"🔁 Retry {desc or endpoint}"):
        st.rerun()

    return {}

# ============================================
# 🤖 DeepInfra AI Helper (Streamlit Secrets Only)
# ============================================
import time, random, requests, streamlit as st

# ✅ Correct DeepInfra Chat Completion endpoint
DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# 🔐 Load secrets directly from Streamlit Cloud
DEEPINFRA_API_KEY = st.secrets.get("DEEPINFRA_API_KEY")
DEEPINFRA_MODEL = st.secrets.get("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")


# --------------------------------------------------
# 🔍 Optional: Sidebar connection status check
# --------------------------------------------------
def check_deepinfra_connection():
    if not DEEPINFRA_API_KEY:
        st.sidebar.warning("⚠️ No DeepInfra API key found in Streamlit Secrets.")
        return False

    try:
        resp = requests.get(
            "https://api.deepinfra.com/v1/openai/models",
            headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
            timeout=8
        )
        if resp.status_code == 200:
            st.sidebar.success("✅ DeepInfra Connected — AI Narratives Ready!")
            return True
        elif resp.status_code == 401:
            st.sidebar.error("🚫 Unauthorized — invalid API key.")
        else:
            st.sidebar.warning(f"⚠️ DeepInfra reachable but returned {resp.status_code}.")
    except Exception as e:
        st.sidebar.error(f"❌ DeepInfra connection error: {e}")
    return False


# --------------------------------------------------
# 💬 Main DeepInfra Chat Function
# --------------------------------------------------
def deepinfra_chat(system_prompt: str, user_prompt: str,
                   max_tokens: int = 512, temperature: float = 0.3,
                   retries: int = 3, delay: float = 2.0):
    """
    DeepInfra Chat Wrapper (for Streamlit Cloud)
    - Reads keys only from st.secrets
    - Handles 401, 405, 429, timeout, and empty responses
    - Retries automatically with exponential delay
    - Streamlit-friendly feedback
    """

    if not DEEPINFRA_API_KEY:
        st.warning("⚠️ Missing DeepInfra API key in Streamlit Secrets.")
        return {"error": "Missing API key"}

    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    st.markdown(
        f"<div style='padding:8px 15px;border-left:4px solid #7d3cff;"
        f"background:rgba(125,60,255,0.08);border-radius:8px;margin:5px 0;'>"
        f"🧠 <b>AI Generating Insight...</b> (Model: <code>{DEEPINFRA_MODEL}</code>)</div>",
        unsafe_allow_html=True
    )

    for attempt in range(1, retries + 1):
        try:
            with st.spinner(f"💬 DeepInfra generating response (attempt {attempt}/{retries})..."):
                response = requests.post(DEEPINFRA_CHAT_URL, headers=headers, json=payload, timeout=90)

                # Handle unauthorized or unsupported methods
                if response.status_code == 401:
                    st.error("🚫 Unauthorized — check `DEEPINFRA_API_KEY` in Streamlit Secrets.")
                    return {"error": "Unauthorized"}

                if response.status_code == 405:
                    st.error("⚠️ DeepInfra returned 405 — wrong endpoint or request format.")
                    return {"error": "405 Method Not Allowed"}

                response.raise_for_status()
                data = response.json()

                if "choices" in data and data["choices"]:
                    text = data["choices"][0]["message"]["content"].strip()
                    st.toast("✅ DeepInfra AI response ready!", icon="🤖")
                    st.markdown(
                        f"<div style='background:#f9fafb;padding:15px;border-radius:10px;"
                        f"border:1px solid #ddd;margin-top:8px;'>"
                        f"<b>🔍 AI Insight:</b><br><pre style='white-space:pre-wrap;'>{text}</pre></div>",
                        unsafe_allow_html=True
                    )
                    return {"text": text, "raw": data}

                st.warning("⚠️ No content returned by AI.")
                return {"text": "No AI output generated.", "raw": data}

        except Exception as e:
            st.error(f"❌ DeepInfra error: {e}")
            time.sleep(delay * attempt * random.uniform(1.0, 1.5))

    st.error("⛔ DeepInfra AI failed after multiple attempts.")
    if st.button("🔁 Retry DeepInfra AI"):
        st.rerun()
    return {"error": "DeepInfra API failed after retries."}


# --------------------------------------------------
# 🧪 Optional Test UI (for Streamlit Debugging)
# --------------------------------------------------
def deepinfra_test_ui():
    """Tiny test block to validate DeepInfra key inside Streamlit."""
    st.subheader("🧠 Test DeepInfra Connection")
    if st.button("🔍 Run Test Prompt"):
        resp = deepinfra_chat(
            "You are an AI summarizer.",
            "Summarize this message: DeepInfra integration test for Streamlit."
        )
        if "text" in resp:
            st.success("✅ AI test successful!")
        else:
            st.error("❌ AI test failed — check logs above.")

# ================================
# 1️⃣ Category Distribution
# ================================
with st.container():
    st.markdown("""
    <div style="padding:12px 20px;border-left:5px solid #6C63FF;
                background:linear-gradient(90deg,#f3f1ff,#ffffff);
                border-radius:12px;margin-bottom:15px;">
        <h3 style="margin:0;">📊 Category Distribution</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14px;">
            Comparative distribution of registered vehicles by category.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching category distribution from Vahan API..."):
        cat_json = fetch_json("vahandashboard/categoriesdonutchart", desc="Category distribution")
    df_cat = to_df(cat_json)

    if not df_cat.empty:
        col1, col2 = st.columns(2)

        with col1:
            try:
                bar_from_df(df_cat, title="Category Distribution (Bar)")
            except Exception as e:
                st.error(f"⚠️ Bar chart failed: {e}")
                st.dataframe(df_cat)

        with col2:
            try:
                pie_from_df(df_cat, title="Category Distribution (Donut)", donut=True)
            except Exception as e:
                st.error(f"⚠️ Pie chart failed: {e}")
                st.dataframe(df_cat)

        # 📈 KPI snapshot
        top_cat = df_cat.loc[df_cat['value'].idxmax(), 'label']
        total = df_cat['value'].sum()
        top_val = df_cat['value'].max()
        pct = round((top_val / total) * 100, 2)
        st.markdown(
            f"""
            <div style="margin-top:10px;padding:10px 15px;
                        background:rgba(108,99,255,0.1);
                        border:1px solid #6C63FF;border-radius:10px;">
                <b>🏆 Top Category:</b> {top_cat} — {pct}% of total registrations ({total:,} total)
            </div>
            """,
            unsafe_allow_html=True
        )

        # 🤖 AI Summary
        if enable_ai:
            with st.expander("🤖 AI Summary — Category Insights", expanded=True):
                with st.spinner("Generating DeepInfra AI insights for Category Distribution..."):
                    system = (
                        "You are a senior automotive analytics assistant summarizing vehicle category trends. "
                        "Focus on dominant categories, anomalies, and data-backed recommendations."
                    )
                    sample = df_cat.head(10).to_dict(orient='records')
                    user = (
                        f"Dataset sample (top 10): {json.dumps(sample, default=str)}\n"
                        "Summarize this dataset in 3-5 sentences: "
                        "Highlight top categories, their relative shares, and one actionable insight."
                    )

                    ai_resp = deepinfra_chat(system, user, max_tokens=300, temperature=0.4)

                    if "text" in ai_resp and ai_resp["text"]:
                        st.markdown(
                            f"""
                            <div style="margin-top:8px;padding:12px 16px;
                                        background:#fafafa;border-left:4px solid #6C63FF;
                                        border-radius:10px;">
                                <b>AI Analysis:</b><br>
                                <div style="margin-top:6px;font-size:15px;">
                                    {ai_resp["text"]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("AI summary unavailable — DeepInfra response empty.")

    else:
        st.warning("No category data returned from Vahan API.")

# ================================
# 2️⃣ Top Makers (Auto-Safe + Maxed)
# ================================
with st.container():
    st.markdown("""
    <div style="padding:12px 20px;border-left:5px solid #FF6B6B;
                background:linear-gradient(90deg,#fff5f5,#ffffff);
                border-radius:12px;margin-bottom:15px;">
        <h3 style="margin:0;">🏭 Top Vehicle Makers</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14px;">
            Market share of top-performing manufacturers based on registration data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching Top Makers data from Vahan API..."):
        mk_json = fetch_json("vahandashboard/top5Makerchart", desc="Top Makers")
        df_mk = parse_makers(mk_json)

    if not df_mk.empty:
        # Normalize columns for flexible matching
        df_mk.columns = [c.strip().lower() for c in df_mk.columns]

        # Identify maker/value columns automatically
        maker_col = next((c for c in ["maker", "makename", "manufacturer", "label", "name"] if c in df_mk.columns), None)
        value_col = next((c for c in ["value", "count", "total", "registeredvehiclecount", "y"] if c in df_mk.columns), None)

        if not maker_col or not value_col:
            st.warning("⚠️ Unable to identify maker/value columns in dataset.")
            st.dataframe(df_mk)
        else:
            # --- Visualization section ---
            col1, col2 = st.columns(2)

            with col1:
                try:
                    bar_from_df(df_mk.rename(columns={maker_col: "label", value_col: "value"}), title="Top Makers (Bar)")
                except Exception as e:
                    st.error(f"⚠️ Bar chart error: {e}")
                    st.dataframe(df_mk)

            with col2:
                try:
                    pie_from_df(df_mk.rename(columns={maker_col: "label", value_col: "value"}), title="Top Makers (Pie)", donut=True)
                except Exception as e:
                    st.error(f"⚠️ Pie chart error: {e}")
                    st.dataframe(df_mk)

            # --- KPI Snapshot ---
            try:
                top_maker = df_mk.loc[df_mk[value_col].idxmax(), maker_col]
                total_val = df_mk[value_col].sum()
                top_val = df_mk[value_col].max()
                pct_share = round((top_val / total_val) * 100, 2)

                st.markdown(
                    f"""
                    <div style="margin-top:10px;padding:10px 15px;
                                background:rgba(255,107,107,0.08);
                                border:1px solid #FF6B6B;border-radius:10px;">
                        <b>🏆 Leading Maker:</b> {top_maker}<br>
                        <b>Market Share:</b> {pct_share}% of total registrations ({total_val:,} total)
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.warning(f"⚠️ Could not compute top maker: {e}")
                st.dataframe(df_mk)

            # --- 🤖 AI Summary (DeepInfra) ---
            if enable_ai:
                with st.expander("🤖 AI Summary — Maker Insights", expanded=True):
                    with st.spinner("Generating DeepInfra AI summary for Top Makers..."):
                        try:
                            system = (
                                "You are an automotive market analyst. "
                                "Summarize market trends, dominant manufacturers, and competitive insights based on maker data."
                            )
                            sample = df_mk[[maker_col, value_col]].head(10).to_dict(orient='records')
                            user = (
                                f"Dataset sample: {json.dumps(sample, default=str)}\n"
                                "Generate a concise summary (3–5 lines) identifying leading makers, rising competitors, "
                                "and one strategic insight for the Indian vehicle market."
                            )

                            ai_resp = deepinfra_chat(system, user, max_tokens=300, temperature=0.4)

                            if ai_resp and "text" in ai_resp and ai_resp["text"]:
                                st.markdown(
                                    f"""
                                    <div style="margin-top:8px;padding:12px 16px;
                                                background:#fafafa;border-left:4px solid #FF6B6B;
                                                border-radius:10px;">
                                        <b>AI Market Insight:</b><br>
                                        <div style="margin-top:6px;font-size:15px;">
                                            {ai_resp["text"]}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.info("AI summary unavailable — no DeepInfra response received.")
                        except Exception as e:
                            st.error(f"AI generation error: {e}")
    else:
        st.warning("No maker data returned from Vahan API.")


# =============================================
# 3️⃣ Registration Trends + YoY/QoQ + AI
# =============================================
with st.container():
    st.markdown("""
    <div style="padding:12px 20px;border-left:5px solid #007BFF;
                background:linear-gradient(90deg,#f0f8ff,#ffffff);
                border-radius:12px;margin-bottom:15px;">
        <h3 style="margin:0;">📈 Registration Trends</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14px;">
            Yearly and Monthly trends in new vehicle registrations, including growth metrics and DeepInfra AI insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching registration trends..."):
        tr_json = fetch_json("vahandashboard/vahanyearwiseregistrationtrend", desc="Registration Trend")

    try:
        df_trend = normalize_trend(tr_json)
    except Exception as e:
        st.error(f"Trend parsing failed: {e}")
        df_trend = pd.DataFrame(columns=["date", "value"])

    if not df_trend.empty:
        # ================= Line Chart =================
        st.markdown("### 📊 Registration Trend Line")
        try:
            line_from_trend(df_trend, title="Registrations Trend Line")
        except Exception as e:
            st.warning(f"Fallback to Streamlit chart due to error: {e}")
            st.line_chart(df_trend.set_index("date")["value"])

        # ================= KPI & Metrics =================
        yoy_df = compute_yoy(df_trend)
        qoq_df = compute_qoq(df_trend)

        latest_yoy = yoy_df["YoY%"].dropna().iloc[-1] if not yoy_df.empty and yoy_df["YoY%"].dropna().size else None
        latest_qoq = qoq_df["QoQ%"].dropna().iloc[-1] if not qoq_df.empty and qoq_df["QoQ%"].dropna().size else None

        total_days = (df_trend["date"].max() - df_trend["date"].min()).days or 1
        daily_avg = df_trend["value"].sum() / total_days

        # KPI Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Total Registrations", f"{int(df_trend['value'].sum()):,}")
        with col2:
            st.metric("📅 Daily Avg Orders", f"{daily_avg:,.0f}")
        with col3:
            st.metric("📈 Latest YoY%", f"{latest_yoy:.2f}%" if latest_yoy else "N/A")
        with col4:
            st.metric("📊 Latest QoQ%", f"{latest_qoq:.2f}%" if latest_qoq else "N/A")

        # Display YoY / QoQ Tables
        with st.expander("📑 Year-over-Year and Quarter-over-Quarter Analysis", expanded=False):
            st.markdown("#### 📆 Year-over-Year (YoY)")
            st.dataframe(yoy_df, use_container_width=True)
            st.markdown("#### 🧭 Quarter-over-Quarter (QoQ)")
            st.dataframe(qoq_df, use_container_width=True)

        # ================= AI Narrative =================
        if enable_ai:
            with st.expander("🤖 AI Narrative — Trend Insights", expanded=True):
                with st.spinner("Generating DeepInfra AI analysis for trends..."):
                    system = (
                        "You are an automotive data analyst summarizing registration trends. "
                        "Explain growth pattern, highlight anomalies, give 2 concise actionable insights. "
                        "Write clearly with numerical context."
                    )

                    sample_rows = df_trend.tail(12).to_dict(orient='records')
                    daily_avg_display = f"{daily_avg:.1f}" if daily_avg is not None else "N/A"

                    user = (
                        f"Recent 12 periods data: {json.dumps(sample_rows, default=str)}\n"
                        f"YoY: {latest_yoy}, QoQ: {latest_qoq}, DailyAvg: {daily_avg_display}\n"
                        "Summarize in 5–7 lines with 2 short recommendations for policymakers or manufacturers."
                    )

                    ai_resp = deepinfra_chat(system, user, max_tokens=450, temperature=0.3)

                    if isinstance(ai_resp, dict) and "text" in ai_resp:
                        st.markdown(
                            f"""
                            <div style="margin-top:8px;padding:12px 16px;
                                        background:#fafafa;border-left:4px solid #007BFF;
                                        border-radius:10px;">
                                <b>AI Trend Insight:</b><br>
                                <div style="margin-top:6px;font-size:15px;">
                                    {ai_resp["text"]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("AI summary unavailable — no valid response received.")

        # ================= Optional: Forecasting =================
        if enable_forecast:
            try:
                with st.expander("🔮 Forecast — Next Periods", expanded=False):
                    st.markdown("Forecasting next 3–12 months (based on recent trend)...")
                    df_forecast = forecast_trend(df_trend, forecast_periods)
                    line_from_trend(df_forecast, title="Forecasted Trend")
            except Exception as e:
                st.warning(f"Forecast failed: {e}")

    else:
        st.warning("No registration trend data returned from API.")

# ================================================================
# 4️⃣ Duration-wise Growth + 5️⃣ Top 5 Revenue States
# ================================================================

st.markdown("""
<div style="padding:12px 20px;border-left:5px solid #28a745;
            background:linear-gradient(90deg,#f5fff6,#ffffff);
            border-radius:12px;margin-bottom:15px;">
    <h3 style="margin:0;">📊 Duration-wise Growth & Revenue Insights</h3>
    <p style="margin:4px 0 0;color:#555;font-size:14px;">
        Monthly, quarterly, and yearly registration growth trends with revenue analytics and AI commentary.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------- Duration-wise Growth ---------------------
def fetch_duration_growth(calendar_type, label, color):
    with st.spinner(f"Fetching {label} growth data..."):
        json_data = fetch_json(
            "vahandashboard/durationWiseRegistrationTable",
            {**params_common, "calendarType": calendar_type},
            desc=f"{label} growth"
        )
        df = parse_duration_table(json_data)

        if not df.empty:
            st.markdown(f"### {label} Vehicle Registration Growth")
            col1, col2 = st.columns(2)
            with col1:
                try:
                    bar_from_df(df, title=f"{label} Growth (Bar)")
                except Exception:
                    st.write(df)
            with col2:
                try:
                    pie_from_df(df, title=f"{label} Growth (Pie)", donut=True)
                except Exception:
                    st.write(df)

            # Mini KPI Summary
            with st.container():
                max_label = df.loc[df["value"].idxmax(), "label"]
                max_val = df["value"].max()
                avg_val = df["value"].mean()
                st.markdown(
                    f"""
                    <div style="background:#f9f9f9;padding:10px 16px;
                                border-left:4px solid {color};
                                border-radius:10px;margin-bottom:12px;">
                        <b>Top Period:</b> {max_label} — <b>{max_val:,.0f}</b> registrations<br>
                        <b>Average:</b> {avg_val:,.0f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # AI insight for this duration
            if enable_ai:
                with st.expander(f"🤖 AI Summary — {label} Growth", expanded=False):
                    with st.spinner(f"Generating AI summary for {label} growth..."):
                        system = (
                            f"You are a data analyst explaining {label.lower()} growth of vehicle registrations. "
                            "Mention top periods, growth momentum, and one recommendation."
                        )
                        sample = df.head(10).to_dict(orient="records")
                        user = (
                            f"Dataset sample: {json.dumps(sample, default=str)}\n"
                            f"Summarize in 3–5 sentences and suggest 1 action to improve consistency."
                        )
                        ai_resp = deepinfra_chat(system, user, max_tokens=250)
                        if isinstance(ai_resp, dict) and "text" in ai_resp:
                            st.markdown(
                                f"""
                                <div style="padding:10px 14px;background:#fefefe;
                                            border-left:3px solid {color};
                                            border-radius:8px;margin-top:6px;">
                                    {ai_resp["text"]}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            return df
        else:
            st.warning(f"No {label.lower()} data available.")
        return pd.DataFrame()

# Run all duration sections with color coding
df_monthly   = fetch_duration_growth(3, "📅 Monthly",  "#007bff")
df_quarterly = fetch_duration_growth(2, "🧭 Quarterly", "#6f42c1")
df_yearly    = fetch_duration_growth(1, "📆 Yearly",   "#28a745")

# --------------------- Top 5 Revenue States ---------------------
st.markdown("""
<div style="padding:12px 20px;border-left:5px solid #ffc107;
            background:linear-gradient(90deg,#fffdf5,#ffffff);
            border-radius:12px;margin-top:30px;margin-bottom:15px;">
    <h3 style="margin:0;">💰 Top 5 Revenue States</h3>
    <p style="margin:4px 0 0;color:#555;font-size:14px;">
        State-level comparison of total vehicle-related revenue and collection performance.
    </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Fetching Top 5 Revenue States..."):
    top5_rev_json = fetch_json("vahandashboard/top5chartRevenueFee", desc="Top 5 Revenue States")

df_top5_rev = parse_top5_revenue(top5_rev_json if top5_rev_json else {})

if not df_top5_rev.empty:
    col1, col2 = st.columns(2)
    with col1:
        try:
            bar_from_df(df_top5_rev, title="Top 5 Revenue States (Bar)")
        except Exception:
            st.write(df_top5_rev)
    with col2:
        try:
            pie_from_df(df_top5_rev, title="Top 5 Revenue States (Pie)", donut=True)
        except Exception:
            st.write(df_top5_rev)

    # KPI summary for top state
    top_state = df_top5_rev.loc[df_top5_rev["value"].idxmax(), "label"]
    top_value = df_top5_rev["value"].max()
    total_rev = df_top5_rev["value"].sum()
    st.markdown(
        f"""
        <div style="background:#fffbe6;padding:12px 16px;
                    border-left:4px solid #ffc107;border-radius:10px;
                    margin-top:12px;margin-bottom:10px;">
            <b>🏆 Top Revenue State:</b> {top_state} — ₹{top_value:,.0f}<br>
            <b>💵 Combined Revenue (Top 5):</b> ₹{total_rev:,.0f}
        </div>
        """,
        unsafe_allow_html=True
    )

    # AI summary for revenue
    if enable_ai:
        with st.expander("🤖 AI Summary — Revenue Insights", expanded=True):
            with st.spinner("Generating AI summary for Revenue..."):
                system = "You are a financial analytics assistant summarizing state-level vehicle revenue trends."
                sample = df_top5_rev.head(10).to_dict(orient="records")
                user = (
                    f"Dataset: {json.dumps(sample, default=str)}\n"
                    "Provide a concise summary (3–4 lines) highlighting top-performing state, revenue gap, and one actionable insight."
                )
                ai_resp = deepinfra_chat(system, user, max_tokens=220)
                if isinstance(ai_resp, dict) and "text" in ai_resp:
                    st.markdown(
                        f"""
                        <div style="padding:10px 14px;background:#fffef9;
                                    border-left:3px solid #ffc107;
                                    border-radius:8px;margin-top:6px;">
                            {ai_resp["text"]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
else:
    st.warning("No revenue data available from API.")

# ================================================================
# 6️⃣ MAXED Revenue Trend + Forecast + Anomaly Detection + Clustering
# ================================================================
with st.container():
    st.markdown("""
    <div style="padding:12px 20px;border-left:5px solid #ff5722;
                background:linear-gradient(90deg,#fff7f3,#ffffff);
                border-radius:12px;margin-top:18px;margin-bottom:15px;">
        <h3 style="margin:0;">💹 Revenue Trend & Advanced Analytics</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14px;">
            Revenue time-series, forecasts, anomaly detection and clustering — with AI narratives.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ----- Fetch revenue trend -----
    with st.spinner("Fetching Revenue Trend..."):
        rev_trend_json = fetch_json("vahandashboard/revenueFeeLineChart", desc="Revenue Trend")
    df_rev_trend = parse_revenue_trend(rev_trend_json if rev_trend_json else {})

    if df_rev_trend.empty:
        st.warning("No revenue trend data available.")
    else:
        # --- Trend chart (Altair) ---
        st.subheader("Revenue Trend Comparison")
        try:
            chart = (
                alt.Chart(df_rev_trend)
                .mark_line(point=True, interpolate="monotone")
                .encode(
                    x=alt.X("period:O", title="Period"),
                    y=alt.Y("value:Q", title="Revenue (₹)"),
                    color=alt.Color("year:N", legend=alt.Legend(title="Year")),
                    tooltip=["year", "period", "value"]
                )
                .properties(height=380, title="Revenue Trend Comparison")
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.write(df_rev_trend)

        # --- KPIs ---
        try:
            total_rev = float(df_rev_trend["value"].sum())
            avg_rev = float(df_rev_trend["value"].mean())
            latest_rev = float(df_rev_trend["value"].iloc[-1])
            prev_rev = float(df_rev_trend["value"].iloc[-2]) if len(df_rev_trend) > 1 else latest_rev
            growth_pct = ((latest_rev - prev_rev) / prev_rev) * 100 if prev_rev else 0.0
        except Exception:
            total_rev = avg_rev = latest_rev = growth_pct = None

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Total Revenue", f"₹{total_rev:,.0f}" if total_rev is not None else "N/A")
        with col2:
            st.metric("📈 Latest Revenue", f"₹{latest_rev:,.0f}" if latest_rev is not None else "N/A")
        with col3:
            st.metric("📊 Avg per Period", f"₹{avg_rev:,.0f}" if avg_rev is not None else "N/A")
        with col4:
            st.metric("📅 Latest Growth %", f"{growth_pct:.2f}%" if growth_pct is not None else "N/A")

        # ========== FORECAST ==========
        forecast_df = pd.DataFrame()
        if enable_forecast:
            st.markdown("### 🔮 Forecasting")
            try:
                if PROPHET_AVAILABLE and 'date' in df_trend.columns and 'value' in df_trend.columns:
                    with st.spinner("Running Prophet forecast..."):
                        dfp = df_trend[['date','value']].rename(columns={'date':'ds','value':'y'})
                        m = Prophet()
                        m.fit(dfp)
                        future = m.make_future_dataframe(periods=forecast_periods, freq='MS')
                        forecast = m.predict(future)
                        forecast_df = forecast[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'ds':'date','yhat':'value'})
                        st.markdown("Prophet forecast (tail):")
                        st.dataframe(forecast_df.tail(forecast_periods))
                        fig = m.plot(forecast)
                        st.pyplot(fig, bbox_inches='tight')
                else:
                    # linear regression fallback
                    with st.spinner("Running linear regression forecast..."):
                        if 'date' in df_trend.columns and 'value' in df_trend.columns:
                            forecast_df = linear_forecast(df_trend, months=forecast_periods)
                            if not forecast_df.empty:
                                combined = pd.concat([df_trend.set_index('date')['value'], forecast_df.set_index('date')['value']])
                                st.line_chart(combined)
                        else:
                            st.info("Forecast requires a time series with 'date' and 'value'.")
            except Exception as e:
                st.warning(f"Forecast failed: {e}")
                # fallback
                try:
                    forecast_df = linear_forecast(df_trend, months=forecast_periods)
                    if not forecast_df.empty:
                        st.line_chart(pd.concat([df_trend.set_index('date')['value'], forecast_df.set_index('date')['value']]))
                except Exception:
                    pass

            # AI commentary for forecast
            if enable_ai and not forecast_df.empty:
                with st.spinner("Generating AI forecast commentary..."):
                    system = "You are an analytics assistant that summarizes short-term forecasts and confidence."
                    sample = forecast_df.head(6).to_dict(orient='records')
                    user = f"Forecasted values (date,value): {json.dumps(sample, default=str)}\nProvide a 3-sentence summary with risk/confidence commentary."
                    ai_resp = deepinfra_chat(system, user, max_tokens=220)
                    if isinstance(ai_resp, dict) and "text" in ai_resp:
                        st.markdown("**AI Forecast Commentary**")
                        st.write(ai_resp["text"])

        # ========== ANOMALY DETECTION ==========
        if enable_anomaly:
            st.markdown("### 🚨 Anomaly Detection (Registrations)")
            if not SKLEARN_AVAILABLE:
                st.info("scikit-learn not available. Install scikit-learn to enable IsolationForest anomaly detection.")
            elif 'value' not in df_trend.columns:
                st.info("No 'value' column detected for anomaly detection.")
            else:
                try:
                    contamination = st.slider("Contamination (expected outliers fraction)", 0.001, 0.2, 0.02)
                    model = IsolationForest(contamination=contamination, random_state=42)
                    vals = df_trend[['value']].fillna(0)
                    model.fit(vals)
                    df_trend['anomaly_score'] = model.decision_function(vals)
                    df_trend['anomaly'] = model.predict(vals)
                    anomalies = df_trend[df_trend['anomaly'] == -1]
                    st.write(f"Detected {anomalies.shape[0]} anomalies")
                    if not anomalies.empty:
                        st.dataframe(anomalies[['date','value','anomaly_score']].sort_values('anomaly_score'))

                    # show anomalies on chart
                    if 'date' in df_trend.columns:
                        base = alt.Chart(df_trend).encode(x='date:T')
                        line = base.mark_line().encode(y='value:Q')
                        points = base.mark_circle(size=60).encode(
                            y='value:Q',
                            color=alt.condition(alt.datum.anomaly == -1, alt.value('red'), alt.value('black')),
                            tooltip=['date', 'value', 'anomaly_score']
                        )
                        st.altair_chart((line + points).properties(height=350), use_container_width=True)
                except Exception as e:
                    st.error(f"Anomaly detection failed: {e}")

            # AI note on anomalies
            if enable_ai and 'anomalies' in locals() and not anomalies.empty:
                with st.spinner("Generating AI anomaly insights..."):
                    system = "You are an analytics investigator. Review anomalies and offer likely causes and remediation suggestions."
                    sample = anomalies.head(10).to_dict(orient='records')
                    user = f"Anomalies (date,value,score): {json.dumps(sample, default=str)}\nProvide likely causes (3 items) and 2 suggested next steps."
                    ai_resp = deepinfra_chat(system, user, max_tokens=240)
                    if isinstance(ai_resp, dict) and "text" in ai_resp:
                        st.markdown("**AI Anomaly Insights**")
                        st.write(ai_resp["text"])

        # ========== CLUSTERING & CORRELATION ==========
        if enable_clustering:
            st.markdown("### 🧭 Clustering & Correlation (AI + Visualized)")
            if not SKLEARN_AVAILABLE:
                st.info("scikit-learn not available. Install scikit-learn for clustering features.")
            else:
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.cluster import KMeans
                    from sklearn.decomposition import PCA
                    from sklearn.metrics import silhouette_score
                    import numpy as np
                    import plotly.express as px

                    # Build feature matrix: prefer revenue-based features
                    features_df = None
                    if 'df_top5_rev' in globals() and not df_top5_rev.empty:
                        try:
                            features_df = df_top5_rev.copy().reset_index(drop=True)
                            if 'value' not in features_df.columns and 'revenue' in features_df.columns:
                                features_df['value'] = pd.to_numeric(features_df['revenue'], errors='coerce').fillna(0)
                        except Exception:
                            features_df = None

                    # Fallback: rolling stats of trend
                    if features_df is None and not df_trend.empty:
                        df_roll = df_trend.set_index('date').resample('M').sum().fillna(0)
                        df_roll['rolling_mean_3'] = df_roll['value'].rolling(3).mean().fillna(0)
                        df_roll['rolling_std_3'] = df_roll['value'].rolling(3).std().fillna(0)
                        features_df = df_roll[['rolling_mean_3', 'rolling_std_3']].reset_index().dropna().reset_index(drop=True)

                    if features_df is None or features_df.empty:
                        st.info("No suitable data for clustering (need revenue or trend rolling stats).")
                    else:
                        numeric_cols = [c for c in features_df.columns if pd.api.types.is_numeric_dtype(features_df[c])]
                        if not numeric_cols:
                            st.info("No numeric columns available for clustering.")
                        else:
                            X = features_df[numeric_cols].fillna(0).astype(float)

                            # Scale
                            scaler = StandardScaler()
                            Xs = scaler.fit_transform(X)

                            # clusters slider
                            max_k = min(8, max(2, len(Xs)))
                            n_clusters = st.slider("Clusters (k)", 2, max_k, 3)
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            labels = kmeans.fit_predict(Xs)
                            features_df['cluster'] = labels

                            st.markdown("#### Cluster centroids (numeric features)")
                            st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols))

                            # PCA projection & visualization
                            try:
                                n_comp = min(3, Xs.shape[1])
                                pca = PCA(n_components=n_comp)
                                proj = pca.fit_transform(Xs)

                                if n_comp >= 3 and proj.shape[0] >= 3:
                                    fig3d = px.scatter_3d(
                                        x=proj[:,0], y=proj[:,1], z=proj[:,2],
                                        color=labels.astype(str),
                                        title="3D PCA Cluster Visualization",
                                    )
                                    st.plotly_chart(fig3d, use_container_width=True)
                                else:
                                    scatter = pd.DataFrame({'x': proj[:,0], 'y': proj[:,1] if proj.shape[1] > 1 else 0, 'cluster': labels})
                                    chart = alt.Chart(scatter).mark_circle(size=70).encode(
                                        x='x', y='y', color='cluster:N', tooltip=['x','y','cluster']
                                    ).properties(title="PCA Projection of Clusters")
                                    st.altair_chart(chart, use_container_width=True)
                            except Exception as e:
                                st.warning(f"PCA projection failed: {e}")
                                st.line_chart(features_df[numeric_cols])

                            # silhouette
                            try:
                                if len(Xs) >= n_clusters + 1:
                                    sc = silhouette_score(Xs, labels)
                                    st.metric("Silhouette Score", f"{sc:.3f}")
                            except Exception as e:
                                st.warning(f"Silhouette score failed: {e}")

                            # show cluster samples
                            st.markdown("#### Sample rows by cluster")
                            st.dataframe(features_df.head(200))

                            # correlation heatmap
                            try:
                                corr = features_df.select_dtypes(include=[np.number]).corr()
                                fig_corr = px.imshow(corr, text_auto=".2f", title="Correlation Matrix")
                                st.plotly_chart(fig_corr, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Correlation calc failed: {e}")

                            # AI summary for clusters
                            if enable_ai:
                                try:
                                    with st.spinner("Generating AI clustering insights..."):
                                        cluster_summary = features_df.groupby('cluster')[numeric_cols].mean().to_dict()
                                        system = "You are an expert data analyst. Summarize cluster patterns and differences briefly."
                                        user = (
                                            f"Cluster summary: {json.dumps(cluster_summary, indent=2)}\n"
                                            "Provide 5–6 sentences of insights comparing clusters and 2 actionable recommendations."
                                        )
                                        ai_resp = deepinfra_chat(system, user, max_tokens=350)
                                        if isinstance(ai_resp, dict) and "text" in ai_resp:
                                            st.markdown("**🤖 AI Cluster Narrative**")
                                            st.write(ai_resp["text"])
                                except Exception as e:
                                    st.warning(f"AI insight generation failed: {e}")
                except Exception as e:
                    st.error(f"Clustering pipeline failed: {e}")

# ============================================================
# 💾 SMART EXCEL EXPORT — Unified, Safe, and AI-enhanced
# ============================================================
st.markdown("## 💾 Smart Excel Export")
st.caption("Automatically export all KPIs, trends, forecasts, and AI insights into a single, styled Excel workbook.")

with st.container():
    with st.expander("📈 Generate & Download Smart Excel Report", expanded=True):

        # --- ✅ Safe defaults for all DataFrames
        df_cat = locals().get("df_cat", pd.DataFrame())
        df_mk = locals().get("df_mk", pd.DataFrame())
        df_trend = locals().get("df_trend", pd.DataFrame())
        yoy_df = locals().get("yoy_df", pd.DataFrame())
        qoq_df = locals().get("qoq_df", pd.DataFrame())
        df_top5_rev = locals().get("df_top5_rev", pd.DataFrame())
        df_rev_trend = locals().get("df_rev_trend", pd.DataFrame())

        # --- Collect all datasets
        datasets = {
            "Category": df_cat,
            "Top Makers": df_mk,
            "Registrations Trend": df_trend,
            "YoY Trend": yoy_df,
            "QoQ Trend": qoq_df,
            "Top 5 Revenue States": df_top5_rev,
            "Revenue Trend": df_rev_trend,
        }

        # --- 🔍 Forecast & Anomaly Detection
        with st.spinner("🔍 Performing Forecast & Anomaly Detection..."):
            try:
                if not df_trend.empty:
                    df_forecast = df_trend.copy()
                    df_forecast["Forecast"] = df_forecast["value"].rolling(3, min_periods=1).mean()
                    df_forecast["Anomaly"] = (
                        (df_forecast["value"] - df_forecast["Forecast"]).abs()
                        > df_forecast["Forecast"] * 0.15
                    )
                    datasets["Forecast & Anomaly Detection"] = df_forecast
                    st.success("✅ Forecast & anomaly detection completed!")
                    st.dataframe(df_forecast.tail(5), use_container_width=True)
                else:
                    st.info("ℹ️ No trend data available for forecast.")
            except Exception as e:
                st.warning(f"⚠️ Forecast step skipped: {e}")

        # --- 🧠 AI Summaries (if DeepInfra key enabled)
        summaries = {}
        if 'enable_ai' in locals() and enable_ai:
            try:
                st.info("🤖 Generating AI-driven summaries for each dataset...")
                for name, df in datasets.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        try:
                            system = f"You are a business analyst summarizing KPIs for '{name}'."
                            user = f"Data sample: {df.head(10).to_dict(orient='records')}.\nGive concise insights (2–3 lines)."
                            ai_resp = deepinfra_chat(system, user, max_tokens=180)
                            summaries[name] = ai_resp.get("text", "No summary generated.")
                        except Exception as e:
                            summaries[name] = f"AI summary failed: {e}"

                if summaries:
                    ai_df = pd.DataFrame(list(summaries.items()), columns=["Dataset", "AI Summary"])
                    datasets["AI Insights"] = ai_df

                    with st.expander("🧠 View AI Insights"):
                        for name, text in summaries.items():
                            st.markdown(f"**{name}**")
                            st.write(text)
                            st.markdown("---")
            except Exception as e:
                st.warning(f"⚠️ AI summaries skipped: {e}")

        # --- Warn if no data at all
        if all((not isinstance(df, pd.DataFrame)) or df.empty for df in datasets.values()):
            st.warning("⚠️ No data available for export. Creating summary sheet instead.")

        # --- 📦 Compile Excel workbook
        with st.spinner("📦 Compiling Excel workbook..."):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                any_written = False
                for name, df in datasets.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df.to_excel(writer, sheet_name=name[:31], index=False)
                        any_written = True

                # Always ensure at least one visible sheet
                if not any_written:
                    pd.DataFrame({"Info": ["No data available for export."]}).to_excel(
                        writer, sheet_name="Summary", index=False
                    )

            output.seek(0)

            # --- Style workbook
            from openpyxl import load_workbook
            from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
            from openpyxl.utils import get_column_letter
            from openpyxl.chart import LineChart, Reference

            wb = load_workbook(output)
            thin = Border(left=Side(style="thin"), right=Side(style="thin"),
                          top=Side(style="thin"), bottom=Side(style="thin"))

            for sheet in wb.sheetnames:
                ws = wb[sheet]

                # Header styling
                for cell in ws[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="305496", end_color="305496", fill_type="solid")
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = thin

                # Body styling
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                        cell.border = thin

                # Auto-fit columns
                for col in ws.columns:
                    max_len = max(len(str(c.value or "")) for c in col)
                    ws.column_dimensions[get_column_letter(col[0].column)].width = max_len + 3

                # Add simple trend chart (if applicable)
                if ws.max_row > 2 and ws.max_column >= 2:
                    try:
                        val_ref = Reference(ws, min_col=2, min_row=1, max_row=ws.max_row)
                        cat_ref = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
                        chart = LineChart()
                        chart.title = f"{sheet} Trend"
                        chart.y_axis.title = "Value"
                        chart.x_axis.title = "Category"
                        chart.add_data(val_ref, titles_from_data=True)
                        chart.set_categories(cat_ref)
                        chart.height = 8
                        chart.width = 16
                        ws.add_chart(chart, "H4")
                    except Exception:
                        pass

            styled = io.BytesIO()
            wb.save(styled)
            styled.seek(0)

        # --- Download button
        ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
        st.download_button(
            label="⬇️ Download Complete Excel Report",
            data=styled.getvalue(),
            file_name=f"Vahan_SmartReport_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.success("✅ Export successful — all KPIs, forecasts, AI summaries, and charts included.")

# ---------------- 🧩 RAW JSON PREVIEW (Debug Mode) ----------------
with st.expander("🛠️ Raw JSON Preview (Developer Debug Mode)", expanded=False):
    st.caption("Inspect raw API responses returned from each Vahan endpoint for debugging or verification.")
    
    debug_cols = st.columns(2)

    with debug_cols[0]:
        st.markdown("**📦 Category JSON**")
        st.json(cat_json if cat_json else {"info": "No category data"})
        
        st.markdown("**🏭 Top Makers JSON**")
        st.json(mk_json if mk_json else {"info": "No maker data"})
        
        st.markdown("**📊 Trend JSON**")
        st.json(tr_json if tr_json else {"info": "No trend data"})

    with debug_cols[1]:
        st.markdown("**💰 Top 5 Revenue JSON**")
        st.json(top5_rev_json if top5_rev_json else {"info": "No revenue data"})

        st.markdown("**📈 Revenue Trend JSON**")
        st.json(rev_trend_json if rev_trend_json else {"info": "No trend data"})

    st.info("✅ All API responses are safely displayed below. Use this only for internal diagnostics.")

# ---------------- ⚡ FOOTER KPIs + EXECUTIVE SUMMARY ----------------
st.markdown("---")
st.subheader("📊 Dashboard Summary & Insights")

# --- KPI Section with styled layout
kpi_cols = st.columns(4)

with kpi_cols[0]:
    if not df_trend.empty:
        total_reg = int(df_trend["value"].sum())
        st.metric("🧾 Total Registrations", f"{total_reg:,}")
    else:
        st.metric("🧾 Total Registrations", "N/A")

with kpi_cols[1]:
    if "daily_avg" in locals() and daily_avg is not None:
        st.metric("📅 Daily Avg Orders", f"{daily_avg:.0f}")
    else:
        st.metric("📅 Daily Avg Orders", "N/A")

with kpi_cols[2]:
    if latest_yoy is not None:
        yoy_arrow = "🔼" if latest_yoy > 0 else "🔽"
        st.metric("📈 Latest YoY%", f"{yoy_arrow} {latest_yoy:.2f}%")
    else:
        st.metric("📈 Latest YoY%", "N/A")

with kpi_cols[3]:
    if latest_qoq is not None:
        qoq_arrow = "🔼" if latest_qoq > 0 else "🔽"
        st.metric("📉 Latest QoQ%", f"{qoq_arrow} {latest_qoq:.2f}%")
    else:
        st.metric("📉 Latest QoQ%", "N/A")

# --- Additional KPI if revenue data exists
if not df_top5_rev.empty:
    try:
        top_state = df_top5_rev.iloc[0].get("label", df_top5_rev.iloc[0].get("state", "N/A"))
        top_val = df_top5_rev.iloc[0].get("value", "N/A")
        st.success(f"🏆 **Top Revenue State:** {top_state} — ₹{top_val:,}")
    except Exception:
        st.info("🏆 Top Revenue State: Data unavailable")

# --- AI-Powered Executive Summary
if enable_ai:
    st.markdown("### 🤖 Executive AI Summary")
    with st.spinner("Synthesizing executive-level narrative..."):
        try:
            # Prepare structured context for AI
            context = {
                "total_registrations": int(df_trend["value"].sum()) if not df_trend.empty else None,
                "latest_yoy": float(latest_yoy) if latest_yoy is not None else None,
                "latest_qoq": float(latest_qoq) if latest_qoq is not None else None,
                "top_revenue_state": top_state if not df_top5_rev.empty else None,
                "daily_avg": float(daily_avg) if "daily_avg" in locals() and daily_avg is not None else None,
            }

            system = (
                "You are an executive analytics assistant summarizing key performance indicators "
                "for a national vehicle registration and revenue dashboard. "
                "Focus on major trends, anomalies, growth momentum, and actionable business takeaways."
            )
            user = (
                f"Context data: {json.dumps(context, default=str)}\n"
                "Generate a 5-sentence executive-level summary covering trends, revenue, and growth signals, "
                "then conclude with one strategic recommendation."
            )

            ai_resp = deepinfra_chat(system, user, max_tokens=320)

            if isinstance(ai_resp, dict) and "text" in ai_resp:
                st.info(ai_resp["text"])
            else:
                st.warning("AI executive summary unavailable.")
        except Exception as e:
            st.error(f"AI summary generation failed: {e}")

# --- Footer Aesthetic
st.markdown("""
<hr style="border: 1px solid #444; margin-top: 2em; margin-bottom: 1em;">
<div style="text-align:center; color:gray; font-size:0.9em;">
    🚀 <b>Parivahan Analytics 2025</b> — Automated Insights | AI Narratives | Smart KPIs
</div>
""", unsafe_allow_html=True)
