# app/streamlit_app.py
import os
import json
import time
import requests
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io
from datetime import date, timedelta

from dotenv import load_dotenv
import os
load_dotenv()

# Excel formatting
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# Your existing vahan modules (kept as-is)
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
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="Vahan Registrations ", layout="wide")
st.title("🚀 Vahan Registrations")
st.markdown("**Parivahan Analytics — KPIs, AI Narratives (DeepInfra), Forecasting, Clustering, Anomaly Detection & Smart Exports**")

# ---------------- Sidebar filters ----------------
today = date.today()
default_from_year = max(2017, today.year - 1)

st.sidebar.header("Filters & Options")
from_year = st.sidebar.number_input("From Year", min_value=2012, max_value=today.year, value=default_from_year)
to_year = st.sidebar.number_input("To Year", min_value=from_year, max_value=today.year, value=today.year)
state_code = st.sidebar.text_input("State Code (blank=All-India)", value="")
rto_code = st.sidebar.text_input("RTO Code (0=aggregate)", value="0")
vehicle_classes = st.sidebar.text_input("Vehicle Classes (e.g., 2W,3W,4W)", value="")
vehicle_makers = st.sidebar.text_input("Vehicle Makers (comma-separated or IDs)", value="")
time_period = st.sidebar.selectbox("Time Period", options=[0,1,2], index=0)
fitness_check = st.sidebar.selectbox("Fitness Check", options=[0,1], index=0)
vehicle_type = st.sidebar.text_input("Vehicle Type (optional)", value="")

# Advanced toggles
st.sidebar.markdown("---")
enable_forecast = st.sidebar.checkbox("Enable Forecasting", value=True)
enable_anomaly = st.sidebar.checkbox("Enable Anomaly Detection", value=True)
enable_clustering = st.sidebar.checkbox("Enable Clustering", value=True)
enable_ai = st.sidebar.checkbox("Enable DeepInfra AI Narratives", value=True)
forecast_periods = st.sidebar.number_input("Forecast horizon (months)", min_value=1, max_value=36, value=3)

# DeepInfra settings (env var or st.secrets)
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY") or (st.secrets.get("DEEPINFRA_API_KEY") if "DEEPINFRA_API_KEY" in st.secrets else None)
# Recommended model (balanced): Mixtral 8x7B instruct
DEEPINFRA_MODEL = os.environ.get("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

if enable_ai and not DEEPINFRA_API_KEY:
    st.sidebar.warning("DeepInfra API key not found. Set DEEPINFRA_API_KEY env var or st.secrets['DEEPINFRA_API_KEY'] to enable AI features.")

# ---------------- Build Vahan params ----------------
params_common = build_params(
    from_year, to_year,
    state_code=state_code, rto_code=rto_code,
    vehicle_classes=vehicle_classes, vehicle_makers=vehicle_makers,
    time_period=time_period, fitness_check=fitness_check, vehicle_type=vehicle_type
)

# ---------------- Helper: Safe API call ----------------
def fetch_json(endpoint, params=params_common, desc=""):
    try:
        json_data, _ = get_json(endpoint, params)
        if json_data is None:
            return {}
        return json_data
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch {desc} ({endpoint}): {e}")
        return {}

# ---------------- DeepInfra helper (OpenAI-compatible endpoint) ----------------
DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

def deepinfra_chat(system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.1):
    """
    Simple wrapper calling DeepInfra's OpenAI-compatible chat completions endpoint.
    Requires DEEPINFRA_API_KEY to be set in env or st.secrets.
    """
    if not DEEPINFRA_API_KEY:
        return {"error":"DeepInfra API key not configured."}
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role":"system","content": system_prompt},
            {"role":"user","content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # keep fallback values; tune as needed
    }
    try:
        resp = requests.post(DEEPINFRA_CHAT_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # DeepInfra's OpenAI-compatible response structure: choices[0].message.content
        if "choices" in data and data["choices"]:
            return {"text": data["choices"][0]["message"]["content"], "raw": data}
        # fallback
        return {"text": json.dumps(data)}
    except Exception as e:
        return {"error": str(e), "raw_resp": getattr(e, 'response', None)}

# ---------------- 1️⃣ Category Distribution ----------------
with st.spinner("Fetching category distribution..."):
    cat_json = fetch_json("vahandashboard/categoriesdonutchart", desc="Category distribution")
df_cat = to_df(cat_json)
if not df_cat.empty:
    st.subheader("Category Distribution")
    col1, col2 = st.columns(2)
    with col1:
        try:
            bar_from_df(df_cat, title="Category Distribution (Bar)")
        except Exception:
            st.write(df_cat)
    with col2:
        try:
            pie_from_df(df_cat, title="Category Distribution (Pie)", donut=True)
        except Exception:
            st.write(df_cat)

# AI summary for category
if enable_ai and not df_cat.empty:
    with st.spinner("Generating AI summary for Category Distribution..."):
        system = "You are an analytics assistant summarizing vehicle category distribution and trends."
        sample = df_cat.head(10).to_dict(orient='records')
        user = f"Dataset: Category distribution sample rows: {json.dumps(sample, default=str)}\nProvide a short (3-5 line) data-driven summary highlighting top categories, percent shares, growth signals, and one recommendation."
        ai_resp = deepinfra_chat(system, user, max_tokens=250)
        if "text" in ai_resp:
            st.markdown("**AI Category Summary**")
            st.write(ai_resp["text"])
        else:
            st.info("AI summary unavailable.")

# ---------------- 2️⃣ Top Makers ----------------
with st.spinner("Fetching Top Makers..."):
    mk_json = fetch_json("vahandashboard/top5Makerchart", desc="Top Makers")
    df_mk = parse_makers(mk_json)  # use the new parser here

if not df_mk.empty:
    st.subheader("Top Makers")
    col1, col2 = st.columns(2)
    with col1:
        bar_from_df(df_mk.rename(columns={"maker": "label"}), title="Top Makers (Bar)")
    with col2:
        pie_from_df(df_mk.rename(columns={"maker": "label"}), title="Top Makers (Pie)", donut=True)

# AI summary for makers
if enable_ai and not df_mk.empty:
    with st.spinner("Generating AI summary for Top Makers..."):
        system = "You are an analytics assistant summarizing maker market share and growth."
        sample = df_mk.head(10).to_dict(orient='records')
        user = f"Dataset: Top makers sample rows: {json.dumps(sample, default=str)}\nProvide a short summary with top maker, fastest growing maker (if determinable), and one strategic insight."
        ai_resp = deepinfra_chat(system, user, max_tokens=220)
        if "text" in ai_resp:
            st.markdown("**AI Makers Summary**")
            st.write(ai_resp["text"])



# ---------------- 3️⃣ Yearly/MONTHLY Trend + YoY/QoQ ----------------
with st.spinner("Fetching registration trends..."):
    tr_json = fetch_json("vahandashboard/vahanyearwiseregistrationtrend", desc="Registration Trend")
try:
    df_trend = normalize_trend(tr_json)
except Exception as e:
    st.error(f"Trend parsing failed: {e}")
    df_trend = pd.DataFrame(columns=["date","value"])

if not df_trend.empty:
    st.subheader("Registration Trend")
    try:
        line_from_trend(df_trend, title="Registrations Trend Line")
    except Exception:
        st.line_chart(df_trend.set_index('date')['value'])
    # compute metrics
    yoy_df = compute_yoy(df_trend)
    qoq_df = compute_qoq(df_trend)

    latest_yoy = yoy_df["YoY%"].dropna().iloc[-1] if not yoy_df.empty and yoy_df["YoY%"].dropna().size else None
    latest_qoq = qoq_df["QoQ%"].dropna().iloc[-1] if not qoq_df.empty and qoq_df["QoQ%"].dropna().size else None

    # daily orders (approx)
    if 'date' in df_trend.columns:
        total_days = (df_trend['date'].max() - df_trend['date'].min()).days or 1
        daily_avg = df_trend['value'].sum() / total_days
    else:
        daily_avg = None

    # Show metrics
    show_metrics(latest_yoy, latest_qoq)
    show_tables(yoy_df, qoq_df)

    # KPI cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Total Registrations", f"{int(df_trend['value'].sum()):,}")
    with col_k2:
        st.metric("Daily Avg Orders", f"{daily_avg:.0f}" if daily_avg is not None else "N/A")
    with col_k3:
        if latest_yoy is not None:
            st.metric("Latest YoY%", f"{latest_yoy:.2f}%")
        else:
            st.metric("Latest YoY%", "N/A")
    with col_k4:
        if latest_qoq is not None:
            st.metric("Latest QoQ%", f"{latest_qoq:.2f}%")
        else:
            st.metric("Latest QoQ%", "N/A")
else:
    yoy_df = pd.DataFrame()
    qoq_df = pd.DataFrame()
    latest_yoy = latest_qoq = None

# AI narrative for trends
if enable_ai and not df_trend.empty:
    with st.spinner("Generating AI trend narrative..."):
        system = (
            "You are a concise analytics assistant: provide trend summary, anomalies, "
            "top drivers, and 2 actionable recommendations."
        )

        # Craft a compact dataset summary (last 12 periods)
        sample_rows = df_trend.tail(12).to_dict(orient='records')

        # Safe formatting for daily average
        daily_avg_display = f"{daily_avg:.1f}" if daily_avg is not None else "N/A"

        # Build user prompt
        user = (
            f"Dataset: last 12 periods (date,value): {json.dumps(sample_rows, default=str)}\n"
            f"Latest YoY: {latest_yoy}, Latest QoQ: {latest_qoq}, DailyAvg: {daily_avg_display}\n"
            "Provide a short (5–7 sentences) analysis and 2 short recommendations."
        )

        ai_resp = deepinfra_chat(system, user, max_tokens=400)

        if isinstance(ai_resp, dict) and "text" in ai_resp:
            st.markdown("**AI Trend Narrative**")
            st.write(ai_resp["text"])
        else:
            st.info("No AI response received.")

# ---------------- 4️⃣ Duration-wise Growth ----------------
def fetch_duration_growth(calendar_type, label):
    with st.spinner(f"Fetching {label} growth..."):
        json_data = fetch_json("vahandashboard/durationWiseRegistrationTable",
                               {**params_common, "calendarType": calendar_type}, desc=f"{label} growth")
        df = parse_duration_table(json_data)
        if not df.empty:
            st.subheader(f"{label} Vehicle Registration Growth")
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
            return df
        return pd.DataFrame()

df_monthly = fetch_duration_growth(3, "Monthly")
df_quarterly = fetch_duration_growth(2, "Quarterly")
df_yearly = fetch_duration_growth(1, "Yearly")

# ---------------- 5️⃣ Top 5 Revenue States ----------------
with st.spinner("Fetching Top 5 Revenue States..."):
    top5_rev_json = fetch_json("vahandashboard/top5chartRevenueFee", desc="Top 5 Revenue States")
df_top5_rev = parse_top5_revenue(top5_rev_json if top5_rev_json else {})
if not df_top5_rev.empty:
    st.subheader("Top 5 Revenue States")
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

# AI summary for revenue
if enable_ai and not df_top5_rev.empty:
    with st.spinner("Generating AI summary for Revenue..."):
        system = "You are a financial analytics assistant summarizing state-level revenue and trends."
        sample = df_top5_rev.head(10).to_dict(orient='records')
        user = f"Dataset: Top 5 revenue states sample: {json.dumps(sample, default=str)}\nProvide a short summary (3 lines) and 1 recommendation."
        ai_resp = deepinfra_chat(system, user, max_tokens=200)
        if "text" in ai_resp:
            st.markdown("**AI Revenue Summary**")
            st.write(ai_resp["text"])

# ---------------- 6️⃣ Revenue Trend ----------------
with st.spinner("Fetching Revenue Trend..."):
    rev_trend_json = fetch_json("vahandashboard/revenueFeeLineChart", desc="Revenue Trend")
df_rev_trend = parse_revenue_trend(rev_trend_json if rev_trend_json else {})
if not df_rev_trend.empty:
    st.subheader("Revenue Trend Comparison")
    try:
        chart = alt.Chart(df_rev_trend).mark_line(point=True).encode(
            x=alt.X('period:O', title='Period'),
            y=alt.Y('value:Q', title='Revenue'),
            color='year:N'
        ).properties(title="Revenue Trend Comparison")
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.write(df_rev_trend)

# ---------------- Forecasting ----------------
def linear_forecast(df_ts, months=3):
    """Lightweight linear regression forecast on monthly aggregated data"""
    df = df_ts.copy()
    df = df.sort_values('date')
    df['period_num'] = (df['date'].dt.year - df['date'].dt.year.min()) * 12 + df['date'].dt.month
    X = df[['period_num']].values
    y = df['value'].values
    if len(df) < 3:
        return pd.DataFrame()
    model = LinearRegression()
    model.fit(X, y)
    last_period = df['period_num'].max()
    future_periods = np.arange(last_period + 1, last_period + months + 1).reshape(-1, 1)
    preds = model.predict(future_periods)
    # build future dates (monthly)
    last_date = df['date'].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, months+1)]
    return pd.DataFrame({'date': future_dates, 'value': preds})

forecast_df = pd.DataFrame()
if enable_forecast and not df_trend.empty:
    st.subheader("🔮 Forecasting")
    if PROPHET_AVAILABLE and 'date' in df_trend.columns and 'value' in df_trend.columns:
        try:
            dfp = df_trend[['date','value']].rename(columns={'date':'ds','value':'y'})
            m = Prophet()
            m.fit(dfp)
            future = m.make_future_dataframe(periods=forecast_periods, freq='MS')
            forecast = m.predict(future)
            forecast_df = forecast[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'ds':'date','yhat':'value'})
            st.write("Prophet forecast (tail):")
            st.write(forecast_df.tail(forecast_periods))
            # show plot
            fig = m.plot(forecast)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Prophet forecast failed: {e}. Falling back to linear regression.")
            forecast_df = linear_forecast(df_trend, months=forecast_periods)
            st.line_chart(forecast_df.set_index('date')['value'])
    else:
        if not SKLEARN_AVAILABLE:
            # our linear regression fallback uses numpy & pandas only
            forecast_df = linear_forecast(df_trend, months=forecast_periods)
            if not forecast_df.empty:
                st.line_chart(pd.concat([df_trend.set_index('date')['value'], forecast_df.set_index('date')['value']]))
        else:
            # linear regression via sklearn (same as above)
            forecast_df = linear_forecast(df_trend, months=forecast_periods)
            if not forecast_df.empty:
                st.line_chart(pd.concat([df_trend.set_index('date')['value'], forecast_df.set_index('date')['value']]))

# AI forecast commentary
if enable_ai and not forecast_df.empty:
    with st.spinner("Generating AI forecast commentary..."):
        system = "You are an analytics assistant that summarizes short-term forecasts and confidence."
        sample = forecast_df.head(6).to_dict(orient='records')
        user = f"Forecasted values (date,value): {json.dumps(sample, default=str)}\nProvide a 3-sentence summary with risk/confidence commentary."
        ai_resp = deepinfra_chat(system, user, max_tokens=220)
        if "text" in ai_resp:
            st.markdown("**AI Forecast Commentary**")
            st.write(ai_resp["text"])

# ---------------- Anomaly Detection ----------------
if enable_anomaly:
    st.subheader("🚨 Anomaly Detection")
    if not SKLEARN_AVAILABLE:
        st.info("scikit-learn not available. Install scikit-learn to enable IsolationForest anomaly detection.")
    elif 'value' not in df_trend.columns:
        st.info("No 'value' column detected for anomaly detection.")
    else:
        contamination = st.slider("Contamination (expected outliers fraction)", 0.001, 0.2, 0.02)
        try:
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
                points = base.mark_circle(size=60).encode(y='value:Q', color=alt.condition(alt.datum.anomaly == -1, alt.value('red'), alt.value('black')))
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
        if "text" in ai_resp:
            st.markdown("**AI Anomaly Insights**")
            st.write(ai_resp["text"])

# ---------------- Clustering & Correlation ----------------
if enable_clustering:
    st.subheader("🧭 Clustering & Correlation (AI + Visualized)")
    if not SKLEARN_AVAILABLE:
        st.info("scikit-learn not available. Install scikit-learn for clustering features.")
    else:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        import numpy as np
        import plotly.express as px

        # 1️⃣ Build feature matrix
        features_df = None

        # Prefer revenue data first
        if not df_top5_rev.empty:
            try:
                features_df = df_top5_rev.copy().reset_index(drop=True)
                if 'value' not in features_df.columns and 'revenue' in features_df.columns:
                    features_df['value'] = pd.to_numeric(features_df['revenue'], errors='coerce').fillna(0)
            except Exception:
                features_df = None

        # Fallback: from trend rolling stats
        if features_df is None and not df_trend.empty:
            df_roll = df_trend.set_index('date').resample('M').sum().fillna(0)
            df_roll['rolling_mean_3'] = df_roll['value'].rolling(3).mean().fillna(0)
            df_roll['rolling_std_3'] = df_roll['value'].rolling(3).std().fillna(0)
            features_df = df_roll[['rolling_mean_3', 'rolling_std_3']].reset_index().dropna().reset_index(drop=True)

        if features_df is not None and not features_df.empty:
            numeric_cols = [c for c in features_df.columns if pd.api.types.is_numeric_dtype(features_df[c])]
            if numeric_cols:
                X = features_df[numeric_cols].fillna(0).astype(float)

                # Scale features
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)

                # 2️⃣ Clustering
                n_clusters = st.slider("Clusters (k)", 2, min(8, len(Xs)), 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(Xs)
                features_df['cluster'] = labels

                st.write("Cluster centroids (numeric features):")
                st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols))

                # 3️⃣ PCA projection — adaptive
                try:
                    if Xs.shape[0] > 1 and Xs.shape[1] > 1:
                        n_comp = min(3, Xs.shape[1])
                        pca = PCA(n_components=n_comp)
                        proj = pca.fit_transform(Xs)

                        if n_comp >= 3:
                            # 3D scatter via Plotly
                            fig3d = px.scatter_3d(
                                x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
                                color=labels.astype(str),
                                title="3D PCA Cluster Visualization",
                                labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'}
                            )
                            st.plotly_chart(fig3d, use_container_width=True)
                        else:
                            # 2D scatter via Altair
                            scatter = pd.DataFrame({'x': proj[:, 0], 'y': proj[:, 1], 'cluster': labels})
                            chart = alt.Chart(scatter).mark_circle(size=60).encode(
                                x='x', y='y', color='cluster:N', tooltip=['x', 'y', 'cluster']
                            ).properties(title="PCA Projection of Clusters")
                            st.altair_chart(chart, use_container_width=True)
                    else:
                        st.warning(f"PCA skipped — insufficient data (shape={Xs.shape})")
                        st.line_chart(features_df[numeric_cols])
                except Exception as e:
                    st.warning(f"PCA projection failed: {e}")
                    st.line_chart(features_df[numeric_cols])

                # 4️⃣ Silhouette score
                if len(Xs) >= n_clusters + 1:
                    try:
                        sc = silhouette_score(Xs, labels)
                        st.metric("Silhouette Score", f"{sc:.3f}")
                    except Exception as e:
                        st.warning(f"Silhouette score computation failed: {e}")

                # 5️⃣ Cluster samples
                st.write("Sample rows by cluster:")
                st.dataframe(features_df.head(200))

                # 6️⃣ Correlation heatmap
                try:
                    st.markdown("### 🔗 Correlation Heatmap")
                    corr = features_df.select_dtypes(include=[np.number]).corr()
                    fig_corr = px.imshow(
                        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        title="Correlation Matrix"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                except Exception as e:
                    st.warning(f"Correlation calc failed: {e}")

                # 7️⃣ Optional AI summary (if AI enabled)
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

            else:
                st.info("No numeric columns available for clustering.")
        else:
            st.info("No suitable data for clustering (need trend or revenue data).")
# --- Write to Excel (Safe Export)
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    any_written = False
    for name, df in datasets.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_excel(writer, sheet_name=name[:31], index=False)
            any_written = True

    # ✅ Fallback sheet if everything is empty
    if not any_written:
        pd.DataFrame({"Info": ["No data available for export."]}).to_excel(
            writer, sheet_name="Summary", index=False
        )

output.seek(0)

# --- Load workbook and style
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

    # Cell borders + alignment
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin

    # Auto-fit columns
    for col in ws.columns:
        width = max(len(str(c.value or "")) for c in col) + 3
        ws.column_dimensions[get_column_letter(col[0].column)].width = width

    # Add chart if numeric data exists
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

# --- Save styled workbook
styled = io.BytesIO()
wb.save(styled)
styled.seek(0)

# --- Download button
ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
st.download_button(
    label="⬇️ Download Excel ",
    data=styled.getvalue(),
    file_name=f"Vahan_{ts}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.success("✅ All datasets, AI summaries, ML forecasts, styled sheets, and embedded charts exported successfully!")


# ---------------- Raw JSON preview (debug) ----------------
with st.expander("🛠️ Raw JSON Preview (debug)"):
    st.write("Category JSON", cat_json)
    st.write("Top Makers JSON", mk_json)
    st.write("Trend JSON", tr_json)
    st.write("Top 5 Revenue JSON", top5_rev_json)
    st.write("Revenue Trend JSON", rev_trend_json)

# ---------------- Footer KPIs summary ----------------
st.markdown("---")
st.subheader("📊 Dashboard Summary")
if not df_trend.empty:
    st.metric("Total Registrations", int(df_trend['value'].sum()))
if not df_top5_rev.empty:
    # safe access
    try:
        st.metric("Top Revenue State", df_top5_rev.iloc[0].get('label', df_top5_rev.iloc[0].get('state', 'N/A')))
    except Exception:
        pass
if latest_yoy is not None:
    st.metric("Latest YoY%", f"{latest_yoy:.2f}%")
if latest_qoq is not None:
    st.metric("Latest QoQ%", f"{latest_qoq:.2f}%")

# Final AI executive summary
if enable_ai:
    st.markdown("### 🤖 Executive AI Summary")
    with st.spinner("Generating executive summary..."):
        try:
            system = "You are an executive business analyst. Provide a concise 5-sentence executive summary of registration trends, revenue, anomalies, and forecast direction, with one strategic recommendation."
            # sample summary context
            ctx = {
                "total_registrations": int(df_trend['value'].sum()) if not df_trend.empty else None,
                "latest_yoy": float(latest_yoy) if latest_yoy is not None else None,
                "latest_qoq": float(latest_qoq) if latest_qoq is not None else None,
                "top_revenue_state": df_top5_rev.iloc[0]['label'] if not df_top5_rev.empty else None,
                "daily_avg": float(daily_avg) if 'daily_avg' in locals() and daily_avg is not None else None,
            }
            user = f"Context JSON: {json.dumps(ctx, default=str)}\nProvide an executive 5-sentence summary and one action item."
            ai_resp = deepinfra_chat(system, user, max_tokens=300)
            if "text" in ai_resp:
                st.write(ai_resp["text"])
            else:
                st.info("AI executive summary unavailable.")
        except Exception as e:
            st.error(f"AI summary failed: {e}")

# End of file
