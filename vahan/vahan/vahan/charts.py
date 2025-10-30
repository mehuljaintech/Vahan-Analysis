# # ==============================================================
# # vahan/charts.py — MAXED+ ANALYTICS VISUALIZATION SUITE (2025)
# # ==============================================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import altair as alt
# from datetime import datetime

# # ==============================================================
# # 🌈 GLOBAL STYLE CONFIG
# # ==============================================================
# alt.themes.enable("dark")
# px.defaults.template = "plotly_white"
# px.defaults.color_continuous_scale = "Turbo"
# px.defaults.width = 1100
# px.defaults.height = 500

# # ==============================================================
# # 🧩 SAFE DATA CHECK
# # ==============================================================
# def safe_df(df):
#     """Check if DataFrame is valid."""
#     return df is not None and isinstance(df, pd.DataFrame) and not df.empty


# # ==============================================================
# # 📊 BAR CHART
# # ==============================================================
# def bar_from_df(df, title="", index_col="label", value_col="value", color=None, orientation="v"):
#     if not safe_df(df):
#         st.info(f"No data for {title or 'bar chart'}.")
#         return
#     st.subheader(f"📊 {title}")
#     fig = px.bar(
#         df,
#         x=index_col if orientation == "v" else value_col,
#         y=value_col if orientation == "v" else index_col,
#         color=color,
#         orientation=orientation,
#         text=value_col,
#         color_discrete_sequence=px.colors.qualitative.Bold
#     )
#     fig.update_traces(texttemplate='%{text}', textposition='outside')
#     fig.update_layout(
#         title=title,
#         xaxis_title=index_col.capitalize(),
#         yaxis_title=value_col.capitalize(),
#         bargap=0.2,
#         legend_title=color or "Category",
#     )
#     st.plotly_chart(fig, use_container_width=True)


# # ==============================================================
# # 🥧 PIE / DONUT CHART
# # ==============================================================
# def pie_from_df(df, title="", names="label", values="value", donut=True):
#     if not safe_df(df):
#         st.info(f"No data for {title or 'pie chart'}.")
#         return
#     st.subheader(f"🥧 {title}")
#     fig = px.pie(
#         df,
#         names=names,
#         values=values,
#         hole=0.4 if donut else 0,
#         color_discrete_sequence=px.colors.sequential.Viridis
#     )
#     fig.update_traces(textinfo='percent+label', pull=[0.05]*len(df))
#     fig.update_layout(title=title)
#     st.plotly_chart(fig, use_container_width=True)


# # ==============================================================
# # 📈 LINE / TREND CHART
# # ==============================================================
# def line_from_trend(df, title="Trend Line", x_col="date", y_col="value"):
#     """Simple line chart using Altair (compatible with old references)."""
#     if not safe_df(df):
#         st.info(f"No data for {title}.")
#         return
#     d = df.dropna(subset=[x_col, y_col]).sort_values(x_col)
#     if d.empty:
#         st.info("No valid data to plot.")
#         return
#     st.subheader(f"📈 {title}")
#     chart = alt.Chart(d).mark_line(point=True, interpolate='monotone').encode(
#         x=alt.X(f"{x_col}:T", title=x_col.capitalize()),
#         y=alt.Y(f"{y_col}:Q", title=y_col.capitalize()),
#         tooltip=[x_col, y_col]
#     ).interactive()
#     st.altair_chart(chart, use_container_width=True)


# # ==============================================================
# # 📈 ADVANCED TREND WITH ROLLING AVG / ANOMALY / FORECAST
# # ==============================================================
# def trend_chart(df, title="📈 Trend", x_col="date", y_col="value"):
#     if not safe_df(df):
#         st.info(f"No data for {title}.")
#         return
#     df = df.sort_values(x_col)
#     df["RollingAvg"] = df[y_col].rolling(3, min_periods=1).mean()
#     df["Forecast"] = df["RollingAvg"].shift(1)
#     df["Growth%"] = df[y_col].pct_change() * 100
#     df["Anomaly"] = (df[y_col] - df["RollingAvg"]).abs() > (df["RollingAvg"] * 0.25)

#     base = alt.Chart(df).encode(x=f"{x_col}:T")
#     line = base.mark_line(color="steelblue", interpolate='monotone').encode(
#         y=alt.Y(f"{y_col}:Q", title="Registrations"),
#         tooltip=[x_col, y_col, "RollingAvg", "Forecast", "Growth%", "Anomaly"]
#     )
#     avg = base.mark_line(color="orange", strokeDash=[6, 3]).encode(y="RollingAvg:Q")
#     forecast = base.mark_line(color="green", strokeDash=[3, 3]).encode(y="Forecast:Q")
#     points = base.mark_circle(size=80, color="red").transform_filter("datum.Anomaly == true").encode(y=f"{y_col}:Q")

#     chart = (line + avg + forecast + points).interactive().properties(title=title)
#     st.altair_chart(chart, use_container_width=True)


# # ==============================================================
# # 🌈 STACKED AREA
# # ==============================================================
# def stacked_area(df, x_col="date", y_col="value", color_col="category", title="Stacked Area"):
#     if not safe_df(df):
#         st.info(f"No data for {title}.")
#         return
#     st.subheader(f"🌈 {title}")
#     chart = alt.Chart(df).mark_area(opacity=0.6, interpolate='monotone').encode(
#         x=alt.X(f"{x_col}:T"),
#         y=alt.Y(f"{y_col}:Q", stack="zero"),
#         color=alt.Color(color_col, legend=alt.Legend(title=color_col.capitalize())),
#         tooltip=[x_col, y_col, color_col]
#     ).interactive()
#     st.altair_chart(chart, use_container_width=True)


# # ==============================================================
# # 📉 MULTI-LINE CHART
# # ==============================================================
# def multi_line_chart(df, x_col="date", y_cols=None, title="Multi-Line Chart"):
#     """Supports multiple numeric series."""
#     if not safe_df(df) or not y_cols:
#         st.info(f"No data for {title}.")
#         return
#     st.subheader(f"📊 {title}")
#     df_long = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="series", value_name="value")
#     chart = alt.Chart(df_long).mark_line(point=True, interpolate='monotone').encode(
#         x=alt.X(f"{x_col}:T"),
#         y=alt.Y("value:Q"),
#         color="series:N",
#         tooltip=[x_col, "series", "value"]
#     ).interactive()
#     st.altair_chart(chart, use_container_width=True)


# # ==============================================================
# # 📋 KPI METRICS DISPLAY
# # ==============================================================
# def show_metrics(latest_yoy=None, latest_qoq=None, latest_cumulative=None, additional_metrics=None):
#     metric_cols = st.columns(max(3, 1 + (len(additional_metrics) if additional_metrics else 0)))
#     metric_cols[0].metric("Latest YoY%", f"{latest_yoy:.1f}%" if latest_yoy is not None else "n/a")
#     metric_cols[1].metric("Latest QoQ%", f"{latest_qoq:.1f}%" if latest_qoq is not None else "n/a")
#     metric_cols[2].metric("Cumulative Registrations", f"{latest_cumulative:,}" if latest_cumulative is not None else "n/a")
#     if additional_metrics:
#         for i, (k, v) in enumerate(additional_metrics.items()):
#             if i + 3 < len(metric_cols):
#                 metric_cols[i + 3].metric(k, str(v))


# # ==============================================================
# # 🧾 DATA TABLE DISPLAY
# # ==============================================================
# def show_tables(yoy_df=None, qoq_df=None):
#     col1, col2 = st.columns(2)
#     with col1:
#         if yoy_df is not None and not yoy_df.empty:
#             st.markdown("YoY% by Month")
#             st.dataframe(yoy_df.tail(12), use_container_width=True)
#     with col2:
#         if qoq_df is not None and not qoq_df.empty:
#             tmp = qoq_df.copy()
#             if "date" in tmp and isinstance(tmp["date"].iloc[0], pd.Timestamp):
#                 tmp["Quarter"] = tmp["date"].dt.to_period("Q").astype(str)
#             st.markdown("QoQ% by Quarter")
#             st.dataframe(tmp[["Quarter", "value", "QoQ%"]] if "Quarter" in tmp else tmp, use_container_width=True)


# # ==============================================================
# # 📉 WATERFALL / CUMULATIVE / FORECAST CHARTS
# # ==============================================================
# def waterfall_chart(df, x_col="category", y_col="value", title="Waterfall Chart"):
#     if not safe_df(df):
#         st.info(f"No data for {title}.")
#         return
#     st.subheader(f"💧 {title}")
#     fig = go.Figure(go.Waterfall(
#         name="Registrations",
#         x=df[x_col],
#         y=df[y_col],
#         measure=["relative"] * len(df)
#     ))
#     fig.update_layout(title=title)
#     st.plotly_chart(fig, use_container_width=True)


# def cumulative_line_chart(df, x_col="date", y_col="value", title="Cumulative Line Chart"):
#     if not safe_df(df):
#         st.info(f"No data for {title}.")
#         return
#     df = df.sort_values(x_col)
#     df["cumulative"] = df[y_col].cumsum()
#     st.subheader(f"📈 {title}")
#     fig = px.line(df, x=x_col, y="cumulative", markers=True, title=title)
#     st.plotly_chart(fig, use_container_width=True)


# # ==============================================================
# # 🧮 DATA TABLE EXPORT
# # ==============================================================
# def show_data(df, title="Data Table"):
#     if not safe_df(df): return
#     st.markdown(f"### 🧾 {title}")
#     st.dataframe(df, use_container_width=True)
#     st.download_button(
#         "⬇️ Download CSV",
#         df.to_csv(index=False).encode("utf-8"),
#         file_name=f"{title.replace(' ', '_')}.csv"
#     )


# ==============================================================
# 🚀 vahan/charts.py — MAXED+ ANALYTICS VISUALIZATION SUITE (2025)
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from datetime import datetime

# ==============================================================
# 🌈 GLOBAL STYLE CONFIG
# ==============================================================
alt.themes.enable("dark")
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Turbo"
px.defaults.width = 1100
px.defaults.height = 500

# ==============================================================
# 🧩 SAFE DATA CHECK
# ==============================================================
def safe_df(df):
    """Check if DataFrame is valid."""
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty


# ==============================================================
# 📊 BAR CHART
# ==============================================================
def bar_from_df(df, title="", index_col="label", value_col="value", color=None, orientation="v"):
    if not safe_df(df):
        st.info(f"No data for {title or 'bar chart'}.")
        return
    st.subheader(f"📊 {title}")
    fig = px.bar(
        df,
        x=index_col if orientation == "v" else value_col,
        y=value_col if orientation == "v" else index_col,
        color=color,
        orientation=orientation,
        text=value_col,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        title=title,
        xaxis_title=index_col.capitalize(),
        yaxis_title=value_col.capitalize(),
        bargap=0.2,
        legend_title=color or "Category",
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================
# 🥧 PIE / DONUT CHART
# ==============================================================
def pie_from_df(df, title="", names="label", values="value", donut=True):
    if not safe_df(df):
        st.info(f"No data for {title or 'pie chart'}.")
        return
    st.subheader(f"🥧 {title}")
    fig = px.pie(
        df,
        names=names,
        values=values,
        hole=0.4 if donut else 0,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_traces(textinfo='percent+label', pull=[0.05]*len(df))
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================
# 📈 LINE / TREND CHART
# ==============================================================
def line_from_trend(df, title="Trend Line", x_col="date", y_col="value"):
    """Simple line chart using Altair (compatible with old references)."""
    if not safe_df(df):
        st.info(f"No data for {title}.")
        return
    d = df.dropna(subset=[x_col, y_col]).sort_values(x_col)
    if d.empty:
        st.info("No valid data to plot.")
        return
    st.subheader(f"📈 {title}")
    chart = alt.Chart(d).mark_line(point=True, interpolate='monotone').encode(
        x=alt.X(f"{x_col}:T", title=x_col.capitalize()),
        y=alt.Y(f"{y_col}:Q", title=y_col.capitalize()),
        tooltip=[x_col, y_col]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)


# ==============================================================
# 📈 ADVANCED TREND WITH ROLLING AVG / ANOMALY / FORECAST
# ==============================================================
def trend_chart(df, title="📈 Trend", x_col="date", y_col="value"):
    if not safe_df(df):
        st.info(f"No data for {title}.")
        return
    df = df.sort_values(x_col)
    df["RollingAvg"] = df[y_col].rolling(3, min_periods=1).mean()
    df["Forecast"] = df["RollingAvg"].shift(1)
    df["Growth%"] = df[y_col].pct_change() * 100
    df["Anomaly"] = (df[y_col] - df["RollingAvg"]).abs() > (df["RollingAvg"] * 0.25)

    base = alt.Chart(df).encode(x=f"{x_col}:T")
    line = base.mark_line(color="steelblue", interpolate='monotone').encode(
        y=alt.Y(f"{y_col}:Q", title="Registrations"),
        tooltip=[x_col, y_col, "RollingAvg", "Forecast", "Growth%", "Anomaly"]
    )
    avg = base.mark_line(color="orange", strokeDash=[6, 3]).encode(y="RollingAvg:Q")
    forecast = base.mark_line(color="green", strokeDash=[3, 3]).encode(y="Forecast:Q")
    points = base.mark_circle(size=80, color="red").transform_filter("datum.Anomaly == true").encode(y=f"{y_col}:Q")

    chart = (line + avg + forecast + points).interactive().properties(title=title)
    st.altair_chart(chart, use_container_width=True)


# ==============================================================
# 🌈 STACKED AREA
# ==============================================================
def stacked_area(df, x_col="date", y_col="value", color_col="category", title="Stacked Area"):
    if not safe_df(df):
        st.info(f"No data for {title}.")
        return
    st.subheader(f"🌈 {title}")
    chart = alt.Chart(df).mark_area(opacity=0.6, interpolate='monotone').encode(
        x=alt.X(f"{x_col}:T"),
        y=alt.Y(f"{y_col}:Q", stack="zero"),
        color=alt.Color(color_col, legend=alt.Legend(title=color_col.capitalize())),
        tooltip=[x_col, y_col, color_col]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)


# ==============================================================
# 📉 MULTI-LINE CHART
# ==============================================================
def multi_line_chart(df, x_col="date", y_cols=None, title="Multi-Line Chart"):
    """Supports multiple numeric series."""
    if not safe_df(df) or not y_cols:
        st.info(f"No data for {title}.")
        return
    st.subheader(f"📊 {title}")
    df_long = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="series", value_name="value")
    chart = alt.Chart(df_long).mark_line(point=True, interpolate='monotone').encode(
        x=alt.X(f"{x_col}:T"),
        y=alt.Y("value:Q"),
        color="series:N",
        tooltip=[x_col, "series", "value"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)


# ==============================================================
# 🧮 KPI METRICS DISPLAY
# ==============================================================
def show_metrics(latest_yoy=None, latest_qoq=None, latest_cumulative=None, additional_metrics=None):
    metric_cols = st.columns(max(3, 1 + (len(additional_metrics) if additional_metrics else 0)))
    metric_cols[0].metric("Latest YoY%", f"{latest_yoy:.1f}%" if latest_yoy is not None else "n/a")
    metric_cols[1].metric("Latest QoQ%", f"{latest_qoq:.1f}%" if latest_qoq is not None else "n/a")
    metric_cols[2].metric("Cumulative Registrations", f"{latest_cumulative:,}" if latest_cumulative is not None else "n/a")
    if additional_metrics:
        for i, (k, v) in enumerate(additional_metrics.items()):
            if i + 3 < len(metric_cols):
                metric_cols[i + 3].metric(k, str(v))


# ==============================================================
# 🧾 DATA TABLE DISPLAY
# ==============================================================
def show_tables(yoy_df=None, qoq_df=None):
    col1, col2 = st.columns(2)
    with col1:
        if yoy_df is not None and not yoy_df.empty:
            st.markdown("YoY% by Month")
            st.dataframe(yoy_df.tail(12), use_container_width=True)
    with col2:
        if qoq_df is not None and not qoq_df.empty:
            tmp = qoq_df.copy()
            if "date" in tmp and isinstance(tmp["date"].iloc[0], pd.Timestamp):
                tmp["Quarter"] = tmp["date"].dt.to_period("Q").astype(str)
            st.markdown("QoQ% by Quarter")
            st.dataframe(tmp[["Quarter", "value", "QoQ%"]] if "Quarter" in tmp else tmp, use_container_width=True)


# ==============================================================
# 💧 WATERFALL / CUMULATIVE / FORECAST CHARTS
# ==============================================================
def waterfall_chart(df, x_col="category", y_col="value", title="Waterfall Chart"):
    if not safe_df(df):
        st.info(f"No data for {title}.")
        return
    st.subheader(f"💧 {title}")
    fig = go.Figure(go.Waterfall(
        name="Registrations",
        x=df[x_col],
        y=df[y_col],
        measure=["relative"] * len(df)
    ))
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)


def cumulative_line_chart(df, x_col="date", y_col="value", title="Cumulative Line Chart"):
    if not safe_df(df):
        st.info(f"No data for {title}.")
        return
    df = df.sort_values(x_col)
    df["cumulative"] = df[y_col].cumsum()
    st.subheader(f"📈 {title}")
    fig = px.line(df, x=x_col, y="cumulative", markers=True, title=title)
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================
# 🧠 ADVANCED ANALYTICS — CORRELATION + OUTLIER + INSIGHT
# ==============================================================
def correlation_heatmap(df, title="Correlation Heatmap"):
    if not safe_df(df):
        st.info(f"No data for {title}.")
        return
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        st.info("No numeric columns for correlation.")
        return
    st.subheader(f"🧠 {title}")
    corr = num_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)


def outlier_detection_chart(df, col, threshold=2.5, title="Outlier Detection"):
    if not safe_df(df) or col not in df.columns:
        st.info(f"No data for {title}.")
        return
    mean, std = df[col].mean(), df[col].std()
    df["Z"] = (df[col] - mean) / std
    df["Outlier"] = abs(df["Z"]) > threshold
    st.subheader(f"🚨 {title}")
    fig = px.scatter(df, x=df.index, y=col, color="Outlier",
                     color_discrete_map={True: "red", False: "blue"},
                     title=title)
    st.plotly_chart(fig, use_container_width=True)


def trend_forecast(df, x_col="date", y_col="value", horizon=3, title="Forecast (Simple Linear)"):
    """Simple regression forecast line using NumPy polyfit."""
    if not safe_df(df):
        st.info(f"No data for {title}.")
        return
    df = df.sort_values(x_col).reset_index(drop=True)
    df = df.dropna(subset=[x_col, y_col])
    x = np.arange(len(df))
    y = df[y_col].values
    coef = np.polyfit(x, y, 1)
    trend = np.poly1d(coef)
    forecast_x = np.arange(len(df) + horizon)
    forecast_y = trend(forecast_x)
    df["Forecast"] = trend(x)
    st.subheader(f"🔮 {title}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines+markers", name="Actual"))
    future_dates = pd.date_range(df[x_col].iloc[-1], periods=horizon + 1, freq="M")[1:]
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_y[-horizon:], mode="lines+markers", name="Forecast", line=dict(dash="dot")))
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================
# 🧾 DATA EXPORT
# ==============================================================
def show_data(df, title="Data Table"):
    if not safe_df(df):
        return
    st.markdown(f"### 🧾 {title}")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{title.replace(' ', '_')}.csv"
    )

