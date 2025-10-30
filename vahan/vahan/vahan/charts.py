# # vahan/charts.py
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import altair as alt

# # ---------------- BASIC CHARTS ----------------
# def bar_from_df(df, title="", index_col="label", value_col="value"):
#     """Maxed Bar chart"""
#     if df is None or df.empty:
#         st.info(f"No data for {title or 'bar chart'}.")
#         return
#     st.subheader(title)
#     fig = px.bar(df, x=index_col, y=value_col, text=value_col)
#     fig.update_traces(texttemplate='%{text}', textposition='outside')
#     fig.update_layout(yaxis_title=value_col.capitalize(), xaxis_title=index_col.capitalize())
#     st.plotly_chart(fig, use_container_width=True)

# def pie_from_df(df, title="", names="label", values="value", donut=True):
#     """Maxed Pie/Donut chart with percentages"""
#     if df is None or df.empty:
#         st.info(f"No data for {title or 'pie chart'}.")
#         return
#     st.subheader(title)
#     fig = px.pie(df, names=names, values=values, hole=0.4 if donut else 0)
#     fig.update_traces(textinfo='percent+label', pull=[0.05]*len(df))
#     st.plotly_chart(fig, use_container_width=True)

def line_from_trend(df, title="Trend Line", x_col="date", y_col="value"):
    """Maxed Line chart with Altair"""
    if df is None or df.empty:
        st.info(f"No data for {title}.")
        return
    d = df.dropna(subset=[x_col, y_col]).sort_values(x_col)
    if d.empty:
        st.info("No valid data to plot.")
        return
    st.subheader(title)
    chart = alt.Chart(d).mark_line(point=True, interpolate='monotone').encode(
        x=alt.X(f"{x_col}:T", title=x_col.capitalize()),
        y=alt.Y(f"{y_col}:Q", title=y_col.capitalize()),
        tooltip=[x_col, y_col]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# def area_from_df(df, title="Area Chart", x_col="date", y_col="value"):
#     """Maxed Area chart"""
#     if df is None or df.empty:
#         st.info(f"No data for {title}.")
#         return
#     d = df.dropna(subset=[x_col, y_col]).sort_values(x_col)
#     st.subheader(title)
#     chart = alt.Chart(d).mark_area(opacity=0.4, interpolate='monotone').encode(
#         x=alt.X(f"{x_col}:T", title=x_col.capitalize()),
#         y=alt.Y(f"{y_col}:Q", title=y_col.capitalize()),
#         tooltip=[x_col, y_col]
#     ).interactive()
#     st.altair_chart(chart, use_container_width=True)

# # ---------------- STACKED CHARTS ----------------
# def stacked_bar(df, x_col="state", y_col="value", color_col="maker", title="Stacked Bar Chart"):
#     """Maxed Stacked Bar chart with percentages"""
#     if df is None or df.empty:
#         st.info(f"No data for {title}.")
#         return
#     st.subheader(title)
#     fig = px.bar(df, x=x_col, y=y_col, color=color_col, text=y_col)
#     fig.update_traces(texttemplate='%{text}', textposition='inside')
#     fig.update_layout(barmode='stack', xaxis_title=x_col.capitalize(), yaxis_title=y_col.capitalize())
#     st.plotly_chart(fig, use_container_width=True)

# def stacked_area_chart(df, x_col="date", y_col="value", color_col="category", title="Stacked Area Chart"):
#     """Maxed stacked area chart"""
#     if df is None or df.empty:
#         st.info(f"No data for {title}")
#         return
#     st.subheader(title)
#     chart = alt.Chart(df).mark_area(opacity=0.5, interpolate='monotone').encode(
#         x=alt.X(f"{x_col}:T"),
#         y=alt.Y(f"{y_col}:Q", stack="zero"),
#         color=color_col,
#         tooltip=[x_col, y_col, color_col]
#     ).interactive()
#     st.altair_chart(chart, use_container_width=True)

# def multi_line_chart(df, x_col="date", y_cols=None, title="Multi-Line Chart"):
#     """Maxed Multi-Line chart for multiple series"""
#     if df is None or df.empty or y_cols is None:
#         st.info(f"No data for {title}")
#         return
#     st.subheader(title)
#     df_long = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="series", value_name="value")
#     chart = alt.Chart(df_long).mark_line(point=True, interpolate='monotone').encode(
#         x=alt.X(f"{x_col}:T"),
#         y=alt.Y("value:Q"),
#         color="series:N",
#         tooltip=[x_col, "series", "value"]
#     ).interactive()
#     st.altair_chart(chart, use_container_width=True)

# # ---------------- KPIs ----------------
# def show_metrics(latest_yoy=None, latest_qoq=None, latest_cumulative=None, additional_metrics=None):
#     """Maxed KPI display for unlimited metrics"""
#     metric_cols = st.columns(max(3, 1+ (len(additional_metrics) if additional_metrics else 0)))
#     metric_cols[0].metric("Latest YoY%", f"{latest_yoy:.1f}%" if latest_yoy is not None else "n/a")
#     metric_cols[1].metric("Latest QoQ%", f"{latest_qoq:.1f}%" if latest_qoq is not None else "n/a")
#     metric_cols[2].metric("Cumulative Registrations", f"{latest_cumulative:,}" if latest_cumulative is not None else "n/a")
#     if additional_metrics:
#         for i, (k, v) in enumerate(additional_metrics.items()):
#             if i+3 < len(metric_cols):
#                 metric_cols[i+3].metric(k, str(v))

# # ---------------- TABLES ----------------
# def show_tables(yoy_df=None, qoq_df=None):
#     """Maxed DataTables for YoY and QoQ with dynamic formatting"""
#     col1, col2 = st.columns(2)
#     with col1:
#         if yoy_df is not None and not yoy_df.empty:
#             st.markdown("YoY% by month")
#             st.dataframe(yoy_df.tail(12), use_container_width=True)
#     with col2:
#         if qoq_df is not None and not qoq_df.empty:
#             tmp = qoq_df.copy()
#             if "date" in tmp and isinstance(tmp["date"].iloc[0], pd.Timestamp):
#                 tmp["Quarter"] = tmp["date"].dt.to_period("Q").astype(str)
#             st.markdown("QoQ% by quarter")
#             st.dataframe(tmp[["Quarter","value","QoQ%"]] if "Quarter" in tmp else tmp, use_container_width=True)

# # ---------------- ADVANCED MAXED CHARTS ----------------
# def waterfall_chart(df, x_col="category", y_col="value", title="Waterfall Chart"):
#     """Maxed Waterfall chart"""
#     if df is None or df.empty:
#         st.info(f"No data for {title}")
#         return
#     st.subheader(title)
#     fig = go.Figure(go.Waterfall(
#         name = "Registrations",
#         x = df[x_col],
#         y = df[y_col],
#         measure = ["relative"]*len(df)
#     ))
#     fig.update_layout(title=title)
#     st.plotly_chart(fig, use_container_width=True)

# def cumulative_line_chart(df, x_col="date", y_col="value", title="Cumulative Line Chart"):
#     """Maxed cumulative chart"""
#     if df is None or df.empty:
#         st.info(f"No data for {title}")
#         return
#     df = df.sort_values(x_col)
#     df["cumulative"] = df[y_col].cumsum()
#     st.subheader(title)
#     fig = px.line(df, x=x_col, y="cumulative", markers=True, title=title)
#     st.plotly_chart(fig, use_container_width=True)


# ===============================================
# vahan/charts.py — 🔥 MAXED Analytics Visualization Suite
# ===============================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from datetime import datetime
import numpy as np

# =============== GLOBAL STYLING ===============
alt.themes.enable("dark")
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Turbo"
px.defaults.width = 1100
px.defaults.height = 500

# ===============================================
# 🔹 GENERIC SAFE HANDLER
# ===============================================
def safe_df(df):
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty

# ===============================================
# 🔸 UNIVERSAL MAXED BAR CHART
# ===============================================
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

# ===============================================
# 🔸 PIE / DONUT CHART
# ===============================================
def pie_from_df(df, title="", names="label", values="value", donut=True):
    if not safe_df(df):
        st.info(f"No data for {title or 'pie chart'}.")
        return
    st.subheader(f"🥧 {title}")
    fig = px.pie(
        df, names=names, values=values,
        hole=0.4 if donut else 0,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_traces(textinfo='percent+label', pull=[0.05]*len(df))
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)

# ===============================================
# 🔸 LINE / TREND CHART — with Growth, Forecast & Anomaly
# ===============================================
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

# ===============================================
# 🔸 STACKED AREA — Category/State/Segment trends
# ===============================================
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

# ===============================================
# 🔸 MULTI-LINE — YoY / QoQ / Yearly Comparison
# ===============================================
def multi_line(df, x_col="date", y_cols=None, title="Multi-Year Comparison"):
    if not safe_df(df) or not y_cols:
        st.info(f"No data for {title}")
        return
    st.subheader(f"📊 {title}")
    df_long = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="Metric", value_name="Value")
    chart = alt.Chart(df_long).mark_line(point=True, interpolate='monotone').encode(
        x=alt.X(f"{x_col}:T"),
        y=alt.Y("Value:Q"),
        color=alt.Color("Metric:N", legend=alt.Legend(title="Series")),
        tooltip=[x_col, "Metric", "Value"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# ===============================================
# 🔸 KPI Summary Block
# ===============================================
def kpi_summary(df, period="Yearly"):
    if not safe_df(df): return
    latest_val = df.iloc[-1]["value"]
    prev_val = df.iloc[-2]["value"] if len(df) > 1 else np.nan
    change = ((latest_val - prev_val) / prev_val * 100) if prev_val else 0
    st.markdown(f"### 📅 {period} Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Value", f"{latest_val:,.0f}")
    c2.metric("Previous Value", f"{prev_val:,.0f}" if not np.isnan(prev_val) else "n/a")
    c3.metric("Change (%)", f"{change:+.2f}%")

# ===============================================
# 🔸 FORECAST CHART — Predict Next Year
# ===============================================
def forecast_chart(df, title="Forecast", x_col="date", y_col="value", periods=12):
    if not safe_df(df): return
    st.subheader(f"🔮 {title}")
    df = df.sort_values(x_col)
    df["Forecast"] = df[y_col].rolling(3, min_periods=1).mean()
    last_date = df[x_col].max()
    freq = pd.infer_freq(df[x_col]) or "M"
    future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
    future_df = pd.DataFrame({x_col: future_dates})
    future_df["Forecast"] = [df["Forecast"].iloc[-1] * (1 + 0.02 * i) for i in range(periods)]
    df["Type"] = "Actual"
    future_df["Type"] = "Predicted"
    full_df = pd.concat([df[[x_col, y_col, "Type"]].rename(columns={y_col:"Value"}), 
                         future_df.rename(columns={"Forecast":"Value"})])
    chart = alt.Chart(full_df).mark_line(point=True).encode(
        x=alt.X(f"{x_col}:T"),
        y="Value:Q",
        color="Type:N",
        tooltip=[x_col, "Value", "Type"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# ===============================================
# 🔸 COMBINED YEAR COMPARISON
# ===============================================
def compare_years(df, title="Yearly Comparison", date_col="date", value_col="value"):
    if not safe_df(df): return
    df["year"] = pd.to_datetime(df[date_col]).dt.year
    df["month"] = pd.to_datetime(df[date_col]).dt.month
    pivot = df.pivot_table(index="month", columns="year", values=value_col, aggfunc="sum")
    pivot = pivot.reset_index().rename_axis(None, axis=1)
    multi_line(pivot, "month", pivot.columns[1:], title=f"{title} (Prev vs Current vs Next Year)")

# ===============================================
# 🔸 EXPORTABLE DATA TABLE
# ===============================================
def show_data(df, title="Data Table"):
    if not safe_df(df): return
    st.markdown(f"### 🧾 {title}")
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"{title.replace(' ','_')}.csv")

