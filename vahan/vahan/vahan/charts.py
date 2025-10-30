# ============================================================
# 🚀 VAHAN / CHARTS — MAXED ANALYTICS VISUAL ENGINE
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from datetime import datetime

# ============================================================
# 🧠 UNIVERSAL UTILITIES
# ============================================================
def safe_df(df):
    """Ensure dataframe is valid."""
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return pd.DataFrame()

def chart_header(title, subtitle=None):
    """Styled chart header with optional subtitle."""
    st.markdown(f"""
    <div style='text-align:center;margin:15px 0;'>
        <h3 style='margin-bottom:3px;'>{title}</h3>
        <p style='opacity:0.6;font-size:13px;margin-top:-5px;'>{subtitle or ''}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 📊 BASIC CHARTS
# ============================================================
def bar_from_df(df, title="", index_col="label", value_col="value"):
    df = safe_df(df)
    if df.empty:
        st.info(f"No data available for {title or 'bar chart'}.")
        return
    chart_header(title, f"{len(df)} categories")
    fig = px.bar(df, x=index_col, y=value_col, text=value_col, title=None)
    fig.update_traces(texttemplate='%{text}', textposition='outside', marker_color='#0F9D58')
    fig.update_layout(
        yaxis_title=value_col.capitalize(),
        xaxis_title=index_col.capitalize(),
        margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

def pie_from_df(df, title="", names="label", values="value", donut=True):
    df = safe_df(df)
    if df.empty:
        st.info(f"No data available for {title or 'pie chart'}.")
        return
    chart_header(title, "Percentage contribution")
    fig = px.pie(df, names=names, values=values, hole=0.4 if donut else 0)
    fig.update_traces(textinfo='percent+label', pull=[0.05]*len(df))
    st.plotly_chart(fig, use_container_width=True)

def line_from_trend(df, title="Trend Line", x_col="date", y_col="value"):
    df = safe_df(df)
    if df.empty or x_col not in df or y_col not in df:
        st.info(f"No data available for {title}.")
        return
    d = df.dropna(subset=[x_col, y_col]).sort_values(x_col)
    chart_header(title, "Trend over time")
    chart = alt.Chart(d).mark_line(point=True, interpolate='monotone').encode(
        x=alt.X(f"{x_col}:T", title=x_col.capitalize()),
        y=alt.Y(f"{y_col}:Q", title=y_col.capitalize()),
        tooltip=[x_col, y_col]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def area_from_df(df, title="Area Chart", x_col="date", y_col="value"):
    df = safe_df(df)
    if df.empty or x_col not in df or y_col not in df:
        st.info(f"No data available for {title}.")
        return
    d = df.dropna(subset=[x_col, y_col]).sort_values(x_col)
    chart_header(title)
    chart = alt.Chart(d).mark_area(opacity=0.4, interpolate='monotone', color='#4285F4').encode(
        x=alt.X(f"{x_col}:T", title=x_col.capitalize()),
        y=alt.Y(f"{y_col}:Q", title=y_col.capitalize()),
        tooltip=[x_col, y_col]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# ============================================================
# 🧩 STACKED / MULTI SERIES
# ============================================================
def stacked_bar(df, x_col="state", y_col="value", color_col="maker", title="Stacked Bar Chart"):
    df = safe_df(df)
    if df.empty:
        st.info(f"No data available for {title}.")
        return
    chart_header(title, "Distribution across categories")
    fig = px.bar(df, x=x_col, y=y_col, color=color_col, text=y_col)
    fig.update_traces(texttemplate='%{text}', textposition='inside')
    fig.update_layout(barmode='stack', xaxis_title=x_col.capitalize(), yaxis_title=y_col.capitalize())
    st.plotly_chart(fig, use_container_width=True)

def stacked_area_chart(df, x_col="date", y_col="value", color_col="category", title="Stacked Area Chart"):
    df = safe_df(df)
    if df.empty:
        st.info(f"No data available for {title}.")
        return
    chart_header(title, "Category composition over time")
    chart = alt.Chart(df).mark_area(opacity=0.5, interpolate='monotone').encode(
        x=alt.X(f"{x_col}:T"),
        y=alt.Y(f"{y_col}:Q", stack="zero"),
        color=color_col,
        tooltip=[x_col, y_col, color_col]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

def multi_line_chart(df, x_col="date", y_cols=None, title="Multi-Line Chart"):
    df = safe_df(df)
    if df.empty or not y_cols:
        st.info(f"No data available for {title}.")
        return
    chart_header(title, "Comparative time series")
    df_long = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="Series", value_name="Value")
    chart = alt.Chart(df_long).mark_line(point=True, interpolate='monotone').encode(
        x=alt.X(f"{x_col}:T"),
        y=alt.Y("Value:Q"),
        color="Series:N",
        tooltip=[x_col, "Series", "Value"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# ============================================================
# 📈 ADVANCED VISUALS
# ============================================================
def waterfall_chart(df, x_col="category", y_col="value", title="Waterfall Chart"):
    df = safe_df(df)
    if df.empty:
        st.info(f"No data available for {title}.")
        return
    chart_header(title)
    fig = go.Figure(go.Waterfall(
        name="Change",
        orientation="v",
        x=df[x_col],
        y=df[y_col],
        measure=["relative"] * len(df),
        textposition="outside",
        connector={"line": {"color": "rgba(63,63,63,0.5)"}}
    ))
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def cumulative_line_chart(df, x_col="date", y_col="value", title="Cumulative Chart"):
    df = safe_df(df)
    if df.empty:
        st.info(f"No data available for {title}.")
        return
    df = df.sort_values(x_col)
    df["Cumulative"] = df[y_col].cumsum()
    chart_header(title, "Cumulative growth trend")
    fig = px.line(df, x=x_col, y="Cumulative", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 💡 KPI + TABLES
# ============================================================
def show_metrics(latest_yoy=None, latest_qoq=None, latest_cumulative=None, additional_metrics=None):
    base_cols = 3 + (len(additional_metrics) if additional_metrics else 0)
    cols = st.columns(base_cols)
    cols[0].metric("YoY%", f"{latest_yoy:.1f}%" if latest_yoy is not None else "n/a")
    cols[1].metric("QoQ%", f"{latest_qoq:.1f}%" if latest_qoq is not None else "n/a")
    cols[2].metric("Total Registrations", f"{latest_cumulative:,}" if latest_cumulative else "n/a")
    if additional_metrics:
        for i, (k, v) in enumerate(additional_metrics.items(), start=3):
            cols[i].metric(k, str(v))

def show_tables(yoy_df=None, qoq_df=None):
    col1, col2 = st.columns(2)
    if yoy_df is not None and not yoy_df.empty:
        with col1:
            st.markdown("### 📆 Year-over-Year (%)")
            st.dataframe(yoy_df.tail(12), use_container_width=True)
    if qoq_df is not None and not qoq_df.empty:
        with col2:
            st.markdown("### 📊 Quarter-over-Quarter (%)")
            tmp = qoq_df.copy()
            if "date" in tmp and pd.api.types.is_datetime64_any_dtype(tmp["date"]):
                tmp["Quarter"] = tmp["date"].dt.to_period("Q").astype(str)
            st.dataframe(tmp, use_container_width=True)

# ============================================================
# 🔍 MAXED SUMMARY EXPORT / ANALYSIS
# ============================================================
def trend_comparison_chart(df, x_col="date", cols=None, title="Trend Comparison"):
    df = safe_df(df)
    if df.empty or not cols:
        st.info("No data available for trend comparison.")
        return
    chart_header(title)
    melted = df.melt(id_vars=x_col, value_vars=cols, var_name="Metric", value_name="Value")
    chart = alt.Chart(melted).mark_line(point=True).encode(
        x=f"{x_col}:T",
        y="Value:Q",
        color="Metric:N",
        tooltip=[x_col, "Metric", "Value"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
