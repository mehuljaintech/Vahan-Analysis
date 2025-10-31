# vahan/charts.py
"""
ALL-MAXED Charts Utilities for VAHAN dashboards
- Streamlit friendly
- Plotly + Altair charts
- Multi-year, daily/monthly/quarterly/yearly aggregations
- Comparison helpers (YoY, MoM, QoQ), top-N, normalization, percent shares
- Table, metrics, export helpers
"""

from typing import List, Optional, Dict, Union, Iterable
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from io import StringIO

# ---------------- Helpers / Validation ----------------
def _empty_info(title: str):
    st.info(f"No data for {title or 'chart'}.")

def _ensure_df(df):
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        try:
            return pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()
    return df.copy()

def _to_datetime(df: pd.DataFrame, col: str = "date"):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def csv_download_link(df: pd.DataFrame, name="data.csv"):
    buf = StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download CSV", data=buf, file_name=name, mime="text/csv")

# ---------------- Aggregators ----------------
def aggregate_time(df: pd.DataFrame, date_col="date", value_col="value", freq="M", agg="sum"):
    """
    freq: 'D' daily, 'M' monthly, 'Q' quarterly, 'Y' yearly
    agg: 'sum'|'mean'|'median'|'max' etc.
    returns DataFrame with columns ['date','value']
    """
    df = _ensure_df(df)
    df = _to_datetime(df, date_col)
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[date_col, value_col])
    df = df.dropna(subset=[date_col])
    series = getattr(df.set_index(date_col)[value_col].resample(freq), agg)()
    out = series.reset_index().rename(columns={value_col: "value"})
    return out

def pivot_topn(df: pd.DataFrame, group_col: str, value_col: str, top_n:int=10, others_label="Other"):
    """
    Returns DataFrame aggregated by group_col, keeps top_n and merges rest into 'Other'
    """
    df = _ensure_df(df)
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[group_col, value_col])
    agg = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    top = agg.head(top_n)
    rest = agg.iloc[top_n:].sum()
    res = top.reset_index()
    if rest > 0:
        res = pd.concat([res, pd.DataFrame({group_col:[others_label], value_col:[rest]})], ignore_index=True)
    return res

# ---------------- BASIC CHARTS ----------------
def bar_from_df(df: pd.DataFrame, title:str="", index_col:str="label", value_col:str="value", top_n:Optional[int]=None, orientation="v"):
    df = _ensure_df(df)
    if df.empty:
        _empty_info(title)
        return
    if top_n:
        if index_col in df.columns and value_col in df.columns:
            df = pivot_topn(df, index_col, value_col, top_n=top_n)
    fig = px.bar(df, x=index_col if orientation=="v" else value_col, y=value_col if orientation=="v" else index_col,
                 text=value_col, title=title)
    fig.update_traces(texttemplate='%{text}', textposition='outside' if orientation=="v" else 'auto')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_title=index_col.capitalize(), yaxis_title=value_col.capitalize())
    st.plotly_chart(fig, use_container_width=True)

def pie_from_df(df: pd.DataFrame, title:str="", names:str="label", values:str="value", donut:bool=True, top_n:Optional[int]=10):
    df = _ensure_df(df)
    if df.empty:
        _empty_info(title)
        return
    if top_n:
        df = pivot_topn(df, names, values, top_n=top_n)
    fig = px.pie(df, names=names, values=values, hole=0.4 if donut else 0, title=title)
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def line_from_trend(df: pd.DataFrame, title:str="Trend Line", date_col:str="date", value_col:str="value", use_altair:bool=True):
    df = _ensure_df(df)
    df = _to_datetime(df, date_col)
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        _empty_info(title)
        return
    d = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if d.empty:
        _empty_info(title)
        return
    st.subheader(title)
    if use_altair:
        chart = alt.Chart(d).mark_line(point=True, interpolate='monotone').encode(
            x=alt.X(f"{date_col}:T", title=date_col.capitalize()),
            y=alt.Y(f"{value_col}:Q", title=value_col.capitalize()),
            tooltip=[date_col, value_col]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        fig = px.line(d, x=date_col, y=value_col, markers=True, title=title)
        st.plotly_chart(fig, use_container_width=True)

def area_from_df(df: pd.DataFrame, title="Area Chart", date_col="date", value_col="value"):
    df = _ensure_df(df)
    df = _to_datetime(df, date_col)
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        _empty_info(title)
        return
    d = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    chart = alt.Chart(d).mark_area(opacity=0.4, interpolate='monotone').encode(
        x=alt.X(f"{date_col}:T", title=date_col.capitalize()),
        y=alt.Y(f"{value_col}:Q", title=value_col.capitalize()),
        tooltip=[date_col, value_col]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# ---------------- STACKED / MULTI-SERIES ----------------
def stacked_bar(df: pd.DataFrame, x_col:str="category", y_col:str="value", color_col:str="segment", title:str="Stacked Bar Chart", normalize:bool=False):
    df = _ensure_df(df)
    if df.empty or any(c not in df.columns for c in (x_col, y_col, color_col)):
        _empty_info(title)
        return
    agg = df.groupby([x_col, color_col])[y_col].sum().reset_index()
    if normalize:
        total = agg.groupby(x_col)[y_col].transform('sum')
        agg['pct'] = agg[y_col] / total
        y_val = 'pct'
    else:
        y_val = y_col
    fig = px.bar(agg, x=x_col, y=y_val, color=color_col, title=title, text=y_val)
    fig.update_layout(barmode='stack', xaxis_title=x_col.capitalize(), yaxis_title=('Share' if normalize else y_col.capitalize()))
    st.plotly_chart(fig, use_container_width=True)

def multi_line_chart(df: pd.DataFrame, date_col="date", y_cols:Iterable[str]=None, title="Multi-Line Chart"):
    df = _ensure_df(df)
    df = _to_datetime(df, date_col)
    if df.empty or y_cols is None:
        _empty_info(title)
        return
    present = [c for c in y_cols if c in df.columns]
    if not present:
        _empty_info(title)
        return
    df_long = df.melt(id_vars=[date_col], value_vars=present, var_name="series", value_name="value").dropna(subset=["value"])
    chart = alt.Chart(df_long).mark_line(point=True, interpolate='monotone').encode(
        x=alt.X(f"{date_col}:T"),
        y=alt.Y("value:Q"),
        color="series:N",
        tooltip=[date_col, "series", "value"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# ---------------- KPIs & TABLES ----------------
def show_metrics(latest_yoy:Optional[float]=None, latest_qoq:Optional[float]=None, latest_cumulative:Optional[Union[int,float]]=None, additional_metrics:Optional[Dict[str,Union[str,int,float]]]=None):
    # dynamic number of columns
    metrics = [("Latest YoY%", f"{latest_yoy:.1f}%" if latest_yoy is not None else "n/a"),
               ("Latest QoQ%", f"{latest_qoq:.1f}%" if latest_qoq is not None else "n/a"),
               ("Cumulative", f"{int(latest_cumulative):,}" if latest_cumulative is not None else "n/a")]
    if additional_metrics:
        metrics.extend([(k, str(v)) for k, v in additional_metrics.items()])
    cols = st.columns(len(metrics))
    for c, (label, val) in zip(cols, metrics):
        c.metric(label, val)

def show_tables(yoy_df:Optional[pd.DataFrame]=None, qoq_df:Optional[pd.DataFrame]=None, allow_download:bool=True):
    col1, col2 = st.columns(2)
    with col1:
        if yoy_df is not None and not yoy_df.empty:
            st.markdown("YoY% — recent")
            st.dataframe(yoy_df.tail(12), use_container_width=True)
            if allow_download:
                csv_download_link(yoy_df.tail(12), name="yoy_recent.csv")
    with col2:
        if qoq_df is not None and not qoq_df.empty:
            tmp = qoq_df.copy()
            if "date" in tmp.columns:
                tmp = _to_datetime(tmp, "date")
                if isinstance(tmp["date"].iloc[0], pd.Timestamp):
                    tmp["Quarter"] = tmp["date"].dt.to_period("Q").astype(str)
            st.markdown("QoQ% — recent")
            st.dataframe(tmp.tail(12), use_container_width=True)
            if allow_download:
                csv_download_link(tmp.tail(12), name="qoq_recent.csv")

# ---------------- ADVANCED CHARTS ----------------
def waterfall_chart(df: pd.DataFrame, x_col:str="category", y_col:str="value", title:str="Waterfall"):
    df = _ensure_df(df)
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        _empty_info(title)
        return
    fig = go.Figure(go.Waterfall(
        x=df[x_col].astype(str),
        y=df[y_col],
        measure=["relative"] * len(df),
        text=df[y_col],
        connector={"line":{"color":"rgb(63, 63, 63)"}}
    ))
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True)

def cumulative_line_chart(df: pd.DataFrame, date_col="date", y_col="value", title="Cumulative"):
    df = _ensure_df(df)
    df = _to_datetime(df, date_col)
    if df.empty or date_col not in df.columns or y_col not in df.columns:
        _empty_info(title)
        return
    d = df.dropna(subset=[date_col, y_col]).sort_values(date_col)
    d["cumulative"] = d[y_col].cumsum()
    fig = px.line(d, x=date_col, y="cumulative", markers=True, title=title)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- COMPARISONS: YoY / MoM / QoQ / Multi-year ----------------
def compute_period_change(df: pd.DataFrame, date_col="date", value_col="value", period="Y"):
    """
    period: 'Y' year-over-year, 'Q' quarter-over-quarter, 'M' month-over-month
    Returns a DataFrame with date, value, pct_change
    """
    df = _ensure_df(df)
    df = _to_datetime(df, date_col)
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[date_col, value_col, "pct_change"])
    d = df.dropna(subset=[date_col, value_col]).set_index(date_col).sort_index()
    if period == "Y":
        shifted = d[value_col].shift(365)  # approximate; prefer resample compare externally
    elif period == "Q":
        shifted = d[value_col].shift(90)
    elif period == "M":
        shifted = d[value_col].shift(30)
    else:
        shifted = d[value_col].shift(1)
    out = d[[value_col]].copy()
    out["pct_change"] = (d[value_col] - shifted) / shifted.replace({0: np.nan}) * 100
    out = out.reset_index()
    return out

def prepare_multiyear_compare(df: pd.DataFrame, date_col="date", value_col="value", pivot_freq="M"):
    """
    Returns a pivoted DataFrame where columns are years and rows are month/period within year,
    making it easy to draw multi-year comparisons.
    pivot_freq: frequency used to normalize series for comparison (D/M/Q/Y)
    """
    df = _ensure_df(df)
    df = _to_datetime(df, date_col)
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=[date_col, value_col])
    df['Year'] = df[date_col].dt.year
    if pivot_freq == "M":
        df['Period'] = df[date_col].dt.month
    elif pivot_freq == "Q":
        df['Period'] = df[date_col].dt.quarter
    elif pivot_freq == "D":
        df['Period'] = df[date_col].dt.dayofyear
    else:
        df['Period'] = df[date_col].dt.month
    pivot = df.groupby(['Year','Period'])[value_col].sum().reset_index().pivot(index='Period', columns='Year', values=value_col).fillna(0)
    return pivot

def plot_multiyear_compare(pivot_df: pd.DataFrame, title="Multi-year Comparison", normalize:bool=False):
    if pivot_df is None or pivot_df.empty:
        _empty_info(title)
        return
    df = pivot_df.copy()
    if normalize:
        df = df.div(df.sum(axis=0), axis=1) * 100
    df = df.reset_index().melt(id_vars="Period", var_name="Year", value_name="value")
    chart = alt.Chart(df).mark_line(point=True).encode(
        x="Period:O",
        y="value:Q",
        color="Year:N",
        tooltip=["Year","Period","value"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# ---------------- UTILS ----------------
def quick_summary(df: pd.DataFrame, date_col="date", value_col="value"):
    df = _ensure_df(df)
    df = _to_datetime(df, date_col)
    if df.empty:
        return {}
    total = df[value_col].sum() if value_col in df.columns else None
    latest = df.sort_values(date_col).iloc[-1] if date_col in df.columns else None
    return {
        "total": int(total) if total is not None and not np.isnan(total) else None,
        "latest_date": str(latest[date_col]) if isinstance(latest, pd.Series) and date_col in latest else None,
        "latest_value": float(latest[value_col]) if isinstance(latest, pd.Series) and value_col in latest else None
    }

# ---------------- Example: compound utility to show full dashboard slice ----------------
def render_timeseries_slice(df: pd.DataFrame, title_prefix="Registrations", date_col="date", value_col="value", compare_years:List[int]=None, top_n:int=10):
    """
    High-level helper that renders:
    - main time-series line
    - cumulative line
    - multi-year comparison (if possible)
    - top-N pie and bar if 'category' present
    """
    df = _ensure_df(df)
    if df.empty:
        _empty_info(title_prefix)
        return

    # summary metrics
    summary = quick_summary(df, date_col=date_col, value_col=value_col)
    show_metrics(
        latest_yoy=None,  # left for caller to compute precisely with fiscal/calendar awareness
        latest_qoq=None,
        latest_cumulative=summary.get("total"),
        additional_metrics={"Latest value": summary.get("latest_value")}
    )

    # time-series
    line_from_trend(df, title=f"{title_prefix} — Time Series", date_col=date_col, value_col=value_col)

    # cumulative
    cumulative_line_chart(df, date_col=date_col, y_col=value_col, title=f"{title_prefix} — Cumulative")

    # multi-year compare (auto-detect)
    pivot = prepare_multiyear_compare(df, date_col=date_col, value_col=value_col, pivot_freq="M")
    if not pivot.empty:
        plot_multiyear_compare(pivot, title=f"{title_prefix} — Multi-year (by month)")

    # category/topN if category exists
    for cat in ("maker", "manufacturer", "state", "category", "segment"):
        if cat in df.columns:
            pie_from_df(df.groupby(cat)[value_col].sum().reset_index().rename(columns={cat:"label", value_col:"value"}), title=f"Share by {cat.capitalize()}", names="label", values="value", top_n=top_n)
            bar_from_df(df.groupby(cat)[value_col].sum().reset_index().rename(columns={cat:"label", value_col:"value"}).sort_values("value", ascending=False), title=f"Top {top_n} {cat.capitalize()}", index_col="label", value_col="value", top_n=top_n)
            break

# ---------------- End of file ----------------
