# vahan/metrics.py
import pandas as pd
import numpy as np

# ---------------- BASIC METRICS ----------------
def compute_yoy(df, date_col="date", value_col="value"):
    """Year-over-Year percentage growth"""
    if df.empty:
        return df.assign(**{"YoY%": None})
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if d.empty:
        return df.assign(**{"YoY%": None})
    m = d[value_col].resample("MS").sum()
    yoy = (m - m.shift(12)) / m.shift(12) * 100
    return pd.DataFrame({"date": m.index, "value": m.values, "YoY%": yoy.values})

def compute_qoq(df, date_col="date", value_col="value"):
    """Quarter-over-Quarter percentage growth"""
    if df.empty:
        return df.assign(**{"QoQ%": None})
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if d.empty:
        return df.assign(**{"QoQ%": None})
    q = d[value_col].resample("QS").sum()
    qoq = (q - q.shift(1)) / q.shift(1) * 100
    return pd.DataFrame({"date": q.index, "value": q.values, "QoQ%": qoq.values})

def compute_mom(df, date_col="date", value_col="value"):
    """Month-over-Month percentage growth"""
    if df.empty:
        return df.assign(**{"MoM%": None})
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if d.empty:
        return df.assign(**{"MoM%": None})
    m = d[value_col].resample("MS").sum()
    mom = (m - m.shift(1)) / m.shift(1) * 100
    return pd.DataFrame({"date": m.index, "value": m.values, "MoM%": mom.values})

# ---------------- CUMULATIVE & ROLLING ----------------
def compute_cumulative(df, value_col="value"):
    """Cumulative sum"""
    if df.empty:
        return df.assign(**{"Cumulative": None})
    d = df.copy()
    d["Cumulative"] = d[value_col].cumsum()
    return d

def compute_rolling(df, value_col="value", window=3):
    """Rolling average"""
    if df.empty:
        return df.assign(**{f"Rolling{window}": None})
    d = df.copy()
    d[f"Rolling{window}"] = d[value_col].rolling(window=window).mean()
    return d

# ---------------- SUMMARY METRICS ----------------
def top_bottom(df, value_col="value", top_n=5):
    """Top and bottom N entries"""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    top = df.nlargest(top_n, value_col)
    bottom = df.nsmallest(top_n, value_col)
    return top, bottom

def percentage_contribution(df, value_col="value", group_col="category"):
    """Percentage contribution per group"""
    if df.empty or group_col not in df.columns:
        return df.assign(**{"%Contribution": None})
    total = df[value_col].sum()
    df["%Contribution"] = df[value_col] / total * 100
    return df

# ---------------- QUARTER LABEL ----------------
def quarter_label(ts: pd.Timestamp):
    """Returns Q1-2025, Q2-2025 style labels"""
    q = (ts.month - 1) // 3 + 1
    return f"Q{q}-{ts.year}"

def add_quarter_column(df, date_col="date"):
    """Adds quarter column for reporting"""
    if df.empty:
        df["Quarter"] = None
        return df
    d = df.copy()
    d["Quarter"] = d[date_col].apply(lambda x: quarter_label(pd.to_datetime(x)))
    return d

# ---------------- MAXED AGGREGATION ----------------
def compute_all_metrics(df, date_col="date", value_col="value"):
    """Maxed: returns DataFrame with YoY, QoQ, MoM, cumulative, rolling, quarter"""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.set_index(date_col)
    df_res = pd.DataFrame(index=df.index)
    df_res["value"] = df[value_col]

    # Compute metrics
    df_res["YoY%"] = compute_yoy(df, date_col, value_col)["YoY%"].values
    df_res["QoQ%"] = compute_qoq(df, date_col, value_col)["QoQ%"].values
    df_res["MoM%"] = compute_mom(df, date_col, value_col)["MoM%"].values
    df_res["Cumulative"] = df_res["value"].cumsum()
    df_res["Rolling3"] = df_res["value"].rolling(3).mean()
    df_res["Quarter"] = df_res.index.to_series().apply(quarter_label)
    df_res.reset_index(inplace=True)
    df_res.rename(columns={"index": date_col}, inplace=True)
    return df_res
