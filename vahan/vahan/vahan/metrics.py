# ================================================================
# 📊 vahan/metrics.py — ALL-MAXED UNIVERSAL METRIC ENGINE
# Supports: Daily, Weekly, Monthly, Quarterly, Yearly
# Includes: YoY, QoQ, MoM, WoW, DoD, Cumulative, Rolling, Normalized, etc.
# ================================================================
import pandas as pd
import numpy as np

# ================================================================
# 🔧 BASIC HELPERS
# ================================================================
def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce")

def quarter_label(ts: pd.Timestamp):
    q = (ts.month - 1) // 3 + 1
    return f"Q{q}-{ts.year}"

def add_time_columns(df, date_col="date"):
    df = df.copy()
    df[date_col] = safe_to_datetime(df[date_col])
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Week"] = df[date_col].dt.isocalendar().week
    df["Quarter"] = df[date_col].apply(lambda x: quarter_label(x))
    df["Day"] = df[date_col].dt.day
    df["MonthName"] = df[date_col].dt.strftime("%b")
    return df

# ================================================================
# 📈 CORE METRIC FUNCTIONS
# ================================================================
def compute_growth(df, period=1, freq="M", date_col="date", value_col="value"):
    """Generic growth over previous period"""
    if df.empty:
        return df.assign(**{f"Growth{period}{freq}%": None})
    d = df.copy()
    d[date_col] = safe_to_datetime(d[date_col])
    d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if d.empty:
        return df.assign(**{f"Growth{period}{freq}%": None})
    if freq == "D":
        s = d[value_col].resample("D").sum()
    elif freq == "W":
        s = d[value_col].resample("W").sum()
    elif freq == "Q":
        s = d[value_col].resample("QS").sum()
    elif freq == "Y":
        s = d[value_col].resample("YS").sum()
    else:
        s = d[value_col].resample("MS").sum()
    growth = (s - s.shift(period)) / s.shift(period) * 100
    return pd.DataFrame({date_col: s.index, "value": s.values, f"Growth{period}{freq}%": growth.values})

def compute_yoy(df, **kw): return compute_growth(df, period=12, freq="M", **kw)
def compute_qoq(df, **kw): return compute_growth(df, period=1, freq="Q", **kw)
def compute_mom(df, **kw): return compute_growth(df, period=1, freq="M", **kw)
def compute_wow(df, **kw): return compute_growth(df, period=1, freq="W", **kw)
def compute_dod(df, **kw): return compute_growth(df, period=1, freq="D", **kw)

# ================================================================
# 🔁 CUMULATIVE, ROLLING, NORMALIZATION
# ================================================================
def compute_cumulative(df, value_col="value"):
    if df.empty: return df.assign(**{"Cumulative": None})
    d = df.copy()
    d["Cumulative"] = d[value_col].cumsum()
    return d

def compute_rolling(df, value_col="value", windows=(3,6,12)):
    if df.empty: return df
    d = df.copy()
    for w in windows:
        d[f"Rolling{w}"] = d[value_col].rolling(window=w, min_periods=1).mean()
    return d

def compute_normalized(df, value_col="value"):
    if df.empty: return df.assign(**{"Normalized": None})
    d = df.copy()
    minv, maxv = d[value_col].min(), d[value_col].max()
    if maxv == minv: d["Normalized"] = 0
    else: d["Normalized"] = (d[value_col] - minv) / (maxv - minv)
    return d

# ================================================================
# 🧮 CONTRIBUTIONS & RANKS
# ================================================================
def percentage_contribution(df, value_col="value", group_col="category"):
    if df.empty or group_col not in df.columns:
        return df.assign(**{"%Contribution": None})
    d = df.copy()
    total = d[value_col].sum()
    d["%Contribution"] = d[value_col] / total * 100 if total else 0
    return d

def rank_by_value(df, value_col="value", ascending=False):
    if df.empty: return df.assign(**{"Rank": None})
    d = df.copy()
    d["Rank"] = d[value_col].rank(ascending=ascending, method="dense")
    return d

def top_bottom(df, value_col="value", top_n=5):
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    top = df.nlargest(top_n, value_col)
    bottom = df.nsmallest(top_n, value_col)
    return top, bottom

# ================================================================
# 🌐 FREQUENCY AGGREGATION
# ================================================================
def aggregate_by_freq(df, freq="M", date_col="date", value_col="value"):
    """Aggregate to given frequency (D/W/M/Q/Y)"""
    if df.empty:
        return df
    d = df.copy()
    d[date_col] = safe_to_datetime(d[date_col])
    d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
    mapping = {"D":"D","W":"W","M":"MS","Q":"QS","Y":"YS"}
    rule = mapping.get(freq.upper(), "MS")
    agg = d[value_col].resample(rule).sum().reset_index()
    return agg

# ================================================================
# 🧠 MASTER METRIC ENGINE
# ================================================================
def compute_all_metrics(df, date_col="date", value_col="value"):
    """Maxed: computes *everything* across frequencies"""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df[date_col] = safe_to_datetime(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # add base time parts
    df = add_time_columns(df, date_col)

    # core metrics
    yoy = compute_yoy(df, date_col=date_col, value_col=value_col)
    qoq = compute_qoq(df, date_col=date_col, value_col=value_col)
    mom = compute_mom(df, date_col=date_col, value_col=value_col)
    wow = compute_wow(df, date_col=date_col, value_col=value_col)
    dod = compute_dod(df, date_col=date_col, value_col=value_col)

    # merge all metrics by date
    merged = pd.DataFrame({date_col: df[date_col], value_col: df[value_col]})
    for subdf in [yoy, qoq, mom, wow, dod]:
        merged = pd.merge(merged, subdf, on=[date_col, value_col], how="left")

    # cumulative, rolling, normalization
    merged = compute_cumulative(merged, value_col)
    merged = compute_rolling(merged, value_col)
    merged = compute_normalized(merged, value_col)

    # add extra time parts and ranks
    merged = add_time_columns(merged, date_col)
    merged = rank_by_value(merged, value_col)

    return merged

# ================================================================
# 🧩 SUMMARY UTILITIES
# ================================================================
def summarize_metrics(df, value_col="value"):
    """Quick summary of key stats"""
    if df.empty:
        return {}
    stats = {
        "Mean": df[value_col].mean(),
        "Median": df[value_col].median(),
        "StdDev": df[value_col].std(),
        "Total": df[value_col].sum(),
        "Min": df[value_col].min(),
        "Max": df[value_col].max(),
        "LatestValue": df[value_col].iloc[-1] if not df.empty else None
    }
    return stats
