import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

# ---------------- BASIC METRICS ----------------
def compute_yoy(df: pd.DataFrame, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
    """Year-over-Year percentage growth (monthly resample).

    Returns a DataFrame with columns: date, value, YoY%
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "value", "YoY%"])

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if d.empty:
        return pd.DataFrame(columns=["date", "value", "YoY%"])

    m = d[value_col].resample("MS").sum()
    yoy = (m - m.shift(12)) / m.shift(12) * 100
    return pd.DataFrame({"date": m.index, "value": m.values, "YoY%": yoy.values})


def compute_qoq(df: pd.DataFrame, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
    """Quarter-over-Quarter percentage growth (quarterly start resample).

    Returns a DataFrame with columns: date, value, QoQ%
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "value", "QoQ%"])

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if d.empty:
        return pd.DataFrame(columns=["date", "value", "QoQ%"])

    q = d[value_col].resample("QS").sum()
    qoq = (q - q.shift(1)) / q.shift(1) * 100
    return pd.DataFrame({"date": q.index, "value": q.values, "QoQ%": qoq.values})


def compute_mom(df: pd.DataFrame, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
    """Month-over-Month percentage growth (monthly resample).

    Returns a DataFrame with columns: date, value, MoM%
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "value", "MoM%"])

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if d.empty:
        return pd.DataFrame(columns=["date", "value", "MoM%"])

    m = d[value_col].resample("MS").sum()
    mom = (m - m.shift(1)) / m.shift(1) * 100
    return pd.DataFrame({"date": m.index, "value": m.values, "MoM%": mom.values})


# ---------------- CUMULATIVE & ROLLING ----------------
def compute_cumulative(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Cumulative sum (preserves index/columns)."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["Cumulative"] = d[value_col].cumsum()
    return d


def compute_rolling(df: pd.DataFrame, value_col: str = "value", window: int = 3) -> pd.DataFrame:
    """Rolling average (adds Rolling{window} column)."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d[f"Rolling{window}"] = d[value_col].rolling(window=window).mean()
    return d


# ---------------- SUMMARY METRICS ----------------
def top_bottom(df: pd.DataFrame, value_col: str = "value", top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Top and bottom N entries."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    top = df.nlargest(top_n, value_col)
    bottom = df.nsmallest(top_n, value_col)
    return top, bottom


def percentage_contribution(df: pd.DataFrame, value_col: str = "value", group_col: str = "category") -> pd.DataFrame:
    """Percentage contribution per group."""
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame()
    total = df[value_col].sum()
    if total == 0:
        df["%Contribution"] = 0.0
    else:
        df["%Contribution"] = df[value_col] / total * 100
    return df


# ---------------- QUARTER LABEL ----------------
def quarter_label(ts: pd.Timestamp) -> str:
    """Returns Q1-2025, Q2-2025 style labels."""
    if pd.isna(ts):
        return None
    q = (ts.month - 1) // 3 + 1
    return f"Q{q}-{ts.year}"


def add_quarter_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Adds quarter column for reporting."""
    if df is None or df.empty:
        df = df.copy() if df is not None else pd.DataFrame()
        df["Quarter"] = None
        return df
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d["Quarter"] = d[date_col].apply(lambda x: quarter_label(x) if not pd.isna(x) else None)
    return d


# ---------------- ADDITIONAL 'MAXED' UTILITIES ----------------
def compute_growth(df: pd.DataFrame, value_col: str = "value", base_col: str = None) -> Any:
    """Generic growth rate computation.

    - If base_col provided, computes (sum(value) - sum(base)) / sum(base) * 100
    - Else computes growth% between first and last non-null entries.
    Returns None when not computable.
    """
    if df is None or df.empty:
        return None
    try:
        if base_col and base_col in df.columns:
            base = df[base_col].sum()
            new = df[value_col].sum()
            if base == 0:
                return None
            return (new - base) / base * 100
        series = df[value_col].dropna()
        if series.empty:
            return None
        first = series.iloc[0]
        last = series.iloc[-1]
        if first == 0:
            return None
        return (last - first) / first * 100
    except Exception:
        return None


def compare_periods(df: pd.DataFrame, date_col: str = "date", value_col: str = "value", period: str = "M") -> pd.DataFrame:
    """Compare metrics across time periods.

    period: 'M' (month), 'Q' (quarter), 'Y' (year)
    Returns DataFrame with Period, value, %Change.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        return pd.DataFrame()

    if period == "Q":
        d["Period"] = d[date_col].dt.to_period("Q").astype(str)
    elif period == "Y":
        d["Period"] = d[date_col].dt.year.astype(str)
    else:
        d["Period"] = d[date_col].dt.to_period("M").astype(str)

    summary = d.groupby("Period")[value_col].sum().reset_index()
    summary["%Change"] = summary[value_col].pct_change() * 100
    return summary


def summarize_trends(df: pd.DataFrame, value_col: str = "value") -> Dict[str, Any]:
    """Basic descriptive summary of numeric trend data.

    Returns dictionary with min, max, mean, std, count, growth%.
    """
    if df is None or df.empty:
        return {"min": None, "max": None, "mean": None, "std": None, "count": 0, "growth%": None}
    try:
        s = df[value_col].dropna().astype(float)
        return {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "count": int(s.count()),
            "growth%": compute_growth(df, value_col),
        }
    except Exception:
        return {"error": "Failed to compute summary"}


# ---------------- MAXED AGGREGATION ----------------
def compute_all_metrics(df: pd.DataFrame, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
    """Maxed: returns DataFrame with YoY, QoQ, MoM, cumulative, rolling, quarter.

    This function is defensive: it aligns resampled indexes and fills missing where needed.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    if df.empty:
        return pd.DataFrame()

    # create monthly baseline index
    baseline = df.set_index(date_col)[value_col].resample("MS").sum()
    out = pd.DataFrame(index=baseline.index)
    out["value"] = baseline.values

    # attach YoY/MoM/QoQ safely by reindexing the results
    try:
        yoy = compute_yoy(df, date_col, value_col).set_index("date")["YoY%"].reindex(out.index)
        out["YoY%"] = yoy.values
    except Exception:
        out["YoY%"] = np.nan

    try:
        mom = compute_mom(df, date_col, value_col).set_index("date")["MoM%"].reindex(out.index)
        out["MoM%"] = mom.values
    except Exception:
        out["MoM%"] = np.nan

    try:
        # QoQ is quarterly; align by taking latest QoQ value for months within that quarter
        qoq_df = compute_qoq(df, date_col, value_col)
        if not qoq_df.empty:
            qoq_map = qoq_df.set_index("date")["QoQ%"].to_dict()
            # map each month to its quarter QoQ (approx by period)
            out["QoQ%"] = [
                qoq_df.loc[qoq_df["date"] <= idx].iloc[-1]["QoQ%"]
                if not qoq_df[qoq_df["date"] <= idx].empty else np.nan
                for idx in out.index
            ]
        else:
            out["QoQ%"] = np.nan
    except Exception:
        out["QoQ%"] = np.nan

    out["Cumulative"] = out["value"].cumsum()
    out["Rolling3"] = out["value"].rolling(3, min_periods=1).mean()
    out["Quarter"] = out.index.to_series().apply(lambda x: quarter_label(x))

    out = out.reset_index().rename(columns={"index": date_col})
    return out


# Export API
__all__ = [
    "compute_yoy",
    "compute_qoq",
    "compute_mom",
    "compute_cumulative",
    "compute_rolling",
    "top_bottom",
    "percentage_contribution",
    "quarter_label",
    "add_quarter_column",
    "compute_growth",
    "compare_periods",
    "summarize_trends",
    "compute_all_metrics",
]
