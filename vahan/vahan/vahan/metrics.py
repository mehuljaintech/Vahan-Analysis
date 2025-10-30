# import pandas as pd
# import numpy as np
# from typing import Tuple, Dict, Any

# # ---------------- BASIC METRICS ----------------
# def compute_yoy(df: pd.DataFrame, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
#     """Year-over-Year percentage growth (monthly resample).

#     Returns a DataFrame with columns: date, value, YoY%
#     """
#     if df is None or df.empty:
#         return pd.DataFrame(columns=["date", "value", "YoY%"])

#     d = df.copy()
#     d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
#     d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
#     if d.empty:
#         return pd.DataFrame(columns=["date", "value", "YoY%"])

#     m = d[value_col].resample("MS").sum()
#     yoy = (m - m.shift(12)) / m.shift(12) * 100
#     return pd.DataFrame({"date": m.index, "value": m.values, "YoY%": yoy.values})


# def compute_qoq(df: pd.DataFrame, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
#     """Quarter-over-Quarter percentage growth (quarterly start resample).

#     Returns a DataFrame with columns: date, value, QoQ%
#     """
#     if df is None or df.empty:
#         return pd.DataFrame(columns=["date", "value", "QoQ%"])

#     d = df.copy()
#     d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
#     d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
#     if d.empty:
#         return pd.DataFrame(columns=["date", "value", "QoQ%"])

#     q = d[value_col].resample("QS").sum()
#     qoq = (q - q.shift(1)) / q.shift(1) * 100
#     return pd.DataFrame({"date": q.index, "value": q.values, "QoQ%": qoq.values})


# def compute_mom(df: pd.DataFrame, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
#     """Month-over-Month percentage growth (monthly resample).

#     Returns a DataFrame with columns: date, value, MoM%
#     """
#     if df is None or df.empty:
#         return pd.DataFrame(columns=["date", "value", "MoM%"])

#     d = df.copy()
#     d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
#     d = d.dropna(subset=[date_col]).set_index(date_col).sort_index()
#     if d.empty:
#         return pd.DataFrame(columns=["date", "value", "MoM%"])

#     m = d[value_col].resample("MS").sum()
#     mom = (m - m.shift(1)) / m.shift(1) * 100
#     return pd.DataFrame({"date": m.index, "value": m.values, "MoM%": mom.values})


# # ---------------- CUMULATIVE & ROLLING ----------------
# def compute_cumulative(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
#     """Cumulative sum (preserves index/columns)."""
#     if df is None or df.empty:
#         return pd.DataFrame()
#     d = df.copy()
#     d["Cumulative"] = d[value_col].cumsum()
#     return d


# def compute_rolling(df: pd.DataFrame, value_col: str = "value", window: int = 3) -> pd.DataFrame:
#     """Rolling average (adds Rolling{window} column)."""
#     if df is None or df.empty:
#         return pd.DataFrame()
#     d = df.copy()
#     d[f"Rolling{window}"] = d[value_col].rolling(window=window).mean()
#     return d


# # ---------------- SUMMARY METRICS ----------------
# def top_bottom(df: pd.DataFrame, value_col: str = "value", top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Top and bottom N entries."""
#     if df is None or df.empty:
#         return pd.DataFrame(), pd.DataFrame()
#     top = df.nlargest(top_n, value_col)
#     bottom = df.nsmallest(top_n, value_col)
#     return top, bottom


# def percentage_contribution(df: pd.DataFrame, value_col: str = "value", group_col: str = "category") -> pd.DataFrame:
#     """Percentage contribution per group."""
#     if df is None or df.empty or group_col not in df.columns:
#         return pd.DataFrame()
#     total = df[value_col].sum()
#     if total == 0:
#         df["%Contribution"] = 0.0
#     else:
#         df["%Contribution"] = df[value_col] / total * 100
#     return df


# # ---------------- QUARTER LABEL ----------------
# def quarter_label(ts: pd.Timestamp) -> str:
#     """Returns Q1-2025, Q2-2025 style labels."""
#     if pd.isna(ts):
#         return None
#     q = (ts.month - 1) // 3 + 1
#     return f"Q{q}-{ts.year}"


# def add_quarter_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
#     """Adds quarter column for reporting."""
#     if df is None or df.empty:
#         df = df.copy() if df is not None else pd.DataFrame()
#         df["Quarter"] = None
#         return df
#     d = df.copy()
#     d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
#     d["Quarter"] = d[date_col].apply(lambda x: quarter_label(x) if not pd.isna(x) else None)
#     return d


# # ---------------- ADDITIONAL 'MAXED' UTILITIES ----------------
# def compute_growth(df: pd.DataFrame, value_col: str = "value", base_col: str = None) -> Any:
#     """Generic growth rate computation.

#     - If base_col provided, computes (sum(value) - sum(base)) / sum(base) * 100
#     - Else computes growth% between first and last non-null entries.
#     Returns None when not computable.
#     """
#     if df is None or df.empty:
#         return None
#     try:
#         if base_col and base_col in df.columns:
#             base = df[base_col].sum()
#             new = df[value_col].sum()
#             if base == 0:
#                 return None
#             return (new - base) / base * 100
#         series = df[value_col].dropna()
#         if series.empty:
#             return None
#         first = series.iloc[0]
#         last = series.iloc[-1]
#         if first == 0:
#             return None
#         return (last - first) / first * 100
#     except Exception:
#         return None


# def compare_periods(df: pd.DataFrame, date_col: str = "date", value_col: str = "value", period: str = "M") -> pd.DataFrame:
#     """Compare metrics across time periods.

#     period: 'M' (month), 'Q' (quarter), 'Y' (year)
#     Returns DataFrame with Period, value, %Change.
#     """
#     if df is None or df.empty:
#         return pd.DataFrame()
#     d = df.copy()
#     d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
#     d = d.dropna(subset=[date_col])
#     if d.empty:
#         return pd.DataFrame()

#     if period == "Q":
#         d["Period"] = d[date_col].dt.to_period("Q").astype(str)
#     elif period == "Y":
#         d["Period"] = d[date_col].dt.year.astype(str)
#     else:
#         d["Period"] = d[date_col].dt.to_period("M").astype(str)

#     summary = d.groupby("Period")[value_col].sum().reset_index()
#     summary["%Change"] = summary[value_col].pct_change() * 100
#     return summary


# def summarize_trends(df: pd.DataFrame, value_col: str = "value") -> Dict[str, Any]:
#     """Basic descriptive summary of numeric trend data.

#     Returns dictionary with min, max, mean, std, count, growth%.
#     """
#     if df is None or df.empty:
#         return {"min": None, "max": None, "mean": None, "std": None, "count": 0, "growth%": None}
#     try:
#         s = df[value_col].dropna().astype(float)
#         return {
#             "min": float(s.min()),
#             "max": float(s.max()),
#             "mean": float(s.mean()),
#             "std": float(s.std()),
#             "count": int(s.count()),
#             "growth%": compute_growth(df, value_col),
#         }
#     except Exception:
#         return {"error": "Failed to compute summary"}


# # ---------------- MAXED AGGREGATION ----------------
# def compute_all_metrics(df: pd.DataFrame, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
#     """Maxed: returns DataFrame with YoY, QoQ, MoM, cumulative, rolling, quarter.

#     This function is defensive: it aligns resampled indexes and fills missing where needed.
#     """
#     if df is None or df.empty:
#         return pd.DataFrame()

#     df = df.copy()
#     df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
#     df = df.dropna(subset=[date_col]).sort_values(date_col)
#     if df.empty:
#         return pd.DataFrame()

#     # create monthly baseline index
#     baseline = df.set_index(date_col)[value_col].resample("MS").sum()
#     out = pd.DataFrame(index=baseline.index)
#     out["value"] = baseline.values

#     # attach YoY/MoM/QoQ safely by reindexing the results
#     try:
#         yoy = compute_yoy(df, date_col, value_col).set_index("date")["YoY%"].reindex(out.index)
#         out["YoY%"] = yoy.values
#     except Exception:
#         out["YoY%"] = np.nan

#     try:
#         mom = compute_mom(df, date_col, value_col).set_index("date")["MoM%"].reindex(out.index)
#         out["MoM%"] = mom.values
#     except Exception:
#         out["MoM%"] = np.nan

#     try:
#         # QoQ is quarterly; align by taking latest QoQ value for months within that quarter
#         qoq_df = compute_qoq(df, date_col, value_col)
#         if not qoq_df.empty:
#             qoq_map = qoq_df.set_index("date")["QoQ%"].to_dict()
#             # map each month to its quarter QoQ (approx by period)
#             out["QoQ%"] = [
#                 qoq_df.loc[qoq_df["date"] <= idx].iloc[-1]["QoQ%"]
#                 if not qoq_df[qoq_df["date"] <= idx].empty else np.nan
#                 for idx in out.index
#             ]
#         else:
#             out["QoQ%"] = np.nan
#     except Exception:
#         out["QoQ%"] = np.nan

#     out["Cumulative"] = out["value"].cumsum()
#     out["Rolling3"] = out["value"].rolling(3, min_periods=1).mean()
#     out["Quarter"] = out.index.to_series().apply(lambda x: quarter_label(x))

#     out = out.reset_index().rename(columns={"index": date_col})
#     return out


# # Export API
# __all__ = [
#     "compute_yoy",
#     "compute_qoq",
#     "compute_mom",
#     "compute_cumulative",
#     "compute_rolling",
#     "top_bottom",
#     "percentage_contribution",
#     "quarter_label",
#     "add_quarter_column",
#     "compute_growth",
#     "compare_periods",
#     "summarize_trends",
#     "compute_all_metrics",
# ]

# # import pandas as pd
# # import numpy as np
# # from typing import Tuple, Dict, Any
# # from datetime import timedelta

# # try:
# #     from statsmodels.tsa.holtwinters import ExponentialSmoothing
# # except ImportError:
# #     ExponentialSmoothing = None

# # # ===============================================================
# # # 📊 MAXED ANALYTICS MODULE — ULTIMATE ALL-IN-ONE ENGINE
# # # ===============================================================

# # def safe_to_datetime(df, col="date"):
# #     df[col] = pd.to_datetime(df[col], errors="coerce")
# #     return df.dropna(subset=[col])

# # def safe_numeric(series):
# #     return pd.to_numeric(series, errors="coerce")

# # # ---------------- BASIC GROWTH METRICS ----------------
# # def compute_change(df, period="M", date_col="date", value_col="value"):
# #     if df.empty:
# #         return pd.DataFrame()
# #     df = safe_to_datetime(df.copy(), date_col)
# #     df[value_col] = safe_numeric(df[value_col])
# #     df = df.set_index(date_col).sort_index()

# #     mapping = {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS", "Y": "YS"}
# #     if period not in mapping:
# #         period = "M"

# #     res = df[value_col].resample(mapping[period]).sum()
# #     pct = res.pct_change() * 100
# #     label = {"D": "DoD%", "W": "WoW%", "M": "MoM%", "Q": "QoQ%", "Y": "YoY%"}[period]
# #     return pd.DataFrame({"date": res.index, "value": res.values, label: pct.values})

# # def compute_yoy(df, **kwargs): return compute_change(df, "Y", **kwargs)
# # def compute_qoq(df, **kwargs): return compute_change(df, "Q", **kwargs)
# # def compute_mom(df, **kwargs): return compute_change(df, "M", **kwargs)
# # def compute_wow(df, **kwargs): return compute_change(df, "W", **kwargs)
# # def compute_dod(df, **kwargs): return compute_change(df, "D", **kwargs)

# # # ---------------- ADVANCED METRICS ----------------
# # def compute_cumulative(df, value_col="value"):
# #     if df.empty: return df
# #     df = df.copy()
# #     df["Cumulative"] = safe_numeric(df[value_col]).cumsum()
# #     return df

# # def compute_rolling(df, value_col="value", window=3):
# #     if df.empty: return df
# #     df = df.copy()
# #     df[f"Rolling{window}"] = safe_numeric(df[value_col]).rolling(window, min_periods=1).mean()
# #     return df

# # def exponential_smooth(df, value_col="value", alpha=0.3):
# #     if df.empty: return df
# #     df = df.copy()
# #     df["ExpSmooth"] = safe_numeric(df[value_col]).ewm(alpha=alpha).mean()
# #     return df

# # # ---------------- ANOMALY DETECTION ----------------
# # def detect_anomalies(df, value_col="value", z_thresh=2.5):
# #     if df.empty: return df
# #     d = df.copy()
# #     vals = safe_numeric(d[value_col])
# #     zscore = (vals - vals.mean()) / (vals.std() or 1)
# #     d["Zscore"] = zscore
# #     d["Anomaly"] = (np.abs(zscore) > z_thresh)
# #     return d

# # # ---------------- FORECASTING ----------------
# # def forecast_next(df, periods=12, date_col="date", value_col="value"):
# #     if df.empty or ExponentialSmoothing is None:
# #         return pd.DataFrame()
# #     df = safe_to_datetime(df.copy(), date_col)
# #     df[value_col] = safe_numeric(df[value_col])
# #     df = df.set_index(date_col).asfreq("MS").fillna(method="ffill")
# #     try:
# #         model = ExponentialSmoothing(df[value_col], trend="add", seasonal="add", seasonal_periods=12)
# #         fit = model.fit()
# #         future = fit.forecast(periods)
# #         forecast_df = pd.DataFrame({date_col: future.index, "Predicted": future.values})
# #         return forecast_df
# #     except Exception:
# #         return pd.DataFrame()

# # # ---------------- DESCRIPTIVE SUMMARY ----------------
# # def summarize_trends(df, value_col="value"):
# #     if df.empty:
# #         return {}
# #     s = safe_numeric(df[value_col].dropna())
# #     if s.empty: return {}
# #     return {
# #         "min": float(s.min()),
# #         "max": float(s.max()),
# #         "mean": float(s.mean()),
# #         "std": float(s.std()),
# #         "count": int(s.count()),
# #         "median": float(s.median()),
# #         "growth%": ((s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100) if s.iloc[0] != 0 else None,
# #     }

# # # ---------------- CORRELATION & TREND ----------------
# # def compute_correlation(df, col1, col2):
# #     if col1 not in df or col2 not in df: return None
# #     x, y = safe_numeric(df[col1]), safe_numeric(df[col2])
# #     valid = (~x.isna()) & (~y.isna())
# #     if valid.sum() == 0: return None
# #     return float(np.corrcoef(x[valid], y[valid])[0, 1])

# # def compute_trend_slope(df, date_col="date", value_col="value"):
# #     if df.empty: return None
# #     df = safe_to_datetime(df.copy(), date_col)
# #     df[value_col] = safe_numeric(df[value_col])
# #     x = np.arange(len(df))
# #     y = df[value_col].values
# #     if len(x) < 2: return None
# #     slope = np.polyfit(x, y, 1)[0]
# #     return float(slope)

# # # ---------------- CONTRIBUTION ----------------
# # def percentage_contribution(df, value_col="value", group_col="category"):
# #     if df.empty or group_col not in df: return df
# #     df = df.copy()
# #     total = df[value_col].sum()
# #     df["%Contribution"] = df[value_col] / total * 100 if total != 0 else 0
# #     return df

# # # ---------------- PERIOD COMPARISON ----------------
# # def compare_periods(df, date_col="date", value_col="value", period="M"):
# #     if df.empty: return pd.DataFrame()
# #     df = safe_to_datetime(df.copy(), date_col)
# #     period_map = {"M": "M", "Q": "Q", "Y": "Y"}
# #     df["Period"] = df[date_col].dt.to_period(period_map.get(period, "M")).astype(str)
# #     summary = df.groupby("Period")[value_col].sum().reset_index()
# #     summary["%Change"] = summary[value_col].pct_change() * 100
# #     return summary

# # # ---------------- MASTER AGGREGATOR ----------------
# # def compute_all_metrics(df, date_col="date", value_col="value"):
# #     if df.empty: return pd.DataFrame()
# #     df = safe_to_datetime(df.copy(), date_col)
# #     base = df.set_index(date_col).asfreq("MS").fillna(method="ffill")
# #     out = pd.DataFrame({"date": base.index, "value": base[value_col].values})
# #     out = out.merge(compute_mom(df), on="date", how="outer")
# #     out = out.merge(compute_qoq(df), on="date", how="outer")
# #     out = out.merge(compute_yoy(df), on="date", how="outer")
# #     out = compute_cumulative(out)
# #     out = compute_rolling(out)
# #     out = exponential_smooth(out)
# #     out = detect_anomalies(out)
# #     forecast_df = forecast_next(out)
# #     if not forecast_df.empty:
# #         out = pd.concat([out, forecast_df], ignore_index=True)
# #     return out

# # # ---------------- EXPORTS ----------------
# # __all__ = [
# #     "compute_yoy", "compute_qoq", "compute_mom", "compute_wow", "compute_dod",
# #     "compute_cumulative", "compute_rolling", "exponential_smooth",
# #     "detect_anomalies", "forecast_next", "summarize_trends",
# #     "compute_correlation", "compute_trend_slope",
# #     "percentage_contribution", "compare_periods", "compute_all_metrics"
# # ]

# ===============================================================
# 📊 vahan/metrics.py — ALL MAXED ANALYTICS ENGINE (2025)
# ===============================================================

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from datetime import timedelta

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ExponentialSmoothing = None


# ===============================================================
# 🧠 SAFE UTILITIES
# ===============================================================
def safe_to_datetime(df, col="date"):
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col])

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ===============================================================
# 📈 BASIC CHANGE METRICS
# ===============================================================
def compute_change(df, period="M", date_col="date", value_col="value"):
    """Generic percentage change for given period (M=month, Q=quarter, Y=year)."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = safe_to_datetime(df.copy(), date_col)
    df[value_col] = safe_numeric(df[value_col])
    df = df.set_index(date_col).sort_index()

    mapping = {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS", "Y": "YS"}
    if period not in mapping:
        period = "M"

    res = df[value_col].resample(mapping[period]).sum()
    pct = res.pct_change() * 100
    label = {"D": "DoD%", "W": "WoW%", "M": "MoM%", "Q": "QoQ%", "Y": "YoY%"}[period]
    return pd.DataFrame({"date": res.index, "value": res.values, label: pct.values})


def compute_yoy(df, **kw): return compute_change(df, "Y", **kw)
def compute_qoq(df, **kw): return compute_change(df, "Q", **kw)
def compute_mom(df, **kw): return compute_change(df, "M", **kw)
def compute_wow(df, **kw): return compute_change(df, "W", **kw)
def compute_dod(df, **kw): return compute_change(df, "D", **kw)


# ===============================================================
# 🧮 CUMULATIVE / ROLLING / SMOOTHING
# ===============================================================
def compute_cumulative(df, value_col="value"):
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    d["Cumulative"] = safe_numeric(d[value_col]).cumsum()
    return d


def compute_rolling(df, value_col="value", window=3):
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    d[f"Rolling{window}"] = safe_numeric(d[value_col]).rolling(window, min_periods=1).mean()
    return d


def exponential_smooth(df, value_col="value", alpha=0.3):
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    d["ExpSmooth"] = safe_numeric(d[value_col]).ewm(alpha=alpha).mean()
    return d


# ===============================================================
# ⚠️ ANOMALY DETECTION
# ===============================================================
def detect_anomalies(df, value_col="value", z_thresh=2.5):
    """Marks anomalies where |zscore| > threshold."""
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    vals = safe_numeric(d[value_col])
    z = (vals - vals.mean()) / (vals.std() or 1)
    d["Zscore"] = z
    d["Anomaly"] = np.abs(z) > z_thresh
    return d


# ===============================================================
# 🔮 FORECASTING (HOLT-WINTERS)
# ===============================================================
def forecast_next(df, periods=12, date_col="date", value_col="value"):
    """Predict next N periods using Holt-Winters exponential smoothing."""
    if df is None or df.empty or ExponentialSmoothing is None:
        return pd.DataFrame()

    df = safe_to_datetime(df.copy(), date_col)
    df[value_col] = safe_numeric(df[value_col])
    df = df.set_index(date_col).asfreq("MS").fillna(method="ffill")

    try:
        model = ExponentialSmoothing(df[value_col], trend="add", seasonal="add", seasonal_periods=12)
        fit = model.fit()
        future = fit.forecast(periods)
        return pd.DataFrame({date_col: future.index, "Predicted": future.values})
    except Exception:
        return pd.DataFrame()


# ===============================================================
# 🧾 DESCRIPTIVE STATS
# ===============================================================
def summarize_trends(df, value_col="value"):
    """Summarize numeric trend data."""
    if df is None or df.empty:
        return {}
    s = safe_numeric(df[value_col].dropna())
    if s.empty: return {}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "count": int(s.count()),
        "median": float(s.median()),
        "growth%": ((s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100) if s.iloc[0] != 0 else None,
    }


# ===============================================================
# 📉 CORRELATION / TREND SLOPE
# ===============================================================
def compute_correlation(df, col1, col2):
    if df is None or df.empty or col1 not in df or col2 not in df:
        return None
    x, y = safe_numeric(df[col1]), safe_numeric(df[col2])
    valid = (~x.isna()) & (~y.isna())
    if valid.sum() == 0: return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def compute_trend_slope(df, date_col="date", value_col="value"):
    if df is None or df.empty: return None
    df = safe_to_datetime(df.copy(), date_col)
    df[value_col] = safe_numeric(df[value_col])
    x = np.arange(len(df))
    y = df[value_col].values
    if len(x) < 2: return None
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


# ===============================================================
# 🧩 GROUP CONTRIBUTION / COMPARISON
# ===============================================================
def percentage_contribution(df, value_col="value", group_col="category"):
    if df is None or df.empty or group_col not in df: return pd.DataFrame()
    df = df.copy()
    total = df[value_col].sum()
    df["%Contribution"] = (df[value_col] / total * 100) if total != 0 else 0
    return df


def compare_periods(df, date_col="date", value_col="value", period="M"):
    """Compare metrics across months, quarters or years."""
    if df is None or df.empty: return pd.DataFrame()
    df = safe_to_datetime(df.copy(), date_col)
    df["Period"] = (
        df[date_col].dt.to_period({"M": "M", "Q": "Q", "Y": "Y"}.get(period, "M")).astype(str)
    )
    summary = df.groupby("Period")[value_col].sum().reset_index()
    summary["%Change"] = summary[value_col].pct_change() * 100
    return summary


# ===============================================================
# 🧠 MASTER AGGREGATOR — ALL METRICS IN ONE
# ===============================================================
def compute_all_metrics(df, date_col="date", value_col="value"):
    if df is None or df.empty: return pd.DataFrame()

    df = safe_to_datetime(df.copy(), date_col)
    base = df.set_index(date_col).asfreq("MS").fillna(method="ffill")

    out = pd.DataFrame({"date": base.index, "value": base[value_col].values})

    for fn in [compute_mom, compute_qoq, compute_yoy]:
        try:
            tmp = fn(df, date_col=date_col, value_col=value_col)
            out = out.merge(tmp, on="date", how="outer")
        except Exception:
            pass

    out = compute_cumulative(out)
    out = compute_rolling(out)
    out = exponential_smooth(out)
    out = detect_anomalies(out)

    fc = forecast_next(out)
    if not fc.empty:
        out = pd.concat([out, fc], ignore_index=True)

    return out


# ===============================================================
# 🧾 EXPORTS
# ===============================================================
__all__ = [
    "compute_yoy", "compute_qoq", "compute_mom", "compute_wow", "compute_dod",
    "compute_cumulative", "compute_rolling", "exponential_smooth",
    "detect_anomalies", "forecast_next", "summarize_trends",
    "compute_correlation", "compute_trend_slope",
    "percentage_contribution", "compare_periods", "compute_all_metrics"
]

