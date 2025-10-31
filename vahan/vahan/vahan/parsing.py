# ===============================================================
# vahan/parsing.py — MAXED ULTIMATE UNIVERSAL PARSER
# ===============================================================
"""
Universal + Defensive JSON Parsing Utilities for VAHAN Analytics
---------------------------------------------------------------
Supports:
✅ Parallel arrays (labels/data)
✅ Nested "data"/"rows"/"payload" wrappers
✅ Dict-of-dicts (period:value)
✅ List-of-dicts with arbitrary key names
✅ Daily, Monthly, Quarterly, Yearly date parsing
✅ Auto numeric coercion, missing-key recovery
✅ Maker/state/revenue/duration trend parsers
✅ Ready for "ALL MAXED" multi-year, multi-period comparisons
"""

from typing import Any, Iterable, Tuple
import pandas as pd
import numpy as np
import math
import logging

logger = logging.getLogger("vahan.parsing")

# ===============================================================
# 🔹 HELPER UTILITIES
# ===============================================================

def _coerce_num(x):
    """Safely convert to float; remove commas, %, spaces."""
    if x is None:
        return None
    try:
        s = str(x).replace(",", "").replace("%", "").strip()
        return float(s) if s not in ("", "None", "nan") else None
    except Exception:
        return None

def _get_first_present(d: dict, keys: Iterable[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _ensure_list_like(obj):
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return [obj]

# ===============================================================
# 🔹 UNIVERSAL PARSER (label/value)
# ===============================================================

def to_df(json_obj: Any,
          label_keys: Tuple[str, ...] = ("label", "name", "makerName", "manufacturer", "x", "Month-Year", "period"),
          value_keys: Tuple[str, ...] = ("value", "count", "total", "registeredVehicleCount", "y")) -> pd.DataFrame:
    """Convert arbitrary JSON -> DataFrame(label, value)."""
    if not json_obj:
        return pd.DataFrame(columns=["label", "value"])

    # Case 1: Parallel arrays
    if isinstance(json_obj, dict) and "labels" in json_obj and "data" in json_obj:
        labels = json_obj.get("labels", [])
        values = json_obj.get("data", [])
        n = min(len(labels), len(values))
        return pd.DataFrame({
            "label": labels[:n],
            "value": [_coerce_num(v) for v in values[:n]]
        }).dropna(subset=["value"]).reset_index(drop=True)

    # Case 2: Wrapped data
    if isinstance(json_obj, dict) and "data" in json_obj:
        return to_df(json_obj["data"], label_keys, value_keys)

    # Case 3: List of dicts
    if isinstance(json_obj, (list, tuple)):
        rows = []
        for it in json_obj:
            if not isinstance(it, dict):
                continue
            label = _get_first_present(it, label_keys)
            value = _get_first_present(it, value_keys)
            if label is None:
                label = it.get("category") or it.get("stateName") or it.get("region")
            if value is None and "metrics" in it:
                value = _get_first_present(it["metrics"], value_keys)
            num = _coerce_num(value)
            if label is not None and num is not None:
                rows.append({"label": str(label), "value": num})
        if rows:
            return pd.DataFrame(rows).drop_duplicates(subset=["label"]).reset_index(drop=True)

    # Case 4: Dict of label:value
    if isinstance(json_obj, dict):
        rows = []
        for k, v in json_obj.items():
            if k in ("labels", "data"):
                continue
            num = _coerce_num(v)
            if num is not None:
                rows.append({"label": str(k), "value": num})
        if rows:
            return pd.DataFrame(rows).reset_index(drop=True)

    return pd.DataFrame(columns=["label", "value"])

# ===============================================================
# 🔹 DATE PARSING (supports yearly, monthly, daily)
# ===============================================================

def parse_date(y=None, m=None, d=None, my=None):
    """Convert year/month/day strings into pd.Timestamp."""
    try:
        if y and m and d:
            return pd.Timestamp(int(y), int(m), int(d))
        if y and m:
            mm = None
            try:
                mm = int(m)
            except Exception:
                for fmt in ("%b", "%B"):
                    try:
                        mm = pd.to_datetime(m, format=fmt).month
                        break
                    except Exception:
                        continue
            if mm:
                return pd.Timestamp(int(y), mm, 1)
        if my:
            for fmt in ("%b-%Y", "%Y-%m", "%B %Y", "%b %Y", "%Y/%m", "%Y%m", "%Y"):
                try:
                    dt = pd.to_datetime(str(my), format=fmt, errors="coerce")
                    if pd.notna(dt):
                        return pd.Timestamp(dt.year, dt.month, 1)
                except Exception:
                    continue
            dt = pd.to_datetime(str(my), errors="coerce")
            if pd.notna(dt):
                return pd.Timestamp(dt.year, dt.month, 1)
        if y:
            return pd.Timestamp(int(y), 1, 1)
    except Exception:
        pass
    return None

# ===============================================================
# 🔹 TREND NORMALIZATION (Daily / Monthly / Yearly)
# ===============================================================

def normalize_trend(trend_json: Any) -> pd.DataFrame:
    """
    Normalize trend JSON → DataFrame(date, value)
    Works for:
      • {'labels': [...], 'data': [...]}
      • list of dicts
      • dict of period:value
    """
    if not trend_json:
        return pd.DataFrame(columns=["date", "value"])

    # Case 1: Parallel arrays
    if isinstance(trend_json, dict) and "labels" in trend_json and "data" in trend_json:
        labels = trend_json.get("labels", [])
        values = trend_json.get("data", [])
        rows = []
        for lbl, val in zip(labels, values):
            dt = parse_date(my=lbl)
            if dt is None:
                dt = pd.to_datetime(str(lbl), errors="coerce")
            num = _coerce_num(val)
            if pd.notna(dt) and num is not None:
                rows.append({"date": pd.Timestamp(dt), "value": num})
        if rows:
            return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Case 2: list of dicts
    if isinstance(trend_json, (list, tuple)):
        rows = []
        for it in trend_json:
            if not isinstance(it, dict):
                continue
            y = _get_first_present(it, ("year", "Year", "yr"))
            m = _get_first_present(it, ("month", "Month", "mn", "monthNo"))
            d = _get_first_present(it, ("day", "Day", "dateNo"))
            my = _get_first_present(it, ("Month-Year", "monthYear", "label", "period", "date"))
            val = _get_first_present(it, ("count", "value", "registrations", "total", "y"))
            dt = parse_date(y=y, m=m, d=d, my=my)
            num = _coerce_num(val)
            if dt is not None and num is not None:
                rows.append({"date": pd.Timestamp(dt), "value": num})
        if rows:
            return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Case 3: Dict of period:value
    if isinstance(trend_json, dict):
        rows = []
        for k, v in trend_json.items():
            if isinstance(v, (list, dict)):
                continue
            dt = parse_date(my=k)
            if dt is None:
                dt = pd.to_datetime(str(k), errors="coerce")
            num = _coerce_num(v)
            if pd.notna(dt) and num is not None:
                rows.append({"date": pd.Timestamp(dt), "value": num})
        if rows:
            return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    return pd.DataFrame(columns=["date", "value"])

# ===============================================================
# 🔹 DURATION, REVENUE, MAKER, STATE PARSERS
# ===============================================================

def parse_duration_table(json_obj: Any) -> pd.DataFrame:
    """Parses year-wise or duration-wise registration counts."""
    df = to_df(json_obj, label_keys=("yearAsString", "year", "label"), value_keys=("registeredVehicleCount", "count", "value"))
    return df.sort_values("label").reset_index(drop=True)

def parse_top5_revenue(json_obj: Any) -> pd.DataFrame:
    """Extracts top 5 revenue states."""
    df = to_df(json_obj, label_keys=("label", "stateName", "region"), value_keys=("value", "revenue", "amount"))
    if not df.empty:
        df = df.sort_values("value", ascending=False).head(5)
    return df.reset_index(drop=True)

def parse_revenue_trend(json_obj: Any) -> pd.DataFrame:
    """Parses year-wise revenue trend into DataFrame(year, period, value)."""
    rows = []
    if isinstance(json_obj, dict):
        for year, vals in json_obj.items():
            if isinstance(vals, (list, tuple)):
                for i, v in enumerate(vals, start=1):
                    num = _coerce_num(v)
                    if num is not None:
                        rows.append({"year": str(year), "period": i, "value": num})
    return pd.DataFrame(rows).reset_index(drop=True)

def parse_makers(json_obj: Any) -> pd.DataFrame:
    """Extracts top makers/manufacturers and their counts."""
    try:
        if "labels" in json_obj and "datasets" in json_obj:
            labels = json_obj.get("labels", [])
            data = json_obj.get("datasets", [{}])[0].get("data", [])
            df = pd.DataFrame({"label": labels, "value": [_coerce_num(v) for v in data]})
            return df.sort_values("value", ascending=False).reset_index(drop=True)
        return to_df(json_obj, label_keys=("makerName", "manufacturer", "label"), value_keys=("value", "count"))
    except Exception:
        return pd.DataFrame(columns=["label", "value"])

def parse_maker_state(json_obj: Any) -> pd.DataFrame:
    """Parses nested maker→state maps."""
    data = _ensure_list_like(json_obj)
    rows = []
    for it in data:
        if not isinstance(it, dict):
            continue
        makers = it.get("makers") or it.get("makerStateMap") or {}
        if isinstance(makers, dict):
            for maker, stmap in makers.items():
                if isinstance(stmap, dict):
                    for state, val in stmap.items():
                        num = _coerce_num(val)
                        if num is not None:
                            rows.append({"maker": maker, "state": state, "value": num})
    return pd.DataFrame(rows).reset_index(drop=True)

# ===============================================================
# 🔹 UNIVERSAL PREVIEW (for Streamlit or CLI)
# ===============================================================

def preview_df(df: pd.DataFrame, n=5):
    if df is None or df.empty:
        print("<Empty DataFrame>")
        return
    print(df.head(n))

# ===============================================================
# ✅ READY FOR: DAILY / MONTHLY / YEARLY / MULTI-YEAR ALL-PARAM COMPARISON
# ===============================================================
