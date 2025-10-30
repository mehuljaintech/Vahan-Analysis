# # ==============================================================
# vahan/parsing.py — MAXED Universal Parser
# ==============================================================
"""
💪 MAXED JSON → pandas.DataFrame Parser for Vahan Analytics

✅ Handles all known + unknown Parivahan API formats
✅ Auto-detects parallel arrays, list-of-dicts, dict-of-year:value, nested data
✅ Smart month/year parsing with normalization
✅ Defensive numeric coercion and duplicate cleanup
✅ Consistent output for downstream analytics
✅ Includes maker/state, duration, revenue trend parsers
✅ Ready for integration with MAXED API and Streamlit
"""

from typing import Any, Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import re, math, json, logging

logger = logging.getLogger("vahan.parsing_maxed")
logger.setLevel(logging.INFO)

# ==============================================================
# 🔹 BASIC UTILITIES
# ==============================================================

def _coerce_num(x: Any) -> float:
    """Convert to float safely — handles commas, %, ₹, booleans, and empties."""
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return np.nan
        s = str(x).replace(",", "").replace("%", "").replace("₹", "").strip()
        if s.lower() in ("nan", "none", "null", "-", "--", "n/a"):
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def _get_first_present(d: Dict, keys: List[str]):
    """Return first non-empty value for given keys."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in [None, ""]:
            return d[k]
    return None


def _normalize_key(k: str) -> str:
    """Normalize key names."""
    return str(k).strip().lower().replace(" ", "").replace("_", "")


# ==============================================================
# 🔹 DATE HANDLING
# ==============================================================

MONTH_MAP = {
    m.lower(): i
    for i, m in enumerate(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        start=1
    )
}

def parse_date(y=None, m=None, my=None):
    """Universal date parser for monthly/yearly labels."""
    try:
        if y and m:
            try:
                mm = int(m)
            except Exception:
                mm = MONTH_MAP.get(str(m)[:3].lower())
            if mm:
                return pd.Timestamp(int(y), mm, 1)
        if my:
            my = str(my).strip()
            for fmt in ("%Y-%m", "%b-%Y", "%B-%Y", "%b %Y", "%B %Y", "%Y%m", "%Y"):
                dt = pd.to_datetime(my, format=fmt, errors="coerce")
                if pd.notna(dt):
                    return pd.Timestamp(dt.year, dt.month, 1)
            # free parse fallback
            dt = pd.to_datetime(my, errors="coerce")
            if pd.notna(dt):
                return pd.Timestamp(dt.year, dt.month, 1)
        if y:
            return pd.Timestamp(int(y), 1, 1)
    except Exception:
        pass
    return pd.NaT


# ==============================================================
# 🔹 UNIVERSAL to_df()
# ==============================================================

def to_df(json_obj: Any) -> pd.DataFrame:
    """Universal JSON → DataFrame converter."""
    if not json_obj:
        return pd.DataFrame(columns=["label", "value"])

    # Try to parse stringified JSON
    if isinstance(json_obj, str) and json_obj.strip().startswith("{"):
        try:
            json_obj = json.loads(json_obj)
        except Exception:
            pass

    # Case 1: Parallel arrays
    if isinstance(json_obj, dict) and "labels" in json_obj and "data" in json_obj:
        labels = json_obj.get("labels", [])
        values = json_obj.get("data", [])
        df = pd.DataFrame({"label": labels, "value": [_coerce_num(v) for v in values]})
        return df.dropna(subset=["value"]).reset_index(drop=True)

    # Case 2: Nested “data”
    if isinstance(json_obj, dict) and "data" in json_obj:
        return to_df(json_obj["data"])

    # Case 3: Dict of label:value pairs
    if isinstance(json_obj, dict):
        flat = [
            {"label": k, "value": _coerce_num(v)}
            for k, v in json_obj.items()
            if not isinstance(v, (dict, list))
        ]
        if flat:
            return pd.DataFrame(flat).dropna(subset=["value"])

    # Case 4: List of dicts
    if isinstance(json_obj, list):
        records = []
        for item in json_obj:
            if not isinstance(item, dict):
                continue
            label = _get_first_present(
                item,
                ["label", "name", "category", "state", "makerName", "manufacturer", "x"]
            )
            value = _get_first_present(
                item,
                ["value", "count", "total", "registeredVehicleCount", "y"]
            )
            if value is None and "metrics" in item:
                value = _get_first_present(item["metrics"], ["value", "count"])
            if label and value is not None:
                records.append({"label": label, "value": _coerce_num(value)})
        if records:
            return pd.DataFrame(records).drop_duplicates(subset=["label"])
    return pd.DataFrame(columns=["label", "value"])


# ==============================================================
# 🔹 TREND NORMALIZER
# ==============================================================

def normalize_trend(trend_json: Any) -> pd.DataFrame:
    """Normalize any trend structure → DataFrame [date, value]."""
    if not trend_json:
        return pd.DataFrame(columns=["date", "value"])

    # Case 1: Parallel arrays
    if isinstance(trend_json, dict) and "labels" in trend_json and "data" in trend_json:
        labels = trend_json.get("labels", [])
        values = trend_json.get("data", [])
        rows = []
        for lbl, val in zip(labels, values):
            val = _coerce_num(val)
            if pd.isna(val):
                continue
            dt = parse_date(my=lbl)
            rows.append({"date": dt, "value": val})
        return pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date")

    # Case 2: dict year→list
    if isinstance(trend_json, dict):
        rows = []
        for year, vals in trend_json.items():
            if isinstance(vals, (list, tuple)):
                for i, v in enumerate(vals, start=1):
                    rows.append({
                        "date": pd.Timestamp(int(year), min(i, 12), 1),
                        "value": _coerce_num(v)
                    })
            elif not isinstance(vals, (dict, list)):
                rows.append({
                    "date": pd.Timestamp(int(year), 1, 1),
                    "value": _coerce_num(vals)
                })
        if rows:
            return pd.DataFrame(rows).dropna(subset=["value"]).sort_values("date")

    # Case 3: list of dicts
    if isinstance(trend_json, list):
        rows = []
        for it in trend_json:
            if not isinstance(it, dict):
                continue
            y = _get_first_present(it, ["year", "Year"])
            m = _get_first_present(it, ["month", "Month"])
            my = _get_first_present(it, ["Month-Year", "period", "monthYear", "label"])
            v = _get_first_present(it, ["value", "count", "total", "y"])
            dt = parse_date(y=y, m=m, my=my)
            rows.append({"date": dt, "value": _coerce_num(v)})
        return pd.DataFrame(rows).dropna(subset=["date", "value"]).sort_values("date")

    return pd.DataFrame(columns=["date", "value"])


# ==============================================================
# 🔹 SPECIALIZED PARSERS
# ==============================================================

def parse_duration_table(json_obj: Any) -> pd.DataFrame:
    """Parse duration/registration summary tables."""
    df = to_df(json_obj)
    if "label" in df.columns:
        df["label"] = df["label"].astype(str)
    return df.sort_values("label")


def parse_top5_revenue(json_obj: Any) -> pd.DataFrame:
    """Extract top-5 revenue entries."""
    df = to_df(json_obj)
    return df.sort_values("value", ascending=False).head(5).reset_index(drop=True)


def parse_revenue_trend(json_obj: Any) -> pd.DataFrame:
    """Year→period trend normalized."""
    df = normalize_trend(json_obj)
    if df.empty and isinstance(json_obj, dict):
        rows = []
        for y, arr in json_obj.items():
            if isinstance(arr, (list, tuple)):
                for i, v in enumerate(arr, 1):
                    rows.append({
                        "year": str(y),
                        "period": i,
                        "value": _coerce_num(v)
                    })
        return pd.DataFrame(rows)
    return df


def parse_makers(json_obj: Any) -> pd.DataFrame:
    """Parse top makers JSON."""
    if not json_obj:
        return pd.DataFrame(columns=["label", "value"])
    if isinstance(json_obj, dict) and "datasets" in json_obj:
        labels = json_obj.get("labels", [])
        data = json_obj["datasets"][0].get("data", []) if json_obj["datasets"] else []
        df = pd.DataFrame({"label": labels, "value": [_coerce_num(v) for v in data]})
        return df.sort_values("value", ascending=False).reset_index(drop=True)
    return to_df(json_obj)


def parse_maker_state(json_obj: Any) -> pd.DataFrame:
    """Parse maker→state mapping."""
    rows = []
    if isinstance(json_obj, list):
        for item in json_obj:
            makers = item.get("makers") or item.get("makerStateMap") or {}
            for maker, stmap in makers.items():
                if isinstance(stmap, dict):
                    for state, val in stmap.items():
                        rows.append({
                            "maker": maker,
                            "state": state,
                            "value": _coerce_num(val)
                        })
    return pd.DataFrame(rows)


# ==============================================================
# 🔹 VALIDATION + EXPORT
# ==============================================================

def validate_df(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    """Ensure DataFrame has required columns and clean dtypes."""
    if df is None:
        return pd.DataFrame(columns=required)
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    df = df.dropna(subset=[c for c in required if c != "value"], how="all")
    return df.reset_index(drop=True)


def export_all_formats(df: pd.DataFrame, name: str = "vahan_export"):
    """Export DataFrame to CSV, Excel, JSON, HTML for debugging or logs."""
    try:
        df.to_csv(f"{name}.csv", index=False)
        df.to_excel(f"{name}.xlsx", index=False)
        df.to_json(f"{name}.json", orient="records", indent=2)
        df.to_html(f"{name}.html", index=False)
        logger.info(f"✅ Exported {len(df)} rows to all formats under {name}")
    except Exception as e:
        logger.error(f"Export failed: {e}")


# ==============================================================
# 🔹 PUBLIC EXPORTS
# ==============================================================

__all__ = [
    "to_df", "normalize_trend", "parse_duration_table",
    "parse_top5_revenue", "parse_revenue_trend",
    "parse_makers", "parse_maker_state",
    "validate_df", "export_all_formats"
]
