"""
Census data processing — clean raw Nomis data into standardised marginals
and prepare inputs for the IPF/raking step.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Column-name normalization ───────────────────────────────────────────
# Nomis CSV columns vary by dataset. These helpers extract the category
# label and observation value from whatever column naming Nomis uses.


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase all column names for easier matching."""
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _obs_col(df: pd.DataFrame) -> str:
    """Find the observation value column."""
    for candidate in ["obs_value", "observation", "value", "count"]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Cannot find observation column in {list(df.columns)}")


# ── Age marginal ────────────────────────────────────────────────────────


def process_age_marginal(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Process TS007A (age by 6-category bands) into the age bands we use
    for the model.

    Returns DataFrame with columns:
        geography_code, age_band, count
    """
    df = _normalise_columns(raw.copy())
    obs = _obs_col(df)

    # Identify the age category column — Nomis names it something like
    # 'c2021_age_6a_name' or similar. We look for 'age' in column names.
    age_col = _find_category_column(df, "age")

    result = (
        df[["geography_code", age_col, obs]]
        .rename(columns={age_col: "age_band", obs: "count"})
    )
    result["count"] = pd.to_numeric(result["count"], errors="coerce").fillna(0)

    # Filter out totals
    result = result[~result["age_band"].str.lower().str.contains("total|all", na=False)]

    # Recode age bands to our standard set
    result["age_band"] = result["age_band"].apply(_recode_age)
    result = result.groupby(["geography_code", "age_band"], as_index=False)["count"].sum()

    return result


def _recode_age(label: str) -> str:
    """Map Nomis age band labels to our simplified bands."""
    label = label.lower().strip()
    if any(x in label for x in ["15", "16", "17", "18", "19", "20", "21", "22", "23", "24"]):
        if any(x in label for x in ["under", "0 to", "5 to", "10 to"]):
            return "under_18"  # will be dropped
        return "18-24"
    if "25" in label or "30" in label or "34" in label:
        return "25-34"
    if "35" in label or "40" in label or "45" in label or "49" in label:
        return "35-49"
    if "50" in label or "55" in label or "60" in label or "64" in label:
        return "50-64"
    if "65" in label or "70" in label or "75" in label or "80" in label or "85" in label or "90" in label:
        return "65+"
    if any(x in label for x in ["aged 4", "aged 0", "under"]):
        return "under_18"
    return label  # keep as-is for debugging


# ── Tenure marginal ────────────────────────────────────────────────────


def process_tenure_marginal(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Process TS054 (tenure) into simplified tenure categories.

    Returns DataFrame: geography_code, tenure, count
    """
    df = _normalise_columns(raw.copy())
    obs = _obs_col(df)
    tenure_col = _find_category_column(df, "tenure")

    result = (
        df[["geography_code", tenure_col, obs]]
        .rename(columns={tenure_col: "tenure", obs: "count"})
    )
    result["count"] = pd.to_numeric(result["count"], errors="coerce").fillna(0)
    result = result[~result["tenure"].str.lower().str.contains("total|all", na=False)]

    result["tenure"] = result["tenure"].apply(_recode_tenure)
    result = result.groupby(["geography_code", "tenure"], as_index=False)["count"].sum()

    return result


def _recode_tenure(label: str) -> str:
    """Simplify tenure labels."""
    label_lower = label.lower()
    if "outright" in label_lower:
        return "owned_outright"
    if "mortgage" in label_lower or "shared" in label_lower:
        return "owned_mortgage"
    if "social" in label_lower:
        return "social_rent"
    if "private" in label_lower or "rent free" in label_lower:
        return "private_rent"
    return label


# ── NS-SEC marginal ────────────────────────────────────────────────────


def process_nssec_marginal(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Process TS062 (NS-SeC) into simplified ABC1C2DE categories.

    Returns DataFrame: geography_code, nssec, count
    """
    df = _normalise_columns(raw.copy())
    obs = _obs_col(df)
    nssec_col = _find_category_column(df, "ns-sec", "nssec", "ns_sec")

    result = (
        df[["geography_code", nssec_col, obs]]
        .rename(columns={nssec_col: "nssec", obs: "count"})
    )
    result["count"] = pd.to_numeric(result["count"], errors="coerce").fillna(0)
    result = result[~result["nssec"].str.lower().str.contains("total|all", na=False)]

    result["nssec"] = result["nssec"].apply(_recode_nssec)
    result = result.groupby(["geography_code", "nssec"], as_index=False)["count"].sum()

    return result


def _recode_nssec(label: str) -> str:
    """Map NS-SEC categories to simplified ABC1C2DE grouping."""
    label_lower = label.lower()
    for group, prefixes in config.NSSEC_SIMPLIFIED.items():
        for prefix in prefixes:
            if prefix.lower() in label_lower:
                return group
    if "not classified" in label_lower or "not applicable" in label_lower:
        return "unclassified"
    return label


# ── Cross-tab (Tenure × NS-SEC) ────────────────────────────────────────


def process_tenure_nssec_crosstab(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Process RM138 (tenure × NS-SEC) into a usable cross-tabulation.

    Returns DataFrame: geography_code, tenure, nssec, count
    """
    df = _normalise_columns(raw.copy())
    obs = _obs_col(df)
    tenure_col = _find_category_column(df, "tenure")
    nssec_col = _find_category_column(df, "ns-sec", "nssec", "ns_sec")

    result = (
        df[["geography_code", tenure_col, nssec_col, obs]]
        .rename(columns={tenure_col: "tenure", nssec_col: "nssec", obs: "count"})
    )
    result["count"] = pd.to_numeric(result["count"], errors="coerce").fillna(0)

    # Remove totals
    for col in ["tenure", "nssec"]:
        result = result[~result[col].str.lower().str.contains("total|all", na=False)]

    result["tenure"] = result["tenure"].apply(_recode_tenure)
    result["nssec"] = result["nssec"].apply(_recode_nssec)
    result = result.groupby(["geography_code", "tenure", "nssec"], as_index=False)["count"].sum()

    return result


# ── Helpers ─────────────────────────────────────────────────────────────


def _find_category_column(df: pd.DataFrame, *keywords: str) -> str:
    """Find a column whose name contains one of the keywords and ends with _name."""
    for col in df.columns:
        col_lower = col.lower()
        for kw in keywords:
            if kw.lower() in col_lower and ("name" in col_lower or "label" in col_lower):
                return col
    # Fallback: any column containing the keyword
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower() and col not in ("geography_code", "geography_name"):
                return col
    raise KeyError(f"Cannot find category column for {keywords} in {list(df.columns)}")


def build_marginals_for_constituency(
    age_df: pd.DataFrame,
    tenure_df: pd.DataFrame,
    nssec_df: pd.DataFrame,
    geo_code: str,
) -> dict:
    """
    Extract the three 1D marginals for a single constituency.

    Returns dict with keys 'age', 'tenure', 'nssec', each a dict of
    {category: count}.
    """
    def _extract(df, geo_code, cat_col):
        sub = df[df["geography_code"] == geo_code]
        return dict(zip(sub[cat_col], sub["count"]))

    return {
        "age": _extract(age_df, geo_code, "age_band"),
        "tenure": _extract(tenure_df, geo_code, "tenure"),
        "nssec": _extract(nssec_df, geo_code, "nssec"),
    }
