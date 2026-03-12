"""
YouGov survey data processing.

Reads the YouGov voting intention survey (xlsx or csv), extracts
demographic-by-vote-intention cross-tabulations, and maps the demographic
categories to match the census categories used in the raking step.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_survey(path: str | Path) -> pd.DataFrame:
    """
    Load YouGov survey data from xlsx or csv.

    The function auto-detects the format from the file extension.
    """
    path = Path(path)
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info("Loaded survey data: %d rows, %d columns from %s", len(df), len(df.columns), path)
    return df


def convert_xlsx_to_csv(xlsx_path: str | Path, csv_path: str | Path | None = None) -> Path:
    """
    Convert an xlsx file to csv for easier manipulation.

    Parameters
    ----------
    xlsx_path : path to the xlsx file
    csv_path : output path. If None, uses same name with .csv extension.

    Returns
    -------
    Path to the created CSV file.
    """
    xlsx_path = Path(xlsx_path)
    if csv_path is None:
        csv_path = xlsx_path.with_suffix(".csv")
    else:
        csv_path = Path(csv_path)

    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.to_csv(csv_path, index=False)
    logger.info("Converted %s -> %s (%d rows)", xlsx_path, csv_path, len(df))
    return csv_path


def extract_vote_by_demographics(
    survey: pd.DataFrame,
    vote_col: str = "vote_intention",
    age_col: str = "age",
    tenure_col: str = "tenure",
    nssec_col: str = "nssec",
    weight_col: str | None = "weight",
) -> pd.DataFrame:
    """
    Extract vote intention cross-tabulated by demographics.

    This is the core function that prepares survey data for the model.
    It produces a table of (age_band, tenure, nssec, party, weighted_count)
    that can be converted to vote shares per demographic cell.

    Parameters
    ----------
    survey : pd.DataFrame
        The raw survey data.
    vote_col : str
        Column name for vote intention / party choice.
    age_col : str
        Column name for respondent age.
    tenure_col : str
        Column name for housing tenure.
    nssec_col : str
        Column name for NS-SEC / social grade.
    weight_col : str or None
        Column name for survey weights. None if unweighted.

    Returns
    -------
    pd.DataFrame with columns:
        age_band, tenure, nssec, party, weighted_count
    """
    df = survey.copy()

    # Standardise column names (case-insensitive matching)
    col_map = _fuzzy_match_columns(df, {
        "vote": vote_col,
        "age": age_col,
        "tenure": tenure_col,
        "nssec": nssec_col,
        "weight": weight_col,
    })

    vote_col = col_map["vote"]
    age_col = col_map["age"]
    tenure_col = col_map["tenure"]
    nssec_col = col_map["nssec"]
    weight_col = col_map.get("weight")

    # Drop rows with missing vote intention
    df = df.dropna(subset=[vote_col])

    # Recode demographics to match census categories
    df["age_band"] = df[age_col].apply(_recode_survey_age)
    df["tenure"] = df[tenure_col].apply(_recode_survey_tenure)
    df["nssec"] = df[nssec_col].apply(_recode_survey_nssec)
    df["party"] = df[vote_col].apply(_standardise_party)

    # Apply weights
    if weight_col and weight_col in df.columns:
        df["w"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
    else:
        df["w"] = 1.0

    # Aggregate
    result = (
        df.groupby(["age_band", "tenure", "nssec", "party"], as_index=False)["w"]
        .sum()
        .rename(columns={"w": "weighted_count"})
    )

    logger.info("Extracted %d demographic×party cells from survey", len(result))
    return result


def compute_vote_shares(vote_demo: pd.DataFrame) -> pd.DataFrame:
    """
    Convert weighted counts to vote shares within each demographic cell.

    Returns DataFrame: age_band, tenure, nssec, party, vote_share
    """
    totals = (
        vote_demo.groupby(["age_band", "tenure", "nssec"], as_index=False)["weighted_count"]
        .sum()
        .rename(columns={"weighted_count": "cell_total"})
    )
    merged = vote_demo.merge(totals, on=["age_band", "tenure", "nssec"])
    merged["vote_share"] = merged["weighted_count"] / merged["cell_total"]
    return merged[["age_band", "tenure", "nssec", "party", "vote_share"]]


# ── Recoding helpers ────────────────────────────────────────────────────


def _recode_survey_age(val) -> str:
    """Map survey age to census age bands."""
    try:
        age = int(val)
    except (TypeError, ValueError):
        val_str = str(val).lower()
        if "18" in val_str or "24" in val_str:
            return "18-24"
        if "25" in val_str or "34" in val_str:
            return "25-34"
        if "35" in val_str or "49" in val_str:
            return "35-49"
        if "50" in val_str or "64" in val_str:
            return "50-64"
        if "65" in val_str:
            return "65+"
        return "unknown"

    if age < 18:
        return "under_18"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 49:
        return "35-49"
    if age <= 64:
        return "50-64"
    return "65+"


def _recode_survey_tenure(val) -> str:
    """Map survey tenure categories to census categories."""
    if pd.isna(val):
        return "unknown"
    val = str(val).lower()
    if "outright" in val or "own" in val and "mortgage" not in val:
        return "owned_outright"
    if "mortgage" in val or "buying" in val:
        return "owned_mortgage"
    if "social" in val or "council" in val or "housing association" in val:
        return "social_rent"
    if "private" in val or "rent" in val:
        return "private_rent"
    if "own" in val:
        return "owned_outright"
    return "unknown"


def _recode_survey_nssec(val) -> str:
    """Map survey NS-SEC / social grade to census groups."""
    if pd.isna(val):
        return "unknown"
    val = str(val).upper().strip()
    if val in ("A", "B", "AB"):
        return "AB"
    if val in ("C1",):
        return "C1"
    if val in ("C2",):
        return "C2"
    if val in ("D", "E", "DE"):
        return "DE"
    # Try matching the longer NS-SEC labels
    val_lower = val.lower()
    if any(x in val_lower for x in ["higher managerial", "higher professional", "l1", "l2", "l3"]):
        return "AB"
    if any(x in val_lower for x in ["lower managerial", "lower professional", "l4", "l5", "l6"]):
        return "AB"
    if any(x in val_lower for x in ["intermediate", "l7"]):
        return "C1"
    if any(x in val_lower for x in ["small employer", "own account", "l8", "l9"]):
        return "C1"
    if any(x in val_lower for x in ["supervisory", "technical", "l10", "l11"]):
        return "C2"
    if any(x in val_lower for x in ["semi-routine", "routine", "never worked", "l12", "l13", "l14"]):
        return "DE"
    return "unknown"


def _standardise_party(val) -> str:
    """Standardise party names."""
    if pd.isna(val):
        return "undecided"
    val = str(val).lower().strip()
    if "conserv" in val or "tory" in val:
        return "Conservative"
    if "labour" in val:
        return "Labour"
    if "lib" in val and "dem" in val:
        return "Liberal Democrat"
    if "reform" in val:
        return "Reform"
    if "green" in val:
        return "Green"
    if "snp" in val or "scottish national" in val:
        return "SNP"
    if "plaid" in val:
        return "Plaid Cymru"
    if "undecided" in val or "don't know" in val or "dk" in val:
        return "undecided"
    if "would not vote" in val or "wouldn't" in val:
        return "would_not_vote"
    if "refuse" in val:
        return "refused"
    return val.title()


def _fuzzy_match_columns(df: pd.DataFrame, targets: dict) -> dict:
    """
    Try to match target column names to actual DataFrame columns,
    allowing for case-insensitive and partial matching.
    """
    result = {}
    df_cols_lower = {c.lower(): c for c in df.columns}

    for key, target in targets.items():
        if target is None:
            continue
        # Exact match
        if target in df.columns:
            result[key] = target
            continue
        # Case-insensitive
        if target.lower() in df_cols_lower:
            result[key] = df_cols_lower[target.lower()]
            continue
        # Partial match
        matches = [c for c in df.columns if target.lower() in c.lower()]
        if matches:
            result[key] = matches[0]
            continue
        # Keyword match
        matches = [c for c in df.columns if key.lower() in c.lower()]
        if matches:
            result[key] = matches[0]
            continue
        logger.warning("Could not match column '%s' (key=%s) in survey data", target, key)

    return result
