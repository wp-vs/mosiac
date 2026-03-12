"""
YouGov cross-tabulation parser.

Parses the YouGov survey xlsx which contains vote intention cross-tabs
broken down by demographic groups. Each sheet has a specific structure:
- Row 0: Title
- Row 1: Sample size
- Row 2: Fieldwork dates
- Row 4: Dimension group headers
- Row 5: Column (demographic category) labels
- Row 6: Weighted sample sizes per column
- Row 7: Unweighted sample sizes
- Row 8: Section header (MRP model / constituency VI)
- Rows 9-16: Party vote shares (percentages) for MRP headline
- Row 17: Second section header
- Rows 18-28: Party vote shares for constituency VI (includes DK/WNV/Refused)

Values are percentages (vote share within each demographic column).
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ── Sheet parsing ───────────────────────────────────────────────────────


def parse_yougov_xlsx(path: str | Path) -> dict:
    """
    Parse the full YouGov xlsx into structured data.

    Returns a dict keyed by dimension name, each containing:
        {
            "categories": list of demographic category labels,
            "sample_sizes": dict of {category: weighted_n},
            "headline_vi": DataFrame (party × category) of vote shares,
            "constituency_vi": DataFrame (party × category) of vote shares,
        }
    """
    path = Path(path)
    xl = pd.ExcelFile(path, engine="openpyxl")
    result = {}

    for sheet_name in xl.sheet_names:
        raw = pd.read_excel(xl, sheet_name=sheet_name, header=None)
        parsed = _parse_sheet(raw, sheet_name)
        if parsed:
            for dim_name, dim_data in parsed.items():
                result[dim_name] = dim_data

    logger.info("Parsed %d demographic dimensions from %s", len(result), path)
    return result


def _parse_sheet(raw: pd.DataFrame, sheet_name: str) -> dict:
    """Parse a single sheet into one or more dimension blocks."""
    results = {}

    # Row 4 has dimension group headers (merged cells show in first col of span)
    # Row 5 has column labels
    if len(raw) < 20:
        logger.warning("Sheet %s too short (%d rows), skipping", sheet_name, len(raw))
        return results

    header_row = raw.iloc[4]
    label_row = raw.iloc[5]
    weighted_row = raw.iloc[6]
    unweighted_row = raw.iloc[7]

    # Find dimension blocks by looking at non-null cells in row 4
    blocks = _find_dimension_blocks(header_row, label_row)

    for block in blocks:
        dim_name = block["name"]
        col_start = block["col_start"]
        col_end = block["col_end"]

        # Extract category labels
        categories = []
        sample_sizes = {}
        for c in range(col_start, col_end + 1):
            label = str(label_row.iloc[c]) if pd.notna(label_row.iloc[c]) else None
            if label and label != "nan":
                categories.append(label)
                w = weighted_row.iloc[c]
                sample_sizes[label] = int(w) if pd.notna(w) else 0

        if not categories:
            continue

        # Extract vote share tables
        headline_vi = _extract_vote_block(raw, categories, col_start, col_end,
                                           data_start_row=9, data_end_row=16)
        constituency_vi = _extract_vote_block(raw, categories, col_start, col_end,
                                               data_start_row=18, data_end_row=28)

        results[dim_name] = {
            "categories": categories,
            "sample_sizes": sample_sizes,
            "headline_vi": headline_vi,
            "constituency_vi": constituency_vi,
        }
        logger.info("  Parsed dimension '%s': %d categories, %d parties",
                     dim_name, len(categories),
                     len(constituency_vi) if constituency_vi is not None else 0)

    return results


def _find_dimension_blocks(header_row: pd.Series, label_row: pd.Series) -> list[dict]:
    """
    Identify dimension blocks from the header row.

    A block starts where header_row has a non-null value and extends
    until the next non-null header (or end of data).
    """
    blocks = []
    current_block = None

    for c in range(2, len(header_row)):  # Skip cols 0 (Metric) and 1 (Total)
        header_val = header_row.iloc[c] if pd.notna(header_row.iloc[c]) else None
        label_val = label_row.iloc[c] if pd.notna(label_row.iloc[c]) else None

        if header_val:
            # New block starts
            if current_block:
                current_block["col_end"] = c - 1
                blocks.append(current_block)
            current_block = {
                "name": str(header_val).strip(),
                "col_start": c,
                "col_end": c,
            }
        elif current_block and label_val:
            # Extend current block
            current_block["col_end"] = c

    if current_block:
        blocks.append(current_block)

    return blocks


def _extract_vote_block(
    raw: pd.DataFrame,
    categories: list[str],
    col_start: int,
    col_end: int,
    data_start_row: int,
    data_end_row: int,
) -> pd.DataFrame | None:
    """
    Extract a vote share table from a block of rows.

    Returns DataFrame with columns: party, + one column per category,
    values are percentages.
    """
    rows = []
    for r in range(data_start_row, min(data_end_row + 1, len(raw))):
        party = raw.iloc[r, 0]
        if pd.isna(party) or str(party).strip() == "":
            continue
        party = str(party).strip()

        values = []
        for c in range(col_start, col_end + 1):
            v = raw.iloc[r, c]
            if pd.notna(v):
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    values.append(np.nan)
            else:
                values.append(np.nan)

        if len(values) == len(categories):
            row = {"party": party}
            for cat, val in zip(categories, values):
                row[cat] = val
            rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows)


# ── Extracting vote shares for the model ────────────────────────────────


PARTY_NAME_MAP = {
    "Con": "Conservative",
    "Lab": "Labour",
    "Lib Dem": "Liberal Democrat",
    "Liberal Democrat": "Liberal Democrat",
    "Reform UK": "Reform UK",
    "Green": "Green",
    "SNP": "SNP",
    "Plaid Cymru": "Plaid Cymru",
    "Other": "Other",
    "Some other party": "Other",
    "Would not vote": "Would not vote",
    "Don't know": "Don't know",
    "Refused": "Refused",
}

# Target parties for London (exclude SNP, Plaid)
LONDON_PARTIES = ["Conservative", "Labour", "Liberal Democrat", "Reform UK", "Green", "Other"]


def extract_age_vote_shares(parsed: dict, use_constituency_vi: bool = True) -> pd.DataFrame:
    """
    Extract vote shares by age band.

    Returns DataFrame: party, age_band, vote_share, sample_n
    """
    # Find the age dimension (try "Age (1)" which has 18-24, 25-49, 50-64, 65+)
    age_dim = _find_dimension(parsed, "Age (1)", "Age")
    if age_dim is None:
        raise KeyError("Cannot find age dimension in parsed data")

    vi_key = "constituency_vi" if use_constituency_vi else "headline_vi"
    return _to_long_format(age_dim, vi_key, "age_band")


def extract_nssec_vote_shares(parsed: dict, use_constituency_vi: bool = True) -> pd.DataFrame:
    """
    Extract vote shares by NS-SEC.

    Returns DataFrame: party, nssec, vote_share, sample_n
    """
    nssec_dim = _find_dimension(parsed, "NS-SEC", "Socio-economic", "NS-SEC (1)")
    if nssec_dim is None:
        raise KeyError("Cannot find NS-SEC dimension in parsed data")

    vi_key = "constituency_vi" if use_constituency_vi else "headline_vi"
    return _to_long_format(nssec_dim, vi_key, "nssec")


def extract_tenure_vote_shares(parsed: dict, use_constituency_vi: bool = True) -> pd.DataFrame:
    """
    Extract vote shares by tenure.

    Returns DataFrame: party, tenure, vote_share, sample_n
    """
    tenure_dim = _find_dimension(parsed, "Tenure", "House Tenure")
    if tenure_dim is None:
        raise KeyError("Cannot find tenure dimension in parsed data")

    vi_key = "constituency_vi" if use_constituency_vi else "headline_vi"
    return _to_long_format(tenure_dim, vi_key, "tenure")


def _find_dimension(parsed: dict, *keywords: str):
    """Find a dimension by keyword matching."""
    for dim_name, dim_data in parsed.items():
        for kw in keywords:
            if kw.lower() in dim_name.lower():
                return dim_data
    return None


def _to_long_format(dim_data: dict, vi_key: str, dim_col_name: str) -> pd.DataFrame:
    """Convert wide vote share table to long format."""
    vi = dim_data.get(vi_key)
    if vi is None:
        raise ValueError(f"No {vi_key} data available")

    sample_sizes = dim_data["sample_sizes"]

    rows = []
    for _, row in vi.iterrows():
        party_raw = row["party"]
        party = PARTY_NAME_MAP.get(party_raw, party_raw)

        for cat in dim_data["categories"]:
            if cat in row:
                rows.append({
                    "party": party,
                    dim_col_name: cat,
                    "vote_pct": row[cat],
                    "sample_n": sample_sizes.get(cat, 0),
                })

    return pd.DataFrame(rows)


# ── Convenience: extract all three dimensions ───────────────────────────


def extract_all_vote_shares(parsed: dict, use_constituency_vi: bool = True) -> dict:
    """
    Extract vote shares for all three dimensions.

    Returns dict with keys 'age', 'tenure', 'nssec', each a DataFrame.
    """
    return {
        "age": extract_age_vote_shares(parsed, use_constituency_vi),
        "tenure": extract_tenure_vote_shares(parsed, use_constituency_vi),
        "nssec": extract_nssec_vote_shares(parsed, use_constituency_vi),
    }


def convert_xlsx_to_csv(xlsx_path: str | Path, output_dir: str | Path = "data") -> dict[str, Path]:
    """
    Convert the YouGov xlsx to a set of CSV files for easier inspection.

    Returns dict mapping dimension name to CSV path.
    """
    parsed = parse_yougov_xlsx(xlsx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for dim_name, dim_data in parsed.items():
        safe_name = dim_name.replace(" ", "_").replace("/", "_").replace("×", "x")
        for vi_key in ("headline_vi", "constituency_vi"):
            vi = dim_data.get(vi_key)
            if vi is not None:
                fname = f"yougov_{safe_name}_{vi_key}.csv"
                path = output_dir / fname
                vi.to_csv(path, index=False)
                paths[f"{dim_name}_{vi_key}"] = path

    logger.info("Converted to %d CSV files in %s", len(paths), output_dir)
    return paths
