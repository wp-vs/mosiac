"""
MRP-lite voting intention model.

Takes the raked demographic cell table (from raking.py) and the survey-derived
vote shares by demographic cell (from survey_data.py), and produces constituency-
level vote intention estimates for London.

The approach:
1. For each constituency, we have a full (age × tenure × NS-SEC) cell table
   with estimated counts from the raking step.
2. For each cell, we look up the national vote share from the survey.
3. The constituency vote estimate = weighted average of cell-level vote shares,
   where weights are the cell populations.

This is the "post-stratification" step of MRP.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_constituency_votes(
    raked_cells: pd.DataFrame,
    vote_shares: pd.DataFrame,
    constituency_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Estimate vote intention per constituency by post-stratifying the survey
    vote shares onto the raked demographic structure.

    Parameters
    ----------
    raked_cells : pd.DataFrame
        Output of raking. Columns: geography_code, age_band, tenure, nssec,
        count, proportion.
    vote_shares : pd.DataFrame
        Output of compute_vote_shares(). Columns: age_band, tenure, nssec,
        party, vote_share.
    constituency_names : dict, optional
        Mapping from geography_code to human-readable name.

    Returns
    -------
    pd.DataFrame
        Columns: geography_code, geography_name, party, vote_share, vote_count
        Sorted by geography and party.
    """
    # Merge raked cells with vote shares on demographic dimensions
    merged = raked_cells.merge(
        vote_shares,
        on=["age_band", "tenure", "nssec"],
        how="left",
    )

    # Where we have no survey data for a cell, vote_share is NaN.
    # Fill with 0 — these cells contribute no votes.
    n_missing = merged["vote_share"].isna().sum()
    if n_missing > 0:
        n_total = len(merged)
        pct = 100 * n_missing / n_total
        logger.warning(
            "%d / %d cell×party combinations (%.1f%%) had no survey match",
            n_missing, n_total, pct,
        )
    merged["vote_share"] = merged["vote_share"].fillna(0)

    # Weighted vote count per cell
    merged["vote_count"] = merged["count"] * merged["vote_share"]

    # Aggregate by constituency and party
    results = (
        merged.groupby(["geography_code", "party"], as_index=False)
        .agg(vote_count=("vote_count", "sum"), total_pop=("count", "sum"))
    )
    results["vote_share"] = results["vote_count"] / results["total_pop"]

    # Add constituency names
    if constituency_names:
        results["geography_name"] = results["geography_code"].map(constituency_names)
    else:
        results["geography_name"] = results["geography_code"]

    # Sort
    results = results.sort_values(["geography_code", "vote_share"], ascending=[True, False])

    return results[["geography_code", "geography_name", "party", "vote_share", "vote_count", "total_pop"]]


def summarise_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a summary table: leading party per constituency + top-2 margin.

    Returns DataFrame:
        geography_code, geography_name, leading_party, lead_share,
        second_party, second_share, margin
    """
    rows = []
    for geo_code, group in results.groupby("geography_code"):
        # Filter to actual parties (exclude undecided, refused, etc.)
        parties = group[~group["party"].isin(["undecided", "would_not_vote", "refused", "unknown"])]
        if parties.empty:
            continue

        top = parties.nlargest(2, "vote_share")
        leader = top.iloc[0]
        runner = top.iloc[1] if len(top) > 1 else pd.Series({"party": "N/A", "vote_share": 0})

        rows.append({
            "geography_code": geo_code,
            "geography_name": leader.get("geography_name", geo_code),
            "leading_party": leader["party"],
            "lead_share": leader["vote_share"],
            "second_party": runner["party"],
            "second_share": runner["vote_share"],
            "margin": leader["vote_share"] - runner["vote_share"],
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("margin", ascending=True)  # tightest races first
    return summary


def diagnostics(
    raked_cells: pd.DataFrame,
    vote_shares: pd.DataFrame,
    results: pd.DataFrame,
) -> dict:
    """
    Compute diagnostic statistics about the model.

    Returns a dict with:
        - n_constituencies: number of constituencies modelled
        - n_cells_total: total demographic cells
        - n_cells_matched: cells with survey data
        - coverage_pct: percentage of cells matched
        - parties: list of parties in the results
        - london_aggregate: aggregate London vote shares
    """
    merged = raked_cells.merge(
        vote_shares, on=["age_band", "tenure", "nssec"], how="left"
    )
    n_total = len(merged)
    n_matched = merged["vote_share"].notna().sum()

    # Aggregate London-wide vote shares
    parties_df = results[~results["party"].isin(["undecided", "would_not_vote", "refused"])]
    london_agg = (
        parties_df.groupby("party", as_index=False)["vote_count"].sum()
    )
    total_votes = london_agg["vote_count"].sum()
    london_agg["vote_share"] = london_agg["vote_count"] / total_votes
    london_agg = london_agg.sort_values("vote_share", ascending=False)

    return {
        "n_constituencies": results["geography_code"].nunique(),
        "n_cells_total": n_total,
        "n_cells_matched": n_matched,
        "coverage_pct": 100 * n_matched / n_total if n_total > 0 else 0,
        "parties": list(london_agg["party"]),
        "london_aggregate": london_agg[["party", "vote_share"]].to_dict("records"),
    }
