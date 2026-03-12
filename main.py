#!/usr/bin/env python3
"""
London Voting Intention Model — OAC-informed MRP pipeline.

Estimates London constituency-level voting intention by:
1. Downloading OAC 2021 data (output area demographics + cluster assignments)
2. Building constituency-level joint (age × tenure × NS-SEC) distributions
   that capture real covariance between dimensions via OAC clusters
3. Fitting a synthetic MRP model from YouGov marginal cross-tabs
4. Post-stratifying predicted vote shares onto each constituency's
   demographic structure

Usage
-----
    # Full pipeline
    python main.py

    # Skip OAC download (use cached data)
    python main.py --use-cache

    # Convert the YouGov xlsx to CSVs for inspection
    python main.py --convert-survey

    # Fetch OAC data only (no model run)
    python main.py --fetch-only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
import oac_data
import yougov_parser
import mrp_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SURVEY_PATH = "YouGov_Jan2026_v2.xlsx"
DATA_DIR = Path(config.DATA_DIR)


# ── Step 1: Load OAC data ──────────────────────────────────────────────


def load_oac() -> dict:
    """
    Download and load all OAC data sources.

    Returns dict with:
        assignments: OA → cluster mapping
        oa_input: raw demographic % per OA
        oa_lookup: OA → constituency mapping
        oa_demographics: recoded demographics per OA
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Loading OAC 2021 data")
    logger.info("=" * 60)

    assignments = oac_data.load_oac_assignments()
    oa_input = oac_data.load_oac_input_variables()
    oa_lookup = oac_data.load_oa_constituency_lookup()

    # Inspect what we got
    logger.info("OAC assignments: %d OAs, columns: %s", len(assignments), list(assignments.columns))
    logger.info("OAC input: %d OAs × %d vars", *oa_input.shape)
    logger.info("OA lookup: %d rows, columns: %s", len(oa_lookup), list(oa_lookup.columns))

    # Compute recoded demographics
    oa_demographics = oac_data.compute_oa_level_demographics(oa_input)
    logger.info("OA demographics computed: %d rows", len(oa_demographics))

    return {
        "assignments": assignments,
        "oa_input": oa_input,
        "oa_lookup": oa_lookup,
        "oa_demographics": oa_demographics,
    }


# ── Step 2: Build constituency joint distributions ─────────────────────


def build_constituency_joints(oac: dict) -> tuple[dict, dict]:
    """
    Build the joint (age × tenure × NS-SEC) distribution for each London
    constituency using the OAC cluster mixture approach.

    Returns:
        constituency_joints: dict {code: 3D array}
        constituency_names: dict {code: name}
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Building constituency joint distributions")
    logger.info("=" * 60)

    oa_lookup = oac["oa_lookup"]
    assignments = oac["assignments"]
    oa_demographics = oac["oa_demographics"]

    # Identify London constituencies
    london_constituencies = _identify_london_constituencies(oa_lookup)
    logger.info("Identified %d London constituencies", len(london_constituencies))

    # Method 1: OAC cluster mixture (captures covariance)
    logger.info("Computing OAC cluster-level joint distributions...")
    cluster_level = _detect_cluster_column(assignments)
    cluster_joints = oac_data.compute_oac_cluster_covariance(
        oac["oa_input"], assignments, cluster_level
    )
    logger.info("Computed %d cluster joints", len(cluster_joints))

    # Build constituency joints as weighted mixtures of cluster joints
    constituency_joints = {}
    constituency_names = {}
    for code, name in london_constituencies.items():
        seed = oac_data.build_constituency_seed_from_oac(
            assignments, oa_lookup, cluster_joints, code, cluster_level
        )
        if seed is not None:
            constituency_joints[code] = seed
            constituency_names[code] = name

    logger.info("Built joint distributions for %d London constituencies",
                len(constituency_joints))

    # Also compute using direct OA aggregation (Method 2, for comparison)
    logger.info("Also computing direct OA-aggregated joints for comparison...")
    direct_joints = oac_data.compute_all_constituency_joints(
        oa_demographics, oa_lookup, list(london_constituencies.keys())
    )

    # Save both for comparison
    _save_joint_comparison(constituency_joints, direct_joints, constituency_names)

    return constituency_joints, constituency_names


def _identify_london_constituencies(oa_lookup: pd.DataFrame) -> dict:
    """
    Identify London constituencies from the OA lookup.

    London boroughs have specific naming patterns in constituency names.
    """
    london_keywords = [
        "Barking", "Barnet", "Battersea", "Beckenham", "Bermondsey", "Bethnal",
        "Bexley", "Bow", "Brent", "Brixton", "Bromley", "Camberwell",
        "Camden", "Chelsea", "Chingford", "Chiswick", "Clapham", "Croydon",
        "Dagenham", "Deptford", "Dulwich", "Ealing", "East Ham", "Edmonton",
        "Eltham", "Enfield", "Erith", "Feltham", "Finchley", "Greenwich",
        "Hackney", "Hammersmith", "Hampstead", "Harrow", "Hayes", "Hendon",
        "Holborn", "Hornchurch", "Hornsey", "Hounslow", "Ilford", "Islington",
        "Kensington", "Kingston", "Lambeth", "Lewisham", "Leyton", "Mitcham",
        "Orpington", "Peckham", "Poplar", "Putney", "Richmond Park",
        "Romford", "Ruislip", "Southwark", "Stepney", "Stratford",
        "Streatham", "Sutton", "Tooting", "Tottenham", "Twickenham",
        "Uxbridge", "Vauxhall", "Walthamstow", "Wandsworth", "West Ham",
        "Westminster", "Wimbledon", "Woolwich", "Cities of London",
        "Queen's Park", "Carshalton", "Chipping Barnet",
    ]
    pattern = "|".join(london_keywords)

    # Get unique constituencies
    name_col = [c for c in oa_lookup.columns if "nm" in c.lower() or "name" in c.lower()]
    code_col = [c for c in oa_lookup.columns if c in ("PCON_CD", "PCON24CD", "PCON25CD")]

    if not name_col or not code_col:
        logger.warning("Cannot identify constituency columns. Columns: %s", list(oa_lookup.columns))
        # Try all columns
        for c in oa_lookup.columns:
            logger.info("  Column %s: sample values %s", c, list(oa_lookup[c].dropna().unique()[:3]))
        return {}

    name_col = name_col[0]
    code_col = code_col[0]

    constituencies = oa_lookup[[code_col, name_col]].drop_duplicates()
    london = constituencies[constituencies[name_col].str.contains(pattern, case=False, na=False)]

    return dict(zip(london[code_col], london[name_col]))


def _detect_cluster_column(assignments: pd.DataFrame) -> str:
    """Detect the cluster level column in OAC assignments."""
    for col in ["Subgroup", "subgroup", "Group", "group", "Supergroup", "supergroup"]:
        if col in assignments.columns:
            return col
    # Try partial matches
    for col in assignments.columns:
        if "sub" in col.lower():
            return col
    for col in assignments.columns:
        if "group" in col.lower():
            return col
    return assignments.columns[-1]  # last column as fallback


def _save_joint_comparison(
    oac_joints: dict, direct_joints: dict, names: dict,
):
    """Save a comparison of the two methods for diagnostics."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for code in oac_joints:
        if code not in direct_joints:
            continue
        oac_j = oac_joints[code]
        dir_j = direct_joints[code]

        # Normalise both
        oac_n = oac_j / oac_j.sum() if oac_j.sum() > 0 else oac_j
        dir_n = dir_j / dir_j.sum() if dir_j.sum() > 0 else dir_j

        # Compute correlation between the two
        corr = np.corrcoef(oac_n.flatten(), dir_n.flatten())[0, 1]
        # Max absolute difference
        max_diff = np.max(np.abs(oac_n - dir_n))

        rows.append({
            "constituency_code": code,
            "constituency_name": names.get(code, code),
            "correlation": corr,
            "max_abs_diff": max_diff,
        })

    if rows:
        df = pd.DataFrame(rows).sort_values("correlation")
        df.to_csv(DATA_DIR / "joint_method_comparison.csv", index=False)
        logger.info("Joint distribution method comparison saved. "
                     "Mean correlation: %.4f", df["correlation"].mean())


# ── Step 3: Parse YouGov and fit MRP model ──────────────────────────────


def fit_model(survey_path: str) -> tuple[dict, pd.DataFrame]:
    """
    Parse YouGov data and fit the synthetic MRP model.

    Returns:
        model_params: fitted model parameters
        cell_votes: predicted vote shares per demographic cell
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Parsing YouGov survey and fitting MRP model")
    logger.info("=" * 60)

    # Parse the cross-tabs
    parsed = yougov_parser.parse_yougov_xlsx(survey_path)

    # Extract vote shares by dimension
    vote_shares = yougov_parser.extract_all_vote_shares(parsed, use_constituency_vi=True)

    # Save parsed vote shares
    for dim, df in vote_shares.items():
        df.to_csv(DATA_DIR / f"yougov_{dim}_vote_shares.csv", index=False)

    # Fit the additive model
    model_params = mrp_model.fit_additive_model(
        age_shares=vote_shares["age"],
        tenure_shares=vote_shares["tenure"],
        nssec_shares=vote_shares["nssec"],
    )

    # Predict vote shares for all cells
    cell_votes = mrp_model.predict_cell_vote_shares(model_params)
    cell_votes.to_csv(DATA_DIR / "predicted_cell_votes.csv", index=False)
    logger.info("Predicted vote shares saved")

    # Log some diagnostics
    logger.info("Cell vote share ranges by party:")
    for party in mrp_model.PARTIES:
        party_cells = cell_votes[cell_votes["party"] == party]
        logger.info("  %s: %.1f%% — %.1f%%",
                     party,
                     party_cells["vote_share"].min() * 100,
                     party_cells["vote_share"].max() * 100)

    return model_params, cell_votes


# ── Step 4: Post-stratification ─────────────────────────────────────────


def run_poststratification(
    cell_votes: pd.DataFrame,
    constituency_joints: dict,
    constituency_names: dict,
) -> pd.DataFrame:
    """
    Post-stratify: project cell-level vote predictions onto constituency
    demographics.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Post-stratification")
    logger.info("=" * 60)

    results = mrp_model.poststratify_all(
        cell_votes, constituency_joints, constituency_names
    )

    # Save full results
    results.to_csv(DATA_DIR / "london_constituency_results.csv", index=False)

    # Summary
    summary = mrp_model.summarise_constituency_results(results)
    summary.to_csv(DATA_DIR / "london_constituency_summary.csv", index=False)

    # London aggregate
    london_agg = mrp_model.compute_london_aggregate(results)
    london_agg.to_csv(DATA_DIR / "london_aggregate.csv", index=False)

    return results


# ── Output ──────────────────────────────────────────────────────────────


def print_results(results: pd.DataFrame):
    """Print formatted results to stdout."""
    summary = mrp_model.summarise_constituency_results(results)
    london_agg = mrp_model.compute_london_aggregate(results)

    print("\n" + "=" * 70)
    print("LONDON AGGREGATE VOTE SHARES")
    print("=" * 70)
    for _, row in london_agg.iterrows():
        bar = "#" * int(row["vote_share"] * 100)
        print(f"  {row['party']:<20s} {row['vote_share']:6.1%}  {bar}")

    print("\n" + "=" * 70)
    print("CONSTITUENCY ESTIMATES (sorted by margin, tightest first)")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(
            f"  {row['constituency_name']:<45s} "
            f"{row['leading_party']:<18s} {row['lead_pct']:5.1f}%  "
            f"vs {row['second_party']:<18s} {row['second_pct']:5.1f}%  "
            f"(margin: {row['margin_pct']:+.1f}pp)"
        )

    print(f"\n  Constituencies modelled: {len(summary)}")
    print(f"  Results saved to {DATA_DIR}/")

    # Save diagnostics JSON
    diag = {
        "n_constituencies": len(summary),
        "london_aggregate": london_agg.to_dict("records"),
        "tightest_races": summary.head(5).to_dict("records"),
        "safest_seats": summary.tail(5).to_dict("records"),
    }
    with open(DATA_DIR / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2, default=str)


# ── Main pipeline ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="London Voting Intention Model — OAC-informed MRP pipeline"
    )
    parser.add_argument(
        "--survey", type=str, default=SURVEY_PATH,
        help="Path to YouGov survey xlsx (default: %(default)s)",
    )
    parser.add_argument(
        "--fetch-only", action="store_true",
        help="Only download OAC data, don't run model",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Use cached downloads where available",
    )
    parser.add_argument(
        "--convert-survey", action="store_true",
        help="Convert YouGov xlsx to CSVs and exit",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Convert mode
    if args.convert_survey:
        paths = yougov_parser.convert_xlsx_to_csv(args.survey, DATA_DIR)
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return

    # Step 1: Load OAC data
    oac = load_oac()

    if args.fetch_only:
        logger.info("Fetch-only mode. Data cached in %s", config.CACHE_DIR)
        return

    # Step 2: Build constituency joint distributions
    constituency_joints, constituency_names = build_constituency_joints(oac)

    if not constituency_joints:
        logger.error("No constituency joints computed. Check OA-constituency lookup.")
        sys.exit(1)

    # Step 3: Fit MRP model from YouGov data
    model_params, cell_votes = fit_model(args.survey)

    # Step 4: Post-stratification
    results = run_poststratification(cell_votes, constituency_joints, constituency_names)

    # Output
    print_results(results)


if __name__ == "__main__":
    main()
