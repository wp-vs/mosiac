#!/usr/bin/env python3
"""
London Voting Intention Model — MRP-lite pipeline.

Estimates London constituency-level voting intention by:
1. Fetching Census 2021 marginals (age, tenure, NS-SEC) from the Nomis API
2. Using IPF/raking to estimate the joint demographic distribution per constituency
3. Post-stratifying national YouGov survey vote shares onto those demographics

Usage
-----
    # Full pipeline (requires YouGov survey data)
    python main.py --survey data/yougov_survey.xlsx

    # Fetch census data only (no survey needed)
    python main.py --fetch-only

    # Use previously cached census data
    python main.py --survey data/yougov_survey.csv --use-cache

    # Convert xlsx survey to csv
    python main.py --convert data/yougov_survey.xlsx
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

import config
import nomis_api
import census_data
import raking
import survey_data
import model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_census_data(use_cache: bool = True) -> dict:
    """
    Step 1: Fetch Census 2021 marginals from Nomis.

    Returns a dict with processed DataFrames:
        age, tenure, nssec, tenure_nssec, constituencies
    """
    logger.info("=== Step 1: Fetching Census 2021 data from Nomis ===")

    # Get constituency list
    constituencies = nomis_api.get_constituency_geographies()
    london = nomis_api.filter_london_constituencies(constituencies)
    logger.info("Total constituencies: %d, London: %d", len(constituencies), len(london))

    # Fetch the four tables
    raw_age = nomis_api.fetch_age_by_constituency()
    raw_tenure = nomis_api.fetch_tenure_by_constituency()
    raw_nssec = nomis_api.fetch_nssec_by_constituency()

    # RM138 cross-tab (optional but improves the seed)
    try:
        raw_tenure_nssec = nomis_api.fetch_tenure_nssec_by_constituency()
    except Exception as exc:
        logger.warning("Could not fetch RM138 cross-tab: %s. Proceeding without it.", exc)
        raw_tenure_nssec = None

    # Process into standardised marginals
    age_df = census_data.process_age_marginal(raw_age)
    tenure_df = census_data.process_tenure_marginal(raw_tenure)
    nssec_df = census_data.process_nssec_marginal(raw_nssec)

    tenure_nssec_df = None
    if raw_tenure_nssec is not None:
        try:
            tenure_nssec_df = census_data.process_tenure_nssec_crosstab(raw_tenure_nssec)
        except Exception as exc:
            logger.warning("Could not process RM138: %s", exc)

    # Filter to London
    london_codes = set(london["geography_code"])
    age_df = age_df[age_df["geography_code"].isin(london_codes)]
    tenure_df = tenure_df[tenure_df["geography_code"].isin(london_codes)]
    nssec_df = nssec_df[nssec_df["geography_code"].isin(london_codes)]
    if tenure_nssec_df is not None:
        tenure_nssec_df = tenure_nssec_df[tenure_nssec_df["geography_code"].isin(london_codes)]

    logger.info(
        "Census data loaded — age: %d rows, tenure: %d rows, nssec: %d rows",
        len(age_df), len(tenure_df), len(nssec_df),
    )

    return {
        "age": age_df,
        "tenure": tenure_df,
        "nssec": nssec_df,
        "tenure_nssec": tenure_nssec_df,
        "constituencies": london,
    }


def run_raking(census: dict) -> pd.DataFrame:
    """
    Step 2: Run IPF/raking to estimate joint distributions.
    """
    logger.info("=== Step 2: Raking — estimating joint distributions ===")

    # Determine the categories actually present in the data
    age_cats = sorted(census["age"]["age_band"].unique())
    tenure_cats = sorted(census["tenure"]["tenure"].unique())
    nssec_cats = sorted(census["nssec"]["nssec"].unique())

    # Remove non-model categories
    age_cats = [c for c in age_cats if c not in ("under_18", "unknown")]
    tenure_cats = [c for c in tenure_cats if c != "unknown"]
    nssec_cats = [c for c in nssec_cats if c not in ("unknown", "unclassified")]

    logger.info("Categories — age: %s, tenure: %s, nssec: %s", age_cats, tenure_cats, nssec_cats)

    constituency_codes = census["constituencies"]["geography_code"].tolist()

    raked = raking.rake_all_constituencies(
        age_df=census["age"],
        tenure_df=census["tenure"],
        nssec_df=census["nssec"],
        constituency_codes=constituency_codes,
        age_categories=age_cats,
        tenure_categories=tenure_cats,
        nssec_categories=nssec_cats,
        tenure_nssec_df=census.get("tenure_nssec"),
    )

    # Save intermediate output
    out_path = Path(config.DATA_DIR) / "raked_cells.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raked.to_csv(out_path, index=False)
    logger.info("Saved raked cell table to %s (%d rows)", out_path, len(raked))

    return raked


def run_model(
    raked_cells: pd.DataFrame,
    survey_path: str,
    census: dict,
    vote_col: str = "vote_intention",
    age_col: str = "age",
    tenure_col: str = "tenure",
    nssec_col: str = "nssec",
    weight_col: str | None = "weight",
) -> pd.DataFrame:
    """
    Step 3: Post-stratify survey data onto raked demographics.
    """
    logger.info("=== Step 3: Loading survey and running model ===")

    # Load survey
    raw_survey = survey_data.load_survey(survey_path)

    # Extract vote × demographics
    vote_demo = survey_data.extract_vote_by_demographics(
        raw_survey,
        vote_col=vote_col,
        age_col=age_col,
        tenure_col=tenure_col,
        nssec_col=nssec_col,
        weight_col=weight_col,
    )

    # Compute vote shares per cell
    vote_shares = survey_data.compute_vote_shares(vote_demo)

    # Save vote shares for inspection
    out_path = Path(config.DATA_DIR) / "vote_shares_by_cell.csv"
    vote_shares.to_csv(out_path, index=False)
    logger.info("Saved vote shares to %s", out_path)

    # Build constituency name lookup
    name_map = dict(zip(
        census["constituencies"]["geography_code"],
        census["constituencies"]["geography_name"],
    ))

    # Post-stratify
    results = model.estimate_constituency_votes(raked_cells, vote_shares, name_map)

    # Save results
    out_path = Path(config.DATA_DIR) / "constituency_results.csv"
    results.to_csv(out_path, index=False)
    logger.info("Saved constituency results to %s", out_path)

    # Summary
    summary = model.summarise_results(results)
    out_path = Path(config.DATA_DIR) / "constituency_summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Saved summary to %s", out_path)

    # Diagnostics
    diag = model.diagnostics(raked_cells, vote_shares, results)
    logger.info("=== Model Diagnostics ===")
    logger.info("Constituencies modelled: %d", diag["n_constituencies"])
    logger.info("Cell coverage: %.1f%%", diag["coverage_pct"])
    logger.info("London aggregate vote shares:")
    for row in diag["london_aggregate"]:
        logger.info("  %s: %.1f%%", row["party"], 100 * row["vote_share"])

    out_path = Path(config.DATA_DIR) / "diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(diag, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="London Voting Intention Model — MRP-lite pipeline"
    )
    parser.add_argument(
        "--survey", type=str,
        help="Path to YouGov survey data (xlsx or csv)",
    )
    parser.add_argument(
        "--fetch-only", action="store_true",
        help="Only fetch and process census data (no model run)",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Use cached census data if available",
    )
    parser.add_argument(
        "--convert", type=str,
        help="Convert an xlsx file to csv and exit",
    )
    parser.add_argument(
        "--vote-col", type=str, default="vote_intention",
        help="Survey column name for vote intention",
    )
    parser.add_argument(
        "--age-col", type=str, default="age",
        help="Survey column name for age",
    )
    parser.add_argument(
        "--tenure-col", type=str, default="tenure",
        help="Survey column name for tenure",
    )
    parser.add_argument(
        "--nssec-col", type=str, default="nssec",
        help="Survey column name for NS-SEC / social grade",
    )
    parser.add_argument(
        "--weight-col", type=str, default="weight",
        help="Survey column name for weights (pass 'none' to disable)",
    )

    args = parser.parse_args()

    # Convert mode
    if args.convert:
        csv_path = survey_data.convert_xlsx_to_csv(args.convert)
        print(f"Converted to: {csv_path}")
        return

    # Fetch census data
    census = fetch_census_data(use_cache=args.use_cache)

    if args.fetch_only:
        logger.info("Fetch-only mode. Saving processed marginals.")
        for key in ("age", "tenure", "nssec"):
            out = Path(config.DATA_DIR) / f"marginal_{key}.csv"
            census[key].to_csv(out, index=False)
            logger.info("Saved %s", out)
        return

    # Need survey data for the full model
    if not args.survey:
        logger.error("No survey data provided. Use --survey <path> or --fetch-only.")
        sys.exit(1)

    # Run raking
    raked_cells = run_raking(census)

    # Run model
    weight_col = None if args.weight_col.lower() == "none" else args.weight_col
    results = run_model(
        raked_cells=raked_cells,
        survey_path=args.survey,
        census=census,
        vote_col=args.vote_col,
        age_col=args.age_col,
        tenure_col=args.tenure_col,
        nssec_col=args.nssec_col,
        weight_col=weight_col,
    )

    # Print top-line results
    print("\n" + "=" * 70)
    print("LONDON CONSTITUENCY VOTING INTENTION ESTIMATES")
    print("=" * 70)

    summary = model.summarise_results(results)
    for _, row in summary.iterrows():
        print(
            f"  {row['geography_name']:<40s} "
            f"{row['leading_party']:<20s} {row['lead_share']:5.1%}  "
            f"(margin: {row['margin']:+.1%})"
        )

    print(f"\nFull results saved to {config.DATA_DIR}/")


if __name__ == "__main__":
    main()
