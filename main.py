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

    # Try to load OA-constituency lookup (may fail if ONS portal is unreachable)
    oa_lookup = None
    try:
        oa_lookup = oac_data.load_oa_constituency_lookup()
        logger.info("OA lookup: %d rows, columns: %s", len(oa_lookup), list(oa_lookup.columns))
    except (oac_data.LookupUnavailableError, RuntimeError) as exc:
        logger.warning("OA-constituency lookup unavailable: %s", exc)
        logger.warning("Will use national OAC covariance as fallback")

    # Inspect what we got
    logger.info("OAC assignments: %d OAs, columns: %s", len(assignments), list(assignments.columns))
    logger.info("OAC input: %d OAs × %d vars", *oa_input.shape)

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

    If the full OA-constituency lookup is available, uses per-constituency
    OAC cluster mixtures. Otherwise falls back to the national OAC covariance
    applied uniformly to all London constituencies.

    Returns:
        constituency_joints: dict {code: 3D array}
        constituency_names: dict {code: name}
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Building constituency joint distributions")
    logger.info("=" * 60)

    oa_lookup = oac["oa_lookup"]
    assignments = oac["assignments"]

    if oa_lookup is not None:
        return _build_joints_with_lookup(oac)
    else:
        return _build_joints_national_fallback(oac)


def _build_joints_with_lookup(oac: dict) -> tuple[dict, dict]:
    """Full approach: per-constituency OAC cluster mixtures using OA lookup."""
    oa_lookup = oac["oa_lookup"]
    assignments = oac["assignments"]
    oa_demographics = oac["oa_demographics"]

    london_constituencies = _identify_london_constituencies(oa_lookup)
    logger.info("Identified %d London constituencies", len(london_constituencies))

    cluster_level = _detect_cluster_column(assignments)
    cluster_joints = oac_data.compute_oac_cluster_covariance(
        oac["oa_input"], assignments, cluster_level
    )
    logger.info("Computed %d cluster joints", len(cluster_joints))

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

    # Also compute direct OA aggregation for comparison
    logger.info("Also computing direct OA-aggregated joints for comparison...")
    direct_joints = oac_data.compute_all_constituency_joints(
        oa_demographics, oa_lookup, list(london_constituencies.keys())
    )
    _save_joint_comparison(constituency_joints, direct_joints, constituency_names)

    return constituency_joints, constituency_names


def _build_joints_national_fallback(oac: dict) -> tuple[dict, dict]:
    """
    Fallback: use national OAC covariance for all London constituencies.

    When the OA-constituency lookup is unavailable, we:
    1. Compute the national joint distribution from all 239k OAs
    2. Load London constituency names from mysociety data
    3. Apply the national joint uniformly

    This means constituency variation comes entirely from the MRP model's
    demographic effects, not from constituency-specific demographic
    compositions. The covariance structure (how age/tenure/NS-SEC
    correlate) is still captured from the OAC data.
    """
    logger.info("Using NATIONAL FALLBACK: computing joint from all OAs")

    # Compute national joint
    national_joint = oac_data.compute_national_joint(oac["oa_input"])

    # Get London constituencies from mysociety
    try:
        london_df = oac_data.load_london_constituencies_from_mysociety()
    except Exception as exc:
        logger.error("Cannot load London constituencies: %s", exc)
        # Ultimate fallback: use hardcoded London constituency list
        london_df = _hardcoded_london_constituencies()

    constituency_joints = {}
    constituency_names = {}
    for _, row in london_df.iterrows():
        code = row["PCON_CD"]
        name = row["PCON_NM"]
        if pd.notna(code):
            constituency_joints[code] = national_joint.copy()
            constituency_names[code] = name

    logger.info(
        "Applied national joint to %d London constituencies (fallback mode)",
        len(constituency_joints),
    )

    # Save the national joint for inspection
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _save_national_joint(national_joint)

    return constituency_joints, constituency_names


def _save_national_joint(joint: np.ndarray):
    """Save the national joint distribution as a readable CSV."""
    rows = []
    for a_idx, age in enumerate(oac_data.AGE_CATEGORIES):
        for t_idx, tenure in enumerate(oac_data.TENURE_CATEGORIES):
            for n_idx, nssec in enumerate(oac_data.NSSEC_CATEGORIES):
                rows.append({
                    "age": age,
                    "tenure": tenure,
                    "nssec": nssec,
                    "proportion": joint[a_idx, t_idx, n_idx],
                })
    pd.DataFrame(rows).to_csv(DATA_DIR / "national_joint_distribution.csv", index=False)


def _hardcoded_london_constituencies() -> pd.DataFrame:
    """Hardcoded list of London 2025 constituencies as ultimate fallback."""
    constituencies = [
        ("E14001073", "Barking"), ("E14001081", "Battersea"),
        ("E14001082", "Beckenham and Penge"), ("E14001083", "Bermondsey and Old Southwark"),
        ("E14001084", "Bethnal Green and Stepney"), ("E14001086", "Bexleyheath and Crayford"),
        ("E14001094", "Brent East"), ("E14001095", "Brent West"),
        ("E14001098", "Bromley and Biggin Hill"), ("E14001103", "Camberwell and Peckham"),
        ("E14001106", "Carshalton and Wallington"), ("E14001110", "Chelsea and Fulham"),
        ("E14001113", "Chingford and Woodford Green"), ("E14001114", "Chipping Barnet"),
        ("E14001117", "Cities of London and Westminster"),
        ("E14001132", "Croydon East"), ("E14001133", "Croydon South"),
        ("E14001134", "Croydon West"), ("E14001136", "Dagenham and Rainham"),
        ("E14001141", "Dulwich and West Norwood"), ("E14001143", "Ealing Central and Acton"),
        ("E14001144", "Ealing North"), ("E14001145", "Ealing Southall"),
        ("E14001146", "East Ham"), ("E14001149", "Edmonton and Winchmore Hill"),
        ("E14001150", "Eltham and Chislehurst"), ("E14001151", "Enfield North"),
        ("E14001153", "Erith and Thamesmead"), ("E14001157", "Feltham and Heston"),
        ("E14001158", "Finchley and Golders Green"), ("E14001171", "Greenwich and Woolwich"),
        ("E14001173", "Hackney North and Stoke Newington"), ("E14001174", "Hackney South and Shoreditch"),
        ("E14001176", "Hammersmith and Chiswick"), ("E14001177", "Hampstead and Highgate"),
        ("E14001180", "Harrow East"), ("E14001181", "Harrow West"),
        ("E14001183", "Hayes and Harlington"), ("E14001185", "Hendon"),
        ("E14001190", "Holborn and St Pancras"), ("E14001191", "Hornchurch and Upminster"),
        ("E14001192", "Hornsey and Friern Barnet"), ("E14001194", "Hounslow West"),
        ("E14001196", "Ilford North"), ("E14001197", "Ilford South"),
        ("E14001198", "Islington North"), ("E14001199", "Islington South and Finsbury"),
        ("E14001206", "Kensington and Bayswater"), ("E14001207", "Kingston and Surbiton"),
        ("E14001215", "Lewisham East"), ("E14001216", "Lewisham North"),
        ("E14001217", "Lewisham West and East Dulwich"), ("E14001218", "Leyton and Wanstead"),
        ("E14001228", "Mitcham and Morden"), ("E14001250", "Old Bexley and Sidcup"),
        ("E14001251", "Orpington"), ("E14001261", "Peckham"),  # placeholder
        ("E14001263", "Poplar and Limehouse"), ("E14001266", "Putney"),
        ("E14001268", "Queen's Park and Maida Vale"), ("E14001275", "Richmond Park"),
        ("E14001278", "Romford"), ("E14001281", "Ruislip, Northwood and Pinner"),
        ("E14001311", "Stratford and Bow"), ("E14001312", "Streatham and Croydon North"),
        ("E14001318", "Sutton and Cheam"), ("E14001325", "Tooting"),
        ("E14001327", "Tottenham"), ("E14001332", "Twickenham"),
        ("E14001335", "Uxbridge and South Ruislip"), ("E14001337", "Vauxhall and Camberwell Green"),
        ("E14001339", "Walthamstow"), ("E14001345", "West Ham and Beckton"),
        ("E14001355", "Wimbledon"),
    ]
    return pd.DataFrame(constituencies, columns=["PCON_CD", "PCON_NM"])


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
