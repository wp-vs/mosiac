"""
Synthetic MRP (Multilevel Regression and Post-stratification) model.

Since we have marginal cross-tabs from YouGov (vote share by age, by tenure,
by NS-SEC separately) rather than individual-level data, we build a synthetic
joint model using a log-linear approach:

    log P(party | age, tenure, nssec) ∝ α_party + β_{party,age} + γ_{party,tenure} + δ_{party,nssec}

This is fitted from the marginal tables. The key challenge is estimating the
joint from marginals — this is where the OAC data helps by providing:
1. An informed seed for the interaction structure
2. Validation of the resulting joint distributions

The post-stratification step then projects these vote probabilities onto
each constituency's demographic structure (from the OAC-informed joint
distributions) to produce constituency-level vote estimates.
"""

import logging

import numpy as np
import pandas as pd
from scipy.special import softmax

import oac_data

logger = logging.getLogger(__name__)


# ── Category definitions (must match oac_data dimensions) ───────────────

AGE_CATS = oac_data.AGE_CATEGORIES       # ["18-24", "25-49", "50-64", "65+"]
TENURE_CATS = oac_data.TENURE_CATEGORIES  # ["Owned", "Social rent", "Private rent"]
NSSEC_CATS = oac_data.NSSEC_CATEGORIES    # ["AB", "C1", "C2", "DE"]

# Parties to model (London-relevant)
PARTIES = ["Conservative", "Labour", "Liberal Democrat", "Reform UK", "Green", "Other"]


# ── YouGov category mapping ────────────────────────────────────────────
# Map YouGov category labels to our standardised labels.

AGE_MAP = {
    "18-24": "18-24",
    "25-49": "25-49",
    "50-64": "50-64",
    "65+": "65+",
}

TENURE_MAP = {
    "Own outright": "Owned",
    "Own with mortgage": "Owned",
    "Rent from private landlord": "Private rent",
    "Rent from LA / housing association": "Social rent",
    "Live with parents/family/friends": "Private rent",  # approximate
}

NSSEC_MAP = {
    "Large employers & higher managerial": "AB",
    "Higher professional": "AB",
    "Lower managerial, admin & professional": "AB",
    "Intermediate occupations": "C1",
    "Small employers & own account workers": "C1",
    "Lower supervisory & technical": "C2",
    "Semi-routine occupations": "DE",
    "Routine occupations": "DE",
    # Simplified NS-SEC (2) from sheet 4
    "Higher managerial, admin & professional": "AB",
    "Routine & manual occupations": "DE",
}


# ── Model fitting ───────────────────────────────────────────────────────


def fit_additive_model(
    age_shares: pd.DataFrame,
    tenure_shares: pd.DataFrame,
    nssec_shares: pd.DataFrame,
) -> dict:
    """
    Fit an additive log-linear model from marginal vote share tables.

    For each party p and demographic cell (a, t, n):
        η(p, a, t, n) = α_p + β_{p,a} + γ_{p,t} + δ_{p,n}

    The vote probability is then:
        P(p | a, t, n) = exp(η(p, a, t, n)) / Σ_p' exp(η(p', a, t, n))

    Parameters
    ----------
    age_shares : DataFrame with columns: party, age_band, vote_pct, sample_n
    tenure_shares : DataFrame with columns: party, tenure, vote_pct, sample_n
    nssec_shares : DataFrame with columns: party, nssec, vote_pct, sample_n

    Returns
    -------
    dict with keys:
        'alpha': dict {party: float} — party intercepts
        'beta_age': dict {(party, age): float} — age effects
        'gamma_tenure': dict {(party, tenure): float} — tenure effects
        'delta_nssec': dict {(party, nssec): float} — NS-SEC effects
    """
    # Extract marginal vote shares, mapping to our categories
    age_vs = _aggregate_vote_shares(age_shares, "age_band", AGE_MAP)
    tenure_vs = _aggregate_vote_shares(tenure_shares, "tenure", TENURE_MAP)
    nssec_vs = _aggregate_vote_shares(nssec_shares, "nssec", NSSEC_MAP)

    # Compute log-odds relative to a reference category
    # Use the overall mean as the reference for each dimension
    alpha = {}
    beta_age = {}
    gamma_tenure = {}
    delta_nssec = {}

    for party in PARTIES:
        # Overall mean vote share for this party (from age marginal, weighted by sample)
        party_age = age_vs[age_vs["party"] == party]
        if party_age.empty:
            alpha[party] = -5.0  # very low baseline
            for cat in AGE_CATS:
                beta_age[(party, cat)] = 0.0
            for cat in TENURE_CATS:
                gamma_tenure[(party, cat)] = 0.0
            for cat in NSSEC_CATS:
                delta_nssec[(party, cat)] = 0.0
            continue

        # Use sample-weighted mean across age groups as baseline
        total_n = party_age["sample_n"].sum()
        mean_pct = (party_age["vote_pct"] * party_age["sample_n"]).sum() / total_n
        mean_pct = np.clip(mean_pct / 100.0, 0.001, 0.999)
        alpha[party] = np.log(mean_pct)

        # Age effects (deviation from mean on log scale)
        for _, row in party_age.iterrows():
            cat = row["category"]
            pct = np.clip(row["vote_pct"] / 100.0, 0.001, 0.999)
            beta_age[(party, cat)] = np.log(pct) - alpha[party]

        # Tenure effects
        party_tenure = tenure_vs[tenure_vs["party"] == party]
        tenure_mean_pct = mean_pct  # use same baseline
        for _, row in party_tenure.iterrows():
            cat = row["category"]
            pct = np.clip(row["vote_pct"] / 100.0, 0.001, 0.999)
            gamma_tenure[(party, cat)] = np.log(pct) - np.log(tenure_mean_pct)

        # NS-SEC effects
        party_nssec = nssec_vs[nssec_vs["party"] == party]
        for _, row in party_nssec.iterrows():
            cat = row["category"]
            pct = np.clip(row["vote_pct"] / 100.0, 0.001, 0.999)
            delta_nssec[(party, cat)] = np.log(pct) - np.log(mean_pct)

    # Fill missing combinations with 0 effect
    for party in PARTIES:
        for cat in AGE_CATS:
            beta_age.setdefault((party, cat), 0.0)
        for cat in TENURE_CATS:
            gamma_tenure.setdefault((party, cat), 0.0)
        for cat in NSSEC_CATS:
            delta_nssec.setdefault((party, cat), 0.0)

    params = {
        "alpha": alpha,
        "beta_age": beta_age,
        "gamma_tenure": gamma_tenure,
        "delta_nssec": delta_nssec,
    }

    logger.info("Fitted additive model with %d parties", len(PARTIES))
    return params


def predict_cell_vote_shares(params: dict) -> pd.DataFrame:
    """
    Predict vote shares for every (age × tenure × NS-SEC) cell.

    Uses softmax over parties within each cell to ensure shares sum to 1.

    Returns DataFrame: age, tenure, nssec, party, vote_share
    """
    alpha = params["alpha"]
    beta = params["beta_age"]
    gamma = params["gamma_tenure"]
    delta = params["delta_nssec"]

    rows = []
    for a_idx, age in enumerate(AGE_CATS):
        for t_idx, tenure in enumerate(TENURE_CATS):
            for n_idx, nssec in enumerate(NSSEC_CATS):
                # Compute log-linear predictor for each party
                eta = np.array([
                    alpha.get(p, -5) +
                    beta.get((p, age), 0) +
                    gamma.get((p, tenure), 0) +
                    delta.get((p, nssec), 0)
                    for p in PARTIES
                ])

                # Softmax to get probabilities
                probs = softmax(eta)

                for p_idx, party in enumerate(PARTIES):
                    rows.append({
                        "age": age,
                        "tenure": tenure,
                        "nssec": nssec,
                        "party": party,
                        "vote_share": probs[p_idx],
                    })

    df = pd.DataFrame(rows)
    logger.info("Predicted vote shares for %d cells × %d parties = %d rows",
                len(AGE_CATS) * len(TENURE_CATS) * len(NSSEC_CATS),
                len(PARTIES), len(df))
    return df


# ── Post-stratification ────────────────────────────────────────────────


def poststratify_constituency(
    cell_votes: pd.DataFrame,
    constituency_joint: np.ndarray,
    constituency_code: str,
    constituency_name: str = "",
) -> pd.DataFrame:
    """
    Post-stratify: weight cell-level vote shares by the constituency's
    demographic composition to produce constituency vote estimates.

    Parameters
    ----------
    cell_votes : DataFrame from predict_cell_vote_shares()
    constituency_joint : 3D array (age × tenure × nssec) from OAC data
    constituency_code : GSS code
    constituency_name : human-readable name

    Returns
    -------
    DataFrame: constituency_code, constituency_name, party, vote_share, effective_n
    """
    # Normalise joint to proportions
    total = constituency_joint.sum()
    if total <= 0:
        return pd.DataFrame()
    joint_props = constituency_joint / total

    party_votes = {p: 0.0 for p in PARTIES}

    for a_idx, age in enumerate(AGE_CATS):
        for t_idx, tenure in enumerate(TENURE_CATS):
            for n_idx, nssec in enumerate(NSSEC_CATS):
                cell_weight = joint_props[a_idx, t_idx, n_idx]

                # Look up vote shares for this cell
                cell_mask = (
                    (cell_votes["age"] == age) &
                    (cell_votes["tenure"] == tenure) &
                    (cell_votes["nssec"] == nssec)
                )
                cell_data = cell_votes[cell_mask]

                for _, row in cell_data.iterrows():
                    party_votes[row["party"]] += cell_weight * row["vote_share"]

    rows = [
        {
            "constituency_code": constituency_code,
            "constituency_name": constituency_name,
            "party": party,
            "vote_share": share,
            "effective_n": total,
        }
        for party, share in party_votes.items()
    ]

    return pd.DataFrame(rows)


def poststratify_all(
    cell_votes: pd.DataFrame,
    constituency_joints: dict[str, np.ndarray],
    constituency_names: dict[str, str],
) -> pd.DataFrame:
    """
    Post-stratify for all constituencies.

    Returns combined DataFrame of vote estimates.
    """
    results = []
    for code, joint in constituency_joints.items():
        name = constituency_names.get(code, code)
        result = poststratify_constituency(cell_votes, joint, code, name)
        results.append(result)

    combined = pd.concat(results, ignore_index=True)
    combined = combined.sort_values(["constituency_code", "vote_share"], ascending=[True, False])
    logger.info("Post-stratified %d constituencies", len(constituency_joints))
    return combined


# ── Summary and diagnostics ─────────────────────────────────────────────


def summarise_constituency_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Produce summary: leading party per constituency + margin.
    """
    rows = []
    for code, group in results.groupby("constituency_code"):
        top = group.nlargest(2, "vote_share")
        leader = top.iloc[0]
        runner = top.iloc[1] if len(top) > 1 else pd.Series({"party": "N/A", "vote_share": 0})

        rows.append({
            "constituency_code": code,
            "constituency_name": leader.get("constituency_name", code),
            "leading_party": leader["party"],
            "lead_pct": leader["vote_share"] * 100,
            "second_party": runner["party"],
            "second_pct": runner["vote_share"] * 100,
            "margin_pct": (leader["vote_share"] - runner["vote_share"]) * 100,
        })

    return pd.DataFrame(rows).sort_values("margin_pct")


def compute_london_aggregate(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate London-wide vote shares by weighting constituencies
    by their effective population.
    """
    weighted = results.copy()
    weighted["weighted_vote"] = weighted["vote_share"] * weighted["effective_n"]

    agg = weighted.groupby("party", as_index=False).agg(
        total_weighted_vote=("weighted_vote", "sum"),
        total_n=("effective_n", "sum"),
    )
    agg["vote_share"] = agg["total_weighted_vote"] / agg["total_n"]
    return agg[["party", "vote_share"]].sort_values("vote_share", ascending=False)


# ── Helpers ─────────────────────────────────────────────────────────────


def _aggregate_vote_shares(
    df: pd.DataFrame,
    dim_col: str,
    cat_map: dict,
) -> pd.DataFrame:
    """
    Map raw YouGov categories to our standardised categories and
    aggregate (sample-weighted mean) where multiple raw categories
    map to the same target.
    """
    df = df.copy()
    df["category"] = df[dim_col].map(cat_map)

    # Drop unmapped
    df = df.dropna(subset=["category"])

    # Filter to our target parties
    df = df[df["party"].isin(PARTIES)]

    # Weighted aggregation where multiple categories map to same target
    result = (
        df.groupby(["party", "category"], as_index=False)
        .apply(lambda g: pd.Series({
            "vote_pct": np.average(g["vote_pct"], weights=g["sample_n"]),
            "sample_n": g["sample_n"].sum(),
        }), include_groups=False)
    )

    return result
