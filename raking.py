"""
Raking / Iterative Proportional Fitting (IPF) to estimate the joint
distribution of (age × tenure × NS-SEC) from marginal totals.

Uses the `ipfn` package. The output is a full demographic cell table per
constituency that can then be used to weight vote intentions.
"""

import logging
from itertools import product

import numpy as np
import pandas as pd
from ipfn import ipfn

logger = logging.getLogger(__name__)


def build_seed_matrix(
    age_categories: list[str],
    tenure_categories: list[str],
    nssec_categories: list[str],
) -> np.ndarray:
    """
    Create a uniform seed matrix for IPF.

    The seed represents our initial "guess" at the joint distribution before
    raking to match marginals. A uniform seed (all ones) is the maximum-
    entropy starting point, which is standard practice when no prior
    cross-tabulation is available.

    Shape: (n_age, n_tenure, n_nssec)
    """
    n_age = len(age_categories)
    n_tenure = len(tenure_categories)
    n_nssec = len(nssec_categories)
    return np.ones((n_age, n_tenure, n_nssec))


def build_seed_from_crosstab(
    age_categories: list[str],
    tenure_categories: list[str],
    nssec_categories: list[str],
    tenure_nssec_crosstab: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """
    Build a seed matrix informed by the tenure × NS-SEC cross-tabulation
    from RM138. This is better than uniform because we incorporate known
    structure from the 2D cross-tab.

    For the age dimension we remain uniform (spread evenly), but the
    tenure × NS-SEC slice is seeded from actual data.

    Parameters
    ----------
    tenure_nssec_crosstab : dict mapping (tenure, nssec) -> count
        The observed 2D cross-tabulation. If None, falls back to uniform.
    """
    n_age = len(age_categories)
    n_tenure = len(tenure_categories)
    n_nssec = len(nssec_categories)

    if tenure_nssec_crosstab is None:
        return build_seed_matrix(age_categories, tenure_categories, nssec_categories)

    # Build 2D tenure × nssec matrix
    tn_matrix = np.ones((n_tenure, n_nssec))
    for i, t in enumerate(tenure_categories):
        for j, n in enumerate(nssec_categories):
            tn_matrix[i, j] = tenure_nssec_crosstab.get((t, n), 1.0)

    # Expand to 3D by replicating across age (uniform in that dimension)
    seed = np.zeros((n_age, n_tenure, n_nssec))
    for a in range(n_age):
        seed[a, :, :] = tn_matrix / n_age

    # Ensure no zeros (IPF needs positive values)
    seed = np.maximum(seed, 0.01)

    return seed


def rake_constituency(
    age_marginal: dict[str, float],
    tenure_marginal: dict[str, float],
    nssec_marginal: dict[str, float],
    age_categories: list[str],
    tenure_categories: list[str],
    nssec_categories: list[str],
    tenure_nssec_crosstab: dict[tuple[str, str], float] | None = None,
    max_iterations: int = 500,
    convergence_rate: float = 1e-6,
) -> pd.DataFrame:
    """
    Use IPF (raking) to estimate the joint distribution of
    (age × tenure × NS-SEC) for a single constituency.

    Parameters
    ----------
    age_marginal : dict
        {age_band: count} for this constituency.
    tenure_marginal : dict
        {tenure: count} for this constituency.
    nssec_marginal : dict
        {nssec: count} for this constituency.
    age_categories, tenure_categories, nssec_categories : list
        Ordered category labels for each dimension.
    tenure_nssec_crosstab : dict, optional
        Known 2D cross-tab from RM138, as {(tenure, nssec): count}.
    max_iterations : int
        Maximum IPF iterations.
    convergence_rate : float
        Convergence threshold.

    Returns
    -------
    pd.DataFrame
        Columns: age_band, tenure, nssec, count, proportion
        One row per demographic cell.
    """
    # Build marginal arrays in the same order as categories
    age_target = np.array([age_marginal.get(c, 0) for c in age_categories], dtype=float)
    tenure_target = np.array([tenure_marginal.get(c, 0) for c in tenure_categories], dtype=float)
    nssec_target = np.array([nssec_marginal.get(c, 0) for c in nssec_categories], dtype=float)

    # Ensure marginals are positive (replace zeros with small value)
    for arr in [age_target, tenure_target, nssec_target]:
        arr[arr <= 0] = 0.01

    # Build seed
    seed = build_seed_from_crosstab(
        age_categories, tenure_categories, nssec_categories, tenure_nssec_crosstab
    )

    # Run IPF
    aggregates = [age_target, tenure_target, nssec_target]
    dimensions = [[0], [1], [2]]

    try:
        IPF = ipfn.ipfn(
            seed,
            aggregates,
            dimensions,
            convergence_rate=convergence_rate,
            max_iteration=max_iterations,
        )
        result_matrix = IPF.iteration()
    except Exception as exc:
        logger.error("IPF failed: %s. Falling back to outer-product estimate.", exc)
        result_matrix = _outer_product_fallback(age_target, tenure_target, nssec_target)

    # Convert to DataFrame
    total = result_matrix.sum()
    rows = []
    for i, age in enumerate(age_categories):
        for j, tenure in enumerate(tenure_categories):
            for k, nssec in enumerate(nssec_categories):
                count = result_matrix[i, j, k]
                rows.append({
                    "age_band": age,
                    "tenure": tenure,
                    "nssec": nssec,
                    "count": count,
                    "proportion": count / total if total > 0 else 0,
                })

    return pd.DataFrame(rows)


def _outer_product_fallback(
    age: np.ndarray, tenure: np.ndarray, nssec: np.ndarray
) -> np.ndarray:
    """
    Fallback: assume independence between dimensions and construct
    the joint distribution as the normalised outer product.
    """
    age_p = age / age.sum()
    tenure_p = tenure / tenure.sum()
    nssec_p = nssec / nssec.sum()
    total = age.sum()  # use age total as the overall count
    return total * np.einsum("i,j,k->ijk", age_p, tenure_p, nssec_p)


def rake_all_constituencies(
    age_df: pd.DataFrame,
    tenure_df: pd.DataFrame,
    nssec_df: pd.DataFrame,
    constituency_codes: list[str],
    age_categories: list[str],
    tenure_categories: list[str],
    nssec_categories: list[str],
    tenure_nssec_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run raking for all constituencies and return a combined DataFrame.

    Returns DataFrame: geography_code, age_band, tenure, nssec, count, proportion
    """
    all_results = []

    for geo_code in constituency_codes:
        # Extract marginals for this constituency
        age_marg = _extract_marginal(age_df, geo_code, "age_band")
        tenure_marg = _extract_marginal(tenure_df, geo_code, "tenure")
        nssec_marg = _extract_marginal(nssec_df, geo_code, "nssec")

        # Extract tenure×nssec cross-tab if available
        tn_crosstab = None
        if tenure_nssec_df is not None:
            tn_sub = tenure_nssec_df[tenure_nssec_df["geography_code"] == geo_code]
            if not tn_sub.empty:
                tn_crosstab = {
                    (row["tenure"], row["nssec"]): row["count"]
                    for _, row in tn_sub.iterrows()
                }

        result = rake_constituency(
            age_marginal=age_marg,
            tenure_marginal=tenure_marg,
            nssec_marginal=nssec_marg,
            age_categories=age_categories,
            tenure_categories=tenure_categories,
            nssec_categories=nssec_categories,
            tenure_nssec_crosstab=tn_crosstab,
        )
        result["geography_code"] = geo_code
        all_results.append(result)

        logger.debug("Raked %s: %d cells", geo_code, len(result))

    combined = pd.concat(all_results, ignore_index=True)
    logger.info("Raked %d constituencies, %d total cells", len(constituency_codes), len(combined))
    return combined


def _extract_marginal(df: pd.DataFrame, geo_code: str, cat_col: str) -> dict:
    """Extract marginal counts for one constituency."""
    sub = df[df["geography_code"] == geo_code]
    return dict(zip(sub[cat_col], sub["count"]))
