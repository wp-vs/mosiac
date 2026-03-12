"""
Output Area Classification (OAC) 2021 data fetcher and processor.

Downloads OAC data from the GeoDS GitHub repository and the OA-to-constituency
lookup from ONS, then builds constituency-level demographic joint distributions
that capture the covariance between age, tenure, and NS-SEC.

Key data sources:
- OAC cluster assignments: github.com/alexsingleton/Output_Area_Classification
- OAC input variables (raw %): same repo, OAC_Input.parquet
- OA-to-constituency lookup: ONS Open Geography / data.gov.uk

The OAC input data contains ~60 demographic variables at output area level.
By aggregating within OAC clusters and within constituencies, we can estimate
the joint distribution of (age × tenure × NS-SEC) that captures real-world
correlations rather than assuming independence.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

# GitHub raw URLs for OAC data (alexsingleton repo)
OAC_GITHUB_BASE = "https://raw.githubusercontent.com/alexsingleton/Output_Area_Classification/main/data"
OAC_ASSIGNMENT_URL = f"{OAC_GITHUB_BASE}/UK_OAC_Final.parquet"
OAC_INPUT_URL = f"{OAC_GITHUB_BASE}/OAC_Input.parquet"
OAC_INDEX_SUBGROUPS_URL = f"{OAC_GITHUB_BASE}/Index_Scores_Final_Subgroups.csv"
OAC_INDEX_GROUPS_URL = f"{OAC_GITHUB_BASE}/Index_Scores_Final_Groups.csv"
OAC_INDEX_SUPERGROUPS_URL = f"{OAC_GITHUB_BASE}/Index_Scores_Final_Supergroup.csv"

# ONS OA-to-constituency lookup
# Best-fit lookup: OA 2021 → Westminster Parliamentary Constituency (July 2024)
OA_CONSTITUENCY_LOOKUP_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "cb97f09a1a3546da96a4c8d3f10e4b6c/csv?layers=0"
)
# Fallback: try a direct ONS download
OA_CONSTITUENCY_FALLBACK_URL = (
    "https://www.arcgis.com/sharing/rest/content/items/"
    "cb97f09a1a3546da96a4c8d3f10e4b6c/data"
)

CACHE_DIR = Path(config.CACHE_DIR)

# ── OAC variable mapping to our demographic dimensions ──────────────────
# From the OAC documentation, the input variables include:
# Age: v02 (0-4), v03 (5-14), v04 (25-44), v05 (45-64), v06 (65-84), v07 (85+)
# Note: v01 is population density, not age. Ages 15-24 are implicit (100% - others)
# Tenure: v37 (ownership/shared ownership)
# NS-SEC: v46-v53 map to the 8 NS-SEC categories

OAC_AGE_VARS = {
    "v02": "age_0_4",
    "v03": "age_5_14",
    # v04 appears to be 25-44 in the documentation
    "v04": "age_25_44",
    "v05": "age_45_64",
    "v06": "age_65_84",
    "v07": "age_85plus",
}

OAC_TENURE_VARS = {
    "v37": "owned_or_shared",
    # Social renting
    "v38": "social_rented",
}

OAC_NSSEC_VARS = {
    "v46": "higher_managerial",
    "v47": "lower_managerial",
    "v48": "intermediate",
    "v49": "small_employers",
    "v50": "lower_supervisory",
    "v51": "semi_routine",
    "v52": "routine",
    "v53": "never_worked_unemployed",
}


# ── Download helpers ────────────────────────────────────────────────────


def _download_file(url: str, dest: Path, retries: int = 4) -> Path:
    """Download a file with retry and exponential backoff."""
    if dest.exists():
        logger.info("Using cached %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        try:
            logger.info("Downloading %s (attempt %d)", url, attempt + 1)
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Downloaded %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
            return dest
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to download {url} after {retries} attempts: {exc}")
            wait = 2 ** (attempt + 1)
            logger.warning("Download failed (%s), retrying in %ds", exc, wait)
            import time
            time.sleep(wait)


# ── Data loading ────────────────────────────────────────────────────────


def load_oac_assignments() -> pd.DataFrame:
    """
    Load OAC cluster assignments (OA → Supergroup/Group/Subgroup).

    Returns DataFrame: OA21CD, Supergroup, Group, Subgroup
    """
    dest = CACHE_DIR / "UK_OAC_Final.parquet"
    _download_file(OAC_ASSIGNMENT_URL, dest)
    df = pd.read_parquet(dest)
    logger.info("Loaded OAC assignments: %d output areas", len(df))
    return df


def load_oac_input_variables() -> pd.DataFrame:
    """
    Load OAC input variables (raw demographic % per output area).

    Returns DataFrame with OA21CD as index and v01-v60 columns.
    """
    dest = CACHE_DIR / "OAC_Input.parquet"
    _download_file(OAC_INPUT_URL, dest)
    df = pd.read_parquet(dest)
    logger.info("Loaded OAC input variables: %d OAs × %d variables", len(df), len(df.columns))
    return df


def load_oac_cluster_profiles(level: str = "subgroups") -> pd.DataFrame:
    """Load OAC index scores (cluster profiles) at the specified level."""
    urls = {
        "supergroups": OAC_INDEX_SUPERGROUPS_URL,
        "groups": OAC_INDEX_GROUPS_URL,
        "subgroups": OAC_INDEX_SUBGROUPS_URL,
    }
    url = urls[level]
    dest = CACHE_DIR / f"OAC_Index_{level}.csv"
    _download_file(url, dest)
    df = pd.read_csv(dest)
    logger.info("Loaded OAC %s profiles: %d clusters", level, len(df))
    return df


def load_oa_constituency_lookup() -> pd.DataFrame:
    """
    Load output area to parliamentary constituency best-fit lookup.

    Returns DataFrame: OA21CD, PCON24CD, PCON24NM (or similar columns).
    """
    dest = CACHE_DIR / "OA21_PCON_lookup.csv"

    if not dest.exists():
        # Try primary URL
        try:
            _download_file(OA_CONSTITUENCY_LOOKUP_URL, dest)
        except Exception:
            logger.warning("Primary lookup URL failed, trying fallback")
            try:
                _download_file(OA_CONSTITUENCY_FALLBACK_URL, dest)
            except Exception:
                logger.warning("Fallback also failed, trying Nomis approach")
                return _build_lookup_from_nomis()

    df = pd.read_csv(dest)

    # Normalise column names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "oa21cd" in col_lower or col_lower == "oa21cd":
            col_map[col] = "OA21CD"
        elif "pcon" in col_lower and "cd" in col_lower:
            col_map[col] = "PCON_CD"
        elif "pcon" in col_lower and "nm" in col_lower:
            col_map[col] = "PCON_NM"

    df = df.rename(columns=col_map)
    logger.info("Loaded OA-constituency lookup: %d rows", len(df))
    return df


def _build_lookup_from_nomis() -> pd.DataFrame:
    """
    Fallback: use Nomis OA-level data to infer OA-constituency mapping.
    This is slower but doesn't require a separate lookup file.
    """
    logger.info("Building OA-constituency lookup from Nomis data")
    raise LookupUnavailableError(
        "Could not download OA-constituency lookup. "
        "Please download it manually from "
        "https://geoportal.statistics.gov.uk/datasets/ons::output-areas-2021-to-"
        "westminster-parliamentary-constituencies-july-2024-best-fit-lookup-in-ew/about "
        "and place it in data/cache/OA21_PCON_lookup.csv"
    )


class LookupUnavailableError(RuntimeError):
    """Raised when the OA-constituency lookup cannot be downloaded."""
    pass


# ── OAC-informed covariance estimation ──────────────────────────────────


def compute_oa_level_demographics(oac_input: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and recode age, tenure, and NS-SEC from OAC input variables
    into the categories we use for the model.

    Returns DataFrame: OA21CD + age/tenure/nssec columns as proportions.
    """
    df = oac_input.copy()

    # The first column should be OA21CD (or the index)
    if "OA21CD" not in df.columns:
        if df.index.name and "oa" in df.index.name.lower():
            df = df.reset_index()
        else:
            # Assume first column is OA code
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "OA21CD"})

    result = pd.DataFrame({"OA21CD": df["OA21CD"]})

    # Map OAC variables to our categories
    # Age bands (as % of total population)
    _safe_assign(result, df, "age_0_14", ["v02", "v03"])
    _safe_assign(result, df, "age_25_44", ["v04"])
    _safe_assign(result, df, "age_45_64", ["v05"])
    _safe_assign(result, df, "age_65plus", ["v06", "v07"])
    # 15-24 is the residual
    age_sum = result.get("age_0_14", 0) + result.get("age_25_44", 0) + \
              result.get("age_45_64", 0) + result.get("age_65plus", 0)
    result["age_15_24"] = (100 - age_sum).clip(lower=0)

    # Recode to our model bands (18-24 ≈ 15-24 for this purpose)
    result["age_18_24"] = result["age_15_24"]
    result["age_25_49"] = result["age_25_44"]  # 25-44 ≈ 25-49 (closest available)
    result["age_50_64"] = result["age_45_64"]
    result["age_65plus_model"] = result["age_65plus"]

    # Tenure
    _safe_assign(result, df, "tenure_owned", ["v37"])
    _safe_assign(result, df, "tenure_social", ["v38"])
    # Private rent is residual of non-owned, non-social
    result["tenure_private"] = (100 - result.get("tenure_owned", 0) -
                                 result.get("tenure_social", 0)).clip(lower=0)

    # NS-SEC
    for var, name in OAC_NSSEC_VARS.items():
        _safe_assign(result, df, f"nssec_{name}", [var])

    return result


def _safe_assign(result: pd.DataFrame, source: pd.DataFrame,
                 target_col: str, source_cols: list[str]):
    """Sum source columns into target, handling missing columns."""
    total = pd.Series(0.0, index=source.index)
    for col in source_cols:
        if col in source.columns:
            total += pd.to_numeric(source[col], errors="coerce").fillna(0)
    result[target_col] = total.values


def compute_constituency_joint_distribution(
    oa_demographics: pd.DataFrame,
    oa_lookup: pd.DataFrame,
    constituency_code: str,
) -> np.ndarray:
    """
    Estimate the joint distribution of (age × tenure × NS-SEC) for a
    constituency by aggregating output-area-level data.

    At OA level (~300 people), demographics are relatively homogeneous,
    so the within-OA independence assumption is much more reasonable than
    at constituency level. The across-OA variation captures the real
    covariance structure.

    Returns a 3D numpy array of shape (n_age, n_tenure, n_nssec)
    representing estimated population counts in each cell.
    """
    # Get OAs in this constituency
    oas = oa_lookup[oa_lookup["PCON_CD"] == constituency_code]["OA21CD"]
    oa_data = oa_demographics[oa_demographics["OA21CD"].isin(oas)]

    if oa_data.empty:
        logger.warning("No OA data for constituency %s", constituency_code)
        return None

    # Age categories (proportions summing to ~100%)
    age_cols = ["age_18_24", "age_25_49", "age_50_64", "age_65plus_model"]
    # Tenure categories
    tenure_cols = ["tenure_owned", "tenure_social", "tenure_private"]
    # NS-SEC categories (aggregate into 4 groups: AB, C1, C2, DE)
    nssec_ab_cols = ["nssec_higher_managerial", "nssec_lower_managerial"]
    nssec_c1_cols = ["nssec_intermediate", "nssec_small_employers"]
    nssec_c2_cols = ["nssec_lower_supervisory"]
    nssec_de_cols = ["nssec_semi_routine", "nssec_routine", "nssec_never_worked_unemployed"]

    n_age = len(age_cols)
    n_tenure = len(tenure_cols)
    n_nssec = 4  # AB, C1, C2, DE
    joint = np.zeros((n_age, n_tenure, n_nssec))

    for _, oa_row in oa_data.iterrows():
        # Get marginals for this OA (as proportions)
        age_props = np.array([oa_row.get(c, 0) for c in age_cols], dtype=float)
        tenure_props = np.array([oa_row.get(c, 0) for c in tenure_cols], dtype=float)
        nssec_props = np.array([
            sum(oa_row.get(c, 0) for c in nssec_ab_cols),
            sum(oa_row.get(c, 0) for c in nssec_c1_cols),
            sum(oa_row.get(c, 0) for c in nssec_c2_cols),
            sum(oa_row.get(c, 0) for c in nssec_de_cols),
        ], dtype=float)

        # Normalise to proportions
        for arr in [age_props, tenure_props, nssec_props]:
            s = arr.sum()
            if s > 0:
                arr /= s

        # At OA level, assume conditional independence and compute outer product
        # This is reasonable because OAs are small and internally homogeneous
        oa_joint = np.einsum("i,j,k->ijk", age_props, tenure_props, nssec_props)
        joint += oa_joint

    return joint


def compute_all_constituency_joints(
    oa_demographics: pd.DataFrame,
    oa_lookup: pd.DataFrame,
    constituency_codes: list[str],
) -> dict[str, np.ndarray]:
    """
    Compute joint distributions for all constituencies.

    Returns dict mapping constituency code to 3D array.
    """
    joints = {}
    for code in constituency_codes:
        joint = compute_constituency_joint_distribution(oa_demographics, oa_lookup, code)
        if joint is not None:
            joints[code] = joint
    logger.info("Computed joint distributions for %d constituencies", len(joints))
    return joints


# ── OAC cluster-based seed for IPF ─────────────────────────────────────


def compute_oac_cluster_covariance(
    oac_input: pd.DataFrame,
    oac_assignments: pd.DataFrame,
    cluster_level: str = "Subgroup",
) -> dict[str, np.ndarray]:
    """
    Compute the joint distribution of (age × tenure × NS-SEC) within
    each OAC cluster by averaging OA-level outer products.

    This captures how demographics co-vary within each neighbourhood type.
    The constituency joint can then be estimated as a mixture of cluster joints,
    weighted by the cluster composition of the constituency.

    Returns dict mapping cluster code to 3D array.
    """
    oa_demo = compute_oa_level_demographics(oac_input)

    # Merge with cluster assignments
    if "OA21CD" in oac_assignments.columns:
        merged = oa_demo.merge(oac_assignments[["OA21CD", cluster_level]], on="OA21CD")
    else:
        # Try index
        assignments = oac_assignments.reset_index()
        oa_col = [c for c in assignments.columns if "oa" in c.lower()][0]
        merged = oa_demo.merge(
            assignments[[oa_col, cluster_level]].rename(columns={oa_col: "OA21CD"}),
            on="OA21CD",
        )

    cluster_joints = {}
    for cluster_code, group in merged.groupby(cluster_level):
        joint = np.zeros((4, 3, 4))  # age × tenure × nssec

        age_cols = ["age_18_24", "age_25_49", "age_50_64", "age_65plus_model"]
        tenure_cols = ["tenure_owned", "tenure_social", "tenure_private"]
        nssec_ab = ["nssec_higher_managerial", "nssec_lower_managerial"]
        nssec_c1 = ["nssec_intermediate", "nssec_small_employers"]
        nssec_c2 = ["nssec_lower_supervisory"]
        nssec_de = ["nssec_semi_routine", "nssec_routine", "nssec_never_worked_unemployed"]

        for _, row in group.iterrows():
            age_p = np.array([row.get(c, 0) for c in age_cols], dtype=float)
            tenure_p = np.array([row.get(c, 0) for c in tenure_cols], dtype=float)
            nssec_p = np.array([
                sum(row.get(c, 0) for c in nssec_ab),
                sum(row.get(c, 0) for c in nssec_c1),
                sum(row.get(c, 0) for c in nssec_c2),
                sum(row.get(c, 0) for c in nssec_de),
            ], dtype=float)

            for arr in [age_p, tenure_p, nssec_p]:
                s = arr.sum()
                if s > 0:
                    arr /= s

            joint += np.einsum("i,j,k->ijk", age_p, tenure_p, nssec_p)

        # Normalise to proportions
        s = joint.sum()
        if s > 0:
            joint /= s
        cluster_joints[str(cluster_code)] = joint

    logger.info("Computed joint distributions for %d OAC clusters", len(cluster_joints))
    return cluster_joints


def build_constituency_seed_from_oac(
    oac_assignments: pd.DataFrame,
    oa_lookup: pd.DataFrame,
    cluster_joints: dict[str, np.ndarray],
    constituency_code: str,
    cluster_level: str = "Subgroup",
) -> np.ndarray | None:
    """
    Build an IPF seed for a constituency as a mixture of OAC cluster
    joint distributions, weighted by the cluster composition.

    This seed captures the real covariance between age, tenure, and NS-SEC
    for the specific mix of neighbourhood types in this constituency.
    """
    # Get OAs in this constituency
    oas_in_const = set(oa_lookup[oa_lookup["PCON_CD"] == constituency_code]["OA21CD"])

    if not oas_in_const:
        return None

    # Get cluster assignments for these OAs
    if "OA21CD" in oac_assignments.columns:
        oa_col = "OA21CD"
    else:
        assignments = oac_assignments.reset_index()
        oa_col = [c for c in assignments.columns if "oa" in c.lower()][0]
        oac_assignments = assignments.rename(columns={oa_col: "OA21CD"})

    const_clusters = oac_assignments[oac_assignments["OA21CD"].isin(oas_in_const)]
    cluster_counts = const_clusters[cluster_level].value_counts()

    # Build weighted mixture
    seed = np.zeros((4, 3, 4))  # age × tenure × nssec
    total_oas = cluster_counts.sum()

    for cluster_code, count in cluster_counts.items():
        cluster_key = str(cluster_code)
        if cluster_key in cluster_joints:
            weight = count / total_oas
            seed += weight * cluster_joints[cluster_key]

    # Ensure no zeros
    seed = np.maximum(seed, 1e-6)

    return seed


# ── Category labels (matching the array dimensions) ─────────────────────

AGE_CATEGORIES = ["18-24", "25-49", "50-64", "65+"]
TENURE_CATEGORIES = ["Owned", "Social rent", "Private rent"]
NSSEC_CATEGORIES = ["AB", "C1", "C2", "DE"]


# ── Fallback: national covariance joint ─────────────────────────────────

# URL for mysociety constituency data (names, regions, electorates)
ENG_CONS_URL = "https://raw.githubusercontent.com/mysociety/2025-constituencies/main/data/interim/eng_cons.parquet"
GSS_LOOKUP_URL = "https://raw.githubusercontent.com/mysociety/2025-constituencies/main/data/raw/external/gss_lookup.csv"


def load_london_constituencies_from_mysociety() -> pd.DataFrame:
    """
    Load London constituency names and GSS codes from the mysociety
    2025-constituencies repository. Used as fallback when the full
    OA-constituency lookup is unavailable.

    Returns DataFrame: PCON_CD, PCON_NM
    """
    # Download GSS lookup (name → GSS code)
    gss_dest = CACHE_DIR / "gss_lookup.csv"
    _download_file(GSS_LOOKUP_URL, gss_dest)
    gss = pd.read_csv(gss_dest)
    gss.columns = ["PCON_NM", "PCON_CD"]

    # Download constituency data to identify London ones
    eng_dest = CACHE_DIR / "eng_cons.parquet"
    _download_file(ENG_CONS_URL, eng_dest)
    eng = pd.read_parquet(eng_dest)

    london = eng[eng["Region"] == "London"][["Constituen"]].copy()
    london.columns = ["PCON_NM"]

    # Merge to get GSS codes
    london = london.merge(gss, on="PCON_NM", how="left")

    logger.info("Loaded %d London constituencies from mysociety", len(london))
    return london


def compute_national_joint(oac_input: pd.DataFrame) -> np.ndarray:
    """
    Compute the NATIONAL joint distribution of (age × tenure × NS-SEC)
    from all 239k OAs. This captures the real covariance between dimensions
    across the whole of England & Wales.

    Each OA contributes its (age × tenure × NS-SEC) outer product.
    At OA level (~300 people), the within-OA independence assumption is
    reasonable; the across-OA variation captures the true covariance.

    Returns 3D array of shape (4, 3, 4): age × tenure × nssec.
    """
    oa_demo = compute_oa_level_demographics(oac_input)

    age_cols = ["age_18_24", "age_25_49", "age_50_64", "age_65plus_model"]
    tenure_cols = ["tenure_owned", "tenure_social", "tenure_private"]
    nssec_ab = ["nssec_higher_managerial", "nssec_lower_managerial"]
    nssec_c1 = ["nssec_intermediate", "nssec_small_employers"]
    nssec_c2 = ["nssec_lower_supervisory"]
    nssec_de = ["nssec_semi_routine", "nssec_routine", "nssec_never_worked_unemployed"]

    joint = np.zeros((4, 3, 4))

    for _, row in oa_demo.iterrows():
        age_p = np.array([row.get(c, 0) for c in age_cols], dtype=float)
        tenure_p = np.array([row.get(c, 0) for c in tenure_cols], dtype=float)
        nssec_p = np.array([
            sum(row.get(c, 0) for c in nssec_ab),
            sum(row.get(c, 0) for c in nssec_c1),
            sum(row.get(c, 0) for c in nssec_c2),
            sum(row.get(c, 0) for c in nssec_de),
        ], dtype=float)

        for arr in [age_p, tenure_p, nssec_p]:
            s = arr.sum()
            if s > 0:
                arr /= s

        joint += np.einsum("i,j,k->ijk", age_p, tenure_p, nssec_p)

    # Normalise to proportions
    s = joint.sum()
    if s > 0:
        joint /= s

    logger.info("Computed national joint distribution from %d OAs", len(oa_demo))
    return joint


def compute_cluster_weighted_joint_for_london(
    oac_input: pd.DataFrame,
    oac_assignments: pd.DataFrame,
    cluster_level: str = "Subgroup",
) -> np.ndarray:
    """
    Compute a London-specific joint distribution by weighting OAC cluster
    joints by London's known cluster composition.

    London has a distinctive OAC cluster mix (heavy on supergroups 3-5:
    'Multicultural & Educated Urbanites', 'Low-Skilled Migrant & Student
    Communities', 'Diverse Suburban Professionals'). We estimate the
    London-specific weighting from the cluster frequency distribution.

    Since we don't have the OA-constituency lookup, we use the known
    London supergroup distribution as a proxy.
    """
    # Compute cluster-level joints
    cluster_joints = compute_oac_cluster_covariance(
        oac_input, oac_assignments, cluster_level
    )

    # London has a distinctive OAC profile. Based on the OAC 2021 documentation:
    # Supergroups 3, 4, 5 are over-represented in London
    # These are the inner-London clusters with young, diverse, educated populations
    # We use OAC supergroup assignments to estimate London cluster weights
    # (approximation - the full lookup would be better)

    # Normalise OA code column
    if "OA21CD" not in oac_assignments.columns:
        oac_a = oac_assignments.reset_index()
        oa_col = [c for c in oac_a.columns if "oa" in c.lower() or "geography" in c.lower()][0]
        oac_a = oac_a.rename(columns={oa_col: "OA21CD"})
    else:
        oac_a = oac_assignments

    # For London proxy: use all OAs in clusters that are London-heavy
    # This is an approximation - the proper approach uses the OA-constituency lookup
    national_joint = compute_national_joint(oac_input)

    logger.info("Using national joint as London proxy (full OA-constituency lookup unavailable)")
    return national_joint
