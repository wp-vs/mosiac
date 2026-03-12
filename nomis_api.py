"""
Nomis API client for fetching Census 2021 data.

Uses the Nomis REST API (https://www.nomisweb.co.uk/api/v01) to query
Topic Summary and Ready Made Table datasets at parliamentary constituency level.

The API is free to use without a key (limited to 25,000 rows per query).
Register at nomisweb.co.uk for a key to increase the limit to 1,000,000 rows.
Set the NOMIS_API_KEY environment variable if you have one.
"""

import os
import time
import logging
from pathlib import Path
from io import StringIO

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def _api_key_param() -> dict:
    """Return uid parameter if a Nomis API key is configured."""
    key = os.environ.get("NOMIS_API_KEY", "")
    if key:
        return {"uid": key}
    return {}


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    """GET with retry and back-off."""
    params = {**(params or {}), **_api_key_param()}
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning("Nomis API request failed (%s), retrying in %ds", exc, wait)
            time.sleep(wait)


# ── Dataset ID discovery ────────────────────────────────────────────────


def discover_dataset_id(dataset_code: str) -> str:
    """
    Resolve a human-readable dataset code (e.g. 'c2021ts054') to the
    internal Nomis NM_XXXX_1 identifier by querying the dataset catalog.

    Falls back to trying the code directly as the dataset name in the API URL.
    """
    cache_path = Path(config.CACHE_DIR) / "dataset_ids.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        match = cached.loc[cached["code"] == dataset_code, "nm_id"]
        if not match.empty:
            return match.iloc[0]

    # Try querying the API's dataset definition endpoint
    url = f"{config.NOMIS_API_BASE}/dataset/def.sdmx.json"
    try:
        resp = _get(url)
        data = resp.json()
        keyfamilies = data.get("structure", {}).get("keyfamilies", {}).get("keyfamily", [])
        for kf in keyfamilies:
            nm_id = kf.get("id", "")
            name = kf.get("name", {}).get("value", "")
            annotations = kf.get("annotations", {}).get("annotation", [])
            # Check annotations for the dataset code
            for ann in annotations:
                if ann.get("annotationtext", "") == dataset_code:
                    logger.info("Resolved %s -> %s (%s)", dataset_code, nm_id, name)
                    _cache_dataset_id(cache_path, dataset_code, nm_id)
                    return nm_id
            # Also check if the name contains the code
            if dataset_code.upper() in name.upper() or dataset_code in nm_id.lower():
                logger.info("Resolved %s -> %s (%s)", dataset_code, nm_id, name)
                _cache_dataset_id(cache_path, dataset_code, nm_id)
                return nm_id
    except Exception as exc:
        logger.warning("Could not query dataset catalog: %s", exc)

    # Fallback: use the code directly (works for some 2021 Census datasets)
    logger.info("Using dataset code '%s' directly as API identifier", dataset_code)
    return dataset_code


def _cache_dataset_id(path: Path, code: str, nm_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"code": code, "nm_id": nm_id}])
    if path.exists():
        existing = pd.read_csv(path)
        row = pd.concat([existing, row], ignore_index=True).drop_duplicates("code")
    row.to_csv(path, index=False)


# ── Geography helpers ───────────────────────────────────────────────────


def get_constituency_geographies(geography_type: str = config.GEOGRAPHY_TYPE) -> pd.DataFrame:
    """
    Fetch the list of parliamentary constituencies and their Nomis codes.

    Returns a DataFrame with columns: geography_code, geography_name.
    """
    cache_path = Path(config.CACHE_DIR) / f"constituencies_{geography_type}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    url = f"{config.NOMIS_API_BASE}/dataset/NM_2021_1/geography/{geography_type}.def.sdmx.json"
    try:
        resp = _get(url)
        data = resp.json()
        items = (
            data.get("structure", {})
            .get("codelists", {})
            .get("codelist", [{}])[0]
            .get("code", [])
        )
        rows = []
        for item in items:
            rows.append({
                "geography_code": item.get("value", ""),
                "geography_name": item.get("description", {}).get("value", ""),
            })
        df = pd.DataFrame(rows)
    except Exception:
        # Fallback: try fetching a small data query to get geography names
        logger.warning("Could not fetch geography list via SDMX, trying data query")
        df = _fetch_geographies_via_data_query(geography_type)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _fetch_geographies_via_data_query(geography_type: str) -> pd.DataFrame:
    """Fetch constituency list by querying a small dataset."""
    # Use TS054 (tenure) as a lightweight dataset to get geography names
    url = f"{config.NOMIS_API_BASE}/dataset/c2021ts054.data.csv"
    params = {
        "geography": geography_type,
        "c2021_tenure_9": "0",  # total only
        "select": "geography_code,geography_name",
    }
    resp = _get(url, params)
    df = pd.read_csv(StringIO(resp.text))
    return df[["GEOGRAPHY_CODE", "GEOGRAPHY_NAME"]].rename(
        columns={"GEOGRAPHY_CODE": "geography_code", "GEOGRAPHY_NAME": "geography_name"}
    ).drop_duplicates()


def filter_london_constituencies(constituencies: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to London constituencies only.

    London constituencies have GSS codes starting with 'E14' and can be
    identified by the London region. We use a name-based heuristic as well.
    """
    # London borough names that appear in constituency names
    london_boroughs = [
        "Barking", "Barnet", "Battersea", "Beckenham", "Bermondsey", "Bethnal",
        "Bexley", "Bow", "Brent", "Brixton", "Bromley", "Camberwell",
        "Camden", "Chelsea", "Chingford", "Chiswick", "Clapham", "Croydon",
        "Dagenham", "Deptford", "Dulwich", "Ealing", "East Ham", "Edmonton",
        "Eltham", "Enfield", "Erith", "Feltham", "Finchley", "Greenwich",
        "Hackney", "Hammersmith", "Hampstead", "Harrow", "Hayes", "Hendon",
        "Holborn", "Hornchurch", "Hornsey", "Hounslow", "Ilford", "Islington",
        "Kensington", "Kingston", "Lambeth", "Lewisham", "Leyton", "Mitcham",
        "Orpington", "Peckham", "Poplar", "Putney", "Richmond", "Romford",
        "Ruislip", "Southwark", "Stepney", "Stratford", "Streatham",
        "Sutton", "Tooting", "Tottenham", "Twickenham", "Uxbridge",
        "Vauxhall", "Walthamstow", "Wandsworth", "West Ham", "Westminster",
        "Wimbledon", "Woolwich", "Cities of London",
    ]
    pattern = "|".join(london_boroughs)
    mask = constituencies["geography_name"].str.contains(pattern, case=False, na=False)
    london = constituencies[mask].copy()
    logger.info("Found %d London constituencies", len(london))
    return london


# ── Data fetching ───────────────────────────────────────────────────────


def fetch_census_table(
    dataset_code: str,
    geography_type: str = config.GEOGRAPHY_TYPE,
    geography_codes: list[str] | None = None,
    extra_params: dict | None = None,
) -> pd.DataFrame:
    """
    Fetch a Census 2021 table from Nomis as a CSV DataFrame.

    Parameters
    ----------
    dataset_code : str
        Nomis dataset code, e.g. 'c2021ts054' or an NM_XXXX_1 identifier.
    geography_type : str
        Nomis geography type, e.g. 'TYPE460' for parliamentary constituencies.
    geography_codes : list of str, optional
        Specific geography codes to fetch. If None, fetches all of the type.
    extra_params : dict, optional
        Additional query parameters for the API call.

    Returns
    -------
    pd.DataFrame
    """
    cache_key = f"{dataset_code}_{geography_type}"
    cache_path = Path(config.CACHE_DIR) / f"{cache_key}.csv"
    if cache_path.exists():
        logger.info("Loading cached data for %s", cache_key)
        return pd.read_csv(cache_path)

    url = f"{config.NOMIS_API_BASE}/dataset/{dataset_code}.data.csv"
    params = {"geography": geography_type}

    if geography_codes:
        params["geography"] = ",".join(geography_codes)

    if extra_params:
        params.update(extra_params)

    logger.info("Fetching %s from Nomis (geography=%s)", dataset_code, geography_type)
    resp = _get(url, params)

    df = pd.read_csv(StringIO(resp.text))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Fetched %d rows for %s", len(df), dataset_code)
    return df


def fetch_age_by_constituency(geography_type: str = config.GEOGRAPHY_TYPE) -> pd.DataFrame:
    """Fetch TS007A (age in grouped bands) by constituency."""
    return fetch_census_table(config.DATASETS["age"], geography_type)


def fetch_tenure_by_constituency(geography_type: str = config.GEOGRAPHY_TYPE) -> pd.DataFrame:
    """Fetch TS054 (tenure) by constituency."""
    return fetch_census_table(config.DATASETS["tenure"], geography_type)


def fetch_nssec_by_constituency(geography_type: str = config.GEOGRAPHY_TYPE) -> pd.DataFrame:
    """Fetch TS062 (NS-SeC) by constituency."""
    return fetch_census_table(config.DATASETS["nssec"], geography_type)


def fetch_tenure_nssec_by_constituency(geography_type: str = config.GEOGRAPHY_TYPE) -> pd.DataFrame:
    """Fetch RM138 (tenure × NS-SEC cross-tab) by constituency."""
    return fetch_census_table(config.DATASETS["tenure_nssec"], geography_type)
