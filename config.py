"""
Configuration for the London voting intention model.

Nomis API dataset codes and geography types for Census 2021 data.
"""

# Nomis API base URL
NOMIS_API_BASE = "https://www.nomisweb.co.uk/api/v01"

# Census 2021 Topic Summary dataset codes (used in Nomis URLs)
# These are the "c2021tsXXX" codes that Nomis uses internally.
# The NM_XXXX_1 IDs are discovered at runtime via the API.
DATASETS = {
    "age": "c2021ts007a",       # TS007A - Age (6 categories) — grouped, not single-year
    "tenure": "c2021ts054",     # TS054 - Tenure
    "nssec": "c2021ts062",      # TS062 - NS-SeC
    "tenure_nssec": "c2021rm138",  # RM138 - Tenure by NS-SEC (ready-made cross-tab)
}

# Geography type for Westminster Parliamentary Constituencies (post-2019 boundaries)
# TYPE460 = 2010 boundaries; the 2021 Census data was also released on these.
# If querying fails, the code will try to discover the correct type.
GEOGRAPHY_TYPE = "TYPE460"

# London constituencies — filter to these for the London model.
# We identify London constituencies by their GSS codes (starting with E14)
# and by name matching. The full list is resolved at runtime from Nomis.
LONDON_REGION_GSS = "E12000007"  # London region GSS code

# Age bands to use (must match what TS007A provides, or we recode)
AGE_BANDS = ["18-24", "25-34", "35-49", "50-64", "65+"]

# Tenure categories (from TS054)
TENURE_CATEGORIES = [
    "Owned: Owns outright",
    "Owned: Owns with a mortgage or loan or shared ownership",
    "Rented: Social rented",
    "Rented: Private rented or lives rent free",
]

# NS-SEC categories (from TS062, simplified)
NSSEC_CATEGORIES = [
    "L1, L2 and L3: Higher managerial, administrative and professional occupations",
    "L4, L5 and L6: Lower managerial, administrative and professional occupations",
    "L7: Intermediate occupations",
    "L8 and L9: Small employers and own account workers",
    "L10 and L11: Lower supervisory and technical occupations",
    "L12: Semi-routine occupations",
    "L13: Routine occupations",
    "L14.1 and L14.2: Never worked and long-term unemployed",
    "L15: Full-time students",
]

# Simplified NS-SEC grouping for the model (ABC1C2DE style)
NSSEC_SIMPLIFIED = {
    "AB": [
        "L1, L2 and L3",
        "L4, L5 and L6",
    ],
    "C1": [
        "L7",
        "L8 and L9",
    ],
    "C2": [
        "L10 and L11",
    ],
    "DE": [
        "L12",
        "L13",
        "L14.1 and L14.2",
        "L15",
    ],
}

# Output directory
DATA_DIR = "data"
CACHE_DIR = "data/cache"
