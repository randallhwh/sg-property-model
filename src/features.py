"""
features.py — Feature engineering for the SG property fair value model.

Key feature groups:
  1. Location          — district tier, CCR/RCR/OCR, MRT distances
  2. Property          — type, floor, age, tenure / lease remaining
  3. Time              — year, quarter, cyclical encoding
  4. Market context    — rolling PSF by district/project
  5. Geocoding         — OneMap API (optional, cached)
"""

import os
import json
import math
import time
import warnings
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── MRT station coordinates (SG 2024 network, major stations) ────────────────
# Format: (name, lat, lng, line_tier)
# line_tier: 1 = CBD/premium lines (MRT), 2 = suburban

MRT_STATIONS = [
    # EW / NS (oldest, most premium)
    ("Jurong East",       1.3330, 103.7436, 1),
    ("Clementi",          1.3152, 103.7651, 1),
    ("Dover",             1.3110, 103.7784, 1),
    ("Buona Vista",       1.3072, 103.7900, 1),
    ("Queenstown",        1.2940, 103.8062, 1),
    ("Commonwealth",      1.3014, 103.7981, 1),
    ("Redhill",           1.2894, 103.8170, 1),
    ("Tiong Bahru",       1.2862, 103.8272, 1),
    ("Outram Park",       1.2800, 103.8398, 1),
    ("Tanjong Pagar",     1.2765, 103.8448, 1),
    ("Raffles Place",     1.2842, 103.8513, 1),
    ("City Hall",         1.2931, 103.8519, 1),
    ("Bugis",             1.3003, 103.8563, 1),
    ("Lavender",          1.3072, 103.8629, 1),
    ("Kallang",           1.3121, 103.8717, 1),
    ("Aljunied",          1.3165, 103.8828, 1),
    ("Paya Lebar",        1.3176, 103.8924, 1),
    ("Eunos",             1.3196, 103.9028, 1),
    ("Kembangan",         1.3213, 103.9126, 1),
    ("Bedok",             1.3240, 103.9300, 1),
    ("Tanah Merah",       1.3273, 103.9462, 1),
    ("Simei",             1.3432, 103.9531, 1),
    ("Tampines",          1.3527, 103.9454, 1),
    ("Pasir Ris",         1.3731, 103.9494, 1),
    # NS line
    ("Orchard",           1.3043, 103.8317, 1),
    ("Somerset",          1.3006, 103.8396, 1),
    ("Dhoby Ghaut",       1.2995, 103.8456, 1),
    ("Novena",            1.3201, 103.8436, 1),
    ("Newton",            1.3126, 103.8380, 1),
    ("Toa Payoh",         1.3325, 103.8474, 1),
    ("Braddell",          1.3401, 103.8470, 1),
    ("Bishan",            1.3508, 103.8483, 1),
    ("Ang Mo Kio",        1.3700, 103.8495, 1),
    ("Yio Chu Kang",      1.3817, 103.8447, 2),
    ("Khatib",            1.4172, 103.8333, 2),
    ("Yishun",            1.4290, 103.8353, 2),
    ("Woodlands",         1.4370, 103.7866, 2),
    ("Marsiling",         1.4325, 103.7740, 2),
    ("Admiralty",         1.4409, 103.8006, 2),
    # CC line
    ("Harbourfront",      1.2655, 103.8199, 1),
    ("Telok Blangah",     1.2714, 103.8085, 1),
    ("Labrador Park",     1.2723, 103.8022, 1),
    ("Pasir Panjang",     1.2761, 103.7924, 1),
    ("Haw Par Villa",     1.2811, 103.7826, 1),
    ("Kent Ridge",        1.2937, 103.7843, 1),
    ("one-north",         1.2994, 103.7874, 1),
    ("Holland Village",   1.3109, 103.7963, 1),
    ("Farrer Road",       1.3175, 103.8079, 1),
    ("Botanic Gardens",   1.3224, 103.8154, 1),
    ("Caldecott",         1.3386, 103.8390, 1),
    ("Marymount",         1.3487, 103.8394, 1),
    ("Bishan CC",         1.3509, 103.8483, 1),
    ("Serangoon",         1.3498, 103.8733, 1),
    ("Bartley",           1.3427, 103.8797, 2),
    ("Tai Seng",          1.3356, 103.8886, 2),
    ("MacPherson",        1.3268, 103.8893, 2),
    ("Mattar",            1.3203, 103.8866, 2),
    ("Dakota",            1.3078, 103.8882, 2),
    ("Mountbatten",       1.3011, 103.8839, 2),
    ("Stadium",           1.3026, 103.8762, 2),
    ("Nicoll Highway",    1.2990, 103.8641, 1),
    ("Promenade",         1.2933, 103.8607, 1),
    ("Esplanade",         1.2897, 103.8556, 1),
    ("Bras Basah",        1.2969, 103.8503, 1),
    ("Dhoby Ghaut CC",    1.2995, 103.8456, 1),
    # DT line
    ("Bukit Panjang",     1.3790, 103.7625, 2),
    ("Cashew",            1.3704, 103.7685, 2),
    ("Hillview",          1.3619, 103.7674, 2),
    ("Beauty World",      1.3411, 103.7758, 1),
    ("King Albert Park",  1.3329, 103.7840, 1),
    ("Sixth Avenue",      1.3317, 103.7982, 1),
    ("Tan Kah Kee",       1.3264, 103.8075, 1),
    ("Stevens DT",        1.3199, 103.8254, 1),
    ("Napier",            1.3093, 103.8172, 1),
    ("Fort Canning",      1.2916, 103.8447, 1),
    ("Bencoolen",         1.2980, 103.8498, 1),
    ("Rochor",            1.3041, 103.8517, 1),
    ("Little India",      1.3066, 103.8496, 1),
    ("Farrer Park",       1.3124, 103.8496, 1),
    ("Jalan Besar",       1.3101, 103.8588, 2),
    ("Bendemeer",         1.3181, 103.8623, 2),
    ("Geylang Bahru",     1.3220, 103.8717, 2),
    ("Ubi",               1.3290, 103.9014, 2),
    ("Kaki Bukit",        1.3350, 103.9100, 2),
    ("Bedok North",       1.3315, 103.9198, 2),
    ("Bedok Reservoir",   1.3360, 103.9319, 2),
    ("Tampines West",     1.3457, 103.9380, 2),
    ("Tampines East",     1.3559, 103.9533, 2),
    ("Upper Changi",      1.3414, 103.9614, 2),
    ("Expo",              1.3354, 103.9613, 2),
    ("Changi Airport",    1.3573, 103.9888, 2),
    # NE line
    ("Woodleigh",         1.3387, 103.8706, 2),
    ("Potong Pasir",      1.3316, 103.8688, 2),
    ("Boon Keng",         1.3198, 103.8616, 2),
    ("Farrer Park NE",    1.3124, 103.8496, 2),
    ("Little India NE",   1.3066, 103.8496, 2),
    ("Clarke Quay",       1.2884, 103.8462, 1),
    ("Chinatown",         1.2840, 103.8443, 1),
    ("Hougang",           1.3716, 103.8924, 2),
    ("Kovan",             1.3600, 103.8845, 2),
    ("Buangkok",          1.3829, 103.8936, 2),
    ("Sengkang",          1.3916, 103.8951, 2),
    ("Punggol",           1.4047, 103.9022, 2),
    # TEL line
    ("Stevens TEL",       1.3199, 103.8254, 1),
    ("Springleaf",        1.3979, 103.8175, 2),
    ("Lentor",            1.3859, 103.8344, 2),
    ("Mayflower",         1.3757, 103.8359, 2),
    ("Bright Hill",       1.3687, 103.8372, 2),
    ("Upper Thomson",     1.3543, 103.8310, 2),
    ("Caldecott TEL",     1.3386, 103.8390, 1),
    ("Mount Pleasant",    1.3300, 103.8298, 1),
    ("Marine Parade",     1.3025, 103.9056, 2),
    ("Marine Terrace",    1.3077, 103.9121, 2),
    ("Siglap",            1.3135, 103.9264, 2),
    ("Bayshore",          1.3170, 103.9354, 2),
    # Western extension
    ("Boon Lay",          1.3393, 103.7061, 2),
    ("Pioneer",           1.3380, 103.7195, 2),
    ("Joo Koon",          1.3270, 103.6783, 2),
    ("Gul Circle",        1.3167, 103.6647, 2),
    ("Tuas Crescent",     1.3212, 103.6480, 2),
    ("Tuas West Road",    1.3312, 103.6391, 2),
    ("Tuas Link",         1.3408, 103.6370, 2),
    # CCK/BP
    ("Choa Chu Kang",     1.3852, 103.7449, 2),
    ("Bukit Gombak",      1.3590, 103.7519, 2),
    ("Bukit Batok",       1.3490, 103.7496, 2),
]

MRT_DF = pd.DataFrame(
    MRT_STATIONS, columns=["name", "lat", "lng", "line_tier"]
)
MRT_COORDS = MRT_DF[["lat", "lng"]].values  # shape (N, 2)
MRT_TIERS  = MRT_DF["line_tier"].values

# CBD centroid (Raffles Place)
CBD_LAT, CBD_LNG = 1.2842, 103.8513

# ── SVY21 → WGS84 conversion ─────────────────────────────────────────────────

def svy21_to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """
    Convert SVY21 projected coordinates (metres) to WGS84 lat/lng.
    Linear approximation valid for Singapore — error < 1 m.
    URA API: x = Easting, y = Northing.
    """
    lat = 1.3666667 + (northing - 38744.572) / 110574.0
    lng = 103.8333333 + (easting  - 28001.642) / 111279.0
    return lat, lng


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two lat/lng points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _haversine_batch(lat, lon, coords):
    """Vectorised haversine: coords is (N,2) array. Returns distances in metres."""
    R = 6_371_000
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = np.radians(coords[:, 0])
    lon2 = np.radians(coords[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def mrt_distances(lat, lon):
    """Return (dist_nearest_mrt_m, mrt_tier_of_nearest) for a lat/lng point."""
    if lat is None or lon is None or np.isnan(lat) or np.isnan(lon):
        return np.nan, np.nan
    dists = _haversine_batch(lat, lon, MRT_COORDS)
    idx = int(np.argmin(dists))
    return float(dists[idx]), int(MRT_TIERS[idx])

# ── OneMap geocoding (optional) ───────────────────────────────────────────────

_GEOCODE_CACHE: dict[str, tuple[float, float]] = {}
_GEOCODE_CACHE_FILE = "data/external/geocode_cache.json"


def _load_geocode_cache():
    if os.path.exists(_GEOCODE_CACHE_FILE):
        with open(_GEOCODE_CACHE_FILE) as f:
            _GEOCODE_CACHE.update(json.load(f))


def _save_geocode_cache():
    os.makedirs(os.path.dirname(_GEOCODE_CACHE_FILE), exist_ok=True)
    with open(_GEOCODE_CACHE_FILE, "w") as f:
        json.dump(_GEOCODE_CACHE, f)


def geocode_onemap(query: str, retries: int = 2) -> tuple[float | None, float | None]:
    """
    Geocode a Singapore address/project name via OneMap API.
    Returns (lat, lng) or (None, None) if not found.
    """
    key = query.strip().upper()
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]
    url = (
        "https://www.onemap.gov.sg/api/common/elastic/search"
        f"?searchVal={requests.utils.quote(query)}"
        "&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    )
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            results = data.get("results", [])
            if results:
                lat = float(results[0]["LATITUDE"])
                lng = float(results[0]["LONGITUDE"])
                _GEOCODE_CACHE[key] = (lat, lng)
                return lat, lng
            break
        except Exception:
            time.sleep(1)
    _GEOCODE_CACHE[key] = (None, None)
    return None, None


def geocode_projects(df: pd.DataFrame, batch_size: int = 200) -> pd.DataFrame:
    """
    Add lat/lng to df by geocoding project_name via OneMap.
    Results are cached to disk. Only calls API for uncached entries.
    """
    _load_geocode_cache()
    projects = df["project_name"].dropna().unique()
    uncached = [p for p in projects if p.strip().upper() not in _GEOCODE_CACHE]
    print(f"Geocoding {len(uncached)}/{len(projects)} projects via OneMap…")
    for i, proj in enumerate(uncached):
        geocode_onemap(proj)
        if (i + 1) % 50 == 0:
            _save_geocode_cache()
            print(f"  {i+1}/{len(uncached)}")
        time.sleep(0.1)  # polite rate limit
    _save_geocode_cache()
    # Map back to df
    df["lat"] = df["project_name"].apply(
        lambda p: _GEOCODE_CACHE.get(str(p).strip().upper(), (None, None))[0]
    )
    df["lng"] = df["project_name"].apply(
        lambda p: _GEOCODE_CACHE.get(str(p).strip().upper(), (None, None))[1]
    )
    return df

# ── Distance features ─────────────────────────────────────────────────────────

def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires lat/lng columns. Adds:
      - dist_mrt_m       : metres to nearest MRT
      - mrt_tier         : 1 (city/premium) or 2 (suburban)
      - dist_cbd_m       : metres to Raffles Place
    """
    if "lat" not in df.columns or df["lat"].isna().all():
        df["dist_mrt_m"] = np.nan
        df["mrt_tier"]   = np.nan
        df["dist_cbd_m"] = np.nan
        return df

    mrt_results = df.apply(
        lambda r: mrt_distances(r.get("lat"), r.get("lng")),
        axis=1
    )
    df["dist_mrt_m"] = mrt_results.apply(lambda x: x[0])
    df["mrt_tier"]   = mrt_results.apply(lambda x: x[1])
    df["dist_cbd_m"] = df.apply(
        lambda r: haversine_m(r.get("lat", CBD_LAT), r.get("lng", CBD_LNG),
                              CBD_LAT, CBD_LNG)
        if pd.notna(r.get("lat")) else np.nan,
        axis=1
    )
    return df

# ── Time features ─────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Year, quarter, cyclical month encoding."""
    df["year"]    = df["date_of_sale"].dt.year
    df["quarter"] = df["date_of_sale"].dt.quarter
    df["month"]   = df["date_of_sale"].dt.month
    # Cyclical: sin/cos of month to capture seasonality
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Age of property at time of sale (requires year_built — approximated by tenure_start_year)
    if "tenure_start_year" in df.columns:
        df["age_at_sale"] = df["year"] - df["tenure_start_year"].fillna(df["year"] - 10)
        df["age_at_sale"] = df["age_at_sale"].clip(0, 80)
    else:
        df["age_at_sale"] = np.nan
    # Years since TOP (if top_year provided as a column or scalar)
    if "top_year" in df.columns:
        df["years_since_top"] = (df["year"] - pd.to_numeric(df["top_year"], errors="coerce"))
        df["years_since_top"] = df["years_since_top"].clip(0, 50)
    else:
        df["years_since_top"] = np.nan
    return df

# ── Rolling market context features ──────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling PSF stats within district and project, using only past transactions
    (no data leakage). Sorts by date before computing.
    """
    df = df.sort_values("date_of_sale").copy()

    # 3-month rolling median PSF per district
    df["date_period"] = df["date_of_sale"].dt.to_period("M")
    monthly_district = (
        df.groupby(["postal_district", "date_period"])["psf"]
        .median()
        .rename("monthly_district_psf")
    )
    df = df.join(monthly_district, on=["postal_district", "date_period"])

    # Project-level: rolling 6-month median (expanding if < 6 txns)
    def _rolling_project_psf(group):
        group = group.sort_values("date_of_sale")
        # Use expanding median shifted by 1 (no self-reference)
        group["project_rolling_psf"] = (
            group["psf"].shift(1).expanding(min_periods=1).median()
        )
        return group

    df = (
        df.groupby("project_name", group_keys=False)
        .apply(_rolling_project_psf)
        .reset_index(drop=True)  # prevent project_name from becoming index level
    )

    # Number of transactions in project in trailing 6 months
    df["txn_count_project_6m"] = (
        df.groupby("project_name", group_keys=False)["date_of_sale"]
        .transform(lambda x: x.expanding().count().shift(1).fillna(0))
    )

    # PSF vs district median (relative value signal)
    df["psf_vs_district"] = df["psf"] / df["monthly_district_psf"].replace(0, np.nan) - 1

    return df

# ── Property features ─────────────────────────────────────────────────────────

def add_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform area, floor midpoint imputation, type dummies.
    """
    df["log_area"] = np.log1p(df["area_sqft"])

    # Floor midpoint: impute missing with district median
    if "floor_midpoint" in df.columns:
        district_floor_median = df.groupby("postal_district")["floor_midpoint"].transform("median")
        df["floor_midpoint"] = df["floor_midpoint"].fillna(district_floor_median).fillna(5.0)
        df["floor_midpoint_sq"] = df["floor_midpoint"] ** 2  # non-linear floor premium
    else:
        df["floor_midpoint"] = 5.0
        df["floor_midpoint_sq"] = 25.0

    # Property type dummies
    type_dummies = pd.get_dummies(df["property_type"], prefix="type", drop_first=False)
    df = pd.concat([df, type_dummies], axis=1)

    # Floor band dummies
    if "floor_band" in df.columns:
        band_dummies = pd.get_dummies(df["floor_band"], prefix="band", drop_first=False)
        df = pd.concat([df, band_dummies], axis=1)

    return df

# ── Project-level target encoding ─────────────────────────────────────────────

def add_project_encoding(df: pd.DataFrame, train_mask: pd.Series | None = None) -> pd.DataFrame:
    """
    Mean-target encoding for project_name (PSF mean).
    Uses train_mask to prevent leakage: only training rows inform the encoding.
    """
    if train_mask is None:
        train_mask = pd.Series(True, index=df.index)

    project_means = (
        df.loc[train_mask]
        .groupby("project_name")["psf"]
        .mean()
        .rename("project_mean_psf")
    )
    global_mean = df.loc[train_mask, "psf"].mean()
    df["project_mean_psf"] = df["project_name"].map(project_means).fillna(global_mean)
    return df

# ── Master feature builder ────────────────────────────────────────────────────

FEATURE_COLS = [
    # Location
    "postal_district", "district_tier", "market_segment_code",
    "dist_mrt_m", "mrt_tier", "dist_cbd_m",
    # Property
    "log_area", "area_sqft",
    "floor_midpoint", "floor_midpoint_sq",
    "is_freehold", "remaining_lease",
    # Time
    "year", "quarter", "month_sin", "month_cos", "age_at_sale",
    "years_since_top",
    # Sale type
    "sale_type_code",
    # Market context
    "monthly_district_psf", "project_rolling_psf",
    "txn_count_project_6m",
    # Project encoding
    "project_mean_psf",
    # Dummies (added dynamically)
]

TYPE_DUMMY_COLS   = ["type_condo", "type_ec", "type_semi_d", "type_detached",
                     "type_terrace", "type_strata_landed", "type_cluster"]
BAND_DUMMY_COLS   = ["band_low", "band_mid", "band_high", "band_penthouse",
                     "band_basement", "band_unknown"]


def build_features(
    df: pd.DataFrame,
    geocode: bool = False,
    train_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline. Returns df with all feature columns added.
    Set geocode=True to also call OneMap API for rows without SVY21 coords.

    SVY21 → lat/lng derivation runs automatically when x_svy21/y_svy21 are
    present (URA API data), so MRT distance features are always populated for
    those rows without any API calls.
    """
    # ── Auto-derive lat/lng from SVY21 coordinates (URA API provides these) ──
    if "x_svy21" in df.columns and "y_svy21" in df.columns:
        has_svy = (
            pd.to_numeric(df["x_svy21"], errors="coerce").notna() &
            pd.to_numeric(df["y_svy21"], errors="coerce").notna()
        )
        need_ll = has_svy & (
            df.get("lat", pd.Series(np.nan, index=df.index)).isna() |
            ("lat" not in df.columns)
        )
        if need_ll.any():
            e = pd.to_numeric(df.loc[need_ll, "x_svy21"], errors="coerce")
            n = pd.to_numeric(df.loc[need_ll, "y_svy21"], errors="coerce")
            ll = pd.DataFrame({"lat": 1.3666667 + (n - 38744.572) / 110574.0,
                               "lng": 103.8333333 + (e - 28001.642) / 111279.0})
            if "lat" not in df.columns:
                df["lat"] = np.nan
                df["lng"] = np.nan
            df.loc[need_ll, "lat"] = ll["lat"].values
            df.loc[need_ll, "lng"] = ll["lng"].values
            n_derived = need_ll.sum()
            if n_derived > 0:
                print(f"  SVY21 -> lat/lng derived for {n_derived:,} rows")

    # ── Always compute distance features (uses lat/lng if available) ──────────
    df = add_distance_features(df)

    # ── Optional: geocode remaining rows without coords via OneMap ────────────
    if geocode:
        no_ll = df["lat"].isna() if "lat" in df.columns else pd.Series(True, index=df.index)
        if no_ll.any():
            df = geocode_projects(df)
            df = add_distance_features(df)

    df = add_time_features(df)
    df = add_rolling_features(df)
    df = add_property_features(df)
    df = add_project_encoding(df, train_mask)

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract the final feature matrix from a processed+featurised df.
    Returns (X, feature_names).
    """
    # Build dynamic list of dummy columns actually present
    # Exclude raw string columns that happen to start with "type_" (e.g. type_of_sale)
    dummy_cols = [
        c for c in df.columns
        if (c.startswith("type_") or c.startswith("band_"))
        and df[c].dtype != object
    ]
    all_cols = FEATURE_COLS + dummy_cols
    present = [c for c in all_cols if c in df.columns]
    X = df[present].copy()
    # Keep only numeric columns — guard against any stray object columns
    X = X.select_dtypes(include="number")
    # Fill remaining NAs with column medians
    X = X.fillna(X.median())
    return X, list(X.columns)
