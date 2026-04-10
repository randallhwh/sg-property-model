"""
pipeline.py — URA private residential transaction CSV ingestion and cleaning.

URA data is downloaded from: https://www.ura.gov.sg/reis/dataDL
Drop all CSVs into data/raw/ and call load_ura_folder().

URA column name variants handled:
  Old format (pre-2021): Project Name, Street Name, Type, Postal District,
      Market Segment, Tenure, Type of Sale, No. of Units, Price ($),
      Area (Sqft), Unit Price ($ psf), Date of Sale
  New format (2021+): adds Floor Range, Planning Region, Planning Area,
      Type of Area (strata/land), Property Type of Buyer,
      Purchaser Address Indicator
"""

import os
import re
import glob
import warnings
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ── Column name normaliser ────────────────────────────────────────────────────

_COL_MAP = {
    # project
    "project name":           "project_name",
    "project":                "project_name",
    # street
    "street name":            "street_name",
    "street":                 "street_name",
    # property type
    "type":                   "property_type",
    "property type":          "property_type",
    # district
    "postal district":        "postal_district",
    "district":               "postal_district",
    # market segment
    "market segment":         "market_segment",
    # tenure
    "tenure":                 "tenure_raw",
    # type of sale
    "type of sale":           "type_of_sale",
    # units
    "no. of units":           "num_units",
    "no of units":            "num_units",
    "number of units":        "num_units",
    # price
    "price ($)":              "price",
    "transacted price ($)":   "price",
    "price":                  "price",
    # area
    "area (sqft)":            "area_sqft",
    "area (sqm)":             "area_sqm",
    "area":                   "area_sqft",
    # psf
    "unit price ($ psf)":     "psf",
    "unit price ($psf)":      "psf",
    "unit price (psf)":       "psf",
    "unit price":             "psf",
    # date
    "date of sale":           "date_of_sale",
    "sale date":              "date_of_sale",
    # floor range (new format)
    "floor range":            "floor_range",
    "floor level":            "floor_range",
    # planning
    "planning region":        "planning_region",
    "planning area":          "planning_area",
    # type of area (strata/land)
    "type of area":           "area_type",
    # buyer type
    "property type of buyer": "buyer_type",
    "purchaser address indicator": "purchaser_indicator",
}

REQUIRED_COLS = {
    "project_name", "property_type", "postal_district",
    "tenure_raw", "type_of_sale", "price", "area_sqft",
    "date_of_sale",
}

# ── Tenure parsing ────────────────────────────────────────────────────────────

_TENURE_PATTERNS = [
    # "99 yrs from 01/01/2005"
    (re.compile(r"(\d+)\s*yr[s]?\s*from\s*(\d{2}/\d{2}/\d{4})", re.I),
     lambda m: (int(m.group(1)), _parse_date(m.group(2)).year)),
    # "99 years from 2005"
    (re.compile(r"(\d+)\s*year[s]?\s*from\s*(\d{4})", re.I),
     lambda m: (int(m.group(1)), int(m.group(2)))),
    # "99 yrs lease commencing from 2012" (URA API format)
    (re.compile(r"(\d+)\s*yr[s]?\s+(?:lease\s+)?commencing\s+from\s+(\d{4})", re.I),
     lambda m: (int(m.group(1)), int(m.group(2)))),
    # "99 yrs from 2005" (yrs without years)
    (re.compile(r"(\d+)\s*yr[s]?\s+from\s+(\d{4})", re.I),
     lambda m: (int(m.group(1)), int(m.group(2)))),
]

def _parse_date(s):
    for fmt in ("%d/%m/%Y", "%m/%Y", "%Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")


def parse_tenure(raw: str):
    """
    Returns (tenure_type, lease_years, start_year).
    tenure_type: 'freehold' | '999yr' | '9999yr' | '99yr' | 'other'
    lease_years: int or None
    start_year:  int or None
    """
    if not isinstance(raw, str):
        return "unknown", None, None
    r = raw.strip().lower()
    if "freehold" in r:
        return "freehold", None, None
    if r.startswith("9999"):
        return "9999yr", 9999, None
    if r.startswith("999"):
        return "999yr", 999, None
    for pattern, extractor in _TENURE_PATTERNS:
        m = pattern.search(r)
        if m:
            years, start = extractor(m)
            return "99yr", years, start
    # fallback: try to extract leading number
    m = re.match(r"(\d+)", r)
    if m:
        years = int(m.group(1))
        tenure_type = "99yr" if years <= 103 else "999yr"
        return tenure_type, years, None
    return "other", None, None


def compute_remaining_lease_vec(
    tenure_type: pd.Series,
    lease_years: pd.Series,
    tenure_start_year: pd.Series,
    sale_year: pd.Series,
) -> pd.Series:
    """Vectorised remaining-lease computation (no row-wise apply)."""
    is_long = tenure_type.isin(["freehold", "999yr", "9999yr"])
    has_data = lease_years.notna() & tenure_start_year.notna()
    remaining = (lease_years - (sale_year - tenure_start_year)).clip(lower=0)
    return np.where(is_long, 99.0, np.where(has_data, remaining, np.nan))

# ── Floor range parsing ───────────────────────────────────────────────────────

def parse_floor_range(raw) -> tuple[float | None, str]:
    """
    Returns (floor_midpoint, floor_band).
    floor_band: 'basement' | 'low' | 'mid' | 'high' | 'penthouse' | 'unknown'
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, "unknown"
    r = raw.strip().upper()
    if r in ("B1", "B2", "B3", "BASEMENT"):
        return -1.0, "basement"
    m = re.match(r"(\d+)\s*[-–TO]+\s*(\d+)", r)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        mid = (lo + hi) / 2
    else:
        m2 = re.match(r"(\d+)", r)
        if m2:
            mid = float(m2.group(1))
        else:
            return None, "unknown"
    if mid <= 5:
        band = "low"
    elif mid <= 15:
        band = "mid"
    elif mid <= 30:
        band = "high"
    else:
        band = "penthouse"
    return mid, band

# ── District to CCR/RCR/OCR ──────────────────────────────────────────────────

_DISTRICT_SEGMENT = {
    1: "CCR", 2: "CCR", 3: "CCR", 4: "CCR",
    5: "RCR",
    6: "CCR",
    7: "RCR", 8: "RCR",
    9: "CCR", 10: "CCR", 11: "CCR",
    12: "RCR", 13: "RCR", 14: "RCR", 15: "RCR",
    16: "OCR", 17: "OCR", 18: "OCR", 19: "OCR",
    20: "RCR",
    21: "OCR", 22: "OCR", 23: "OCR",
    24: "OCR", 25: "OCR", 26: "OCR", 27: "OCR", 28: "OCR",
}

# Prestige tier (for ordinal encoding as proxy for location quality)
_DISTRICT_TIER = {
    # Tier 1 — prime (D9, D10, D11 Orchard/Holland/Newton)
    9: 1, 10: 1, 11: 1,
    # Tier 2 — near-prime city fringe
    1: 2, 2: 2, 3: 2, 4: 2, 6: 2,
    # Tier 3 — RCR fringe
    5: 3, 7: 3, 8: 3, 12: 3, 13: 3, 14: 3, 15: 3, 20: 3,
    # Tier 4 — OCR mass market
    16: 4, 17: 4, 18: 4, 19: 4, 21: 4, 22: 4, 23: 4,
    24: 4, 25: 4, 26: 4, 27: 4, 28: 4,
}

# ── Type normalisation ────────────────────────────────────────────────────────

_TYPE_MAP = {
    "condominium":           "condo",
    "apartment":             "condo",
    "executive condominium": "ec",
    "ec":                    "ec",
    "semi-detached house":   "semi_d",
    "semi detached house":   "semi_d",
    "detached house":        "detached",
    "terraced house":        "terrace",
    "strata semi-detached":  "strata_landed",
    "strata detached":       "strata_landed",
    "strata terrace":        "strata_landed",
    "cluster house":         "cluster",
}

def _normalise_type(raw):
    if not isinstance(raw, str):
        return "unknown"
    return _TYPE_MAP.get(raw.strip().lower(), raw.strip().lower())

# ── Column normaliser ─────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in _COL_MAP:
            rename[col] = _COL_MAP[key]
    df = df.rename(columns=rename)
    # Ensure area_sqft: convert from sqm if needed
    if "area_sqft" not in df.columns and "area_sqm" in df.columns:
        df["area_sqft"] = df["area_sqm"] * 10.7639
    return df

# ── Main loading functions ────────────────────────────────────────────────────

def load_ura_csv(path: str) -> pd.DataFrame:
    """Load a single URA transaction CSV and return a normalised DataFrame."""
    # URA files sometimes have a header row with metadata — skip rows until header
    raw = pd.read_csv(path, nrows=3, header=None)
    # Find the row index that contains "Project" or "District"
    skip = 0
    for i, row in raw.iterrows():
        row_str = " ".join(str(v) for v in row.values).lower()
        if "project" in row_str or "district" in row_str or "tenure" in row_str:
            skip = i
            break
    df = pd.read_csv(path, skiprows=skip, dtype=str)
    df = _normalise_columns(df)
    return df


def load_ura_folder(folder: str = "data/raw") -> pd.DataFrame:
    """Load all URA CSVs from a folder and concatenate."""
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {folder!r}. "
            "Download from https://www.ura.gov.sg/reis/dataDL"
        )
    frames = [load_ura_csv(f) for f in sorted(files)]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(files)} files.")
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns a cleaned DataFrame ready for
    feature engineering.
    """
    df = df.copy()

    # ── 1. Drop fully empty rows ──────────────────────────────────────────────
    df.dropna(how="all", inplace=True)

    # ── 2. Numeric conversions ────────────────────────────────────────────────
    for col in ("price", "area_sqft", "psf", "postal_district", "num_units"):
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce"
            )

    # ── 3. Parse date ─────────────────────────────────────────────────────────
    # URA REIS dates are typically "MM/YYYY" (e.g. "01/2024") with no day.
    # We try multiple formats in order of specificity.
    if "date_of_sale" in df.columns:
        raw_dates = df["date_of_sale"].astype(str).str.strip()
        # Diagnostic: show unique sample so format issues are obvious
        sample = raw_dates.dropna().unique()[:5]
        print(f"      Date sample: {list(sample)}")

        parsed = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")
        if parsed.isna().all():
            # Try URA's common MM/YYYY format
            parsed = pd.to_datetime(raw_dates, format="%m/%Y", errors="coerce")
        if parsed.isna().all():
            # Try MMM-YY  (e.g. "Jan-24")
            parsed = pd.to_datetime(raw_dates, format="%b-%y", errors="coerce")
        if parsed.isna().all():
            # Try MMM YYYY (e.g. "Jan 2024")
            parsed = pd.to_datetime(raw_dates, format="%b %Y", errors="coerce")
        if parsed.isna().all():
            # Try URA API MMYY format (e.g. "0921" = September 2021)
            parsed = pd.to_datetime(raw_dates, format="%m%y", errors="coerce")
        if parsed.isna().all():
            # Try Q1 2024 / 1Q2024 style → extract year+quarter manually
            def _parse_quarter(s):
                m = re.search(r"(\d)[Qq]\s*(\d{4})|(\d{4})\s*[Qq](\d)", str(s))
                if m:
                    q = int(m.group(1) or m.group(4))
                    y = int(m.group(2) or m.group(3))
                    return pd.Timestamp(year=y, month=(q - 1) * 3 + 1, day=1)
                return pd.NaT
            parsed = raw_dates.apply(_parse_quarter)
        df["date_of_sale"] = parsed
        n_parsed = parsed.notna().sum()
        print(f"      Dates parsed: {n_parsed}/{len(df)} rows")
    df.dropna(subset=["date_of_sale"], inplace=True)

    # ── 4. Derive PSF if missing ──────────────────────────────────────────────
    if "psf" not in df.columns:
        df["psf"] = np.nan
    mask = df["psf"].isna() & df["price"].notna() & df["area_sqft"].notna() & (df["area_sqft"] > 0)
    df.loc[mask, "psf"] = df.loc[mask, "price"] / df.loc[mask, "area_sqft"]

    # ── 5. Drop rows without PSF or area ─────────────────────────────────────
    df.dropna(subset=["psf", "area_sqft"], inplace=True)
    df = df[(df["psf"] > 100) & (df["psf"] < 10_000)]
    df = df[(df["area_sqft"] > 100) & (df["area_sqft"] < 30_000)]

    # ── 6. Normalise property type ────────────────────────────────────────────
    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].apply(_normalise_type)

    # ── 7. Filter: private condo + landed only ────────────────────────────────
    valid_types = {"condo", "ec", "semi_d", "detached", "terrace",
                   "strata_landed", "cluster"}
    df = df[df["property_type"].isin(valid_types)]

    # ── 8. Parse tenure ───────────────────────────────────────────────────────
    if "tenure_raw" in df.columns:
        parsed = df["tenure_raw"].apply(parse_tenure)
        df["tenure_type"]       = parsed.apply(lambda x: x[0])
        df["lease_years"]       = pd.to_numeric(parsed.apply(lambda x: x[1]), errors="coerce")
        df["tenure_start_year"] = pd.to_numeric(parsed.apply(lambda x: x[2]), errors="coerce")
        df["remaining_lease"]   = compute_remaining_lease_vec(
            df["tenure_type"], df["lease_years"],
            df["tenure_start_year"], df["date_of_sale"].dt.year,
        )
        df["is_freehold"]       = (df["tenure_type"] == "freehold").astype(int)
    else:
        df["tenure_type"] = "unknown"
        df["remaining_lease"] = np.nan
        df["is_freehold"] = 0

    # ── 9. Parse floor range ──────────────────────────────────────────────────
    if "floor_range" in df.columns:
        parsed_floor = df["floor_range"].apply(parse_floor_range)
        df["floor_midpoint"] = parsed_floor.apply(lambda x: x[0])
        df["floor_band"]     = parsed_floor.apply(lambda x: x[1])
    else:
        df["floor_midpoint"] = np.nan
        df["floor_band"]     = "unknown"

    # ── 10. Postal district — derive segment if not present ───────────────────
    if "postal_district" in df.columns:
        df["postal_district"] = df["postal_district"].astype("Int64")
        if "market_segment" not in df.columns or df["market_segment"].isna().all():
            df["market_segment"] = df["postal_district"].map(_DISTRICT_SEGMENT)
        df["district_tier"] = df["postal_district"].map(_DISTRICT_TIER)
    else:
        df["district_tier"] = np.nan

    # ── 11. Market segment encoding ───────────────────────────────────────────
    seg_map = {"CCR": 0, "RCR": 1, "OCR": 2}
    df["market_segment_code"] = df["market_segment"].map(seg_map)

    # ── 12. Type of sale encoding ─────────────────────────────────────────────
    if "type_of_sale" in df.columns:
        ts_map = {"new sale": 0, "sub sale": 1, "resale": 2}
        df["sale_type_code"] = df["type_of_sale"].str.strip().str.lower().map(ts_map)
    else:
        df["sale_type_code"] = np.nan

    # ── 13. Project-level stats (will be updated in features.py) ─────────────
    df["project_name"] = df["project_name"].str.strip().str.upper()

    # ── 14. Reset index ───────────────────────────────────────────────────────
    df.reset_index(drop=True, inplace=True)

    print(
        f"Clean: {len(df):,} rows | "
        f"{df['date_of_sale'].min().date()} to {df['date_of_sale'].max().date()} | "
        f"{df['project_name'].nunique():,} projects | "
        f"PSF median ${df['psf'].median():,.0f}"
    )
    return df


def save_processed(df: pd.DataFrame, path: str = "data/processed/transactions.parquet"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} rows -> {path}")


def load_processed(path: str = "data/processed/transactions.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
