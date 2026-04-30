"""
ura_api.py — Fetch private residential transactions from URA Data Service.

Authentication flow:
  1. POST to Token endpoint with AccessKey header → get daily token
  2. GET PMI_Resi_Transaction batches 1–4 with AccessKey + Token headers

The token is valid for 24 hours. It is cached in data/external/ura_token.json
so repeated runs in the same day don't re-request it.

Usage:
    from src.ura_api import fetch_all_transactions
    df = fetch_all_transactions()          # returns cleaned DataFrame
    df.to_csv("data/raw/ura_api.csv", index=False)

Or from the CLI:
    python -m src.ura_api

Set your access key in .env (or as an environment variable):
    URA_ACCESS_KEY=cdeea5b0-ef64-48ad-9562-40ac8b414f02
"""

import json
import os
import time
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL     = "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1"
TOKEN_URL    = "https://eservice.ura.gov.sg/uraDataService/insertNewToken/v1"
TOKEN_CACHE        = Path("data/external/ura_token.json")
RENTAL_CACHE       = Path("data/external/rental_median.csv")
RENTAL_CACHE_DATE  = Path("data/external/rental_median_date.txt")
BATCHES      = [1, 2, 3, 4]
REQUEST_PAUSE = 1.5   # seconds between batch requests (be polite to the API)

# Browser-like headers to pass the WAF bot-protection layer
_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://eservice.ura.gov.sg/maps/api/",
    "Origin":          "https://eservice.ura.gov.sg",
}

# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_access_key() -> str:
    """Read access key from Streamlit secrets, env var, or .env file."""
    # 1. Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        key = st.secrets.get("URA_ACCESS_KEY", "").strip()
        if key:
            return key
    except Exception:
        pass
    # 2. Environment variable
    key = os.environ.get("URA_ACCESS_KEY", "").strip()
    if key:
        return key
    # 3. .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("URA_ACCESS_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise ValueError(
        "URA_ACCESS_KEY not found. Set it in .streamlit/secrets.toml, .env, or as an environment variable."
    )


def _load_cached_token() -> str | None:
    """Return today's cached token, or None if stale/missing."""
    if not TOKEN_CACHE.exists():
        return None
    try:
        data = json.loads(TOKEN_CACHE.read_text())
        if data.get("date") == str(date.today()) and data.get("token"):
            return data["token"]
    except Exception:
        pass
    return None


def _save_token(token: str) -> None:
    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE.write_text(json.dumps({"date": str(date.today()), "token": token}))


def _make_session(access_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    s.headers["AccessKey"] = access_key
    # Warm up the domain so the WAF issues a valid session cookie
    s.get("https://eservice.ura.gov.sg/maps/api/", timeout=15)
    return s


def get_token(access_key: str, session: requests.Session | None = None) -> str:
    """Get a daily auth token (uses cache if still valid today)."""
    cached = _load_cached_token()
    if cached:
        return cached

    if session is None:
        session = _make_session(access_key)

    resp = session.get(TOKEN_URL, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if body.get("Status") != "Success":
        raise RuntimeError(f"Token request failed: {body}")
    token = body["Result"]
    _save_token(token)
    return token


# ── Fetch ─────────────────────────────────────────────────────────────────────

def _fetch_batch(batch: int, token: str, session: requests.Session) -> list[dict]:
    """Fetch one batch of PMI_Resi_Transaction data."""
    resp = session.get(
        BASE_URL,
        params={"service": "PMI_Resi_Transaction", "batch": batch},
        headers={"Token": token},
        timeout=60,
    )
    resp.raise_for_status()
    if resp.text.strip().startswith("<"):
        raise RuntimeError(f"Batch {batch}: WAF challenge — got HTML. Try again later.")
    body = resp.json()
    if body.get("Status") != "Success":
        raise RuntimeError(f"Batch {batch} failed: {body.get('Message', body)}")
    return body.get("Result") or []


def fetch_raw(access_key: str | None = None) -> list[dict]:
    """
    Fetch all 4 transaction batches from the URA API.
    Returns a flat list of raw transaction dicts.
    """
    if access_key is None:
        access_key = _get_access_key()

    print("  Establishing session…")
    session = _make_session(access_key)
    token   = get_token(access_key, session)
    print(f"  Token acquired.")

    records: list[dict] = []
    for batch in BATCHES:
        print(f"  Fetching batch {batch}/{len(BATCHES)}…", end=" ", flush=True)
        rows = _fetch_batch(batch, token, session)
        print(f"{len(rows):,} rows")
        records.extend(rows)
        if batch < len(BATCHES):
            time.sleep(REQUEST_PAUSE)

    print(f"  Total raw records: {len(records):,}")
    return records


# ── Normalise to pipeline schema ──────────────────────────────────────────────

_SALE_TYPE_MAP = {"1": "New Sale", "2": "Sub Sale", "3": "Resale"}

def _sqm_to_sqft(sqm) -> float | None:
    try:
        return round(float(sqm) * 10.7639, 1)
    except (TypeError, ValueError):
        return None


def _parse_floor_range(floor_range: str | None) -> int | None:
    """Extract the lower floor number from a URA floor range string (e.g. '06-10' → 6)."""
    if not floor_range or not isinstance(floor_range, str):
        return None
    parts = floor_range.strip().split("-")
    try:
        return int(parts[0])
    except ValueError:
        return None


def normalise(records: list[dict]) -> pd.DataFrame:
    """
    Convert raw URA API records to the same column schema used by the CSV pipeline.

    The URA API returns one record per PROJECT, with a nested "transaction" list.
    This function flattens each project × transaction into one row.

    Project-level fields:  project, street, x, y, marketSegment
    Transaction-level:     area, floorRange, noOfUnits, contractDate,
                           typeOfSale, price, propertyType, district,
                           typeOfArea, tenure, planningRegion, planningArea
    """
    rows = []
    for r in records:
        # Project-level fields (shared across all transactions for this project)
        project_name    = r.get("project")
        street_name     = r.get("street")
        x_svy21         = r.get("x")
        y_svy21         = r.get("y")
        market_segment  = r.get("marketSegment")

        transactions = r.get("transaction") or []
        if not transactions:
            continue  # skip projects with no transaction data

        for t in transactions:
            area_sqft = _sqm_to_sqft(t.get("area"))
            price_val = None
            try:
                price_val = float(str(t.get("price", "")).replace(",", ""))
            except (TypeError, ValueError):
                pass
            psf = None
            if price_val and area_sqft and area_sqft > 0:
                psf = round(price_val / area_sqft, 2)

            floor_range = t.get("floorRange")
            floor_num   = _parse_floor_range(floor_range)

            type_of_sale_raw = str(t.get("typeOfSale", "")).strip()
            type_of_sale     = _SALE_TYPE_MAP.get(type_of_sale_raw, type_of_sale_raw)

            rows.append({
                "project_name":    project_name,
                "street_name":     street_name,
                "property_type":   t.get("propertyType"),
                "postal_district": t.get("district"),
                "market_segment":  market_segment,
                "tenure_raw":      t.get("tenure"),
                "type_of_sale":    type_of_sale,
                "num_units":       t.get("noOfUnits"),
                "price":           price_val,
                "area_sqft":       area_sqft,
                "psf":             psf,
                "date_of_sale":    t.get("contractDate"),
                "floor_range":     floor_range,
                "floor_num":       floor_num,
                "type_of_area":    t.get("typeOfArea"),
                "planning_region": t.get("planningRegion"),
                "planning_area":   t.get("planningArea"),
                "x_svy21":         x_svy21,
                "y_svy21":         y_svy21,
            })

    df = pd.DataFrame(rows)

    # Numeric coercions
    for col in ("postal_district", "num_units", "price", "area_sqft", "psf", "floor_num"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fetch_all_transactions(access_key: str | None = None) -> pd.DataFrame:
    """End-to-end: fetch + normalise. Returns a DataFrame ready for the pipeline."""
    records = fetch_raw(access_key)
    df = normalise(records)
    print(f"  Normalised: {len(df):,} rows, {df.columns.tolist()}")
    return df


# ── Rental median ─────────────────────────────────────────────────────────────

def fetch_rental_median(access_key: str | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch per-project quarterly median rental PSF from URA PMI_Resi_Rental_Median.

    Returns a DataFrame with one row per project × quarter:
      project_name, street, district, ref_period, ref_date,
      median_psf_month, psf25_month, psf75_month, x_svy21, y_svy21

    Cached to data/external/rental_median.csv for the current day.
    """
    if not force_refresh and RENTAL_CACHE.exists() and RENTAL_CACHE_DATE.exists():
        if RENTAL_CACHE_DATE.read_text().strip() == str(date.today()):
            return pd.read_csv(RENTAL_CACHE)

    if access_key is None:
        access_key = _get_access_key()

    print("  Fetching URA rental median data…")
    session = _make_session(access_key)
    token   = get_token(access_key, session)

    resp = session.get(
        BASE_URL,
        params={"service": "PMI_Resi_Rental_Median", "batch": 1},
        headers={"Token": token},
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("Status") != "Success":
        raise RuntimeError(f"Rental median fetch failed: {body.get('Message', body)}")

    rows = []
    for r in body.get("Result") or []:
        project = r.get("project")
        for q in (r.get("rentalMedian") or []):
            ref = q.get("refPeriod", "")
            try:
                yr, qn = int(ref[:4]), int(ref[5])
                ref_date = pd.Timestamp(year=yr, month=(qn - 1) * 3 + 1, day=1)
            except Exception:
                ref_date = pd.NaT
            rows.append({
                "project_name":     project,
                "street":           r.get("street"),
                "district":         q.get("district"),
                "ref_period":       ref,
                "ref_date":         ref_date,
                "median_psf_month": q.get("median"),
                "psf25_month":      q.get("psf25"),
                "psf75_month":      q.get("psf75"),
                "x_svy21":          r.get("x"),
                "y_svy21":          r.get("y"),
            })

    df = pd.DataFrame(rows)
    RENTAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RENTAL_CACHE, index=False)
    RENTAL_CACHE_DATE.write_text(str(date.today()))
    print(f"  Rental median: {len(df):,} project-quarter rows, {df['project_name'].nunique():,} projects")
    return df


def get_project_rental(
    project_name: str | None,
    district: int | None,
    df_rental: pd.DataFrame,
) -> dict | None:
    """
    Return the most recent rental median for a project, falling back to district average.

    Returns dict: median_psf_month, psf25_month, psf75_month, ref_period, source
    or None if no data available.
    """
    if df_rental is None or len(df_rental) == 0:
        return None

    df = df_rental.copy()
    df["ref_date"] = pd.to_datetime(df["ref_date"], errors="coerce")

    def _safe(v):
        f = float(v)
        return None if (f != f) else f  # NaN check (NaN != NaN)

    # Try exact project match
    if project_name:
        proj_upper = str(project_name).strip().upper()
        sub = df[df["project_name"].str.strip().str.upper() == proj_upper]
        if len(sub) > 0:
            row = sub.sort_values("ref_date").iloc[-1]
            median = _safe(row["median_psf_month"])
            if median is None:
                return None
            return {
                "median_psf_month": median,
                "psf25_month":      _safe(row["psf25_month"]) or median,
                "psf75_month":      _safe(row["psf75_month"]) or median,
                "ref_period":       row["ref_period"],
                "source":           "project",
            }

    # Fallback: district average for the latest available quarter
    if district is not None:
        sub = df[df["district"].astype(str) == str(district)]
        if len(sub) > 0:
            latest_date = sub["ref_date"].max()
            recent = sub[sub["ref_date"] == latest_date]
            ref_str = recent["ref_period"].iloc[0] if len(recent) > 0 else "?"
            median = _safe(recent["median_psf_month"].median())
            if median is None:
                return None
            return {
                "median_psf_month": median,
                "psf25_month":      _safe(recent["psf25_month"].median()) or median,
                "psf75_month":      _safe(recent["psf75_month"].median()) or median,
                "ref_period":       ref_str,
                "source":           "district",
            }

    return None


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download URA transaction data via API")
    parser.add_argument("--key",    default=None, help="URA AccessKey (overrides .env)")
    parser.add_argument("--out",    default="data/raw/ura_api.csv", help="Output CSV path")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV instead of overwriting")
    args = parser.parse_args()

    print("Fetching URA private residential transactions…")
    df = fetch_all_transactions(args.key)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.append and out.exists():
        existing = pd.read_csv(out)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["project_name", "date_of_sale", "area_sqft", "price"]
        )
        print(f"  After dedup merge: {len(df):,} rows")

    df.to_csv(out, index=False)
    print(f"Saved -> {out}  ({len(df):,} rows)")
