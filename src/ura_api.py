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
TOKEN_CACHE  = Path("data/external/ura_token.json")
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
    print(f"Saved → {out}  ({len(df):,} rows)")
