"""
propertyguru.py — Scrape a PropertyGuru listing URL and return a normalised
property spec dict ready for the fair value estimator.

Data is extracted from __NEXT_DATA__ → props.pageProps.pageData.data:
  listingData      — price, area, districtCode, tenure, propertyType, propertyName, postcode
  listingDetail    — property{topYear,topMonth,tenureCode,typeCode}, propertyUnit{floorLevelCode}
  detailsData      — metatable items (TOP, floor, tenure text — human-readable backup)

Returns a dict with keys matching the Estimate tab inputs:
  prop_type, district, area_sqft, tenure_raw, floor_level,
  top_year, sale_type, listing_price, project_name, listing_url, raw
"""

import re
import json
import requests
import cloudscraper
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.propertyguru.com.sg/",
}

# ── Property type normalisation ───────────────────────────────────────────────

# PropertyGuru typeCode → our model type
_TYPE_CODE_MAP = {
    "APT":  "condo",   # Apartment
    "CONDO":"condo",   # Condominium
    "EC":   "ec",      # Executive Condominium
    "SEMI": "semi_d",  # Semi-Detached
    "DET":  "detached",# Detached / Bungalow
    "TERRA":"terrace", # Terrace
    "CLUS": "strata_landed",  # Cluster
    "STRAT":"strata_landed",  # Strata landed
    "LAND": "detached",# Landed (generic)
    "GCB":  "detached",# Good Class Bungalow
}

_TYPE_TEXT_MAP = {
    "condominium":           "condo",
    "condo":                 "condo",
    "apartment":             "condo",
    "executive condominium": "ec",
    "ec":                    "ec",
    "semi-detached":         "semi_d",
    "semi detached":         "semi_d",
    "detached":              "detached",
    "bungalow":              "detached",
    "good class bungalow":   "detached",
    "gcb":                   "detached",
    "terraced":              "terrace",
    "terrace":               "terrace",
    "terrace house":         "terrace",
    "terraced house":        "terrace",
    "strata landed":         "strata_landed",
    "cluster house":         "strata_landed",
}

def _normalise_type_code(code: str | None) -> str | None:
    if not code:
        return None
    return _TYPE_CODE_MAP.get(code.strip().upper())

def _normalise_type_text(raw: str | None) -> str | None:
    if not raw:
        return None
    return _TYPE_TEXT_MAP.get(raw.strip().lower())


# ── Tenure ────────────────────────────────────────────────────────────────────

def _tenure_from_code(code: str | None, top_year: int | None = None) -> str:
    """
    PG tenureCode: F=Freehold, L=Leasehold(99yr), 999=999yr, 9999=9999yr
    """
    if not code:
        return "Freehold"
    c = str(code).strip().upper()
    if c in ("F", "FH", "FREEHOLD"):
        return "Freehold"
    if c in ("999",):
        return "999 yrs from 1885"
    if c in ("9999",):
        return "9999 yrs"
    # Leasehold — use top_year as best proxy for lease start
    if top_year:
        return f"99 yrs from {top_year}"
    return "99 yrs from 2000"


# ── Floor level ───────────────────────────────────────────────────────────────

_FLOOR_CODE_MAP = {
    "LOW":        5,
    "MID":       15,
    "MIDDLE":    15,
    "HIGH":      25,
    "PENTHOUSE": 40,
    "GROUND":     1,
    "BASEMENT":  -1,
}

def _floor_from_code(code: str | None) -> int:
    if not code:
        return 10
    return _FLOOR_CODE_MAP.get(code.strip().upper(), 10)


# ── District from districtCode ────────────────────────────────────────────────

def _district_from_code(code: str | None) -> int | None:
    """'D15' → 15"""
    if not code:
        return None
    m = re.search(r"(\d{1,2})$", str(code).strip())
    return int(m.group(1)) if m else None


def _district_from_postal(postal: str | None) -> int | None:
    """Map a 6-digit Singapore postal code to a district number."""
    if not postal:
        return None
    m = re.match(r"(\d{2})", str(postal).strip())
    if not m:
        return None
    prefix = int(m.group(1))
    _PREFIX_DIST = {
        1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
        7: 2, 8: 2,
        14: 3, 15: 3, 16: 3,
        9: 4, 10: 4,
        11: 5, 12: 5, 13: 5,
        17: 6,
        18: 7, 19: 7,
        20: 8, 21: 8,
        22: 9, 23: 9,
        24: 10, 25: 10, 26: 10, 27: 10,
        28: 11, 29: 11, 30: 11,
        31: 12, 32: 12, 33: 12,
        34: 13, 35: 13, 36: 13, 37: 13,
        38: 14, 39: 14, 40: 14, 41: 14,
        42: 15, 43: 15, 44: 15, 45: 15,
        46: 16, 47: 16, 48: 16,
        49: 17, 50: 17, 81: 17,
        51: 18, 52: 18,
        53: 19, 54: 19, 55: 19, 82: 19,
        56: 20, 57: 20,
        58: 21, 59: 21,
        60: 22, 61: 22, 62: 22, 63: 22, 64: 22,
        65: 23, 66: 23, 67: 23, 68: 23,
        69: 24, 70: 24, 71: 24,
        72: 25, 73: 25,
        77: 26, 78: 26,
        75: 27, 76: 27,
        79: 28, 80: 28,
    }
    return _PREFIX_DIST.get(prefix)


# ── URL-only fallback ─────────────────────────────────────────────────────────

def _project_from_url(url: str) -> str:
    """Extract a best-guess project name from the listing URL slug."""
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"-?\d{6,}$", "", slug)
    slug = re.sub(r"^(for-sale|for-rent|for-lease)-?", "", slug, flags=re.IGNORECASE)
    return slug.replace("-", " ").title().strip()


# ── HTTP fetch ────────────────────────────────────────────────────────────────

def _fetch_html(url: str) -> str:
    """
    Try multiple strategies to fetch PropertyGuru HTML.
    Strategy 1: cloudscraper (works on residential IPs / locally)
    Strategy 2: httpx with HTTP/2 (sometimes bypasses cloud IP blocks)
    Raises requests.HTTPError on final failure.
    """
    # Strategy 1: cloudscraper
    try:
        scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows"})
        resp = scraper.get(url, headers=_HEADERS, timeout=20)
        if resp.status_code == 200:
            return resp.text
        first_status = resp.status_code
    except Exception as e:
        first_status = str(e)

    # Strategy 2: httpx with HTTP/2
    try:
        import httpx
        _h2_headers = {
            **_HEADERS,
            "Accept-Encoding": "gzip, deflate, br",
            "Sec-Fetch-Dest":  "document",
            "Sec-Fetch-Mode":  "navigate",
            "Sec-Fetch-Site":  "none",
            "Sec-Fetch-User":  "?1",
            "Upgrade-Insecure-Requests": "1",
        }
        with httpx.Client(http2=True, follow_redirects=True, timeout=20) as client:
            r = client.get(url, headers=_h2_headers)
            if r.status_code == 200:
                return r.text
    except Exception:
        pass

    # Both failed — raise with the original status so callers can handle it
    raise requests.HTTPError(
        f"{first_status} for url: {url}",
        response=type("R", (), {"status_code": first_status if isinstance(first_status, int) else 403})(),
    )


# ── Main scraper ──────────────────────────────────────────────────────────────

def scrape(url: str) -> dict:
    """
    Scrape a PropertyGuru listing and return a spec dict.

    Primary source: __NEXT_DATA__ → pageProps.pageData.data
      - listingData:   price, floorArea, districtCode, tenure, propertyTypeCode, propertyName
      - listingDetail.property: topYear, topMonth, tenureCode, typeCode, newProject
      - listingDetail.propertyUnit: floorLevelCode (LOW/MID/HIGH/PENTHOUSE)
    """
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    result = {
        "prop_type":     "condo",
        "district":      None,
        "area_sqft":     None,
        "tenure_raw":    "Freehold",
        "floor_level":   10,
        "top_year":      None,
        "sale_type":     "Resale",
        "listing_price": None,
        "project_name":  "",
        "address":       "",
        "listing_url":   url,
        "raw":           {},
    }

    # ── Parse __NEXT_DATA__ ───────────────────────────────────────────────────
    next_tag = soup.find("script", id="__NEXT_DATA__")
    if not next_tag:
        result["project_name"] = _project_from_url(url)
        return result

    try:
        nd = json.loads(next_tag.string or "")
    except json.JSONDecodeError:
        result["project_name"] = _project_from_url(url)
        return result

    # Navigate to pageData.data (current PG structure as of 2025)
    page_data = (
        nd.get("props", {})
          .get("pageProps", {})
          .get("pageData", {})
          .get("data", {})
    )

    ld   = page_data.get("listingData", {})       # flat listing fields
    det  = page_data.get("listingDetail", {})      # nested detail object
    prop = det.get("property", {})                 # project-level info
    pu   = det.get("propertyUnit", {})             # unit-level info

    result["raw"] = {k: v for k, v in ld.items() if not isinstance(v, (dict, list))}

    # ── Price ─────────────────────────────────────────────────────────────────
    price_val = ld.get("price") or det.get("price", {}).get("value")
    if price_val:
        try:
            result["listing_price"] = int(price_val)
        except (TypeError, ValueError):
            pass

    # ── Area ──────────────────────────────────────────────────────────────────
    area = ld.get("floorArea") or ld.get("landArea")
    if area:
        try:
            result["area_sqft"] = float(area)
        except (TypeError, ValueError):
            pass

    # ── Project name ──────────────────────────────────────────────────────────
    name = (prop.get("name") or ld.get("propertyName") or ld.get("localizedTitle") or "").strip()
    if name and "propertyguru" not in name.lower():
        result["project_name"] = name
    if not result["project_name"]:
        result["project_name"] = _project_from_url(url)

    # ── District ──────────────────────────────────────────────────────────────
    result["district"] = (
        _district_from_code(ld.get("districtCode"))
        or _district_from_postal(ld.get("postcode"))
    )

    # ── TOP year ──────────────────────────────────────────────────────────────
    top_yr = prop.get("topYear")
    if top_yr:
        try:
            result["top_year"] = int(top_yr)
        except (TypeError, ValueError):
            pass

    # ── Tenure ────────────────────────────────────────────────────────────────
    tenure_code = prop.get("tenureCode") or ld.get("tenure")
    result["tenure_raw"] = _tenure_from_code(tenure_code, result["top_year"])

    # ── Property type ─────────────────────────────────────────────────────────
    # Priority: prop.typeCode > ld.propertyTypeCode > ld.propertyType (text)
    ptype = (
        _normalise_type_code(prop.get("typeCode"))
        or _normalise_type_code(ld.get("propertyTypeCode"))
        or _normalise_type_text(ld.get("propertyType"))
        or "condo"
    )
    result["prop_type"] = ptype

    # ── Floor level ───────────────────────────────────────────────────────────
    floor_code = pu.get("floorLevelCode")
    if floor_code:
        result["floor_level"] = _floor_from_code(floor_code)

    # ── Sale type ─────────────────────────────────────────────────────────────
    is_new = prop.get("newProject") or ld.get("isNewProject")
    listing_type = str(ld.get("listingType") or "").upper()
    if is_new:
        result["sale_type"] = "New Sale"
    elif "RENT" in listing_type:
        result["sale_type"] = "Resale"  # shouldn't happen on sale listings
    # default stays "Resale"

    # ── Address ───────────────────────────────────────────────────────────────
    result["address"] = ld.get("streetName", "")

    return result
