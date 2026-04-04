"""
propertyguru.py — Scrape a PropertyGuru listing URL and return a normalised
property spec dict ready for the fair value estimator.

Extraction priority:
  1. JSON-LD structured data  (fastest, most reliable)
  2. Meta tags / og: tags
  3. HTML element fallbacks

Returns a dict with keys matching the Estimate tab inputs:
  prop_type, district, area_sqft, tenure_raw, floor_level,
  sale_type, listing_price, project_name, listing_url, raw
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

_TYPE_MAP = {
    "condominium":          "condo",
    "condo":                "condo",
    "apartment":            "condo",
    "executive condominium":"ec",
    "ec":                   "ec",
    "semi-detached":        "semi_d",
    "semi detached":        "semi_d",
    "semi-d":               "semi_d",
    "detached":             "detached",
    "bungalow":             "detached",
    "good class bungalow":  "detached",
    "gcb":                  "detached",
    "terraced":             "terrace",
    "terrace":              "terrace",
    "terrace house":        "terrace",
    "strata landed":        "strata_landed",
    "cluster house":        "strata_landed",
}

def _normalise_type(raw: str | None) -> str:
    if not raw:
        return "condo"
    return _TYPE_MAP.get(raw.strip().lower(), "condo")


# ── District extraction ───────────────────────────────────────────────────────

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


def _district_from_text(text: str) -> int | None:
    """Try to extract district number from free text (e.g. 'District 10')."""
    m = re.search(r"[Dd]istrict\s*(\d{1,2})|D(\d{2})", text)
    if m:
        return int(m.group(1) or m.group(2))
    return None


# ── Floor level extraction ────────────────────────────────────────────────────

def _parse_floor(raw: str | None) -> int:
    if not raw:
        return 5
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else 5


# ── Price extraction ──────────────────────────────────────────────────────────

def _parse_price(raw) -> int | None:
    if raw is None:
        return None
    s = re.sub(r"[^\d]", "", str(raw))
    return int(s) if s else None


# ── Area extraction ───────────────────────────────────────────────────────────

def _parse_area(raw, unit: str = "sqft") -> float | None:
    if raw is None:
        return None
    m = re.search(r"[\d,]+\.?\d*", str(raw).replace(",", ""))
    if not m:
        return None
    val = float(m.group().replace(",", ""))
    if "sqm" in str(raw).lower() or unit == "sqm":
        val = round(val * 10.7639, 1)
    return val


# ── Main scraper ──────────────────────────────────────────────────────────────

def scrape(url: str) -> dict:
    """
    Scrape a PropertyGuru listing and return a spec dict.
    Raises on HTTP error or if the page can't be parsed at all.
    """
    scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows"})
    resp = scraper.get(url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    result = {
        "prop_type":     "condo",
        "district":      None,
        "area_sqft":     None,
        "tenure_raw":    "Freehold",
        "floor_level":   5,
        "sale_type":     "Resale",
        "listing_price": None,
        "project_name":  "",
        "address":       "",
        "listing_url":   url,
        "raw":           {},
    }

    # ── 1. __NEXT_DATA__ (primary — PropertyGuru is Next.js) ──────────────────
    next_tag = soup.find("script", id="__NEXT_DATA__")
    listing  = {}
    if next_tag:
        try:
            nd = json.loads(next_tag.string or "")
            # Walk the props tree to find the listing object
            props = nd.get("props", {}).get("pageProps", {})
            # Try common key names
            for key in ("listing", "listingDetail", "data", "property"):
                if key in props and isinstance(props[key], dict):
                    listing = props[key]
                    break
            # Sometimes nested deeper
            if not listing:
                for v in props.values():
                    if isinstance(v, dict) and any(k in v for k in ("listingId", "listing_id", "price")):
                        listing = v
                        break
            result["raw"] = listing
        except (json.JSONDecodeError, AttributeError):
            pass

    if listing:
        # Price
        for pk in ("askingPrice", "price", "priceForDisplay"):
            if listing.get(pk):
                result["listing_price"] = _parse_price(listing[pk])
                break

        # Area
        for ak in ("floorArea", "landArea", "area", "builtUpArea"):
            if listing.get(ak):
                unit = "sqm" if listing.get("floorAreaUnit", "sqft").lower() in ("sqm", "m2") else "sqft"
                result["area_sqft"] = _parse_area(listing[ak], unit)
                break

        # Property type
        for tk in ("propertyType", "property_type", "type", "categoryName"):
            if listing.get(tk):
                result["prop_type"] = _normalise_type(str(listing[tk]))
                break

        # District
        for dk in ("districtId", "district", "postalDistrict"):
            if listing.get(dk):
                try:
                    result["district"] = int(str(listing[dk]).lstrip("D").lstrip("d"))
                except ValueError:
                    pass
                break
        if result["district"] is None:
            postal = listing.get("postalCode") or listing.get("postal_code")
            if postal:
                result["district"] = _district_from_postal(str(postal))

        # Project / building name
        for nk in ("projectName", "project_name", "buildingName", "name", "title"):
            v = listing.get(nk, "")
            if v and "propertyguru" not in str(v).lower():
                result["project_name"] = str(v)
                break

        # Tenure
        tenure_raw = listing.get("tenure") or listing.get("tenureLabel") or ""
        if tenure_raw:
            if "freehold" in tenure_raw.lower():
                result["tenure_raw"] = "Freehold"
            elif "999" in tenure_raw:
                result["tenure_raw"] = "999 yrs from 1885"
            else:
                m = re.search(r"(\d{4})", tenure_raw)
                result["tenure_raw"] = f"99 yrs from {m.group(1)}" if m else "99 yrs from 2000"

        # Floor
        for fk in ("floorLevel", "floor", "level", "floorRange"):
            if listing.get(fk):
                result["floor_level"] = _parse_floor(str(listing[fk]))
                break

        # Sale type
        st_raw = str(listing.get("listingType") or listing.get("saleType") or "")
        if re.search(r"new|launch|developer", st_raw, re.IGNORECASE):
            result["sale_type"] = "New Sale"
        elif re.search(r"sub", st_raw, re.IGNORECASE):
            result["sale_type"] = "Sub Sale"

    # ── 2. JSON-LD fallback ───────────────────────────────────────────────────
    if not listing:
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(tag.string or "")
                if not isinstance(data, dict):
                    continue
                if "offers" in data:
                    result["listing_price"] = _parse_price(
                        data["offers"].get("price") or data["offers"].get("lowPrice")
                    )
                if "floorSize" in data:
                    fs = data["floorSize"]
                    unit = fs.get("unitCode", "sqft") if isinstance(fs, dict) else "sqft"
                    result["area_sqft"] = _parse_area(
                        fs.get("value") if isinstance(fs, dict) else fs, unit
                    )
                addr = data.get("address") or {}
                if isinstance(addr, dict):
                    result["district"] = _district_from_postal(addr.get("postalCode"))
                if data.get("name") and "propertyguru" not in data["name"].lower():
                    result["project_name"] = data["name"]
            except (json.JSONDecodeError, AttributeError):
                continue

    # ── 3. Page text fallbacks ────────────────────────────────────────────────
    page_text = soup.get_text(" ", strip=True)

    # Property type — look specifically for the "Property Type" label in Property Details
    # e.g. "Property Type Terrace House" or "Property Type\nTerrace House"
    _pt_match = re.search(
        r"property\s*type[\s:\-]+([A-Za-z\s\-]+?)(?:\n|floor|tenure|district|price|bedrooms|$)",
        page_text, re.IGNORECASE
    )
    if _pt_match:
        result["prop_type"] = _normalise_type(_pt_match.group(1).strip())
    else:
        # Fallback: scan for specific landed/type keywords only (avoid generic "condo" from nav)
        _type_patterns = [
            (r"good class bungalow|gcb",         "detached"),
            (r"bungalow|detached house",          "detached"),
            (r"semi.?detached",                   "semi_d"),
            (r"terrace\s*house|terraced\s*house", "terrace"),
            (r"cluster house|strata landed",      "strata_landed"),
            (r"executive condominium",            "ec"),
        ]
        for pattern, ptype in _type_patterns:
            if re.search(pattern, page_text, re.IGNORECASE):
                result["prop_type"] = ptype
                break

    if result["listing_price"] is None:
        m = re.search(r"S?\$\s*([\d,]+)", page_text)
        if m:
            result["listing_price"] = _parse_price(m.group(1))

    if result["area_sqft"] is None:
        m = re.search(r"([\d,]+)\s*(sqft|sq ft|sqm|sq m)", page_text, re.IGNORECASE)
        if m:
            result["area_sqft"] = _parse_area(m.group(1), m.group(2).lower().replace(" ", ""))

    # Floor level from property details text
    _floor_m = re.search(
        r"(?:floor\s*level|floor\s*no\.?|level)\s*[:\-]?\s*(\d+)",
        page_text, re.IGNORECASE
    )
    if _floor_m:
        result["floor_level"] = _parse_floor(_floor_m.group(1))

    if result["district"] is None:
        m = re.search(r"Singapore\s+(\d{6})", page_text)
        if m:
            result["district"] = _district_from_postal(m.group(1))
        if result["district"] is None:
            result["district"] = _district_from_text(page_text)

    if not result["project_name"] or "propertyguru" in result["project_name"].lower():
        slug = url.rstrip("/").split("/")[-1]
        slug = re.sub(r"-?\d{6,}$", "", slug)
        slug = re.sub(r"^(for-sale|for-rent|for-lease)-?", "", slug, flags=re.IGNORECASE)
        result["project_name"] = slug.replace("-", " ").title().strip()

    if not re.search(r"freehold", page_text, re.IGNORECASE):
        if re.search(r"999.?yr", page_text, re.IGNORECASE):
            result["tenure_raw"] = "999 yrs from 1885"
        elif re.search(r"leasehold|99.?yr", page_text, re.IGNORECASE):
            m = re.search(r"99.?yr[s]?\s+from\s+(\d{4})", page_text, re.IGNORECASE)
            result["tenure_raw"] = f"99 yrs from {m.group(1)}" if m else "99 yrs from 2000"

    if re.search(r"new launch|new sale|direct developer", page_text, re.IGNORECASE):
        result["sale_type"] = "New Sale"
    elif re.search(r"sub.?sale", page_text, re.IGNORECASE):
        result["sale_type"] = "Sub Sale"

    return result
