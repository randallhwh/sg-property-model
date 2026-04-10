"""
valuation.py — FairValueModel: the main inference class.

Usage:
    from src.valuation import FairValueModel

    fvm = FairValueModel.load()   # loads model + data from default paths
    result = fvm.estimate(spec)   # spec is a dict of property attributes
    print(result["psf_estimate"], result["price_estimate"])
    print(result["comps"])        # DataFrame of comparable transactions
"""

import os
import numpy as np
import pandas as pd
from datetime import date

from src import features as F
from src import model as M

# ── Spec keys (what the user provides) ───────────────────────────────────────
# Minimum required: property_type, postal_district, area_sqft, tenure_raw
# Optional:         floor_midpoint, date_of_sale (defaults to today)

SPEC_DEFAULTS = {
    "property_type":    "condo",
    "postal_district":  10,
    "market_segment":   None,       # auto-derived from district if None
    "tenure_raw":       "Freehold",
    "area_sqft":        1000.0,
    "floor_midpoint":   None,       # will impute from district median
    "floor_range":      None,
    "project_name":     None,
    "top_year":         None,       # year of TOP (Temporary Occupation Permit)
    "lat":              None,
    "lng":              None,
    "date_of_sale":     None,       # defaults to today
    "type_of_sale":     "Resale",
}

# ── District → approximate lat/lng centroid (for distance features) ───────────
_DISTRICT_CENTROIDS = {
    1:  (1.2834, 103.8507), 2:  (1.2792, 103.8442), 3:  (1.2885, 103.8196),
    4:  (1.2637, 103.8207), 5:  (1.3063, 103.7800), 6:  (1.2897, 103.8500),
    7:  (1.3010, 103.8561), 8:  (1.3106, 103.8594), 9:  (1.3028, 103.8337),
    10: (1.3085, 103.8031), 11: (1.3189, 103.8360), 12: (1.3286, 103.8629),
    13: (1.3230, 103.8833), 14: (1.3150, 103.8888), 15: (1.3059, 103.9097),
    16: (1.3246, 103.9411), 17: (1.3735, 103.9505), 18: (1.3518, 103.9420),
    19: (1.3643, 103.8794), 20: (1.3568, 103.8166), 21: (1.3302, 103.7764),
    22: (1.3393, 103.7050), 23: (1.3795, 103.7491), 24: (1.3973, 103.7413),
    25: (1.4201, 103.8084), 26: (1.3942, 103.8470), 27: (1.4042, 103.7985),
    28: (1.3701, 103.8457),
}

_DISTRICT_SEGMENT_MAP = {
    1: "CCR", 2: "CCR", 3: "CCR", 4: "CCR", 6: "CCR",
    9: "CCR", 10: "CCR", 11: "CCR",
    5: "RCR", 7: "RCR", 8: "RCR", 12: "RCR", 13: "RCR",
    14: "RCR", 15: "RCR", 20: "RCR",
}

_DISTRICT_TIER_MAP = {
    9: 1, 10: 1, 11: 1, 1: 2, 2: 2, 3: 2, 4: 2, 6: 2,
    5: 3, 7: 3, 8: 3, 12: 3, 13: 3, 14: 3, 15: 3, 20: 3,
}


def _spec_to_row(spec: dict) -> pd.DataFrame:
    """Convert a property spec dict into a single-row DataFrame."""
    s = {**SPEC_DEFAULTS, **spec}

    # Date default
    if s["date_of_sale"] is None:
        s["date_of_sale"] = pd.Timestamp(date.today())
    else:
        s["date_of_sale"] = pd.Timestamp(s["date_of_sale"])

    # Market segment
    district = int(s["postal_district"])
    if s["market_segment"] is None:
        s["market_segment"] = _DISTRICT_SEGMENT_MAP.get(district, "OCR")

    seg_map = {"CCR": 0, "RCR": 1, "OCR": 2}
    s["market_segment_code"] = seg_map.get(s["market_segment"], 2)
    s["district_tier"] = _DISTRICT_TIER_MAP.get(district, 4)

    # Tenure
    from src.pipeline import parse_tenure
    tenure_type, lease_years, tenure_start_year = parse_tenure(s["tenure_raw"])
    s["tenure_type"]        = tenure_type
    s["lease_years"]        = lease_years
    s["tenure_start_year"]  = tenure_start_year
    s["is_freehold"]        = 1 if tenure_type == "freehold" else 0

    # Floor
    if s["floor_range"] and s["floor_midpoint"] is None:
        from src.pipeline import parse_floor_range
        s["floor_midpoint"], s["floor_band"] = parse_floor_range(s["floor_range"])
    if s["floor_midpoint"] is None:
        s["floor_midpoint"] = 10.0
    if "floor_band" not in s or s["floor_band"] is None:
        fm = float(s["floor_midpoint"])
        s["floor_band"] = (
            "low" if fm <= 5 else "mid" if fm <= 15 else
            "high" if fm <= 30 else "penthouse"
        )

    # Location: lat/lng from district centroid if not provided
    if s["lat"] is None or s["lng"] is None:
        s["lat"], s["lng"] = _DISTRICT_CENTROIDS.get(district, (1.35, 103.82))

    row = pd.DataFrame([s])
    return row


def _build_spec_features(row: pd.DataFrame, df_history: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for a single spec row, using history for context."""
    # Time features (add_time_features reads top_year if present)
    row = F.add_time_features(row)

    # Distance features (lat/lng available)
    row = F.add_distance_features(row)

    # Property features
    row["log_area"] = np.log1p(row["area_sqft"])
    row["floor_midpoint_sq"] = row["floor_midpoint"] ** 2

    # Age at sale
    if "tenure_start_year" in row.columns and row["tenure_start_year"].notna().any():
        row["age_at_sale"] = row["year"] - row["tenure_start_year"].fillna(row["year"] - 10)
        row["age_at_sale"] = row["age_at_sale"].clip(0, 80)
    else:
        row["age_at_sale"] = 10.0

    # Years since TOP — already computed by add_time_features if top_year column present
    if "years_since_top" not in row.columns or row["years_since_top"].isna().all():
        top_yr = row["top_year"].iloc[0] if "top_year" in row.columns else None
        if top_yr is not None and not pd.isna(top_yr):
            row["years_since_top"] = max(0, int(row["year"].iloc[0]) - int(top_yr))
        else:
            row["years_since_top"] = np.nan

    # Remaining lease
    if "remaining_lease" not in row.columns or row["remaining_lease"].isna().all():
        if row["is_freehold"].iloc[0]:
            row["remaining_lease"] = 99.0
        elif row.get("lease_years") is not None:
            row["remaining_lease"] = 99.0
        else:
            row["remaining_lease"] = 60.0

    # Sale type
    ts_map = {"new sale": 0, "sub sale": 1, "resale": 2}
    row["sale_type_code"] = row["type_of_sale"].str.strip().str.lower().map(ts_map).fillna(2)

    # Type dummies
    type_dummies = pd.get_dummies(row["property_type"], prefix="type")
    row = pd.concat([row, type_dummies], axis=1)

    # Band dummies
    band_dummies = pd.get_dummies(row["floor_band"], prefix="band")
    row = pd.concat([row, band_dummies], axis=1)

    # Rolling features from history
    district = int(row["postal_district"].iloc[0])
    sale_date = row["date_of_sale"].iloc[0]

    # Monthly district PSF (trailing 3 months)
    history_window = df_history[
        (df_history["postal_district"] == district) &
        (df_history["date_of_sale"] <= sale_date) &
        (df_history["date_of_sale"] >= sale_date - pd.DateOffset(months=3))
    ]
    row["monthly_district_psf"] = (
        history_window["psf"].median() if len(history_window) > 0
        else df_history[df_history["postal_district"] == district]["psf"].median()
    )

    # Project rolling PSF
    proj = str(row["project_name"].iloc[0]).upper() if row["project_name"].iloc[0] else None
    if proj:
        proj_history = df_history[df_history["project_name"] == proj]
        row["project_rolling_psf"] = (
            proj_history["psf"].median() if len(proj_history) > 0
            else row["monthly_district_psf"].iloc[0]
        )
        row["txn_count_project_6m"] = len(proj_history)
    else:
        row["project_rolling_psf"] = row["monthly_district_psf"]
        row["txn_count_project_6m"] = 0

    # Project mean PSF (target encoding)
    if proj:
        proj_mean = df_history[df_history["project_name"] == proj]["psf"].mean()
        row["project_mean_psf"] = proj_mean if not np.isnan(proj_mean) else row["monthly_district_psf"].iloc[0]
    else:
        row["project_mean_psf"] = row["monthly_district_psf"]

    return row


# ── Comparable selection ──────────────────────────────────────────────────────

def get_project_history(
    spec: dict,
    df_history: pd.DataFrame,
    n: int = 15,
) -> pd.DataFrame:
    """
    Return recent transactions for the same project, sorted newest first.
    Falls back to same-street transactions if project has < 3 records.
    Returns empty DataFrame if no project name given.
    """
    s = {**SPEC_DEFAULTS, **spec}
    proj = str(s.get("project_name") or "").strip().upper()
    if not proj:
        return pd.DataFrame()

    rows = df_history[df_history["project_name"].str.upper() == proj].copy()

    # Fallback: same street if project has very few records
    if len(rows) < 3 and "street_name" in df_history.columns:
        street = df_history.loc[
            df_history["project_name"].str.upper() == proj, "street_name"
        ].iloc[0] if len(rows) > 0 else None
        if street:
            rows = df_history[
                df_history["street_name"].str.upper() == str(street).strip().upper()
            ].copy()

    if len(rows) == 0:
        return pd.DataFrame()

    rows = rows.sort_values("date_of_sale", ascending=False).head(n)

    output_cols = ["property_type", "area_sqft", "floor_range",
                   "psf", "price", "tenure_raw", "date_of_sale", "type_of_sale"]
    available = [c for c in output_cols if c in rows.columns]
    rows = rows[available].copy()
    rows["date_of_sale"] = rows["date_of_sale"].dt.strftime("%b %Y")
    rows = rows.rename(columns={
        "property_type":  "Type",
        "area_sqft":      "Area (sqft)",
        "floor_range":    "Floor",
        "psf":            "PSF ($)",
        "price":          "Price ($)",
        "tenure_raw":     "Tenure",
        "date_of_sale":   "Date",
        "type_of_sale":   "Sale Type",
    })
    if "PSF ($)" in rows.columns:
        rows["PSF ($)"] = rows["PSF ($)"].round(0).astype(int)
    if "Price ($)" in rows.columns:
        rows["Price ($)"] = rows["Price ($)"].fillna(0).astype(int)
    return rows.reset_index(drop=True)


def _resolve_spec_coords(spec: dict, df_history: pd.DataFrame) -> tuple[float, float]:
    """
    Return best (x_svy21, y_svy21) for a spec, in priority order:
      1. Project name  → median SVY21 of that project's transactions in history
      2. Spec x/y      → already provided
      3. District centroid (rough fallback only)
    """
    proj = str(spec.get("project_name") or "").strip().upper()
    if proj and "x_svy21" in df_history.columns:
        proj_rows = df_history[df_history["project_name"].str.upper() == proj]
        px = pd.to_numeric(proj_rows["x_svy21"], errors="coerce").dropna()
        py = pd.to_numeric(proj_rows["y_svy21"], errors="coerce").dropna()
        if len(px) > 0 and len(py) > 0:
            return float(px.median()), float(py.median())

    if spec.get("x_svy21") and spec.get("y_svy21"):
        return float(spec["x_svy21"]), float(spec["y_svy21"])

    # Reverse lat/lng to SVY21 if available
    if spec.get("lat") and spec.get("lng"):
        lat, lng = float(spec["lat"]), float(spec["lng"])
        return (lng - 103.8333333) * 111279.0 + 28001.642, (lat - 1.3666667) * 110574.0 + 38744.572

    # Last resort: district centroid
    dist = int(spec.get("postal_district", 10))
    c = _DISTRICT_CENTROIDS.get(dist, (1.35, 103.82))
    return (c[1] - 103.8333333) * 111279.0 + 28001.642, (c[0] - 1.3666667) * 110574.0 + 38744.572


def get_comps(
    spec: dict,
    df_history: pd.DataFrame,
    n: int = 10,
    lookback_months: int = 36,
) -> pd.DataFrame:
    """
    Find nearby comparable transactions from other projects.

    Location strategy (hard filter, not just scoring):
      - Resolve reference point from project coords in history (most precise)
      - Filter to transactions within 1 km; expand to 2 km / 3 km if < n*2 results
      - Final ranking by: type match, area similarity, tenure, recency, PSF proximity
    """
    from src.pipeline import parse_tenure as _parse_tenure

    s = {**SPEC_DEFAULTS, **spec}
    sale_date = pd.Timestamp(s["date_of_sale"] or date.today())
    cutoff = sale_date - pd.DateOffset(months=lookback_months)

    all_cands = df_history[df_history["date_of_sale"] >= cutoff].copy()
    if len(all_cands) == 0:
        all_cands = df_history.copy()

    # Exclude same project — it gets its own table
    proj = str(s.get("project_name") or "").strip().upper()
    if proj:
        all_cands = all_cands[all_cands["project_name"].str.upper() != proj]

    # ── Resolve reference coordinates ─────────────────────────────────────────
    ref_x, ref_y = _resolve_spec_coords(s, df_history)

    # ── Hard radius filter — expand until we have enough candidates ───────────
    cands = pd.DataFrame()
    if "x_svy21" in all_cands.columns and "y_svy21" in all_cands.columns:
        cx = pd.to_numeric(all_cands["x_svy21"], errors="coerce")
        cy = pd.to_numeric(all_cands["y_svy21"], errors="coerce")
        has_coords = cx.notna() & cy.notna()
        dist_m = np.sqrt((cx - ref_x) ** 2 + (cy - ref_y) ** 2)
        all_cands["_dist_m"] = dist_m

        for radius in (1000, 2000, 3000, 5000):
            within = all_cands[has_coords & (dist_m <= radius)]
            if len(within) >= n * 2:
                cands = within.copy()
                break
        if len(cands) == 0:
            # No coords or nothing within 5 km — fall back to same district
            cands = all_cands[
                ~has_coords | (dist_m > 5000)
            ].copy()
            # merge with whatever was within 5 km
            cands = pd.concat([all_cands[has_coords & (dist_m <= 5000)], cands]).copy()
            if len(cands) == 0:
                cands = all_cands.copy()
    else:
        cands = all_cands.copy()
        cands["_dist_m"] = np.nan

    # ── Score within the radius pool ─────────────────────────────────────────
    cands["_score"] = 0.0

    # Type match (most important filter — only compare like with like)
    cands["_score"] += (cands["property_type"] == s["property_type"]).astype(float) * 5

    # Area similarity: ±15% = 4 pts, ±30% = 2 pts, ±50% = 1 pt
    area = float(s["area_sqft"])
    area_ratio = (cands["area_sqft"] - area).abs() / area
    cands["_score"] += (area_ratio < 0.15).astype(float) * 4
    cands["_score"] += (area_ratio < 0.30).astype(float) * 2
    cands["_score"] += (area_ratio < 0.50).astype(float) * 1

    # Tenure match
    tenure_type, _, _ = _parse_tenure(s["tenure_raw"])
    if "tenure_type" in cands.columns:
        cands["_score"] += (cands["tenure_type"] == tenure_type).astype(float) * 3

    # Recency
    months_ago = (sale_date - cands["date_of_sale"]).dt.days / 30
    cands["_score"] += (months_ago <= 6).astype(float)  * 3
    cands["_score"] += (months_ago <= 12).astype(float) * 2
    cands["_score"] += (months_ago <= 24).astype(float) * 1

    # Closer = better (secondary tiebreaker via distance bucket)
    if "_dist_m" in cands.columns:
        d = cands["_dist_m"].fillna(9999)
        cands["_score"] += (d < 500).astype(float)  * 2
        cands["_score"] += (d < 1000).astype(float) * 1

    # ── Select top-n, deduplicated by project ─────────────────────────────────
    # One row per project to avoid a single project flooding the table
    top_pool = cands.nlargest(n * 5, "_score")
    seen_projects: set = set()
    rows_out = []
    for _, row in top_pool.iterrows():
        p = str(row.get("project_name", "")).upper()
        if p not in seen_projects:
            seen_projects.add(p)
            rows_out.append(row)
        if len(rows_out) >= n:
            break

    if not rows_out:
        return pd.DataFrame()

    output_cols = ["project_name", "property_type", "postal_district",
                   "area_sqft", "psf", "price", "floor_range", "tenure_raw",
                   "tenure_type", "tenure_start_year",
                   "date_of_sale", "type_of_sale"]
    available = [c for c in output_cols if c in cands.columns]
    comps = pd.DataFrame(rows_out)[available].copy()
    comps["date_of_sale"] = comps["date_of_sale"].dt.strftime("%b %Y")

    # ── Derive TOP year column ────────────────────────────────────────────────
    # Leasehold: tenure_start_year is a close proxy for TOP (within ~1-2 years)
    # Freehold / 999yr: show "FH" / "999yr" — no TOP inference possible
    def _top_label(row):
        tt = row.get("tenure_type", "")
        if tt in ("freehold", "9999yr"):
            return "FH"
        if tt == "999yr":
            return "999yr"
        yr = row.get("tenure_start_year")
        if pd.notna(yr):
            return str(int(yr))
        return "-"

    comps["TOP"] = comps.apply(_top_label, axis=1)

    rename_map = {
        "project_name":    "Project",
        "property_type":   "Type",
        "postal_district": "D",
        "area_sqft":       "Area (sqft)",
        "psf":             "PSF ($)",
        "price":           "Price ($)",
        "floor_range":     "Floor",
        "tenure_raw":      "Tenure",
        "date_of_sale":    "Date",
        "type_of_sale":    "Sale Type",
    }
    drop_cols = ["tenure_type", "tenure_start_year"]
    comps = comps.drop(columns=[c for c in drop_cols if c in comps.columns])
    comps = comps.rename(columns={k: v for k, v in rename_map.items() if k in comps.columns})
    # Reorder so TOP appears after Tenure
    col_order = ["Project", "Type", "D", "Area (sqft)", "PSF ($)", "Price ($)",
                 "Floor", "Tenure", "TOP", "Date", "Sale Type"]
    comps = comps[[c for c in col_order if c in comps.columns]]
    if "PSF ($)" in comps.columns:
        comps["PSF ($)"] = comps["PSF ($)"].round(0).astype(int)
    if "Price ($)" in comps.columns:
        comps["Price ($)"] = comps["Price ($)"].fillna(0).astype(int)
    return comps.reset_index(drop=True)


# ── Main FairValueModel class ─────────────────────────────────────────────────

class FairValueModel:
    """
    Wrapper combining the trained LightGBM model with the transaction history
    for comparable lookup and context features.
    """

    def __init__(self, model_artifacts: dict, df_history: pd.DataFrame):
        self.model           = model_artifacts["model"]
        self.feature_cols    = model_artifacts["feature_cols"]
        self.metadata        = model_artifacts["metadata"]
        self.quantile_models = model_artifacts.get("quantile_models")
        self.df_history      = df_history

    @classmethod
    def load(
        cls,
        model_dir:    str = "models",
        history_path: str = "data/processed/transactions.parquet",
    ) -> "FairValueModel":
        artifacts = M.load(model_dir)
        df = pd.read_parquet(history_path)
        return cls(artifacts, df)

    def estimate(self, spec: dict) -> dict:
        """
        Estimate fair value for a property spec.

        Returns dict:
          psf_estimate     : float — point estimate PSF
          price_estimate   : float — point estimate total price
          ci_low_psf       : float — 80% CI lower bound PSF
          ci_high_psf      : float — 80% CI upper bound PSF
          ci_low_price     : float
          ci_high_price    : float
          comps            : pd.DataFrame — comparable transactions
          shap_values      : pd.DataFrame or None
          district_median_psf : float
          pct_vs_district  : float — estimate vs district median (%)
        """
        s = {**SPEC_DEFAULTS, **spec}
        area = float(s["area_sqft"])

        # Build feature row
        row = _spec_to_row(s)
        row = _build_spec_features(row, self.df_history)

        # Align to training feature columns
        X = pd.DataFrame(columns=self.feature_cols)
        X = pd.concat([X, row], ignore_index=True)
        X = X[self.feature_cols].fillna(0).astype(float)

        # Point estimate
        log_psf = self.model.predict(X)[0]
        psf_est = float(np.exp(log_psf))

        # Confidence interval via quantile models
        ci_low_psf = ci_high_psf = None
        if self.quantile_models:
            try:
                ci_low_psf  = float(np.exp(self.quantile_models[0.10].predict(X)[0]))
                ci_high_psf = float(np.exp(self.quantile_models[0.90].predict(X)[0]))
            except Exception:
                pass
        if ci_low_psf is None:
            # Fallback: ±15% heuristic (typical MAPE for this type of model)
            ci_low_psf  = psf_est * 0.85
            ci_high_psf = psf_est * 1.15

        # District context
        district = int(s["postal_district"])
        d_median = self.df_history[
            self.df_history["postal_district"] == district
        ]["psf"].median()
        pct_vs_district = (psf_est / d_median - 1) * 100 if d_median else 0.0

        # SHAP (single row)
        shap_row = None
        try:
            shap_df = M.explain(self.model, X)
            shap_row = shap_df.iloc[0].sort_values(key=abs, ascending=False)
        except Exception:
            pass

        # Comparables
        project_hist = get_project_history(s, self.df_history, n=15)
        comps        = get_comps(s, self.df_history, n=10)

        return {
            "psf_estimate":       round(psf_est),
            "price_estimate":     round(psf_est * area),
            "ci_low_psf":         round(ci_low_psf),
            "ci_high_psf":        round(ci_high_psf),
            "ci_low_price":       round(ci_low_psf * area),
            "ci_high_price":      round(ci_high_psf * area),
            "district_median_psf": round(d_median) if d_median else None,
            "pct_vs_district":    round(pct_vs_district, 1),
            "project_history":    project_hist,
            "comps":              comps,
            "shap_values":        shap_row,
        }

    def model_info(self) -> dict:
        return {
            **self.metadata,
            "feature_count":   len(self.feature_cols),
            "history_rows":    len(self.df_history),
            "history_from":    str(self.df_history["date_of_sale"].min().date()),
            "history_to":      str(self.df_history["date_of_sale"].max().date()),
            "projects":        int(self.df_history["project_name"].nunique()),
        }
