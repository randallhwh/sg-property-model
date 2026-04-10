"""
train.py — CLI script to ingest URA data and train the fair value model.

Usage (from sg_property_model/ directory):
    python train.py                      # ingest + train, no geocoding
    python train.py --geocode            # + geocode via OneMap (slow)
    python train.py --cv                 # + walk-forward CV
    python train.py --data-only          # ingest only, skip training

Example workflow:
    1. Drop URA CSVs into data/raw/
    2. python train.py --cv
    3. streamlit run app.py
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline  import load_ura_folder, clean_transactions, save_processed, load_processed
from src.features  import build_features, get_feature_matrix
from src           import model as M


def run(args):
    print("=" * 60)
    print("SG Property Fair Value Model — Training Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # -- 1. Ingest -------------------------------------------------------------
    if not args.skip_ingest:
        print("\n[1/3] Ingesting URA data…")

        if args.fetch_api:
            print("      Fetching from URA API…")
            from src.ura_api import fetch_all_transactions
            import pathlib, datetime as _dt
            df_api = fetch_all_transactions()
            out = pathlib.Path(args.raw_dir) / f"ura_api_{_dt.date.today()}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df_api.to_csv(out, index=False)
            print(f"      Saved API data -> {out}")

        df_raw = load_ura_folder(args.raw_dir)
        print(f"      Loaded {len(df_raw):,} raw rows")

        print("\n[2/3] Cleaning transactions…")
        df_clean = clean_transactions(df_raw)

        print("\n[3/3] Engineering features…")
        df = build_features(df_clean, geocode=args.geocode)
        save_processed(df, args.data_path)
    else:
        print("\n[1/3] Loading existing processed data…")
        df = load_processed(args.data_path)
        print(f"      {len(df):,} rows loaded")

    if args.data_only:
        print("\nData-only mode. Exiting.")
        return

    # -- 2. Feature matrix -----------------------------------------------------
    print("\n-- Building feature matrix --")
    X, feat_cols = get_feature_matrix(df)
    y = df["psf"].values
    print(f"   Features: {len(feat_cols)}  |  Rows: {len(X):,}")

    # -- 3. Walk-forward CV (optional) -----------------------------------------
    if args.cv:
        print("\n-- Walk-forward cross-validation --")
        cv_results = M.walk_forward_cv(df, X, pd.Series(y))
        print("\nCV Summary:")
        print(cv_results.to_string(index=False))
        avg_mape = cv_results["mape"].mean()
        print(f"\nMean MAPE across years: {avg_mape:.2f}%")

    # -- 4. Train final model --------------------------------------------------
    print("\n-- Training final model --")
    cutoff = df["date_of_sale"].max() - pd.DateOffset(months=6)
    train_mask = df["date_of_sale"] < cutoff
    test_mask  = ~train_mask

    print(f"   Train set: {train_mask.sum():,} rows  ({df.loc[train_mask, 'date_of_sale'].min().date()} -> {cutoff.date()})")
    print(f"   Hold-out:  {test_mask.sum():,} rows  ({cutoff.date()} -> {df['date_of_sale'].max().date()})")

    X_tr, y_tr = X[train_mask], pd.Series(y)[train_mask]
    X_va, y_va = X[test_mask],  pd.Series(y)[test_mask]

    model = M.train(X_tr, y_tr, eval_set=(X_va, y_va))

    # -- 5. Evaluate -----------------------------------------------------------
    metrics = M.evaluate(model, X_va, y_va)
    print("\n-- Hold-out evaluation --")
    print(f"   MAE:         ${metrics['mae']:,.1f} psf")
    print(f"   RMSE:        ${metrics['rmse']:,.1f} psf")
    print(f"   MAPE:        {metrics['mape']:.2f}%")
    print(f"   Within 10%:  {metrics['within_10pct']:.1f}% of transactions")
    print(f"   Within 20%:  {metrics['within_20pct']:.1f}% of transactions")

    # -- 6. Quantile models ----------------------------------------------------
    print("\n-- Training quantile models (for CI) --")
    q_models = M.train_quantile_models(X_tr, y_tr, quantiles=(0.10, 0.90))
    print("   Done (P10 / P90)")

    # -- 7. Save ---------------------------------------------------------------
    metadata = {
        "trained_at":    datetime.now().isoformat(),
        "n_train":       int(train_mask.sum()),
        "n_holdout":     int(test_mask.sum()),
        **{f"holdout_{k}": v for k, v in metrics.items()},
    }
    M.save(model, feat_cols, metadata, q_models, args.model_dir)

    # -- 8. Feature importance -------------------------------------------------
    print("\n-- Top feature importances --")
    imp = pd.Series(model.feature_importances_, index=X.columns)
    top = imp.sort_values(ascending=False).head(15)
    for feat, val in top.items():
        bar = "#" * int(val / top.max() * 20)
        print(f"   {feat:<35} {bar}")

    print("\n" + "=" * 60)
    print(f"Done. Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train SG Property Fair Value Model")
    p.add_argument("--raw-dir",     default="data/raw",                       help="Folder containing URA CSVs")
    p.add_argument("--data-path",   default="data/processed/transactions.parquet", help="Output parquet path")
    p.add_argument("--model-dir",   default="models",                         help="Model output directory")
    p.add_argument("--geocode",     action="store_true",                       help="Geocode projects via OneMap API")
    p.add_argument("--cv",          action="store_true",                       help="Run walk-forward cross-validation")
    p.add_argument("--data-only",   action="store_true",                       help="Run pipeline only, skip training")
    p.add_argument("--skip-ingest", action="store_true",                       help="Skip ingestion, use existing parquet")
    p.add_argument("--fetch-api",   action="store_true",                       help="Fetch fresh data from URA API before training")
    args = p.parse_args()
    run(args)
