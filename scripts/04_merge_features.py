"""
Step 4: occurrence_base.csv + gee_site_features.csv + gee_date_features.csv +
        hydro_features.csv -> SMX_occurrence.csv

Join keys:
  gee_site_features / hydro_features : one row per location_id (static features)
  gee_date_features                  : one row per site_id + sampling_date (varies by date)

IMPORTANT: `country` comes ONLY from occurrence_base.csv (the raw_SMX.csv
value for that exact row). Never merge a `country` column in from any
site-level table. A handful of your sites (border regions like
Georgia/Ukraine, Romania/Ukraine near the Black Sea coast) share *exact*
identical coordinates across two different countries in the raw data --
grouping only by (latitude, longitude) is still fine for environmental
features (same physical point -> same elevation/rainfall/etc. regardless
of which country's monitoring agency reported it), but it means a site-level
`country` column would be wrong for one of the two countries at those 5
locations. Keeping `country` sourced from the row-level base table sidesteps
that entirely.
"""
import pandas as pd

from paths import (
    OCCURRENCE_BASE,
    UNIQUE_SITES,
    GEE_SITE_FEATURES,
    GEE_DATE_FEATURES,
    HYDRO_FEATURES,
    SMX_OCCURRENCE,
)

# Exact 21-column-group order from the schema. Columns 19a-g (pH/EC/TOC/etc.)
# are "extended-only, matched subset only" -- meaning null is the expected
# default for most rows, not a sign something's missing. This script creates
# them as empty placeholders so the schema is structurally complete even
# before you've sourced any national water-quality data; fill them in later
# by matching site+date against a compatible national monitoring database
# (see README) without needing to touch this merge logic again.
EXTENDED_ONLY_COLS = ["pH", "EC", "TOC", "COD", "nutrients", "turbidity", "suspended_solids"]

FINAL_COLUMNS = [
    "site_id", "country", "latitude", "longitude", "sampling_date", "water_matrix",
    "concentration_ng_L", "LoD", "LoQ", "detection_status",
    "month", "year", "day_of_year", "season", "month_sin", "month_cos",
    "rainfall_1d", "rainfall_7d", "rainfall_30d", "rainfall_anomaly",
    "air_temperature", "evapotranspiration",
    "pct_urban_land", "pct_agri_land", "pct_industrial_land", "pct_forest_land",
    "population_density",
    "catchment_area", "river_order", "elevation", "dist_to_WWTP", "dist_to_urban_area",
    *EXTENDED_ONLY_COLS,
    "Y_occ", "Y_det",
]

base = pd.read_csv(OCCURRENCE_BASE, parse_dates=["sampling_date"])
sites = pd.read_csv(UNIQUE_SITES)  # location_id, latitude, longitude, site_ids, n_samples
site_feats = pd.read_csv(GEE_SITE_FEATURES)
date_feats = pd.read_csv(GEE_DATE_FEATURES, parse_dates=["sampling_date"])
hydro_feats = pd.read_csv(HYDRO_FEATURES)

n_before = len(base)

# only pull location_id from `sites` -- never country/site_ids/n_samples,
# those are bookkeeping columns for building unique_sites.csv, not schema fields
df = base.merge(sites[["location_id", "latitude", "longitude"]],
                 on=["latitude", "longitude"], how="left")
df = df.merge(site_feats.drop(columns=[c for c in ["latitude", "longitude", "country"]
                                        if c in site_feats.columns]),
              on="location_id", how="left")
df = df.merge(hydro_feats.drop(columns=[c for c in ["latitude", "longitude", "country",
                                                      "site_ids", "n_samples"]
                                         if c in hydro_feats.columns]),
              on="location_id", how="left")
df = df.merge(date_feats.drop(columns=[c for c in date_feats.columns if c not in
                                        ["site_id", "sampling_date", "rainfall_1d", "rainfall_7d",
                                         "rainfall_30d", "rainfall_anomaly", "air_temperature",
                                         "evapotranspiration"]]),
              on=["site_id", "sampling_date"], how="left")

assert len(df) == n_before, f"Row count changed on merge: {n_before} -> {len(df)} (check for duplicate keys)"

missing_pct = df.isna().mean().sort_values(ascending=False) * 100
print("Missingness by column (%):")
print(missing_pct[missing_pct > 0])

missing_cols = [c for c in FINAL_COLUMNS if c not in df.columns]
if missing_cols:
    print(f"\nNote: schema columns not produced by this pipeline yet: {missing_cols}")
    print("Adding them as empty placeholders (expected for extended-only fields).")
    for c in missing_cols:
        df[c] = pd.NA
extra_cols = [c for c in df.columns if c not in FINAL_COLUMNS and c != "location_id"]
if extra_cols:
    print(f"WARNING: unexpected extra columns found and will be dropped: {extra_cols}")

df = df[FINAL_COLUMNS]
df.to_csv(SMX_OCCURRENCE, index=False)
print(f"\nWrote {len(df)} rows x {len(df.columns)} columns to {SMX_OCCURRENCE}")
