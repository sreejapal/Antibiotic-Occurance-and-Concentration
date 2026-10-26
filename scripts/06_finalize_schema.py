"""
Step 6 (final): SMX_occurrence_with_water_quality.csv -> SMX_occurrence.csv

Assembles every column from the schema, including:
  - nutrients: a single aggregate column (sum of nitrate + nitrite +
    ammonium + phosphate), matching the schema's single "nutrients" field.
    Only computed when ALL FOUR species are present for that row -- summing
    a partial subset would understate total nutrient load and silently
    bias the feature. This means nutrients has lower coverage (33 rows)
    than the individual species (up to 94 rows) -- the four species
    columns are kept alongside it since they're more usable for modeling
    if you don't specifically need one combined number.
  - Y_conc: log10(concentration_ng_L + 1), the regression target alongside
    the existing Y_occ/Y_det classification targets.
  - match-quality diagnostics (water_quality_station, _distance_km,
    _date_diff_days) kept at the end, not mixed into the core schema, so
    you can audit which rows have trustworthy water-quality matches
    without cluttering the main feature block.
"""
import numpy as np
import pandas as pd

from paths import SMX_OCCURRENCE_WQ, SMX_OCCURRENCE

IN_PATH = SMX_OCCURRENCE_WQ
OUT_PATH = SMX_OCCURRENCE

CORE_SCHEMA = [
    "site_id", "country", "latitude", "longitude", "sampling_date", "water_matrix",
    "concentration_ng_L", "LoD", "LoQ", "detection_status",
    "month", "year", "day_of_year", "season", "month_sin", "month_cos",
    "rainfall_1d", "rainfall_7d", "rainfall_30d", "rainfall_anomaly",
    "air_temperature", "evapotranspiration",
    "pct_urban_land", "pct_agri_land", "pct_industrial_land", "pct_forest_land",
    "population_density",
    "catchment_area", "river_order", "elevation", "dist_to_WWTP", "dist_to_urban_area",
    "pH", "EC", "TOC", "COD", "nutrients", "turbidity", "suspended_solids",
    "Y_conc", "Y_occ", "Y_det",
]

# kept, but appended after the core schema -- useful for auditing match
# quality, not part of the modeling feature set itself
DIAGNOSTIC_COLS = ["nitrate", "nitrite", "ammonium", "phosphate",
                    "water_quality_station", "water_quality_distance_km",
                    "water_quality_date", "water_quality_date_diff_days"]


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["sampling_date", "water_quality_date"])

    nutrient_species = ["nitrate", "nitrite", "ammonium", "phosphate"]
    all_present = df[nutrient_species].notna().all(axis=1)
    df["nutrients"] = np.where(all_present, df[nutrient_species].sum(axis=1), np.nan)
    print(f"nutrients: {all_present.sum()}/{len(df)} rows have all 4 species -> summed; "
          f"rest left NaN rather than understating the total.")

    df["Y_conc"] = np.log10(df["concentration_ng_L"] + 1)

    missing = [c for c in CORE_SCHEMA if c not in df.columns]
    if missing:
        print(f"Adding empty placeholders for schema columns not present: {missing}")
        for c in missing:
            df[c] = np.nan

    out = df[CORE_SCHEMA + [c for c in DIAGNOSTIC_COLS if c in df.columns]]
    out.to_csv(OUT_PATH, index=False)

    print(f"\nWrote {len(out)} rows x {len(out.columns)} columns to {OUT_PATH}")
    print("\nMissingness in core schema (%):")
    print((out[CORE_SCHEMA].isna().mean() * 100).round(2))


if __name__ == "__main__":
    main()
