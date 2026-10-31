"""
Enrich the SMX_*_fixed.csv files: fill pct_urban_land, pct_agri_land,
pct_forest_land, elevation, population_density, catchment_area,
river_order, dist_to_WWTP wherever they're currently missing. Never
overwrites a real existing value - only fills gaps.

------------------------------------------------------------------------
WHAT THIS WILL AND WON'T FIX
------------------------------------------------------------------------
  FIXED:     pct_urban_land, pct_agri_land, pct_forest_land - via a
             1km-buffer ESA WorldCover classification around each site.
             NOTE: these were previously filled with a 0.0 placeholder
             (not NaN), so this version nulls out rows where all three
             are simultaneously 0 before filling, otherwise the old
             isna()-only fill logic silently does nothing to them.
  FIXED:     elevation (SRTM), population_density (GPWv4.11)
  FIXED:     catchment_area, river_order - via HydroATLAS sub-basins
  FIXED:     dist_to_WWTP - only if you set HYDROWASTE_PATH below
  FIXED:     dist_to_urban_area - via GHSL built-up-area distance
  NOT FIXED: pct_industrial_land - no clean global dataset reports this
             the way the column implies; left NaN.
  NOT FIXED: pH/EC/TOC/COD/nutrients/turbidity/suspended_solids/nitrate/
             nitrite/ammonium/phosphate/water_quality_* - need a
             national/EU water-quality monitoring database join, not
             something Earth Engine provides.
  NOT TOUCHED: rainfall_anomaly - already fixed with a real z-score in a
             previous pass; this script doesn't touch it again.

------------------------------------------------------------------------
SETUP (run this on YOUR machine, not in chat - needs your own credentials
and live internet access to Earth Engine)
------------------------------------------------------------------------
pip install earthengine-api pandas scikit-learn
python -c "import ee; ee.Authenticate()"     (one-time browser login)
Set EE_PROJECT below to your Cloud project ID.
Optional: download HydroWASTE_v10.csv (hydrosheds.org/products/hydrowaste)
and set HYDROWASTE_PATH to also fill dist_to_WWTP.

------------------------------------------------------------------------
RUN
------------------------------------------------------------------------
Put this script in the same folder as your SMX_*_fixed.csv files, then:
python smx_enrichment.py
"""

import os
import numpy as np
import pandas as pd

# ===========================================================================
# CONFIG
# ===========================================================================

EE_PROJECT = "smx-occurance"
SMX_FILES = [
    "data/output/SMX_concentration.csv",
    "data/output/SMX_occurrence.csv",
    "data/output/SMX_occurrence_with_water_quality.csv",
]
# Output files will be named <original>_geo.csv (see bottom of main()).

# Columns that were previously filled with a 0.0 placeholder instead of a
# real value. If pct_urban_land == pct_agri_land == pct_forest_land == 0.0
# simultaneously for a row, that's not a real measurement (no site is
# 0% urban AND 0% agricultural AND 0% forest) - treat it as missing so the
# Earth Engine fill actually overwrites it.
PLACEHOLDER_LANDCOVER_COLS = ["pct_urban_land", "pct_agri_land", "pct_forest_land"]

# GHSL Global Human Settlement Layer - urban extents, used for
# dist_to_urban_area (not covered by the original script).
GHSL_URBAN = "JRC/GHSL/P2023A/GHS_BUILT_S/2020"
GHSL_BUILT_THRESHOLD = 0.2  # fraction built-up to call a pixel "urban"


HYDROWASTE_PATH = None                    # <-- e.g. "HydroWASTE_v10.csv", or None to skip
HYDROWASTE_LAT_COL = "LAT_WWTP"
HYDROWASTE_LON_COL = "LON_WWTP"

ESA_WORLDCOVER = "ESA/WorldCover/v200"
GPW_POP_DENSITY = "CIESIN/GPWv411/GPW_Population_Density"
SRTM_ELEVATION = "USGS/SRTMGL1_003"
HYDROATLAS_BASINS = "WWF/HydroATLAS/v1/Basins/level12"
HYDROATLAS_FIELDS = ["HYBAS_ID", "UP_AREA", "ORD_STRA"]  # -> catchment_area, river_order

SCALE_ELEVATION = 30
SCALE_LANDCOVER = 10
LANDCOVER_BUFFER_M = 1000  # radius used for pct_urban/agri/forest_land

# ===========================================================================
# Earth Engine helpers
# ===========================================================================


def ee_points(sites, ee):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]), {"site_key": r.site_key})
        for r in sites.itertuples()
    ])


def _scalar(v):
    """EE sometimes returns [] (or other non-scalar) instead of None when
    a property genuinely has no match (e.g. a site outside any HydroATLAS
    basin polygon). Coerce anything non-scalar to None so it becomes a
    proper NaN instead of crashing pandas."""
    if isinstance(v, (list, dict)) or v is None:
        return None
    return v


def ee_static_features(sites, ee):
    """Elevation (SRTM) + population density (GPWv4.11)."""
    fc = ee_points(sites, ee)
    elevation_img = ee.Image(SRTM_ELEVATION).select("elevation")
    population_img = (ee.ImageCollection(GPW_POP_DENSITY)
                       .filterDate("2020-01-01", "2021-01-01").first()
                       .select("population_density"))
    combined = elevation_img.addBands(population_img)
    sampled = combined.reduceRegions(fc, ee.Reducer.first(), scale=SCALE_ELEVATION).getInfo()
    rows = [{"site_key": f["properties"].get("site_key"),
             "elevation_ee": _scalar(f["properties"].get("elevation")),
             "population_density_ee": _scalar(f["properties"].get("population_density"))}
            for f in sampled["features"]]
    return pd.DataFrame(rows)


def ee_landcover_pct(sites, ee):
    """% urban (built-up), agri (cropland), forest (tree cover) within a
    LANDCOVER_BUFFER_M radius of each site, from ESA WorldCover."""
    lc = ee.Image(ee.ImageCollection(ESA_WORLDCOVER).first()).select("Map")
    rows = []
    for r in sites.itertuples():
        geom = ee.Geometry.Point([r.longitude, r.latitude]).buffer(LANDCOVER_BUFFER_M)
        hist = lc.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(), geometry=geom, scale=SCALE_LANDCOVER, maxPixels=1e9
        ).getInfo()
        counts = hist.get("Map", {}) or {}
        total = sum(counts.values()) or 1
        rows.append({
            "site_key": r.site_key,
            "pct_urban_land_ee": 100 * counts.get("50", 0) / total,
            "pct_agri_land_ee": 100 * counts.get("40", 0) / total,
            "pct_forest_land_ee": 100 * counts.get("10", 0) / total,
        })
    return pd.DataFrame(rows)


def ee_hydroatlas(sites, ee):
    """catchment_area (UP_AREA) + river_order (ORD_STRA) from HydroATLAS basins."""
    fc = ee_points(sites, ee)
    basins = ee.FeatureCollection(HYDROATLAS_BASINS).select(HYDROATLAS_FIELDS)
    joined = ee.Join.saveFirst(matchKey="basin").apply(
        primary=fc, secondary=basins,
        condition=ee.Filter.intersects(leftField=".geo", rightField=".geo"),
    )

    def flatten(feat):
        feat = ee.Feature(feat)
        basin = ee.Feature(feat.get("basin"))
        return feat.copyProperties(basin, HYDROATLAS_FIELDS)

    flattened = ee.FeatureCollection(joined.map(flatten)).getInfo()
    rows = []
    for f in flattened["features"]:
        p = f["properties"]
        rows.append({
            "site_key": p.get("site_key"),
            "catchment_area_ee": _scalar(p.get("UP_AREA")),
            "river_order_ee": _scalar(p.get("ORD_STRA")),
        })
    return pd.DataFrame(rows)


def ee_dist_to_urban_area(sites, ee):
    """Distance (km) from each site to the nearest GHSL-classified urban
    built-up pixel. Fills dist_to_urban_area, which the original script
    left NaN (no dataset was wired up for it)."""
    built = ee.Image(GHSL_URBAN)
    urban_mask = built.gt(GHSL_BUILT_THRESHOLD).selfMask()
    # distance() returns meters to the nearest unmasked ("urban") pixel
    dist_img = urban_mask.fastDistanceTransform().sqrt().multiply(ee.Image.pixelArea().sqrt())
    fc = ee_points(sites, ee)
    sampled = dist_img.reduceRegions(fc, ee.Reducer.first(), scale=100).getInfo()
    rows = [{"site_key": f["properties"].get("site_key"),
             "dist_to_urban_area_ee": (f["properties"].get("first") or 0) / 1000.0}
            for f in sampled["features"]]
    return pd.DataFrame(rows)


def nearest_wwtp(sites, path, lat_col, lon_col):
    from sklearn.neighbors import BallTree
    try:
        wwtp = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        wwtp = pd.read_csv(path, encoding="latin1")
    wwtp = wwtp.dropna(subset=[lat_col, lon_col])
    tree = BallTree(np.radians(wwtp[[lat_col, lon_col]].values), metric="haversine")
    dist, _ = tree.query(np.radians(sites[["latitude", "longitude"]].values), k=1)
    out = sites[["site_key"]].copy()
    out["dist_to_WWTP_ee"] = dist[:, 0] * 6371.0  # km
    return out


# ===========================================================================
# Main
# ===========================================================================


def main():
    present = [f for f in SMX_FILES if os.path.exists(f)]
    if not present:
        print("None of the SMX_*_fixed.csv files were found next to this script. "
              f"Expected one or more of: {SMX_FILES}")
        return
    print(f"Enriching: {present}")

    all_coords = [pd.read_csv(f)[["longitude", "latitude"]].dropna() for f in present]
    sites = pd.concat(all_coords, ignore_index=True).drop_duplicates().reset_index(drop=True)
    sites["site_key"] = sites.index.astype(str)
    print(f"{len(sites)} unique coordinate pairs across all files provided.")

    import ee
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

    print("Elevation / population density...")
    static_feats = ee_static_features(sites, ee)
    print(f"Land cover %% ({LANDCOVER_BUFFER_M} m buffer, {len(sites)} sites)...")
    lc_feats = ee_landcover_pct(sites, ee)
    print("HydroATLAS catchment_area / river_order...")
    hydro_feats = ee_hydroatlas(sites, ee)
    print("Distance to nearest urban area (GHSL)...")
    urban_feats = ee_dist_to_urban_area(sites, ee)

    site_feats = (sites
                  .merge(static_feats, on="site_key", how="left")
                  .merge(lc_feats, on="site_key", how="left")
                  .merge(hydro_feats, on="site_key", how="left")
                  .merge(urban_feats, on="site_key", how="left"))

    if HYDROWASTE_PATH:
        print("Nearest WWTP distance...")
        wwtp_feats = nearest_wwtp(site_feats, HYDROWASTE_PATH, HYDROWASTE_LAT_COL, HYDROWASTE_LON_COL)
        site_feats = site_feats.merge(wwtp_feats, on="site_key", how="left")
    else:
        print("HYDROWASTE_PATH not set - dist_to_WWTP will stay NaN.")

    for path in present:
        df = pd.read_csv(path)

        # BUGFIX: pct_urban_land/pct_agri_land/pct_forest_land are not
        # actually missing (isna()) - they were previously filled with a
        # 0.0 placeholder for every row. fill_missing() below only
        # overwrites NaN, so without this step it silently does nothing
        # to these three columns. Null out the placeholder so it does.
        if all(c in df.columns for c in PLACEHOLDER_LANDCOVER_COLS):
            placeholder = (df[PLACEHOLDER_LANDCOVER_COLS] == 0).all(axis=1)
            df.loc[placeholder, PLACEHOLDER_LANDCOVER_COLS] = np.nan
            print(f"{path}: nulled {placeholder.sum()} placeholder-zero landcover rows")

        merged = df.merge(sites, on=["longitude", "latitude"], how="left")
        merged = merged.merge(site_feats.drop(columns=["longitude", "latitude"]), on="site_key", how="left")

        def fill_missing(col_target, col_source):
            if col_source in merged.columns and col_target in merged.columns:
                # river_order etc. are read in as int64, which can't hold
                # NaN. Upcast to float64 first so a missing EE match (NaN)
                # can be assigned without pandas raising LossySetitemError.
                if pd.api.types.is_integer_dtype(merged[col_target]):
                    merged[col_target] = merged[col_target].astype("float64")
                mask = merged[col_target].isna()
                fill_values = merged.loc[mask, col_source]
                # Belt-and-suspenders: if EE ever returns a non-scalar (list,
                # string, etc.) for a numeric column, coerce it to NaN here
                # instead of letting pandas crash on assignment.
                if pd.api.types.is_numeric_dtype(merged[col_target]):
                    fill_values = pd.to_numeric(fill_values, errors="coerce")
                merged.loc[mask, col_target] = fill_values

        fill_missing("pct_forest_land", "pct_forest_land_ee")
        fill_missing("pct_urban_land", "pct_urban_land_ee")
        fill_missing("pct_agri_land", "pct_agri_land_ee")
        fill_missing("elevation", "elevation_ee")
        fill_missing("population_density", "population_density_ee")
        fill_missing("catchment_area", "catchment_area_ee")
        fill_missing("river_order", "river_order_ee")
        fill_missing("dist_to_urban_area", "dist_to_urban_area_ee")
        if "dist_to_WWTP_ee" in merged.columns:
            fill_missing("dist_to_WWTP", "dist_to_WWTP_ee")

        if "geo_landuse_available" in merged.columns:
            merged["geo_landuse_available"] = merged[["pct_forest_land", "pct_urban_land",
                                                        "pct_agri_land"]].notna().any(axis=1)

        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_ee") or c == "site_key"])
        out_path = path.replace(".csv", "_geo.csv")
        merged.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")

    print("\nDone. pct_industrial_land is still NaN (no clean global dataset - "
          "see notes at top of this script). rainfall_anomaly, COD, turbidity, "
          "and the other water-quality columns are NOT touched by this script - "
          "they need a monitoring-station join, not Earth Engine.")


if __name__ == "__main__":
    main()
