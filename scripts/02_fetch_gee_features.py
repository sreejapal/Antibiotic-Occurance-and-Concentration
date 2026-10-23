"""
Step 2 (replaces old scripts 02/03/04): fetch ERA5-Land, WorldCover land-use %,
WorldPop population density, and SRTM elevation -- all through Google Earth
Engine, which does the raster work server-side. This avoids downloading
continent-scale rasters yourself.

Your actual data footprint: 799 unique locations, Portugal to Georgia,
lat 35.9-62.4 / lon -10.0-41.7, sampling dates 2006-02-21 to 2025-09-08.

=== One-time setup (~10 min) ===
1. Go to https://code.earthengine.google.com/ and sign in with a Google account.
   First visit prompts you to create a free "Google Cloud project" for
   Earth Engine -- accept the free/noncommercial tier, no billing needed
   for this volume of requests.
2. pip install earthengine-api --break-system-packages
3. In a terminal: `earthengine authenticate` (or in Python: `ee.Authenticate()`)
   -- opens a browser, log in once, paste the token back. You won't need to
   repeat this on the same machine.
4. Replace PROJECT_ID below with the Cloud project id you created in step 1.

Run: python scripts/02_fetch_gee_features.py
Output: data/processed/gee_site_features.csv and gee_date_features.csv
"""
import ee
import pandas as pd
import numpy as np
import time

from paths import UNIQUE_SITES, OCCURRENCE_BASE, GEE_SITE_FEATURES, GEE_DATE_FEATURES

PROJECT_ID = "smx-occurance"  # <-- edit this

ee.Initialize(project=PROJECT_ID)

sites = pd.read_csv(UNIQUE_SITES)               # location_id, lat, lon
base = pd.read_csv(OCCURRENCE_BASE, parse_dates=["sampling_date"])
base = base.merge(sites[["location_id", "latitude", "longitude"]],
                   on=["latitude", "longitude"], how="left")


# ---------------------------------------------------------------------------
# 1. Static site-level features: elevation, land use %, population density
# ---------------------------------------------------------------------------
def build_site_fc(sites_df):
    feats = [
        ee.Feature(ee.Geometry.Point([row.longitude, row.latitude]),
                   {"location_id": row.location_id})
        for row in sites_df.itertuples()
    ]
    return ee.FeatureCollection(feats)


def fetch_static_features(sites_df, buffer_m=1000, chunk=100):
    elevation = ee.Image("USGS/SRTMGL1_003").rename("elevation")
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().rename("landcover_class")
    worldpop = (ee.ImageCollection("WorldPop/GP/100m/pop")
                .filterDate("2020-01-01", "2021-01-01")
                .mosaic().rename("population"))

    continuous_stack = elevation.addBands(worldpop)  # plain mean is fine here

    results = []
    for i in range(0, len(sites_df), chunk):
        batch = sites_df.iloc[i:i + chunk]
        fc = build_site_fc(batch).map(lambda f: f.buffer(buffer_m))

        # Two separate reduceRegions calls -- combining reducer types on one
        # call renames output fields (e.g. "elevation" -> "elevation_mean"),
        # which silently broke the lookups below. Keeping them separate
        # means the output property name always matches the band name.
        cont_reduced = continuous_stack.reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(), scale=100,
        ).getInfo()
        cat_reduced = worldcover.reduceRegions(
            collection=fc, reducer=ee.Reducer.frequencyHistogram(), scale=100,
        ).getInfo()

        cont_by_id = {f["properties"]["location_id"]: f["properties"] for f in cont_reduced["features"]}
        cat_by_id = {f["properties"]["location_id"]: f["properties"] for f in cat_reduced["features"]}

        for loc_id in batch["location_id"]:
            cp = cont_by_id.get(loc_id, {})
            catp = cat_by_id.get(loc_id, {})
            hist = catp.get("landcover_class", {}) or {}
            total = sum(hist.values()) or 1
            # WorldCover class codes: 10 tree,20 shrub,30 grass,40 cropland,
            # 50 built-up,60 bare,80 water,90 wetland,95 mangrove
            results.append({
                "location_id": loc_id,
                "elevation": cp.get("elevation"),
                "population_density": cp.get("population"),
                "pct_forest_land": 100 * hist.get("10", 0) / total,
                "pct_agri_land": 100 * hist.get("40", 0) / total,
                "pct_urban_land": 100 * hist.get("50", 0) / total,
                # WorldCover has no separate industrial class -- see README note
                "pct_industrial_land": np.nan,
            })
        print(f"  static features: {min(i + chunk, len(sites_df))}/{len(sites_df)} sites done")
        time.sleep(1)  # be polite to the API
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2. Date-varying features: rainfall (1d/7d/30d + anomaly), air temp, ET
# ---------------------------------------------------------------------------
def fetch_weather_for_row(lat, lon, date, era5_collection):
    # Buffer instead of a bare point: ERA5-Land masks ocean/large-waterbody
    # pixels, so a site sitting exactly on a coastline or river mouth can
    # land on a null pixel. Averaging over a small buffer picks up the
    # nearest valid land pixel instead of returning nothing.
    pt = ee.Geometry.Point([lon, lat]).buffer(5000)
    end = ee.Date(date.strftime("%Y-%m-%d"))
    start_30d = end.advance(-30, "day")
    start_7d = end.advance(-7, "day")
    start_1d = end.advance(-1, "day")

    def sum_precip(start, end_):
        img = era5_collection.select("total_precipitation_sum").filterDate(start, end_).sum()
        return img.reduceRegion(ee.Reducer.mean(), pt, 1000, bestEffort=True).get("total_precipitation_sum")

    temp_img = (era5_collection.select("temperature_2m")
                .filterDate(start_1d, end).mean())
    et_img = (era5_collection.select("total_evaporation_sum")
              .filterDate(start_1d, end).sum())

    return ee.Dictionary({
        "rainfall_1d": sum_precip(start_1d, end),
        "rainfall_7d": sum_precip(start_7d, end),
        "rainfall_30d": sum_precip(start_30d, end),
        "air_temperature": temp_img.reduceRegion(ee.Reducer.mean(), pt, 1000, bestEffort=True).get("temperature_2m"),
        "evapotranspiration": et_img.reduceRegion(ee.Reducer.mean(), pt, 1000, bestEffort=True).get("total_evaporation_sum"),
    })


def fetch_weather_features(base_df):
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    rows = base_df.dropna(subset=["sampling_date"])  # skip the 144 year-only rows here
    results = []
    for n, row in enumerate(rows.itertuples(), 1):
        d = fetch_weather_for_row(row.latitude, row.longitude, row.sampling_date, era5).getInfo()
        d["site_id"] = row.site_id
        d["sampling_date"] = row.sampling_date
        results.append(d)
        if n % 25 == 0:
            print(f"  weather features: {n}/{len(rows)} rows done")
    out = pd.DataFrame(results)
    # convert m -> mm for precipitation/ET, K -> C for temperature
    for col in ["rainfall_1d", "rainfall_7d", "rainfall_30d", "evapotranspiration"]:
        out[col] = out[col] * 1000
    out["air_temperature"] = out["air_temperature"] - 273.15
    # anomaly: deviation from this site's own mean 30-day rainfall in that calendar month
    out["month"] = pd.to_datetime(out["sampling_date"]).dt.month
    out["rainfall_anomaly"] = out["rainfall_30d"] - out.groupby(
        ["site_id", "month"])["rainfall_30d"].transform("mean")
    return out.drop(columns="month")


if __name__ == "__main__":
    print("Fetching static site features (elevation, land use, population)...")
    static_feats = fetch_static_features(sites)
    static_feats.to_csv(GEE_SITE_FEATURES, index=False)
    print(f"Wrote {GEE_SITE_FEATURES} ({len(static_feats)} rows)")

    print("\nFetching per-sample weather features (this is the slow part: "
          "~1 API round-trip per row, budget ~1-2 rows/sec)...")
    weather_feats = fetch_weather_features(base)
    weather_feats.to_csv(GEE_DATE_FEATURES, index=False)
    print(f"Wrote {GEE_DATE_FEATURES} ({len(weather_feats)} rows)")
