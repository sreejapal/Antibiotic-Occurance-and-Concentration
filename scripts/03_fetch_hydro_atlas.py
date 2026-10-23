"""
Step 3: unique_sites.csv -> hydro_features.csv

Columns produced: catchment_area, river_order, dist_to_WWTP, dist_to_urban_area
(elevation is already covered by the GEE script -- see gee_site_features.csv)

Data downloads (no login required, but large files — do these once):
  HydroATLAS (RiverATLAS, has catchment_area & river order as Strahler order
  "ORD_STRA"):
    https://www.hydrosheds.org/hydroatlas
    -> download "RiverATLAS_v10.gdb" (or shapefile version), Europe is covered.

dist_to_WWTP needs a wastewater-treatment-plant point layer, which HydroATLAS
does NOT include. For your 27 European countries, the best source is:
  - EEA Urban Waste Water Treatment Directive (UWWTD) database -- has
    coordinates for essentially every WWTP in the EU/EEA:
    https://www.eea.europa.eu/en/datahub/datahubitem-view/UWWTD (free login)
  - For non-EU sites in your data (Ukraine, Georgia, Serbia, Moldova, Turkey,
    Tunisia), UWWTD won't have coverage -- fall back to OpenStreetMap's
    `man_made=wastewater_plant` tag via the Overpass API for those.

dist_to_urban_area reuses the WorldCover built-up class; vectorize the
GEE landcover output, or download WorldCover's raster once and polygonize
locally with rasterio.features.shapes, rather than a second manual download.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from paths import (
    UNIQUE_SITES,
    HYDRO_FEATURES,
    RIVERATLAS_EU,
    WWTP_POINTS_SHP,
    URBAN_POLY_SHP,
)

sites = pd.read_csv(UNIQUE_SITES)  # location_id, latitude, longitude

RIVERATLAS_SHP = RIVERATLAS_EU  # Europe extract from HydroATLAS download


def nearest_distance_km(points_gdf, target_gdf):
    """Distance in km from each point to the nearest geometry in target_gdf."""
    points_m = points_gdf.to_crs(points_gdf.estimate_utm_crs())
    target_m = target_gdf.to_crs(points_m.crs)
    joined = gpd.sjoin_nearest(points_m, target_m, distance_col="dist_m")
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.reindex(points_m.index)
    return joined["dist_m"] / 1000


def main():
    pts = gpd.GeoDataFrame(
        sites,
        geometry=[Point(xy) for xy in zip(sites.longitude, sites.latitude)],
        crs="EPSG:4326",
    )

    # --- catchment_area + river_order: nearest RiverATLAS reach segment ---
    # Reproject to a Europe-wide equal-area CRS (not lat/lon degrees) so
    # "nearest" is measured in real distance. EPSG:3035 covers all of Europe,
    # which fits your site footprint (Portugal to Georgia).
    rivers = gpd.read_file(RIVERATLAS_SHP)
    pts_proj = pts.to_crs("EPSG:3035")
    rivers_proj = rivers.to_crs("EPSG:3035")

    joined = gpd.sjoin_nearest(pts_proj, rivers_proj, distance_col="_snap_dist_m")
    # sjoin_nearest can return >1 match per point when two reaches are exactly
    # equidistant (ties) -- keep only the first match per original point so
    # row counts stay aligned with pts.
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.reindex(pts_proj.index)  # guarantee same order/length as pts

    pts["catchment_area"] = joined["UPLAND_SKM"].values   # upstream drainage area, km2
    pts["river_order"] = joined["ORD_STRA"].values        # Strahler stream order

    # --- distances to nearest WWTP / urban polygon ---
    try:
        wwtp = gpd.read_file(WWTP_POINTS_SHP)
        pts["dist_to_WWTP"] = nearest_distance_km(pts, wwtp).values
    except Exception as e:
        print(f"WWTP layer not available ({e}) — leaving dist_to_WWTP as NaN")
        pts["dist_to_WWTP"] = np.nan

    try:
        urban = gpd.read_file(URBAN_POLY_SHP)
        pts["dist_to_urban_area"] = nearest_distance_km(pts, urban).values
    except Exception as e:
        print(f"Urban polygon layer not available ({e}) — leaving dist_to_urban_area as NaN")
        pts["dist_to_urban_area"] = np.nan

    out = pts[["location_id", "catchment_area", "river_order", "dist_to_WWTP", "dist_to_urban_area"]]
    out.to_csv(HYDRO_FEATURES, index=False)
    print(f"Wrote {len(out)} sites to {HYDRO_FEATURES}")


if __name__ == "__main__":
    main()
