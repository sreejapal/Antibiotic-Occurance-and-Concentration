"""
Step 5a: unique_sites.csv -> nearby_wq_stations.csv

Finds which EEA water-quality monitoring stations are within MAX_DISTANCE_KM
of each of your SMX sites, for each of your 7 target parameters, using a
public ArcGIS REST endpoint (no SQL portal, no login, no manual query
building required). This tells you WHICH stations to pull actual dated
samples for in step 5b, instead of hand-writing a query against a
60-million-row table and hoping it's narrow enough.

Source: EEA's CIP_monitoringSites_peryear_withEQS table, which already
carries lat/lon per station+parameter plus first_year/last_year/count_samples
coverage. Confirmed fields (from the service's own metadata):
  countryCode, monitoringSiteIdentifier, monitoringSiteIdentifierScheme,
  lat, lon, observedPropertyDeterminandCode, parameterWaterBodyCategory,
  procedureAnalysedMatrix, resultUom, first_year, last_year, count_years,
  count_samples
"""
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from paths import UNIQUE_SITES, NEARBY_WQ_STATIONS

ARCGIS_URL = ("https://water.discomap.eea.europa.eu/arcgis/rest/services/"
              "CIP/CIPMapOct/MapServer/3/query")

MAX_DISTANCE_KM = 10  # widen if you get too few matches; EU monitoring
                       # networks are sparse compared to your 799 SMX sites

# EEA determinand codes for your 7 target parameters, verified against the
# official WISE ObservedProperty vocabulary (dd.eionet.europa.eu) where noted.
# CONFIRMED (found directly in the vocabulary browse):
#   pH                  EEA_3152-01-0  "pH"
#   EC                  EEA_3142-01-6  "Electrical conductivity"
#   TOC                 EEA_3133-06-0  "Total organic carbon (TOC)"
#   turbidity           EEA_3112-01-4  "Turbidity"
#   suspended_solids    EEA_31-02-7    "Total suspended solids"
#   COD                 WISE splits this into two methods -- there is no
#                       single "COD" code. Include both and decide later
#                       which (or both) you want:
#                         EEA_3133-03-7  "CODCr" (dichromate method)
#                         EEA_3133-04-8  "CODMn" (permanganate method)
# UNVERIFIED (plausible based on standard CAS numbers for these ions, but
# NOT confirmed against the WISE vocabulary browse -- check these in
# Discodata's own determinand list before trusting results from them):
#   nitrate    CAS_14797-55-8
#   nitrite    CAS_14797-65-0
#   ammonium   CAS_14798-03-9
#   phosphate  CAS_14265-44-2
TARGET_PARAMS = {
    "pH": ["EEA_3152-01-0"],
    "EC": ["EEA_3142-01-6"],
    "TOC": ["EEA_3133-06-0"],
    "COD": ["EEA_3133-03-7", "EEA_3133-04-8"],  # CODCr + CODMn, both methods
    "turbidity": ["EEA_3112-01-4"],
    "suspended_solids": ["EEA_31-02-7"],
    "nitrate": ["CAS_14797-55-8"],     # unverified -- confirm in Discodata
    "nitrite": ["CAS_14797-65-0"],     # unverified -- confirm in Discodata
    "ammonium": ["CAS_14798-03-9"],    # unverified -- confirm in Discodata
    "phosphate": ["CAS_14265-44-2"],   # unverified -- confirm in Discodata
}


def fetch_stations_for_param(codes):
    where = "observedPropertyDeterminandCode IN (" + \
            ",".join(f"'{c}'" for c in codes) + ")"
    params = {
        "where": where,
        "outFields": "countryCode,monitoringSiteIdentifier,monitoringSiteIdentifierScheme,"
                     "lat,lon,observedPropertyDeterminandCode,parameterWaterBodyCategory,"
                     "first_year,last_year,count_years,count_samples",
        "f": "json",
        "returnGeometry": "false",
    }
    resp = requests.get(ARCGIS_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "features" not in data:
        print(f"  No 'features' in response -- check the query. Response: {data}")
        return pd.DataFrame()
    rows = [f["attributes"] for f in data["features"]]
    return pd.DataFrame(rows)


def main():
    sites = pd.read_csv(UNIQUE_SITES)
    sites_gdf = gpd.GeoDataFrame(
        sites,
        geometry=[Point(xy) for xy in zip(sites.longitude, sites.latitude)],
        crs="EPSG:4326",
    ).to_crs("EPSG:3035")

    all_matches = []
    for param_name, codes in TARGET_PARAMS.items():
        print(f"Fetching stations for {param_name} ({codes})...")
        stations = fetch_stations_for_param(codes)
        if stations.empty:
            print(f"  0 stations found for {param_name} -- code may be wrong, verify in Discodata's explorer")
            continue
        stations = stations.dropna(subset=["lat", "lon"])
        st_gdf = gpd.GeoDataFrame(
            stations,
            geometry=[Point(xy) for xy in zip(stations.lon, stations.lat)],
            crs="EPSG:4326",
        ).to_crs("EPSG:3035")

        joined = gpd.sjoin_nearest(sites_gdf, st_gdf, distance_col="dist_m", max_distance=MAX_DISTANCE_KM * 1000)
        joined["parameter"] = param_name
        print(f"  {stations.shape[0]} stations total, {joined.shape[0]} SMX-site matches within {MAX_DISTANCE_KM}km")
        all_matches.append(joined.drop(columns="geometry"))

    if not all_matches:
        print("No matches for any parameter -- widen MAX_DISTANCE_KM or double check TARGET_PARAMS codes.")
        return

    out = pd.concat(all_matches, ignore_index=True)
    out.to_csv(NEARBY_WQ_STATIONS, index=False)
    print(f"\nWrote {len(out)} site-parameter-station matches to {NEARBY_WQ_STATIONS}")
    print("\nNext: take the unique monitoringSiteIdentifier values from this file "
          "and use them to filter a Discodata query against DisaggregatedData -- "
          "see 05b_query_disaggregated.py")


if __name__ == "__main__":
    main()
