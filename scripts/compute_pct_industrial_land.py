"""
Computes pct_industrial_land for each site using OpenStreetMap landuse=industrial
polygons, matching the same 1km-buffer / percentage-of-buffer-area methodology
used for pct_urban_land / pct_agri_land / pct_forest_land (ESA WorldCover, 1km
radius, % of buffer covered).

Run this LOCALLY (needs internet access to the Overpass API) — the sandbox this
was written in cannot reach external APIs.

Usage:
    pip install requests shapely pyproj pandas tqdm
    python compute_pct_industrial_land.py SMX_occurrence_with_water_quality_geo.csv

Outputs a CSV: <input>_with_industrial.csv with two new columns:
    pct_industrial_land
    pct_industrial_land_confidence   ("high" / "medium" / "low", by country,
                                       based on typical OSM landuse tagging density)

Then merge pct_industrial_land back into all three original files by site_id.

---------------------------------------------------------------------------
WHY THIS VERSION IS DIFFERENT FROM THE ORIGINAL
---------------------------------------------------------------------------
The original issued one Overpass HTTP request per unique site, plus a fixed
1.5s sleep between every single one. For 799 sites that's a *minimum* of
~20 minutes of intentional sleep alone, before any actual network time —
which is almost certainly what made the run feel "stuck" at 30 minutes.

This version instead:
  1. Buckets all unique site coordinates into spatial grid cells (~50km,
     configurable via GRID_DEG).
  2. Issues ONE Overpass query per grid cell (with a margin so 1km buffers
     near a cell edge are still fully covered), not one per site.
  3. Clips every site's 1km buffer against the cached polygons for its cell
     locally with shapely — no network call needed for that step.
  4. Writes a checkpoint file after every cell so an interrupted run can be
     resumed without re-querying cells already done.
  5. Shows a live progress bar (tqdm) with cells/sites done so far, so you
     can tell "slow" from "stuck" at a glance.

If your sites are geographically clustered at all (e.g. stations along the
same river or region), this can cut the number of Overpass calls from ~800
down to a handful — turning a 30+ minute run into a couple of minutes.
"""

import sys
import os
import time
import json
import math
import requests
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform as shp_transform
import pyproj
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BUFFER_M = 1000          # matches the 1km radius used for the other three columns
GRID_DEG = 0.5           # ~55km grid cells at the equator; tune smaller if sites are dense
CELL_MARGIN_M = BUFFER_M + 500   # extra margin so buffers near a cell edge are fully covered
SLEEP_BETWEEN_CALLS = 1.5        # only applies per grid-cell query now, not per site
CHECKPOINT_FILE = "industrial_land_checkpoint.json"

# Rough OSM landuse-tagging completeness by country — refine as you see fit.
# This is a heuristic, not a measured value; treat it as a caveat flag, not ground truth.
HIGH_CONF = {"Germany", "Switzerland", "Netherlands", "Austria", "France",
             "Belgium", "Denmark", "Sweden", "Finland"}
MEDIUM_CONF = {"Italy", "Spain", "Portugal", "Czech Republic", "Slovenia",
               "Slovakia", "Croatia", "Hungary", "Greece"}
# everything else (Ukraine, Georgia, Serbia, Bulgaria, Romania, Moldova,
# Tunisia, Turkey, Latvia, etc.) defaults to "low"


def confidence_for(country):
    if country in HIGH_CONF:
        return "high"
    if country in MEDIUM_CONF:
        return "medium"
    return "low"


def local_azimuthal_proj(lat, lon):
    """Equal-area-ish local projection centered on a point, for accurate m^2 area."""
    return pyproj.Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m +ellps=WGS84",
        always_xy=True,
    ).transform


def grid_cell(lat, lon, size=GRID_DEG):
    return (math.floor(lat / size), math.floor(lon / size))


def cell_bbox_with_margin(cell_i, cell_j, size=GRID_DEG, margin_m=CELL_MARGIN_M):
    """Return (south, west, north, east) bbox for a grid cell, padded by margin_m."""
    south, north = cell_i * size, (cell_i + 1) * size
    west, east = cell_j * size, (cell_j + 1) * size
    # rough degree-per-meter padding (good enough for a margin, not precision-critical)
    lat_pad = margin_m / 111_320.0
    center_lat = (south + north) / 2.0
    lon_pad = margin_m / (111_320.0 * max(math.cos(math.radians(center_lat)), 0.1))
    return south - lat_pad, west - lon_pad, north + lat_pad, east + lon_pad


def query_overpass_bbox(south, west, north, east, retries=3):
    query = f"""
    [out:json][timeout:120];
    (
      way["landuse"="industrial"]({south},{west},{north},{east});
      relation["landuse"="industrial"]({south},{west},{north},{east});
    );
    out geom;
    """
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=150)
            if resp.status_code == 200:
                return resp.json()
            time.sleep(5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    return {"elements": []}


def polygon_from_element(el):
    """Build a shapely polygon from an Overpass 'out geom' way/relation element."""
    try:
        if el["type"] == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) >= 3:
                return Polygon(coords)
        elif el["type"] == "relation" and "members" in el:
            polys = []
            for m in el["members"]:
                if m.get("type") == "way" and "geometry" in m:
                    coords = [(pt["lon"], pt["lat"]) for pt in m["geometry"]]
                    if len(coords) >= 3:
                        polys.append(Polygon(coords))
            if polys:
                return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    except Exception:
        return None
    return None


def pct_industrial_for_site(lat, lon, cell_polygons):
    """Clip a site's 1km buffer against the pre-fetched polygons for its grid cell."""
    proj = local_azimuthal_proj(lat, lon)
    center_local = Point(0, 0)
    circle = center_local.buffer(BUFFER_M)
    circle_area = circle.area

    total_industrial_area = 0.0
    for poly in cell_polygons:
        try:
            poly_local = shp_transform(proj, poly)
            clipped = poly_local.intersection(circle)
            if not clipped.is_empty:
                total_industrial_area += clipped.area
        except Exception:
            continue

    return 100.0 * total_industrial_area / circle_area


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"done_cells": {}, "results": {}}


def save_checkpoint(state):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_pct_industrial_land.py <input.csv>")
        sys.exit(1)
    infile = sys.argv[1]

    df = pd.read_csv(infile)

    uniq = df[["site_id", "latitude", "longitude", "country"]].drop_duplicates(
        subset=["latitude", "longitude"]
    )

    # bucket unique sites into grid cells
    cells = defaultdict(list)
    for row in uniq.itertuples():
        cells[grid_cell(row.latitude, row.longitude)].append((row.latitude, row.longitude))

    print(f"{len(uniq)} unique locations -> {len(cells)} grid cells "
          f"(~{len(uniq) / max(len(cells), 1):.1f} sites/cell on average)")

    state = load_checkpoint()
    done_cells = state["done_cells"]      # {"i,j": True}
    results = state["results"]            # {"lat,lon": pct}

    if done_cells:
        print(f"Resuming from checkpoint: {len(done_cells)} cells already done.")

    cell_items = list(cells.items())
    for (ci, cj), points in tqdm(cell_items, desc="Grid cells", unit="cell"):
        cell_key = f"{ci},{cj}"
        if cell_key in done_cells:
            continue

        south, west, north, east = cell_bbox_with_margin(ci, cj)
        data = query_overpass_bbox(south, west, north, east)
        elements = data.get("elements", [])
        cell_polygons = [p for p in (polygon_from_element(el) for el in elements)
                          if p is not None and not p.is_empty]

        for lat, lon in points:
            pct = pct_industrial_for_site(lat, lon, cell_polygons)
            results[f"{lat},{lon}"] = pct

        done_cells[cell_key] = True
        save_checkpoint(state)   # checkpoint after every cell, not every site
        time.sleep(SLEEP_BETWEEN_CALLS)

    def lookup(r):
        return results.get(f"{r.latitude},{r.longitude}")

    df["pct_industrial_land"] = df.apply(lookup, axis=1)
    df["pct_industrial_land_confidence"] = df["country"].apply(confidence_for)

    outfile = infile.rsplit(".", 1)[0] + "_with_industrial.csv"
    df.to_csv(outfile, index=False)
    print(f"Saved: {outfile}")

    # clean up checkpoint on successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()
