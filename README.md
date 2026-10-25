# SMX occurrence dataset pipeline

Builds the modeling dataset from EMPODAT (`raw_SMX.csv`) plus Google Earth Engine, HydroATLAS, and EEA Waterbase.

## Folder layout

```
docs/                         project notes and paper
scripts/                      pipeline and helper scripts
data/raw/empodat/             EMPODAT export and scrape CSVs
data/raw/waterbase/           EEA Waterbase downloads
data/raw/hydroatlas/          RiverATLAS shapefiles
data/processed/               intermediate feature tables
data/output/                  final SMX_* CSVs
venv/                         local Python environment (not source)
```

Shared file locations live in `scripts/paths.py`. Run scripts from anywhere; they resolve paths from that file.

## Pipeline

```
scripts/build_final_csv_v2.py     scrape CSVs -> data/raw/empodat/raw_SMX.csv
scripts/02_fetch_gee_features.py  ERA5-Land + WorldCover + WorldPop + elevation
scripts/03_fetch_hydro_atlas.py   catchment_area / river_order / distances
scripts/04_merge_features.py      join into data/output/SMX_occurrence.csv
scripts/05a_find_nearby_stations.py
scripts/05_match_water_quality.py Waterbase join
scripts/06_finalize_schema.py     final schema, nutrients, Y_conc
```

`01_build_occurrence_base.py` was already run; see `data/processed/occurrence_base.csv`.

Column 19 (pH / EC / TOC / COD / nutrients / turbidity) is EMPODAT / national-monitoring matched-subset only.

## Setup

```bash
pip install pandas numpy geopandas rasterio shapely pyproj earthengine-api --break-system-packages
```

Already generated from `raw_SMX.csv`:

- `data/processed/occurrence_base.csv` — base table (1502 rows)
- `data/processed/unique_sites.csv` — 799 deduplicated locations

## Credentials

| Source | What to get | Where |
|---|---|---|
| Google Earth Engine | Google account + free Cloud project | https://code.earthengine.google.com/ |
| HydroATLAS | none, direct download (large) | https://www.hydrosheds.org/hydroatlas |
| WWTP locations (EU/EEA) | free EEA login | https://www.eea.europa.eu/en/datahub/datahubitem-view/UWWTD |

## Order of operations

1. `occurrence_base.csv` and `unique_sites.csv` already exist under `data/processed/`.
2. `python scripts/02_fetch_gee_features.py` — set up GEE first (see comments in the script).
3. `python scripts/03_fetch_hydro_atlas.py` after RiverATLAS (and optional WWTP layer) are in `data/raw/hydroatlas/`.
4. `python scripts/04_merge_features.py` to assemble `data/output/SMX_occurrence.csv`.
5. Optionally `python scripts/05_match_water_quality.py` then `python scripts/06_finalize_schema.py`.
