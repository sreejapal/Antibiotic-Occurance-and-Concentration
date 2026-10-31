"""Canonical project paths. Import from other scripts in this folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS = ROOT / "docs"

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUT = DATA / "output"

RAW_EMPODAT = RAW / "empodat"
RAW_WATERBASE = RAW / "waterbase"
RAW_HYDROATLAS = RAW / "hydroatlas"

SCRAPE_DIR = RAW_EMPODAT / "scrape"
RAW_SMX = RAW_EMPODAT / "raw_SMX.csv"

WATERBASE_DISAGG = RAW_WATERBASE / "Waterbase_v2025_1_T_WISE6_DisaggregatedData.csv"
WATERBASE_SPATIAL = RAW_WATERBASE / "Waterbase_v2025_1_S_WISE6_SpatialObject_DerivedData.csv"
WQ_PARAM_CODES = RAW_WATERBASE / "water_quality_parameter_codes.csv"

RIVERATLAS_DIR = RAW_HYDROATLAS / "RiverATLAS_Data_v10_shp" / "RiverATLAS_v10_shp"
RIVERATLAS_EU = RIVERATLAS_DIR / "RiverATLAS_v10_eu.shp"
WWTP_POINTS_SHP = RAW_HYDROATLAS / "wwtp_points.shp"
URBAN_POLY_SHP = RAW_HYDROATLAS / "worldcover_urban.shp"

OCCURRENCE_BASE = PROCESSED / "occurrence_base.csv"
UNIQUE_SITES = PROCESSED / "unique_sites.csv"
GEE_SITE_FEATURES = PROCESSED / "gee_site_features.csv"
GEE_DATE_FEATURES = PROCESSED / "gee_date_features.csv"
HYDRO_FEATURES = PROCESSED / "hydro_features.csv"
WQ_DIAGNOSTIC = PROCESSED / "water_quality_diagnostic.csv"
NEARBY_WQ_STATIONS = PROCESSED / "nearby_wq_stations.csv"

SMX_OCCURRENCE = OUTPUT / "SMX_occurrence.csv"
SMX_CONCENTRATION = OUTPUT / "SMX_concentration.csv"
SMX_OCCURRENCE_WQ = OUTPUT / "SMX_occurrence_with_water_quality.csv"
