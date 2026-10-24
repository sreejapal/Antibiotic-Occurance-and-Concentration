import pandas as pd
import numpy as np
from paths import SMX_OCCURRENCE, WATERBASE_DISAGG, WATERBASE_SPATIAL, SMX_OCCURRENCE_WQ


# ============================================================
# FILE PATHS
# ============================================================

SMX_FILE = SMX_OCCURRENCE

WQ_FILE = WATERBASE_DISAGG

SPATIAL_FILE = WATERBASE_SPATIAL

OUTPUT_FILE = SMX_OCCURRENCE_WQ


# ============================================================
# MATCHING SETTINGS
# ============================================================

MAX_DISTANCE_KM = 5.0
MAX_DATE_DIFF_DAYS = 30


# ============================================================
# WATERBASE PARAMETERS
# ============================================================

PARAMETER_MAP = {

    # pH
    "EEA_3152-01-0": "pH",

    # Electrical conductivity
    "EEA_3142-01-6": "EC",

    # Total organic carbon
    "EEA_3133-06-0": "TOC",

    # Turbidity
    "EEA_3112-01-4": "turbidity",

    # Total suspended solids
    "EEA_31-02-7": "suspended_solids",

    # Nutrients
    "CAS_14797-55-8": "nitrate",
    "CAS_14797-65-0": "nitrite",
    "CAS_14798-03-9": "ammonium",
    "CAS_14265-44-2": "phosphate",
}


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# ============================================================
# LOAD SMX DATA
# ============================================================

print("\nLoading SMX dataset...")

smx = pd.read_csv(SMX_FILE)

print("SMX shape:", smx.shape)

original_rows = len(smx)


# ============================================================
# CLEAN SMX DATE
# ============================================================

smx["sampling_date"] = pd.to_datetime(
    smx["sampling_date"],
    errors="coerce"
)


# ============================================================
# CLEAN SMX COORDINATES
# ============================================================

smx["latitude"] = pd.to_numeric(
    smx["latitude"],
    errors="coerce"
)

smx["longitude"] = pd.to_numeric(
    smx["longitude"],
    errors="coerce"
)


print(
    "SMX date range:",
    smx["sampling_date"].min(),
    "to",
    smx["sampling_date"].max()
)


# ============================================================
# LOAD WATERBASE SPATIAL DATA
# ============================================================

print("\nLoading Waterbase spatial data...")

spatial = pd.read_csv(
    SPATIAL_FILE,
    low_memory=False
)

print("Spatial shape:", spatial.shape)


# ============================================================
# KEEP REQUIRED SPATIAL COLUMNS
# ============================================================

spatial = spatial[
    [
        "countryCode",
        "monitoringSiteIdentifier",
        "monitoringSiteIdentifierScheme",
        "monitoringSiteName",
        "lat",
        "lon"
    ]
].copy()


# ============================================================
# RENAME COORDINATES
# ============================================================

spatial = spatial.rename(
    columns={
        "lat": "station_lat",
        "lon": "station_lon"
    }
)


# ============================================================
# CLEAN STATION COORDINATES
# ============================================================

spatial["station_lat"] = pd.to_numeric(
    spatial["station_lat"],
    errors="coerce"
)

spatial["station_lon"] = pd.to_numeric(
    spatial["station_lon"],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID STATIONS
# ============================================================

spatial = spatial.dropna(
    subset=[
        "station_lat",
        "station_lon"
    ]
)


# ============================================================
# REMOVE DUPLICATE STATIONS
# ============================================================

spatial = spatial.drop_duplicates(
    subset=[
        "monitoringSiteIdentifier",
        "monitoringSiteIdentifierScheme"
    ]
)


print(
    "Spatial stations with valid coordinates:",
    len(spatial)
)


# ============================================================
# LOAD WATERBASE MEASUREMENTS
# ============================================================

print("\nReading Waterbase measurements...")

wanted_codes = set(PARAMETER_MAP.keys())

chunks = []

CHUNK_SIZE = 100_000


for i, chunk in enumerate(
    pd.read_csv(
        WQ_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )
):

    print(f"Processing chunk {i + 1}...")


    # ========================================================
    # FILTER PARAMETERS
    # ========================================================

    chunk = chunk[
        chunk[
            "observedPropertyDeterminandCode"
        ].isin(wanted_codes)
    ]


    if len(chunk) == 0:
        continue


    # ========================================================
    # FIX WATERBASE DATE
    # ========================================================
    #
    # Waterbase stores dates in YYYYMMDD format.
    #
    # Example:
    #
    # 20251113
    #
    # must become:
    #
    # 2025-11-13
    #
    # Do NOT let pandas interpret the integer as nanoseconds.
    # ========================================================

    chunk["phenomenonTimeSamplingDate"] = pd.to_datetime(
        chunk["phenomenonTimeSamplingDate"]
        .astype(str)
        .str[:8],
        format="%Y%m%d",
        errors="coerce"
    )


    # ========================================================
    # KEEP ONLY POSSIBLE DATE RANGE
    # ========================================================

    min_smx_date = smx["sampling_date"].min()

    max_smx_date = smx["sampling_date"].max()


    chunk = chunk[
        chunk["phenomenonTimeSamplingDate"].between(
            min_smx_date - pd.Timedelta(
                days=MAX_DATE_DIFF_DAYS
            ),
            max_smx_date + pd.Timedelta(
                days=MAX_DATE_DIFF_DAYS
            )
        )
    ]


    if len(chunk) == 0:
        continue


    # ========================================================
    # KEEP WATER SAMPLES
    # ========================================================

    chunk = chunk[
        chunk["procedureAnalysedMatrix"] == "W"
    ]


    if len(chunk) == 0:
        continue


    # ========================================================
    # KEEP REQUIRED COLUMNS
    # ========================================================

    chunk = chunk[
        [
            "countryCode",
            "monitoringSiteIdentifier",
            "monitoringSiteIdentifierScheme",
            "parameterWaterBodyCategory",
            "observedPropertyDeterminandCode",
            "observedPropertyDeterminandLabel",
            "procedureAnalysedMatrix",
            "resultUom",
            "phenomenonTimeSamplingDate",
            "resultObservedValue",
            "resultQualityObservedValueBelowLOQ",
            "procedureLOQValue"
        ]
    ]


    chunks.append(chunk)


# ============================================================
# CHECK WATERBASE DATA
# ============================================================

if not chunks:

    raise RuntimeError(
        "No Waterbase records matched the selected parameters "
        "and date range. Check the Waterbase date format."
    )


# ============================================================
# COMBINE WATERBASE
# ============================================================

waterbase = pd.concat(
    chunks,
    ignore_index=True
)


print(
    "\nFiltered Waterbase records:",
    len(waterbase)
)


# ============================================================
# SHOW CORRECTED DATE RANGE
# ============================================================

print(
    "Waterbase date range:",
    waterbase[
        "phenomenonTimeSamplingDate"
    ].min(),
    "to",
    waterbase[
        "phenomenonTimeSamplingDate"
    ].max()
)


# ============================================================
# ATTACH STATION COORDINATES
# ============================================================

print("\nAttaching station coordinates...")


waterbase = waterbase.merge(
    spatial,
    on=[
        "countryCode",
        "monitoringSiteIdentifier",
        "monitoringSiteIdentifierScheme"
    ],
    how="inner"
)


print(
    "Waterbase records with coordinates:",
    len(waterbase)
)


# ============================================================
# CLEAN MEASUREMENT VALUES
# ============================================================

waterbase["resultObservedValue"] = pd.to_numeric(
    waterbase["resultObservedValue"],
    errors="coerce"
)


waterbase = waterbase.dropna(
    subset=[
        "resultObservedValue",
        "station_lat",
        "station_lon",
        "phenomenonTimeSamplingDate"
    ]
)


# ============================================================
# CREATE FEATURE NAME
# ============================================================

waterbase["feature"] = (
    waterbase[
        "observedPropertyDeterminandCode"
    ].map(PARAMETER_MAP)
)


waterbase = waterbase.dropna(
    subset=["feature"]
)


# ============================================================
# SHOW AVAILABLE FEATURES
# ============================================================

print("\nAvailable water-quality features:")

print(
    waterbase["feature"]
    .value_counts()
)


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

country_map = {

    "Germany": "DE",
    "Slovakia": "SK",
    "Hungary": "HU",
    "Ukraine": "UA",
    "Croatia": "HR",
    "Serbia": "RS",
    "Switzerland": "CH",
    "Italy": "IT",
    "Bulgaria": "BG",
    "Georgia": "GE",
    "Czech Republic": "CZ",
    "Spain": "ES",
    "Austria": "AT",
    "Portugal": "PT",
    "Slovenia": "SI",
    "Romania": "RO",
    "Sweden": "SE",
    "Netherlands": "NL",
    "Greece": "GR",
    "Belgium": "BE",
    "France": "FR",
    "Denmark": "DK",
    "Latvia": "LV",
    "Finland": "FI",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Moldova": "MD",
}


smx["country_code"] = smx["country"].map(
    country_map
)


# ============================================================
# CREATE FEATURE COLUMNS
# ============================================================

feature_columns = [

    "pH",
    "EC",
    "TOC",
    "turbidity",
    "suspended_solids",
    "nitrate",
    "nitrite",
    "ammonium",
    "phosphate"
]


for col in feature_columns:

    smx[col] = np.nan


# ============================================================
# OPTIONAL NUTRIENTS FEATURE
# ============================================================

smx["nutrients"] = np.nan


# ============================================================
# DIAGNOSTIC COLUMNS
# ============================================================

# Station IDs are strings, so explicitly create a string column
smx["water_quality_station"] = pd.Series(
    pd.NA,
    index=smx.index,
    dtype="string"
)

# These are numerical
smx["water_quality_distance_km"] = np.nan

# This is a datetime
smx["water_quality_date"] = pd.NaT

# This is numerical
smx["water_quality_date_diff_days"] = np.nan


# ============================================================
# GROUP WATERBASE BY COUNTRY
# ============================================================

waterbase_groups = {

    country: group

    for country, group

    in waterbase.groupby("countryCode")
}


# ============================================================
# MATCH SMX SAMPLES
# ============================================================

print("\nStarting water-quality matching...")


matched_count = 0


for idx, row in smx.iterrows():

    country = row["country_code"]

    lat = row["latitude"]

    lon = row["longitude"]

    sample_date = row["sampling_date"]


    # ========================================================
    # CHECK MISSING INFORMATION
    # ========================================================

    if (
        pd.isna(country)
        or pd.isna(lat)
        or pd.isna(lon)
        or pd.isna(sample_date)
    ):

        continue


    # ========================================================
    # CHECK COUNTRY
    # ========================================================

    if country not in waterbase_groups:

        continue


    candidates = waterbase_groups[
        country
    ].copy()


    # ========================================================
    # CALCULATE DISTANCE
    # ========================================================

    distances = haversine_km(
        lat,
        lon,
        candidates["station_lat"].values,
        candidates["station_lon"].values
    )


    candidates["distance_km"] = distances


    # ========================================================
    # SPATIAL FILTER
    # ========================================================

    candidates = candidates[
        candidates["distance_km"]
        <= MAX_DISTANCE_KM
    ]


    if len(candidates) == 0:

        continue


    # ========================================================
    # CALCULATE DATE DIFFERENCE
    # ========================================================

    candidates["date_diff_days"] = (

        candidates[
            "phenomenonTimeSamplingDate"
        ]

        - sample_date

    ).abs().dt.total_seconds() / 86400


    # ========================================================
    # TEMPORAL FILTER
    # ========================================================

    candidates = candidates[
        candidates["date_diff_days"]
        <= MAX_DATE_DIFF_DAYS
    ]


    if len(candidates) == 0:

        continue


    # ========================================================
    # SORT BY DATE THEN DISTANCE
    # ========================================================

    candidates = candidates.sort_values(
        by=[
            "date_diff_days",
            "distance_km"
        ]
    )


    # ========================================================
    # GET BEST VALUE FOR EACH FEATURE
    # ========================================================

    feature_matched = False


    for feature in candidates["feature"].unique():

        feature_rows = candidates[
            candidates["feature"] == feature
        ].copy()


        if len(feature_rows) == 0:

            continue


        feature_rows = feature_rows.sort_values(
            by=[
                "date_diff_days",
                "distance_km"
            ]
        )


        best = feature_rows.iloc[0]


        smx.loc[
            idx,
            feature
        ] = best[
            "resultObservedValue"
        ]


        feature_matched = True


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    if feature_matched:

        matched_count += 1


        best_overall = candidates.iloc[0]


        smx.loc[
            idx,
            "water_quality_station"
        ] = best_overall[
            "monitoringSiteIdentifier"
        ]


        smx.loc[
            idx,
            "water_quality_distance_km"
        ] = best_overall[
            "distance_km"
        ]


        smx.loc[
            idx,
            "water_quality_date"
        ] = best_overall[
            "phenomenonTimeSamplingDate"
        ]


        smx.loc[
            idx,
            "water_quality_date_diff_days"
        ] = best_overall[
            "date_diff_days"
        ]


    # ========================================================
    # PROGRESS
    # ========================================================

    if idx % 100 == 0:

        print(
            f"Processed {idx}/{len(smx)} SMX rows..."
        )


# ============================================================
# CREATE OPTIONAL NUTRIENTS FEATURE
# ============================================================

nutrient_cols = [

    "nitrate",
    "nitrite",
    "ammonium",
    "phosphate"
]


smx["nutrients"] = smx[
    nutrient_cols
].mean(
    axis=1,
    skipna=True
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n========================================")

print("FINAL CHECK")

print("========================================")


print(
    "Original SMX rows:",
    original_rows
)


print(
    "Final SMX rows:",
    len(smx)
)


print(
    "SMX rows with at least one water-quality match:",
    matched_count
)


if len(smx) != original_rows:

    raise RuntimeError(
        "ERROR: Number of SMX rows changed!"
    )


# ============================================================
# FEATURE COVERAGE
# ============================================================

print(
    "\nWater-quality feature coverage:"
)


for col in [

    "pH",
    "EC",
    "TOC",
    "nutrients",
    "turbidity",
    "suspended_solids",
    "nitrate",
    "nitrite",
    "ammonium",
    "phosphate"

]:

    matched = smx[col].notna().sum()

    percentage = (
        matched / len(smx)
    ) * 100

    print(
        f"{col:20s}: "
        f"{matched:6d} / {len(smx)} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# SAVE
# ============================================================

smx.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSaved:",
    OUTPUT_FILE
)


print("\nDone!")