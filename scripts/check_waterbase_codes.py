import pandas as pd

from paths import WATERBASE_DISAGG, WQ_DIAGNOSTIC

WQ_FILE = WATERBASE_DISAGG

TARGET_CODES = {
    "EEA_3152-01-0",   # pH
    "EEA_3142-01-6",   # EC
    "EEA_3133-06-0",   # TOC
    "EEA_3112-01-4",   # Turbidity
    "EEA_31-02-7",     # Suspended solids
    "CAS_14797-55-8",  # Nitrate
    "CAS_14797-65-0",  # Nitrite
    "CAS_14798-03-9",  # Ammonium
    "CAS_14265-44-2",  # Phosphate
}


found = []

print("Scanning Waterbase...\n")


for i, chunk in enumerate(
    pd.read_csv(
        WQ_FILE,
        chunksize=100000,
        low_memory=False
    )
):

    print(f"Processing chunk {i + 1}...")

    # Keep only our target parameters
    chunk = chunk[
        chunk[
            "observedPropertyDeterminandCode"
        ].isin(TARGET_CODES)
    ]

    if len(chunk) == 0:
        continue

    found.append(
        chunk[
            [
                "observedPropertyDeterminandCode",
                "observedPropertyDeterminandLabel",
                "procedureAnalysedMatrix",
                "resultUom",
                "phenomenonTimeSamplingDate",
                "resultObservedValue"
            ]
        ]
    )


# ============================================================
# RESULTS
# ============================================================

if not found:

    print("\nNO TARGET PARAMETERS FOUND.")
    raise SystemExit


df = pd.concat(
    found,
    ignore_index=True
)


print("\n==============================================")
print("TARGET PARAMETER RECORDS")
print("==============================================")

print(
    df[
        [
            "observedPropertyDeterminandCode",
            "observedPropertyDeterminandLabel"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "observedPropertyDeterminandLabel"
    )
    .to_string(index=False)
)


# ============================================================
# MATRIX VALUES
# ============================================================

print("\n==============================================")
print("PROCEDURE ANALYSED MATRIX")
print("==============================================")

print(
    df[
        "procedureAnalysedMatrix"
    ]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# MATRIX BY PARAMETER
# ============================================================

print("\n==============================================")
print("MATRIX BY PARAMETER")
print("==============================================")

matrix_table = pd.crosstab(
    df["observedPropertyDeterminandLabel"],
    df["procedureAnalysedMatrix"],
    dropna=False
)

print(matrix_table.to_string())


# ============================================================
# RECORD COUNTS
# ============================================================

print("\n==============================================")
print("RECORD COUNTS")
print("==============================================")

print(
    df[
        "observedPropertyDeterminandLabel"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# DATE INFORMATION
# ============================================================

print("\n==============================================")
print("DATE INFORMATION")
print("==============================================")

dates = pd.to_datetime(
    df["phenomenonTimeSamplingDate"],
    errors="coerce"
)

print("Minimum date:", dates.min())
print("Maximum date:", dates.max())
print("Missing dates:", dates.isna().sum())


# ============================================================
# SAVE
# ============================================================

output = WQ_DIAGNOSTIC

df.to_csv(
    output,
    index=False
)

print("\nSaved diagnostic:")
print(output)