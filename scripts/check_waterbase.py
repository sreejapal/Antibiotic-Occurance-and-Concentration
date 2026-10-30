import pandas as pd

from paths import WATERBASE_SPATIAL

file = WATERBASE_SPATIAL
df = pd.read_csv(
    file,
    nrows=5,
    low_memory=False
)

print("COLUMNS:")
for col in df.columns:
    print(repr(col))

print("\nFIRST 5 ROWS:")
print(df.head())