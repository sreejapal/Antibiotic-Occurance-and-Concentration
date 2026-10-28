"""
Combines multiple NORMAN EMPODAT matrix exports (Soil, Surface water, Ground
water, Waste water, etc.) into ONE raw_SMX.csv containing:

  - your 14 required features
  - ONE extra "matrix_category" column (Soil / Surface water / Ground water / Waste water)

Then, per input file:
  - drops rows missing any of the required fields
  - randomly samples up to 2000 of the remaining complete rows

USAGE:
    python scripts/build_final_csv_v2.py data/raw/empodat/scrape/surface.csv "Surface water" \
        data/raw/empodat/scrape/ground.csv "Ground water" \
        data/raw/empodat/scrape/waste.csv "Waste water"

    Optional flags (put anywhere in the argument list):
        --sample N        change sample size per file (default 2000)
        --no-sample       keep ALL complete rows, don't sample
        --no-filter       keep all rows, even incomplete ones (skips the filter step)
"""

import csv
import random
import re
import sys

from paths import RAW_SMX

# The 14 required features + one extra matrix-category column
TARGET_COLUMNS = [
    "compound_name",
    "cas_number",
    "country",
    "site_id",
    "site_name",
    "latitude",
    "longitude",
    "sampling_date",
    "water_source",          # <-- the extra feature: Surface water / Ground water / Waste water
    "reported_concentration",
    "concentration_unit",
    "LoD",
    "LoQ",
    "result_qualifier",
    "analytical_method",
    "organisation",
    "laboratory",
    "source_reference",
]

CAS_RE = re.compile(r"CAS_RN:\s*([\d\-]+)", re.IGNORECASE)


def forward_fill(row):
    filled = []
    last = ""
    for cell in row:
        if cell.strip():
            last = cell.strip()
        filled.append(last)
    return filled


def build_column_keys(row0, row1, row2):
    section_ff = forward_fill(row0)
    field_ff = forward_fill(row1)
    keys = []
    for sec, field, sub in zip(section_ff, field_ff, row2):
        keys.append((field.strip(), sub.strip()))
    return keys


def find_col(keys, field_match, sub_match=None, exact=False):
    for i, (field, sub) in enumerate(keys):
        matched = (field.lower() == field_match.lower()) if exact \
            else (field_match.lower() in field.lower())
        if matched:
            if sub_match is None:
                return i
            if sub.lower() == sub_match.lower():
                return i
    return None


def parse_norman_export(path, matrix_category):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    row0, row1, row2 = rows[0], rows[1], rows[2]
    keys = build_column_keys(row0, row1, row2)

    idx = {
        "id":            find_col(keys, "ID"),
        "country":       find_col(keys, "Name of country"),
        "site_name":     find_col(keys, "Station name", "Name"),
        "longitude":     find_col(keys, "Longitude", "Decimal"),
        "latitude":      find_col(keys, "Latitude", "Decimal"),
        "sample_matrix": find_col(keys, "Sample matrix"),
        "compound":      find_col(keys, "Individual compound"),
        "cas":           find_col(keys, "CAS No"),
        "qualifier":     find_col(keys, "Concentration", exact=True),
        "value":         find_col(keys, "Value"),
        "unit":          find_col(keys, "Unit"),
        "day":           find_col(keys, "Sampling date", "Day"),
        "month":         find_col(keys, "Sampling date", "Month"),
        "year":          find_col(keys, "Sampling date", "Year"),
        "organisation":  find_col(keys, "Organisation"),
        "laboratory":    find_col(keys, "Laboratory"),
        "author":        find_col(keys, "Author"),
        "lod":           find_col(keys, "Limit of Detection"),
        "loq":           find_col(keys, "Limit of Quantification"),
        "method":        find_col(keys, "Analytical method"),
    }

    missing = [k for k, v in idx.items() if v is None]
    if missing:
        print(f"  WARNING in {path}: could not locate columns for: {missing}")

    out_rows = []
    for r in rows[3:]:
        if not any(cell.strip() for cell in r):
            continue

        def get(key):
            i = idx.get(key)
            if i is None or i >= len(r):
                return ""
            return r[i].strip()

        cas_raw = get("cas")
        cas_match = CAS_RE.search(cas_raw)
        cas_clean = cas_match.group(1) if cas_match else cas_raw

        day, month, year = get("day"), get("month"), get("year")
        sampling_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}" if (day and month and year) else ""

        out_rows.append({
            "compound_name":          get("compound"),
            "cas_number":              cas_clean,
            "country":                 get("country"),
            "site_id":                 get("id"),
            "site_name":               get("site_name"),
            "latitude":                get("latitude"),
            "longitude":               get("longitude"),
            "sampling_date":          sampling_date,
            "water_source":           matrix_category,
            "reported_concentration": get("value"),
            "concentration_unit":     get("unit"),
            "LoD":                    get("lod"),
            "LoQ":                    get("loq"),
            "result_qualifier":       get("qualifier"),
            "analytical_method":      get("method"),
            "organisation":           get("organisation"),
            "laboratory":             get("laboratory"),
            "source_reference":       get("author"),
        })

    return out_rows


def is_complete(row, return_reason=False):
    """
    Matches your original 14-feature spec, where a few items allow
    EITHER of two fields to count (e.g. organisation OR laboratory),
    instead of blindly requiring every single output column filled.

    If return_reason=True, returns a list of failed-check names instead
    of True/False (used by --diagnose to find the real bottleneck field).
    """
    def filled(key):
        return bool(str(row.get(key, "")).strip())

    checks = {
        "compound/CAS":        filled("compound_name") or filled("cas_number"),
        "country":             filled("country"),
        "site id/name":        filled("site_id") or filled("site_name"),
        "latitude+longitude":  filled("latitude") and filled("longitude"),
        "sampling_date":       filled("sampling_date"),
        "water_source":        filled("water_source"),
        "reported_concentration": filled("reported_concentration"),
        "concentration_unit":  filled("concentration_unit"),
        "LoD":                 filled("LoD"),
        "LoQ":                 filled("LoQ"),
        "result_qualifier":    filled("result_qualifier"),
        "analytical_method":   filled("analytical_method"),
        "organisation/lab":    filled("organisation") or filled("laboratory"),
        "source_reference":    filled("source_reference"),
    }
    if return_reason:
        return [name for name, ok in checks.items() if not ok]
    return all(checks.values())


def diagnose_file(rows):
    """Count how many rows fail each individual check, to find the bottleneck."""
    from collections import Counter
    fail_counts = Counter()
    for r in rows:
        for reason in is_complete(r, return_reason=True):
            fail_counts[reason] += 1
    print(f"  Failure breakdown (out of {len(rows)} total rows), most common first:")
    for reason, count in fail_counts.most_common():
        pct = 100 * count / len(rows) if rows else 0
        print(f"    - missing '{reason}': {count} rows ({pct:.1f}%)")


def main():
    args = sys.argv[1:]
    sample_size = 2000
    do_sample = True
    do_filter = True

    cleaned_args = []
    i = 0
    diagnose = False
    while i < len(args):
        a = args[i]
        if a == "--sample":
            sample_size = int(args[i+1])
            i += 2
            continue
        if a == "--no-sample":
            do_sample = False
            i += 1
            continue
        if a == "--no-filter":
            do_filter = False
            i += 1
            continue
        if a == "--diagnose":
            diagnose = True
            i += 1
            continue
        cleaned_args.append(a)
        i += 1

    if len(cleaned_args) < 2:
        print("Usage: python build_final_csv.py file1.csv Label1 file2.csv Label2 ... "
              "[--sample N] [--no-sample] [--no-filter]")
        sys.exit(1)

    file_label_pairs = []
    i = 0
    while i < len(cleaned_args):
        path = cleaned_args[i]
        label = cleaned_args[i+1] if i+1 < len(cleaned_args) else path
        file_label_pairs.append((path, label))
        i += 2

    random.seed(42)  # reproducible sampling
    all_rows = []

    for path, label in file_label_pairs:
        print(f"\nParsing {path} (matrix category: {label})...")
        rows = parse_norman_export(path, label)
        print(f"  -> {len(rows)} total records")

        if diagnose:
            diagnose_file(rows)

        if do_filter:
            complete_rows = [r for r in rows if is_complete(r)]
            print(f"  -> {len(complete_rows)} records with ALL required fields filled")
        else:
            complete_rows = rows

        if do_sample and len(complete_rows) > sample_size:
            sampled = random.sample(complete_rows, sample_size)
            print(f"  -> randomly sampled {sample_size} of {len(complete_rows)}")
        else:
            sampled = complete_rows
            if do_sample:
                print(f"  -> kept all {len(sampled)} (fewer than sample size {sample_size})")

        all_rows.extend(sampled)

    out_path = RAW_SMX
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Wrote {len(all_rows)} total records to {out_path}")


if __name__ == "__main__":
    main()