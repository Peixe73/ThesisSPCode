import csv
import json
from pathlib import Path

INPUT_CSV = Path("data/gtsrb_dataset/gtsrb_concepts_filtered_bin.csv")
OUTPUT_JSON = Path("data/gtsrb_class_patterns.json")


def load_rows():
    with open(INPUT_CSV, newline="") as f:
        return list(csv.DictReader(f))


def get_columns(rows):

    cols = list(rows[0].keys())

    # target classes
    class_cols = [c for c in cols if c.startswith("ClassId_")]

    # concept features are between Bar and D3
    start = cols.index("Bar")
    end = cols.index("D3")

    feature_cols = cols[start:end + 1]

    return feature_cols, class_cols


def extract_patterns(rows, feature_cols, class_cols):

    results = {}

    for class_col in class_cols:

        class_rows = [r for r in rows if float(r[class_col]) == 1.0]

        if not class_rows:
            continue

        must_true = []
        must_false = []

        for feat in feature_cols:

            values = [float(r[feat]) for r in class_rows]

            # TRUE if it appears at least once
            if any(v == 1.0 for v in values):
                must_true.append(feat)

            # FALSE only if never appears
            else:
                must_false.append(feat)

        results[class_col] = {
            "true": sorted(must_true),
            "false": sorted(must_false)
        }

    return results


def main():

    print("Loading CSV...")
    rows = load_rows()

    feature_cols, class_cols = get_columns(rows)

    print(f"Found {len(feature_cols)} features")
    print(f"Found {len(class_cols)} classes")

    print("Extracting patterns...")
    patterns = extract_patterns(rows, feature_cols, class_cols)

    print(f"Writing patterns to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(patterns, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()