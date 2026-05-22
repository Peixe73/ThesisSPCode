import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path("data/gtsrb_dataset/gtsrb_concepts_filtered_bin.csv")
OUTPUT_CSV = Path("data/gtsrb_dataset/gtsrb_concepts_merged_12.csv")


def main():

    # Classes to merge into class 12
    MERGED_CLASSES = list(range(18, 32))  # 19 -> 32

    TARGET_CLASS = 11 # ClassId_12 (0-indexed)

    print(f"Loading dataset from: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print(f"Original dataset size: {len(df)}")

    # SHOW ORIGINAL DISTRIBUTION

    print("\nOriginal class distribution:")
    print(df["ClassId"].value_counts().sort_index())

    # FIND ROWS TO MERGE

    merge_mask = df["ClassId"].isin(MERGED_CLASSES)

    print(f"\nRows being merged into class {TARGET_CLASS}:")
    print(merge_mask.sum())

    # UPDATE ClassId COLUMN

    df.loc[merge_mask, "ClassId"] = TARGET_CLASS

    # UPDATE ONE-HOT TARGETS

    # Target one-hot column
    target_col = f"ClassId_{TARGET_CLASS + 1}"

    print(f"\nSetting merged rows to: {target_col}")

    # Set target class to 1
    df.loc[merge_mask, target_col] = 1

    # Remove old merged class activations
    for cls in MERGED_CLASSES:
        old_col = f"ClassId_{cls + 1}"

        if old_col in df.columns:
            df.loc[merge_mask, old_col] = 0

    # REMOVE MERGED CLASS COLUMNS

    print("\nRemoving merged class columns...")

    cols_to_remove = [
        f"ClassId_{cls + 1}"
        for cls in MERGED_CLASSES
    ]

    existing_cols_to_remove = [
        col for col in cols_to_remove
        if col in df.columns
    ]

    df = df.drop(columns=existing_cols_to_remove)

    print(f"Removed columns:")
    print(existing_cols_to_remove)

    # SHOW NEW DISTRIBUTION

    print("\nNew class distribution:")
    print(df["ClassId"].value_counts().sort_index())

    # SAVE CSV

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved merged dataset to:")
    print(OUTPUT_CSV)

    print("\nDone.")


if __name__ == "__main__":
    main()