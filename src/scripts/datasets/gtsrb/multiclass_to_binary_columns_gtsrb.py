from pathlib import Path
import pandas as pd

PATH = Path("data/gtsrb_dataset")
INPUT = PATH / "gtsrb_concepts_filtered.csv"
OUTPUT = PATH / "gtsrb_concepts_filtered_bin.csv"

# Columns to one-hot encode
CLASS_COLUMNS = ["ClassId", "SignClass"]

def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT)

    print("Bin encoding ClassId and SignClass...")
    # Convert to string to avoid numeric issues
    df[CLASS_COLUMNS] = df[CLASS_COLUMNS].astype(str)

    for col in CLASS_COLUMNS:
        bin = pd.get_dummies(df[col], prefix=col).astype(int)
        df = pd.concat([df, bin], axis=1)

    print("Saving binary encoded CSV...")
    df.to_csv(OUTPUT, index=False)
    print("Done.")

if __name__ == "__main__":
    main()