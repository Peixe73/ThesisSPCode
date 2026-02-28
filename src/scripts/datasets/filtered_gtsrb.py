from pathlib import Path
import pandas as pd

PATH = Path("data/gtsrb_dataset")
INPUT = PATH / "gtsrb_concepts.csv"
OUTPUT = PATH / "gtsrb_concepts_filtered.csv"

CONCEPTS = [
    "Bar", "Black_Bar", "White_Bar",
    "Border", "Black_Border", "Red_Border",
    "Red_Ground", "White_Ground", "Yellow_Ground",
    "End_Prohibition", "Start_Prohibition",
    "Circular_Shape", "Diamond_Shape", "Triangular_Shape", "Octagonal_Shape",
    "Symbol", "Black_Symbol", "White_Symbol",
    "Symbol_NoEntryGoods", "Symbol_Overtaking", "Symbol_OvertakingGoods",
    "Symbol_Speed20", "Symbol_Speed30", "Symbol_Speed50", "Symbol_Speed60",
    "Symbol_Speed70", "Symbol_Speed80", "Symbol_Speed100", "Symbol_Speed120",
    "Symbol_Stop",
    "Blue", "White_Border", "Post",
    "D1a1", "D1a4", "D1a5", "D1a6", "D1a7",
    "D2a1", "D2a2", "D3"
]

def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT)

    print("Filtering rows with at least one concept...")
    filtered_df = df[df[CONCEPTS].sum(axis=1) > 0]

    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered_df)}")

    print("Saving filtered CSV...")
    filtered_df.to_csv(OUTPUT, index=False)

    print("Done.")

if __name__ == "__main__":
    main()