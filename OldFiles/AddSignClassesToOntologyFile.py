import pandas as pd


input_file = "gtsrb_ontology_Valid_Final.csv"
output_file = "gtsrb_ontology_Valid_Final_Cat.csv"

df = pd.read_csv(input_file)


classification_classes = [
    "C14_20", "C14_30", "C14_50", "C14_60", "C14_70",
    "C14_80", "C17b_80", "C14_100", "C14_120",
    "C13aa", "C13bb", "A19a", "B3", "B1", "B2a",
    "C2", "C3e3", "C1a", "A32", "A1a", "A1b", "A1c",
    "A7a", "A9", "A4b2", "A16", "A17a", "A33",
    "A13", "A14", "A34", "A15b", "C17a",
    "D1a5", "D1a4", "D1a1", "D1a7", "D1a6",
    "D2a2", "D2a1", "D3", "C17c", "C17d"
]


class_groups = {
    "SignClass_A": [
        "A1a", "A1b", "A1c", "A4b2", "A7a", "A9",
        "A13", "A14", "A15b", "A16", "A17a", "A19a", "A32", "A33", "A34"
    ],

    "SignClass_B": [
        "B1", "B2a", "B3"
    ],

    "SignClass_C": [
        "C14_20", "C14_30", "C14_50", "C14_60", "C14_70",
        "C14_80", "C17b_80", "C14_100", "C14_120",
        "C13aa", "C13bb", "C2", "C3e3", "C1a", "C17a", "C17c", "C17d"
    ],

    "SignClass_D": [
        "D1a5", "D1a4", "D1a1", "D1a7", "D1a6",
        "D2a2", "D2a1", "D3"
    ],
}


for new_column, classes in class_groups.items():
    df[new_column] = df[classes].max(axis=1)


df.to_csv(output_file, index=False)

print(f"Saved: {output_file}")