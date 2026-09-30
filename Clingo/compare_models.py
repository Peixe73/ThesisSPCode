import pandas as pd

# Load both CSVs

clingo = pd.read_csv("clingo_models.csv")
gen = pd.read_csv("../data/gtsrb_ontology_Valid_Final.csv")


# Normalize generator column names to Clingo names

rename_map = {

    # Shape

    "Circular_Shape": "circular",
    "Diamond_Shape": "diamond",
    "Triangular_Shape": "triangular",
    "Octagonal_Shape": "octagonal",

    # Ground

    "Red_Ground": "red_ground",
    "White_Ground": "white_ground",
    "Yellow_Ground": "yellow_ground",
    "Blue_Ground": "blue",

    # Border

    "Border": "border",
    "Black_Border": "black_border",
    "Red_Border": "red_border",
    "White_Border": "white_border",

    # Bar

    "Bar": "bar",
    "Black_Bar": "black_bar",
    "White_Bar": "white_bar",

    # Generic symbol

    "Symbol": "symbol",
    "Black_Symbol": "black_symbol",
    "White_Symbol": "white_symbol",

    # Existing symbols

    "Symbol_Stop": "symbol_stop",
    "Symbol_NoEntryGoods": "symbol_noentrygoods",
    "Symbol_Overtaking": "symbol_overtaking",
    "Symbol_OvertakingGoods": "symbol_overtakinggoods",

    "Symbol_Speed20": "symbol_speed20",
    "Symbol_Speed30": "symbol_speed30",
    "Symbol_Speed50": "symbol_speed50",
    "Symbol_Speed60": "symbol_speed60",
    "Symbol_Speed70": "symbol_speed70",
    "Symbol_Speed80": "symbol_speed80",
    "Symbol_Speed100": "symbol_speed100",
    "Symbol_Speed120": "symbol_speed120",

    # A-category warning symbols

    "Symbol_SingleLeftBend": "symbol_singleleftbend",
    "Symbol_SingleRightBend": "symbol_singlerightbend",
    "Symbol_TwoOrMoreLeftBend": "symbol_twoormoreleftbend",
    "Symbol_RightNarrow": "symbol_rightnarrow",
    "Symbol_RoadDeformities": "symbol_roaddeformities",
    "Symbol_SlipperyRoad": "symbol_slipperyroad",
    "Symbol_Children": "symbol_children",
    "Symbol_Cyclists": "symbol_cyclists",
    "Symbol_WildAnimals": "symbol_wildanimals",
    "Symbol_RoadWorks": "symbol_roadworks",
    "Symbol_VerticalLightSignals": "symbol_verticallightsignals",
    "Symbol_IntersectionPriority": "symbol_intersectionpriority",
    "Symbol_Danger": "symbol_danger",
    "Symbol_Pedestrians": "symbol_pedestrians",
    "Symbol_IceSnow": "symbol_icesnow",

    # Mandatory / directional

    "D1a1": "d1a1",
    "D1a4": "d1a4",
    "D1a5": "d1a5",
    "D1a6": "d1a6",
    "D1a7": "d1a7",

    "D2a1": "d2a1",
    "D2a2": "d2a2",

    "D3": "d3",
}


gen = gen.rename(columns=rename_map)

# Extract class from generator

class_cols = [
    c for c in gen.columns
    if c.startswith("ClassId_")
]


def extract_class(row):
    for c in class_cols:
        if row[c] == 1.0:
            return int(c.replace("ClassId_", ""))
    return None


gen["class"] = gen.apply(extract_class, axis=1)

gen = gen.drop(columns=class_cols)

# Check that all Clingo features exist in generator

feature_cols = [
    c for c in clingo.columns
    if c != "class"
]

missing_in_generator = [
    c for c in feature_cols
    if c not in gen.columns
]

if missing_in_generator:
    print("\nERROR: Missing generator columns:")
    for c in missing_in_generator:
        print("  ", c)

    raise ValueError(
        "Generator CSV is missing features required by Clingo."
    )

# Normalize numeric types

for col in feature_cols:
    clingo[col] = (
        clingo[col]
        .fillna(0)
        .astype(int)
    )

    gen[col] = (
        gen[col]
        .fillna(0)
        .astype(int)
    )

# Feature dataframes

clingo_features = clingo[feature_cols].copy()
gen_features = gen[feature_cols].copy()

# Build class maps

clingo_class_map = dict(
    zip(
        clingo_features.apply(tuple, axis=1),
        clingo["class"]
    )
)

gen_class_map = dict(
    zip(
        gen_features.apply(tuple, axis=1),
        gen["class"]
    )
)

# Row matching

def row_key(df):
    return df.apply(tuple, axis=1)


clingo_keys = set(row_key(clingo_features))
gen_keys = set(row_key(gen_features))

only_clingo = clingo_keys - gen_keys
only_gen = gen_keys - clingo_keys


print("Clingo models:", len(clingo_keys))
print("Generator rows:", len(gen_keys))
print("Only in Clingo:", len(only_clingo))
print("Only in Generator:", len(only_gen))

# Debug helper

def find_best_match(row, df):
    best_idx = None
    best_score = -1

    for i, r in df.iterrows():
        score = (r == row).sum()

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score


def print_diff(a, b):
    diffs = []

    for col in a.index:
        if a[col] != b[col]:
            diffs.append(
                (col, a[col], b[col])
            )

    print("\nDIFFERENCES:")

    for col, v1, v2 in diffs:
        print(
            f"  {col:35s} "
            f"clingo={v1}  gen={v2}"
        )

    print(
        f"Total mismatches: {len(diffs)}"
    )


# CLINGO ONLY

print("\n==============================")
print("CLINGO ONLY (DETAILED)")
print("==============================")


for key in list(only_clingo)[:10]:

    clingo_row = pd.Series(
        key,
        index=feature_cols
    )

    best_idx, score = find_best_match(
        clingo_row,
        gen_features
    )

    gen_row = gen_features.iloc[best_idx]

    clingo_class = clingo_class_map.get(
        tuple(key),
        None
    )

    gen_class = gen_class_map.get(
        tuple(gen_row.values),
        None
    )

    print("\n--- CLINGO ROW ---")
    print("best match score:", score)
    print("clingo class:", clingo_class)
    print("gen class:", gen_class)

    print_diff(
        clingo_row,
        gen_row
    )

# GENERATOR ONLY

print("\n==============================")
print("GENERATOR ONLY (DETAILED)")
print("==============================")


for key in list(only_gen)[:10]:

    gen_row = pd.Series(
        key,
        index=feature_cols
    )

    best_idx, score = find_best_match(
        gen_row,
        clingo_features
    )

    clingo_row = clingo_features.iloc[best_idx]

    clingo_class = clingo_class_map.get(
        tuple(clingo_row.values),
        None
    )

    gen_class = gen_class_map.get(
        tuple(key),
        None
    )

    print("\n--- GENERATOR ROW ---")
    print("best match score:", score)
    print("clingo class:", clingo_class)
    print("gen class:", gen_class)

    print_diff(
        gen_row,
        clingo_row
    )

# Save mismatch files

pd.DataFrame(
    list(only_clingo),
    columns=feature_cols
).to_csv(
    "only_clingo.csv",
    index=False
)

pd.DataFrame(
    list(only_gen),
    columns=feature_cols
).to_csv(
    "only_generator.csv",
    index=False
)


print("\nSaved mismatch files.")