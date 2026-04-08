import pandas as pd

# Load both CSVs
clingo = pd.read_csv("clingo_models.csv")
gen = pd.read_csv("data/gtsrb_ontology_Valid_Final.csv")

# -----------------------------
# 1. Normalize column names
# -----------------------------
rename_map = {
    "Circular_Shape": "circular",
    "Diamond_Shape": "diamond",
    "Triangular_Shape": "triangular",
    "Octagonal_Shape": "octagonal",

    "Red_Ground": "red_ground",
    "White_Ground": "white_ground",
    "Yellow_Ground": "yellow_ground",
    "Blue": "blue",

    "Border": "border",
    "Black_Border": "black_border",
    "Red_Border": "red_border",
    "White_Border": "white_border",

    "Bar": "bar",
    "Black_Bar": "black_bar",
    "White_Bar": "white_bar",

    "Symbol": "symbol",
    "Black_Symbol": "black_symbol",
    "White_Symbol": "white_symbol",

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

# -----------------------------
# 2. Convert generator one-hot → class integer
# -----------------------------
class_cols = [c for c in gen.columns if c.startswith("ClassId_")]

def extract_class(row):
    for c in class_cols:
        if row[c] == 1.0:
            return int(c.replace("ClassId_", ""))
    return None

gen["class"] = gen.apply(extract_class, axis=1)

# Drop one-hot columns
gen = gen.drop(columns=class_cols)

# -----------------------------
# 3. Normalize values to int
# -----------------------------
for col in clingo.columns:
    if col in gen.columns:
        gen[col] = gen[col].astype(int)

# Ensure clingo is also int
for col in clingo.columns:
    clingo[col] = clingo[col].fillna(0).astype(int)

# -----------------------------
# 4. Create unique row keys
# -----------------------------
def row_key(row):
    return tuple(row.values)

clingo["key"] = clingo.apply(row_key, axis=1)
gen["key"] = gen.apply(row_key, axis=1)

# -----------------------------
# 5. Compare sets
# -----------------------------
clingo_set = set(clingo["key"])
gen_set = set(gen["key"])

only_clingo = clingo_set - gen_set
only_gen = gen_set - clingo_set

print("Clingo models:", len(clingo_set))
print("Generator rows:", len(gen_set))

print("Only in Clingo:", len(only_clingo))
print("Only in Generator:", len(only_gen))

# Save differences for inspection
pd.DataFrame(list(only_clingo)).to_csv("only_clingo.csv", index=False)
pd.DataFrame(list(only_gen)).to_csv("only_generator.csv", index=False)

print("Saved mismatch files.")