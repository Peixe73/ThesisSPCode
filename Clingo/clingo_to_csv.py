import json
import csv

# Load Clingo JSON output
with open("models.json", "r") as f:
    data = json.load(f)

models = data["Call"][0]["Witnesses"]


# Features expected from Clingo

FEATURES = [
    # Shape
    "circular",
    "diamond",
    "triangular",
    "octagonal",

    # Ground
    "red_ground",
    "white_ground",
    "yellow_ground",
    "blue",

    # Border
    "border",
    "red_border",
    "white_border",
    "black_border",

    # Bar
    "bar",
    "black_bar",
    "white_bar",

    # Symbol
    "symbol",
    "white_symbol",
    "black_symbol",

    # Existing symbols
    "symbol_stop",
    "symbol_noentrygoods",
    "symbol_overtaking",
    "symbol_overtakinggoods",

    "symbol_speed20",
    "symbol_speed30",
    "symbol_speed50",
    "symbol_speed60",
    "symbol_speed70",
    "symbol_speed80",
    "symbol_speed100",
    "symbol_speed120",

    # A-category warning symbols

    "symbol_singleleftbend",
    "symbol_singlerightbend",
    "symbol_twoormoreleftbend",
    "symbol_rightnarrow",
    "symbol_roaddeformities",
    "symbol_slipperyroad",
    "symbol_children",
    "symbol_cyclists",
    "symbol_wildanimals",
    "symbol_roadworks",
    "symbol_verticallightsignals",
    "symbol_intersectionpriority",
    "symbol_danger",
    "symbol_pedestrians",
    "symbol_icesnow",

    # Mandatory / directional symbols
    "d1a1",
    "d1a4",
    "d1a5",
    "d1a6",
    "d1a7",
    "d2a1",
    "d2a2",
    "d3",
]


rows = []

for m in models:
    atoms = set(m["Value"])

    row = {
        feature: int(feature in atoms)
        for feature in FEATURES
    }

    # Extract class
    row["class"] = None

    for atom in atoms:
        if atom.startswith("class("):
            row["class"] = int(
                atom.replace("class(", "").replace(")", "")
            )

    rows.append(row)


# Save CSV
if rows:
    with open("clingo_models.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FEATURES + ["class"]
        )
        writer.writeheader()
        writer.writerows(rows)

print(f"Exported {len(rows)} models to clingo_models.csv")