import json
import csv

# Load Clingo JSON output
with open("models.json", "r") as f:
    data = json.load(f)

models = data["Call"][0]["Witnesses"]

rows = []

for m in models:
    atoms = set(m["Value"])

    row = {
        "circular": int("circular" in atoms),
        "diamond": int("diamond" in atoms),
        "triangular": int("triangular" in atoms),
        "octagonal": int("octagonal" in atoms),

        "red_ground": int("red_ground" in atoms),
        "white_ground": int("white_ground" in atoms),
        "yellow_ground": int("yellow_ground" in atoms),
        "blue": int("blue" in atoms),

        "border": int("border" in atoms),
        "red_border": int("red_border" in atoms),
        "white_border": int("white_border" in atoms),
        "black_border": int("black_border" in atoms),
        "bar": int("bar" in atoms),
        "black_bar": int("black_bar" in atoms),
        "white_bar": int("white_bar" in atoms),
        
        "symbol": int("symbol" in atoms),
        "white_symbol": int("white_symbol" in atoms),
        "black_symbol": int("black_symbol" in atoms),

        "symbol_stop": int("symbol_stop" in atoms),
        "symbol_noentrygoods": int("symbol_noentrygoods" in atoms),
        "symbol_overtaking": int("symbol_overtaking" in atoms),
        "symbol_overtakinggoods": int("symbol_overtakinggoods" in atoms),
        "symbol_speed20": int("symbol_speed20" in atoms),
        "symbol_speed30": int("symbol_speed30" in atoms),
        "symbol_speed50": int("symbol_speed50" in atoms),
        "symbol_speed60": int("symbol_speed60" in atoms),
        "symbol_speed70": int("symbol_speed70" in atoms),
        "symbol_speed80": int("symbol_speed80" in atoms),
        "symbol_speed100": int("symbol_speed100" in atoms),
        "symbol_speed120": int("symbol_speed120" in atoms),
        "d1a1": int("d1a1" in atoms),
        "d1a4": int("d1a4" in atoms),
        "d1a5": int("d1a5" in atoms),
        "d1a6": int("d1a6" in atoms),
        "d1a7": int("d1a7" in atoms),
        "d2a1": int("d2a1" in atoms),
        "d2a2": int("d2a2" in atoms),
        "d3": int("d3" in atoms),

        "class": None
    }

    # Extract class
    for atom in atoms:
        if atom.startswith("class("):
            row["class"] = atom.replace("class(", "").replace(")", "")

    rows.append(row)

# Save CSV
with open("clingo_models.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Exported {len(rows)} models to clingo_models.csv")