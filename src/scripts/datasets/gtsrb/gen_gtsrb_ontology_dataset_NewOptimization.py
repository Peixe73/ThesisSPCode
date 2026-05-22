from core.init import DO_SCRIPT_IMPORTS
from typing import TYPE_CHECKING
from collections import OrderedDict
from pathlib import Path
import csv
import torch
import logging
from core.util.progress_trackers import LogProgressContextManager
from datetime import timedelta
import importlib.util

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.datasets.binary_generator import BinaryGeneratorBuilder

GENERATED_RULES_PATH = Path("src/analysis_tools/gtsrb_rules.py")

spec = importlib.util.spec_from_file_location("gtsrb_rules", GENERATED_RULES_PATH)
gtsrb_rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gtsrb_rules)

PATH_OUT = Path("data/gtsrb_ontology_Valid_Final.csv")


# ==========================================================
# BUILD GENERATOR
# ==========================================================

def _build_gtsrb_generator():

    gen = BinaryGeneratorBuilder()

    Circular_Shape = gen.free_variable()
    Diamond_Shape = gen.free_variable()
    Triangular_Shape = gen.free_variable()
    Octagonal_Shape = gen.free_variable()

    Red_Ground = gen.free_variable()
    White_Ground = gen.free_variable()
    Yellow_Ground = gen.free_variable()
    Blue = gen.free_variable()

    Border = gen.free_variable()
    Black_Border = gen.free_variable()
    Red_Border = gen.free_variable()
    White_Border = gen.free_variable()

    Bar = gen.free_variable()
    Black_Bar = gen.free_variable()
    White_Bar = gen.free_variable()

    Symbol = gen.free_variable()
    Black_Symbol = gen.free_variable()
    White_Symbol = gen.free_variable()

    gen.features = OrderedDict(
        Circular_Shape=Circular_Shape,
        Diamond_Shape=Diamond_Shape,
        Triangular_Shape=Triangular_Shape,
        Octagonal_Shape=Octagonal_Shape,
        Red_Ground=Red_Ground,
        White_Ground=White_Ground,
        Yellow_Ground=Yellow_Ground,
        Blue=Blue,
        Border=Border,
        Black_Border=Black_Border,
        Red_Border=Red_Border,
        White_Border=White_Border,
        Bar=Bar,
        Black_Bar=Black_Bar,
        White_Bar=White_Bar,
        Symbol=Symbol,
        Black_Symbol=Black_Symbol,
        White_Symbol=White_Symbol,
    )

    return gen.build()


# ==========================================================
# SYMBOL TYPES
# ==========================================================

symbol_specific_names = [

    "Symbol_Stop",

    "Symbol_NoEntryGoods",
    "Symbol_Overtaking",
    "Symbol_OvertakingGoods",

    "Symbol_Speed20",
    "Symbol_Speed30",
    "Symbol_Speed50",
    "Symbol_Speed60",
    "Symbol_Speed70",
    "Symbol_Speed80",
    "Symbol_Speed100",
    "Symbol_Speed120",

    "D1a1",
    "D1a4",
    "D1a5",
    "D1a6",
    "D1a7",
    "D2a1",
    "D2a2",
    "D3",
]


# ==========================================================
# CLASS RULES
# ==========================================================

classid_rules = {

    "ClassId_1": gtsrb_rules.C14_20,
    "ClassId_2": gtsrb_rules.C14_30,
    "ClassId_3": gtsrb_rules.C14_50,
    "ClassId_4": gtsrb_rules.C14_60,
    "ClassId_5": gtsrb_rules.C14_70,
    "ClassId_6": gtsrb_rules.C14_80,
    "ClassId_7": gtsrb_rules.C17b_80,
    "ClassId_8": gtsrb_rules.C14_100,
    "ClassId_9": gtsrb_rules.C14_120,

    "ClassId_10": gtsrb_rules.C13aa,
    "ClassId_11": gtsrb_rules.C13bb,

    "ClassId_12": gtsrb_rules.A19a,

    "ClassId_13": gtsrb_rules.B3,
    "ClassId_14": gtsrb_rules.B1,
    "ClassId_15": gtsrb_rules.B2a,

    "ClassId_16": gtsrb_rules.C2,
    "ClassId_17": gtsrb_rules.C3e3,
    "ClassId_18": gtsrb_rules.C1a,

    "ClassId_33": gtsrb_rules.C17a,

    "ClassId_34": gtsrb_rules.D1a5,
    "ClassId_35": gtsrb_rules.D1a4,
    "ClassId_36": gtsrb_rules.D1a1,
    "ClassId_37": gtsrb_rules.D1a7,
    "ClassId_38": gtsrb_rules.D1a6,
    "ClassId_39": gtsrb_rules.D2a2,
    "ClassId_40": gtsrb_rules.D2a1,
    "ClassId_41": gtsrb_rules.D3,

    "ClassId_42": gtsrb_rules.C17c,
    "ClassId_43": gtsrb_rules.C17d,
}

classid_cols = list(classid_rules.keys())

CLASS12_FORBIDDEN = set([
    "Symbol_Stop",
    "Symbol_NoEntryGoods",
    "Symbol_Overtaking",
    "Symbol_OvertakingGoods",
    "Symbol_Speed20",
    "Symbol_Speed30",
    "Symbol_Speed50",
    "Symbol_Speed60",
    "Symbol_Speed70",
    "Symbol_Speed80",
    "Symbol_Speed100",
    "Symbol_Speed120",
    "D1a1",
    "D1a4",
    "D1a5",
    "D1a6",
    "D1a7",
    "D2a1",
    "D2a2",
    "D3",
])


# ==========================================================
# MAIN
# ==========================================================

def main():

    generator = _build_gtsrb_generator()

    feature_names = generator.feature_names
    label_names = generator.label_names

    feature_count = len(feature_names)
    symbol_count = len(symbol_specific_names)
    label_count = len(label_names)
    class_count = len(classid_cols)

    header = (
        feature_names +
        symbol_specific_names +
        label_names +
        classid_cols +
        ["valid"]
    )

    logger = logging.getLogger(__name__)

    progress_cm = LogProgressContextManager(
        logger,
        cooldown=timedelta(minutes=5)
    )

    symbol_idx = feature_names.index("Symbol")

    with open(PATH_OUT, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(header)

        with progress_cm.track("Dataset Generation", "rows") as progress:

            valid_count = 0
            invalid_count = 0

            for i in range(len(generator)):

                row_tuple = generator.generate_from_int(i, force_valid=False)
                base = torch.cat(row_tuple)

                features = base[:feature_count]
                labels = base[feature_count:feature_count + label_count]

                symbol_active = bool(features[symbol_idx].item())

                symbol_range = range(symbol_count) if symbol_active else [None]

                for sym in symbol_range:

                    symbol_tensor = torch.zeros(symbol_count, dtype=base.dtype)

                    env = {name: bool(features[j].item()) for j, name in enumerate(feature_names)}

                    # initialize all symbol variables, or else it errors.
                    for s in symbol_specific_names:
                        env[s] = False

                    # activate the chosen symbol
                    if sym is not None:
                        symbol_tensor[sym] = 1
                        env[symbol_specific_names[sym]] = True
                        
                    selected_symbol = symbol_specific_names[sym] if sym is not None else None

                    class_values = torch.zeros(class_count, dtype=base.dtype)

                    for j, (cid, rule) in enumerate(classid_rules.items()):
                        class_values[j] = int(rule(env))

                    has_class = int(class_values.sum().item() == 1)

                    visual_valid = (
                        (env["Circular_Shape"] + env["Diamond_Shape"] + env["Triangular_Shape"] + env["Octagonal_Shape"] == 1)
                        and (env["Red_Ground"] + env["White_Ground"] + env["Yellow_Ground"] + env["Blue"] == 1)
                        and ((env["Border"] and (env["Black_Border"] + env["Red_Border"] + env["White_Border"] == 1))
                             or (not env["Border"] and (env["Black_Border"] + env["Red_Border"] + env["White_Border"] == 0)))
                        and ((env["Bar"] and (env["Black_Bar"] + env["White_Bar"] == 1))
                             or (not env["Bar"] and (env["Black_Bar"] + env["White_Bar"] == 0)))
                        and ((env["Symbol"] and (env["Black_Symbol"] + env["White_Symbol"] == 1))
                             or (not env["Symbol"] and (env["Black_Symbol"] + env["White_Symbol"] == 0)))
                    )
                    
                    is_class12 = classid_rules["ClassId_12"](env)
                    
                    class12_valid = (
                        not is_class12
                        or (selected_symbol is not None and selected_symbol not in CLASS12_FORBIDDEN)
                    )

                    valid = int(visual_valid and has_class and class12_valid)

                    if valid:

                        valid_count += 1

                        row = torch.cat([
                            features,
                            symbol_tensor,
                            labels,
                            class_values,
                            torch.tensor([valid], dtype=base.dtype)
                        ])

                        writer.writerow(row.tolist())
                        progress.tick()

                    else:
                        invalid_count += 1
    logger.info(
        "Dataset generation completed with %d valid rows and %d invalid rows",
        valid_count,
        invalid_count
    )


if __name__ == "__main__":
    main()