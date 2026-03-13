from core.init import DO_SCRIPT_IMPORTS
from typing import TYPE_CHECKING
from collections import OrderedDict
from pathlib import Path
import csv
import torch
import random
import logging
from core.util.progress_trackers import LogProgressContextManager
from datetime import timedelta

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.datasets.binary_generator import BinaryGeneratorBuilder

import importlib.util


GENERATED_RULES_PATH = Path("data/gtsrb_rules.py")

spec = importlib.util.spec_from_file_location("gtsrb_rules", GENERATED_RULES_PATH)
gtsrb_rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gtsrb_rules)


PATH_ORIG = Path("data/gtsrb_dataset/gtsrb_concepts_filtered_bin.csv")


def _build_gtsrb_ontology():

    gen = BinaryGeneratorBuilder()

    # ==========================================================
    # FEATURE VARIABLES
    # ==========================================================

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

    # ==========================================================
    # VALIDITY RULES
    # ==========================================================

    # Exactly one shape
    shape_rule = (
        (Circular_Shape & ~Diamond_Shape & ~Triangular_Shape & ~Octagonal_Shape)
        | (~Circular_Shape & Diamond_Shape & ~Triangular_Shape & ~Octagonal_Shape)
        | (~Circular_Shape & ~Diamond_Shape & Triangular_Shape & ~Octagonal_Shape)
        | (~Circular_Shape & ~Diamond_Shape & ~Triangular_Shape & Octagonal_Shape)
    )

    # Exactly one ground
    ground_rule = (
        (Red_Ground & ~White_Ground & ~Yellow_Ground & ~Blue)
        | (~Red_Ground & White_Ground & ~Yellow_Ground & ~Blue)
        | (~Red_Ground & ~White_Ground & Yellow_Ground & ~Blue)
        | (~Red_Ground & ~White_Ground & ~Yellow_Ground & Blue)
    )

    # Border rule
    border_rule = (
        (Border & (
            (Black_Border & ~Red_Border & ~White_Border) |
            (~Black_Border & Red_Border & ~White_Border) |
            (~Black_Border & ~Red_Border & White_Border)
        ))
        |
        (~Border & ~Black_Border & ~Red_Border & ~White_Border)
    )

    # Bar rule
    bar_rule = (
        (Bar & (
            (Black_Bar & ~White_Bar) |
            (~Black_Bar & White_Bar)
        ))
        |
        (~Bar & ~Black_Bar & ~White_Bar)
    )

    # Symbol colour rule
    symbol_colour_rule = (
        (Symbol & (
            (Black_Symbol & ~White_Symbol) |
            (~Black_Symbol & White_Symbol)
        ))
        |
        (~Symbol & ~Black_Symbol & ~White_Symbol)
    )

    valid_rule = (
        shape_rule
        & ground_rule
        & border_rule
        & bar_rule
        & symbol_colour_rule
    )

    #gen.labels.update(Valid=valid_rule)
    gen.valid = valid_rule

    """
    valid_shape = (
        (Circular_Shape & ~Diamond_Shape & ~Triangular_Shape & ~Octagonal_Shape)
        | (~Circular_Shape & Diamond_Shape & ~Triangular_Shape & ~Octagonal_Shape)
        | (~Circular_Shape & ~Diamond_Shape & Triangular_Shape & ~Octagonal_Shape)
        | (~Circular_Shape & ~Diamond_Shape & ~Triangular_Shape & Octagonal_Shape)
    )

    gen.labels.update(ValidShape=valid_shape)

    border_rule = (
        (Border & (
            (Black_Border & ~Red_Border & ~White_Border) |
            (~Black_Border & Red_Border & ~White_Border) |
            (~Black_Border & ~Red_Border & White_Border)
        ))
        |
        (~Border & ~Black_Border & ~Red_Border & ~White_Border)
    )

    gen.labels.update(ValidBorder=border_rule)

    bar_rule = (
        (Bar & ((Black_Bar & ~White_Bar) | (~Black_Bar & White_Bar)))
        |
        (~Bar & ~Black_Bar & ~White_Bar)
    )

    gen.labels.update(ValidBar=bar_rule)
    """

    # ==========================================================
    # STORE FEATURES
    # ==========================================================

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

    # ==========================================================
    # SYMBOL FEATURES
    # ==========================================================

    symbol_specific_names = [

        # B
        "Symbol_Stop",

        # C
        "Symbol_NoEntryGoods",
        "Symbol_Overtaking",
        "Symbol_OvertakingGoods",

        # speed
        "Symbol_Speed20",
        "Symbol_Speed30",
        "Symbol_Speed50",
        "Symbol_Speed60",
        "Symbol_Speed70",
        "Symbol_Speed80",
        "Symbol_Speed100",
        "Symbol_Speed120",

        # D signs (names exactly as in CSV)
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
    # CLASSID RULES (ONTOLOGY MAPPING)
    # ==========================================================
    
    """
    ClassId_1: C14_20
    ClassId_2: C14_30
    ClassId_3: C14_50
    ClassId_4: C14_60
    ClassId_5: C14_70
    ClassId_6: C14_80
    ClassId_7: C17b_80
    ClassId_8: C14_100
    ClassId_9: C14_120
    ClassId_10: C13aa
    ClassId_11: C13bb
    ClassId_12: A19a
    ClassId_13: B3
    ClassId_14: B1
    ClassId_15: B2a
    ClassId_16: C2
    ClassId_17: C3e3
    ClassId_18: C1a
    ClassId_19: A32
    ClassId_20: A1a
    ClassId_21: A1b
    ClassId_22: A1c
    ClassId_23: A7a
    ClassId_24: A9
    ClassId_25: A4b2
    ClassId_26: A16
    ClassId_27: A17a
    ClassId_28: A33
    ClassId_29: A13
    ClassId_30: A14
    ClassId_31: A34
    ClassId_32: A15b
    ClassId_33: C17a
    ClassId_34: D1a5
    ClassId_35: D1a4
    ClassId_36: D1a1
    ClassId_37: D1a7
    ClassId_38: D1a6
    ClassId_39: D2a2
    ClassId_40: D2a1
    ClassId_41: D3
    ClassId_42: C17c
    ClassId_43: C17d
    """

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

        "ClassId_19": gtsrb_rules.A32,
        "ClassId_20": gtsrb_rules.A1a,
        "ClassId_21": gtsrb_rules.A1b,
        "ClassId_22": gtsrb_rules.A1c,
        "ClassId_23": gtsrb_rules.A7a,
        "ClassId_24": gtsrb_rules.A9,
        "ClassId_25": gtsrb_rules.A4b2,
        "ClassId_26": gtsrb_rules.A16,
        "ClassId_27": gtsrb_rules.A17a,
        "ClassId_28": gtsrb_rules.A33,
        "ClassId_29": gtsrb_rules.A13,
        "ClassId_30": gtsrb_rules.A14,
        "ClassId_31": gtsrb_rules.A34,
        "ClassId_32": gtsrb_rules.A15b,

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

    generator = gen.build()

    generator.symbol_specific_names = symbol_specific_names
    generator.classid_rules = classid_rules
    generator.classid_cols = list(classid_rules.keys())

    # ==========================================================
    # OVERRIDE GENERATOR
    # ==========================================================

    original_generate = generator.generate_from_int

    def generate_with_classid(index: int, force_valid=False):

        row_tuple = original_generate(index, force_valid)
        base_tensor = torch.cat(row_tuple)

        feature_count = len(generator.feature_names)
        label_count = len(generator.label_names)

        features = base_tensor[:feature_count]
        labels = base_tensor[feature_count:feature_count + label_count]
        valid = base_tensor[-1:]

        # ---------------------------------
        # create empty symbol tensor
        # ---------------------------------

        symbol_tensor = torch.zeros(len(symbol_specific_names), dtype=base_tensor.dtype)

        # ---------------------------------
        # build ontology environment
        # ---------------------------------

        env = {}

        for j, name in enumerate(generator.feature_names):
            env[name] = bool(features[j].item())

        for j, name in enumerate(symbol_specific_names):
            env[name] = False

        # ---------------------------------
        # compute class values
        # ---------------------------------

        class_values = torch.zeros(len(classid_rules), dtype=base_tensor.dtype)

        for i, (cid, rule) in enumerate(classid_rules.items()):
            class_values[i] = int(rule(env))

        # ---------------------------------
        # check if at least one class fires
        # ---------------------------------

        has_class = int(class_values.sum().item() >= 1)

        # ---------------------------------
        # compute visual validity
        # ---------------------------------

        visual_valid = (
            (env["Circular_Shape"] + env["Diamond_Shape"] + env["Triangular_Shape"] + env["Octagonal_Shape"] == 1)
            and (env["Red_Ground"] + env["White_Ground"] + env["Yellow_Ground"] + env["Blue"] == 1)
            and (
                (env["Border"] and (env["Black_Border"] + env["Red_Border"] + env["White_Border"] == 1))
                or (not env["Border"] and (env["Black_Border"] + env["Red_Border"] + env["White_Border"] == 0))
            )
            and (
                (env["Bar"] and (env["Black_Bar"] + env["White_Bar"] == 1))
                or (not env["Bar"] and (env["Black_Bar"] + env["White_Bar"] == 0))
            )
            and (
                (env["Symbol"] and (env["Black_Symbol"] + env["White_Symbol"] == 1))
                or (not env["Symbol"] and (env["Black_Symbol"] + env["White_Symbol"] == 0))
            )
        )

        valid_value = int(visual_valid and has_class)

        valid = torch.tensor([valid_value], dtype=base_tensor.dtype)

        return torch.cat([
            features,
            symbol_tensor,
            labels,
            class_values,
            valid
        ])

    generator.generate_from_int = generate_with_classid

    return generator


def main():

    generator = _build_gtsrb_ontology()

    feature_names = generator.feature_names
    label_names = generator.label_names
    symbol_specific_names = generator.symbol_specific_names
    classid_cols = generator.classid_cols

    PATH_OUT = Path("data/gtsrb_ontology.csv")

    if PATH_OUT.exists():
        raise FileExistsError(f"{PATH_OUT} already exists")

    header = (
        feature_names
        + symbol_specific_names
        + label_names
        + classid_cols
        + [generator.valid_label]
    )

    logger = logging.getLogger(__name__)

    progress_cm = LogProgressContextManager(
        logger,
        cooldown=timedelta(minutes=5)
    )

    logger.info("Starting dataset generation with %d rows", len(generator))

    with open(PATH_OUT, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(header)

        with progress_cm.track("Dataset Generation", "rows") as progress:
            
            valid_idx = header.index(generator.valid_label)
            
            valid_count = 0
            invalid_count = 0

            for i in range(len(generator)):

                base_row = generator.generate_from_int(i, force_valid=False)
                
                # skip invalid rows
                if bool(base_row[valid_idx].item()):
                    valid_count += 1
                else:
                    invalid_count += 1
                    continue
                
                symbol_idx = feature_names.index("Symbol")
                symbol_active = bool(base_row[symbol_idx].item())

                if symbol_active:

                    for j in range(len(symbol_specific_names)):

                        row_copy = base_row.clone()

                        start = len(feature_names)
                        end = start + len(symbol_specific_names)

                        row_copy[start:end] = 0
                        row_copy[start + j] = 1

                        writer.writerow(row_copy.tolist())
                        progress.tick()

                else:

                    writer.writerow(base_row.tolist())
                    progress.tick()
                    
    logger.info("Dataset generation completed with %d valid rows and %d invalid rows", valid_count, invalid_count)

    logger.info("Dataset generation finished successfully!")