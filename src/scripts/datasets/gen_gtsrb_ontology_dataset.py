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

PATH_ORIG = Path("data/gtsrb_dataset/gtsrb_concepts_filtered_bin.csv")

def _build_gtsrb_ontology():
    gen = BinaryGeneratorBuilder()

    # ==========================================================
    # DEFINE FEATURE VARIABLES
    # Only variables that vary combinatorially
    # ==========================================================
    Circular_Shape = gen.free_variable()
    Diamond_Shape = gen.free_variable()
    Triangular_Shape = gen.free_variable()
    Octagonal_Shape = gen.free_variable()

    Red_Ground = gen.free_variable()
    White_Ground = gen.free_variable()
    Yellow_Ground = gen.free_variable()
    Blue = gen.free_variable()

    Start_Prohibition = gen.free_variable()
    End_Prohibition = gen.free_variable()

    Border = gen.free_variable()
    Black_Border = gen.free_variable()
    Red_Border = gen.free_variable()
    White_Border = gen.free_variable()

    Bar = gen.free_variable()
    Black_Bar = gen.free_variable()
    White_Bar = gen.free_variable()

    Symbol = gen.free_variable()  # master symbol
    Post = gen.free_variable()

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
        Start_Prohibition=Start_Prohibition,
        End_Prohibition=End_Prohibition,
        Post=Post,
    )

    # ==========================================================
    # HIGH-LEVEL SIGN CLASSES
    # ==========================================================
    warning_sign = Triangular_Shape & (White_Ground | Yellow_Ground) & Red_Border
    yield_sign = Triangular_Shape & Red_Border & ~Symbol
    stop_sign = Octagonal_Shape & Red_Ground & Symbol
    priority_road = Diamond_Shape & Yellow_Ground & White_Border
    priority_sign = yield_sign | stop_sign | priority_road
    prohibitory_sign = Circular_Shape & Start_Prohibition & Red_Border
    mandatory_sign = Circular_Shape & Blue & Symbol

    final_classes = [warning_sign, priority_sign, prohibitory_sign, mandatory_sign]
    class_a, class_b, class_c, class_d = final_classes

    # ==========================================================
    # LOAD ORIGINAL CSV FOR CLASSID
    # ==========================================================
    with open(PATH_ORIG, newline="") as f:
        reader = csv.DictReader(f)
        original_rows = [row for row in reader]
        classid_cols = [h for h in reader.fieldnames if h.startswith("ClassId_")]

    # ==========================================================
    # STORE HIGH-LEVEL LABELS
    # ==========================================================
    gen.labels.update(
        SignClass_A=class_a,
        SignClass_B=class_b,
        SignClass_C=class_c,
        SignClass_D=class_d,
        WarningSign=warning_sign,
        PrioritySign=priority_sign,
        ProhibitorySign=prohibitory_sign,
        MandatorySign=mandatory_sign,
        YieldSign=yield_sign,
        StopSign=stop_sign,
        PriorityRoad=priority_road,
    )

    # ==========================================================
    # BUILD GENERATOR
    # ==========================================================
    generator = gen.build()

    # ==========================================================
    # SYMBOL SPECIFICS (dynamic, not free variables)
    # ==========================================================
    symbol_specific_names = [
        "Black_Symbol", "White_Symbol", "Symbol_NoEntryGoods", "Symbol_Overtaking",
        "Symbol_OvertakingGoods", "Symbol_Speed20", "Symbol_Speed30", "Symbol_Speed50",
        "Symbol_Speed60", "Symbol_Speed70", "Symbol_Speed80", "Symbol_Speed100",
        "Symbol_Speed120", "Symbol_Stop",
        "D1a1", "D1a4", "D1a5", "D1a6", "D1a7", "D2a1", "D2a2", "D3"
    ]

    # ==========================================================
    # OVERRIDE generate_from_int
    # ==========================================================
    original_generate = generator.generate_from_int

    def generate_with_classid(index: int, force_valid: bool = False):
        # Clip index to CSV length
        row_index = index % len(original_rows)
        row_tuple = original_generate(row_index, force_valid)
        row_tensor = torch.cat(row_tuple)

        # -----------------------------
        # Symbols logic
        # -----------------------------
        symbol_tensor = torch.zeros(len(symbol_specific_names), dtype=row_tensor.dtype)
        symbol_idx = generator.feature_names.index("Symbol")
        if row_tensor[symbol_idx].item():  # master Symbol active
            chosen_idx = random.randrange(len(symbol_specific_names))
            symbol_tensor[chosen_idx] = 1

        row_tensor = torch.cat([row_tensor, symbol_tensor])

        # -----------------------------
        # Add ClassId dynamically
        # -----------------------------
        for col in classid_cols:
            val = int(original_rows[row_index][col])
            row_tensor = torch.cat([row_tensor, torch.tensor([val], dtype=row_tensor.dtype)])

        return row_tensor

    generator.generate_from_int = generate_with_classid
    generator.__len__ = lambda: len(original_rows)

    return generator


def main():
    generator = _build_gtsrb_ontology()

    # Symbol-specific names (dynamic)
    symbol_specific_names = [
        "Black_Symbol", "White_Symbol", "Symbol_NoEntryGoods", "Symbol_Overtaking",
        "Symbol_OvertakingGoods", "Symbol_Speed20", "Symbol_Speed30", "Symbol_Speed50",
        "Symbol_Speed60", "Symbol_Speed70", "Symbol_Speed80", "Symbol_Speed100",
        "Symbol_Speed120", "Symbol_Stop",
        "D1a1", "D1a4", "D1a5", "D1a6", "D1a7", "D2a1", "D2a2", "D3"
    ]

    feature_names = generator.feature_names + symbol_specific_names
    label_names = generator.label_names
    assert feature_names is not None and label_names is not None

    PATH_OUT = Path("data/gtsrb_ontology.csv")
    if PATH_OUT.exists():
        raise FileExistsError(f"{PATH_OUT} already exists")

    header = feature_names + label_names + [generator.valid_label]

    # Logger and progress manager
    logger = logging.getLogger(__name__)
    progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=5))

    logger.info("Starting dataset generation with %d rows", len(generator))

    # Write CSV incrementally
    with open(PATH_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with progress_cm.track("Dataset Generation", "rows") as progress:
            for i in range(len(generator)):
                row = generator.generate_from_int(i, force_valid=False)
                writer.writerow(row.tolist())
                progress.tick()

    logger.info("Dataset generation finished successfully!")