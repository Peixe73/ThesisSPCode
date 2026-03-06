'''
Since this has a lot of concepts that are free, meaning that it would have the following combinations:
2^number of free variables = 2^41. This is an immense number, which means we have to find another way
of doing this
'''

from core.init import DO_SCRIPT_IMPORTS
from typing import TYPE_CHECKING
from collections import OrderedDict
from pathlib import Path
import csv
import sys
import torch
from datetime import timedelta
import logging
from core.util.progress_trackers import LogProgressContextManager

if TYPE_CHECKING or DO_SCRIPT_IMPORTS:
    from core.datasets.binary_generator import BinaryGeneratorBuilder

PATH_ORIG = Path("data/gtsrb_dataset/gtsrb_concepts_filtered_bin.csv")


def _build_gtsrb_ontology():
    gen = BinaryGeneratorBuilder()

    # ==========================================================
    # DEFINE FEATURE VARIABLES
    # ==========================================================
    # Shapes
    Circular_Shape = gen.free_variable()
    Diamond_Shape = gen.free_variable()
    Triangular_Shape = gen.free_variable()
    Octagonal_Shape = gen.free_variable()

    # Ground colors
    Red_Ground = gen.free_variable()
    White_Ground = gen.free_variable()
    Yellow_Ground = gen.free_variable()
    Blue = gen.free_variable()
    Start_Prohibition = gen.free_variable()
    End_Prohibition = gen.free_variable()

    # Borders
    Border = gen.free_variable()
    Black_Border = gen.free_variable()
    Red_Border = gen.free_variable()
    White_Border = gen.free_variable()

    # Bars
    Bar = gen.free_variable()
    Black_Bar = gen.free_variable()
    White_Bar = gen.free_variable()

    # Symbols
    Symbol = gen.free_variable()
    Black_Symbol = gen.free_variable()
    White_Symbol = gen.free_variable()
    Symbol_NoEntryGoods = gen.free_variable()
    Symbol_Overtaking = gen.free_variable()
    Symbol_OvertakingGoods = gen.free_variable()
    Symbol_Speed20 = gen.free_variable()
    Symbol_Speed30 = gen.free_variable()
    Symbol_Speed50 = gen.free_variable()
    Symbol_Speed60 = gen.free_variable()
    Symbol_Speed70 = gen.free_variable()
    Symbol_Speed80 = gen.free_variable()
    Symbol_Speed100 = gen.free_variable()
    Symbol_Speed120 = gen.free_variable()
    Symbol_Stop = gen.free_variable()
    D1a1 = gen.free_variable()
    D1a4 = gen.free_variable()
    D1a5 = gen.free_variable()
    D1a6 = gen.free_variable()
    D1a7 = gen.free_variable()
    D2a1 = gen.free_variable()
    D2a2 = gen.free_variable()
    D3 = gen.free_variable()

    # Others
    Post = gen.free_variable()

    # ==========================================================
    # STORE FEATURES (match CSV names)
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
        Symbol_NoEntryGoods=Symbol_NoEntryGoods,
        Symbol_Overtaking=Symbol_Overtaking,
        Symbol_OvertakingGoods=Symbol_OvertakingGoods,
        Symbol_Speed20=Symbol_Speed20,
        Symbol_Speed30=Symbol_Speed30,
        Symbol_Speed50=Symbol_Speed50,
        Symbol_Speed60=Symbol_Speed60,
        Symbol_Speed70=Symbol_Speed70,
        Symbol_Speed80=Symbol_Speed80,
        Symbol_Speed100=Symbol_Speed100,
        Symbol_Speed120=Symbol_Speed120,
        Symbol_Stop=Symbol_Stop,
        Start_Prohibition=Start_Prohibition,
        End_Prohibition=End_Prohibition,
        Post=Post,
        D1a1=D1a1,
        D1a4=D1a4,
        D1a5=D1a5,
        D1a6=D1a6,
        D1a7=D1a7,
        D2a1=D2a1,
        D2a2=D2a2,
        D3=D3,
    )

    # ==========================================================
    # HIGH-LEVEL SIGN CLASSES
    # ==========================================================
    warning_sign = Triangular_Shape & (White_Ground | Yellow_Ground) & Red_Border
    yield_sign = Triangular_Shape & Red_Border & ~Symbol
    stop_sign = Octagonal_Shape & Red_Ground & Symbol_Stop
    priority_road = Diamond_Shape & Yellow_Ground & White_Border
    priority_sign = yield_sign | stop_sign | priority_road
    prohibitory_sign = Circular_Shape & Start_Prohibition & Red_Border
    mandatory_sign = Circular_Shape & Blue & Symbol

    class_a = warning_sign
    class_b = priority_sign
    class_c = prohibitory_sign
    class_d = mandatory_sign

    with open(PATH_ORIG, newline="") as f:
        reader = csv.DictReader(f)
        original_rows = [row for row in reader]

    classid_cols = [h for h in reader.fieldnames if h.startswith("ClassId_")]

    # Add ClassId columns as free variables before build()
    for col in classid_cols:
        gen.labels[col] = gen.free_variable()

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
    # OVERRIDE generate_from_int TO FILL CLASSID DYNAMICALLY
    # ==========================================================
    original_generate = generator.generate_from_int

    def generate_with_classid(index: int, force_valid: bool = False):
        if index >= len(original_rows):
            raise IndexError(f"Index {index} out of bounds")
        row = original_generate(index, force_valid)
        for i, col in enumerate(classid_cols):
            row[len(generator.feature_names) + i] = torch.tensor(
                int(original_rows[index][col]), dtype=row[0].dtype
            )
        return row

    generator.generate_from_int = generate_with_classid
    generator.__len__ = lambda: len(original_rows)

    return generator


def main():
    generator = _build_gtsrb_ontology()
    feature_names = generator.feature_names
    label_names = generator.label_names
    assert feature_names is not None and label_names is not None

    PATH_OUT = Path("data/gtsrb_ontology.csv")
    if PATH_OUT.exists():
        raise FileExistsError(f"{PATH_OUT} already exists")

    header = feature_names + label_names + [generator.valid_label]

    # Logger and progress manager
    logger = logging.getLogger(__name__)
    progress_cm = LogProgressContextManager(logger, cooldown=timedelta(minutes=5))

    # Write CSV incrementally
    with open(PATH_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    with progress_cm.track("Dataset Generation", "rows") as progress:
        for i in range(len(generator)):
            row = generator.generate_from_int(i, force_valid=False)
            row = torch.cat(row)
            writer.writerow(row.tolist())
            progress.tick()