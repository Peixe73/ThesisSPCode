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

GENERATED_RULES_PATH = Path("data/generated_gtsrb_rules.py")

spec = importlib.util.spec_from_file_location("gtsrb_rules", GENERATED_RULES_PATH)
gtsrb_rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gtsrb_rules)

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
    
    # Marking Shapes Mutually Exclusive ============================

    valid_shape = (
        (Circular_Shape & ~Diamond_Shape & ~Triangular_Shape & ~Octagonal_Shape) |
        (~Circular_Shape & Diamond_Shape & ~Triangular_Shape & ~Octagonal_Shape) |
        (~Circular_Shape & ~Diamond_Shape & Triangular_Shape & ~Octagonal_Shape) |
        (~Circular_Shape & ~Diamond_Shape & ~Triangular_Shape & Octagonal_Shape)
    )

    gen.labels.update(
        ValidShape=valid_shape
    )
    # ===================================

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

    # Marking Borders Mutually Exclusive ============================
    border_rule = (
        (Border &
            (
                (Black_Border & ~Red_Border & ~White_Border) |
                (~Black_Border & Red_Border & ~White_Border) |
                (~Black_Border & ~Red_Border & White_Border)
            )
        )
        |
        (~Border & ~Black_Border & ~Red_Border & ~White_Border)
    )

    gen.labels.update(
        ValidBorder=border_rule
    )
    # ==========================================================
    
    # Marking Bars Mutually Exclusive ============================
    Bar = gen.free_variable()
    Black_Bar = gen.free_variable()
    White_Bar = gen.free_variable()
    
    bar_rule = (
        (Bar &
            (
                (Black_Bar & ~White_Bar) |
                (~Black_Bar & White_Bar)
            )
        )
        |
        (~Bar & ~Black_Bar & ~White_Bar)
    )

    gen.labels.update(
        ValidBar=bar_rule
    )
    # ==========================================================

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
    
    # ==========================================================
    # MUTUALLY EXCLUSIVE FINAL CLASSES
    # ==========================================================
    class_a, class_b, class_c, class_d = final_classes

    #exactly_one_final_class = (
    #    (class_a | class_b | class_c | class_d) &     # at least one
    #    ~(class_a & class_b) &
    #    ~(class_a & class_c) &
    #    ~(class_a & class_d) &
    #    ~(class_b & class_c) &
    #    ~(class_b & class_d) &
    #    ~(class_c & class_d)
    #)

    #gen.constraints.append(exactly_one_final_class)

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
        #WarningSign=warning_sign,
        #PrioritySign=priority_sign,
        #ProhibitorySign=prohibitory_sign,
        #MandatorySign=mandatory_sign,
        #YieldSign=yield_sign,
        #StopSign=stop_sign,
        #PriorityRoad=priority_road,
    )

    # ==========================================================
    # BUILD GENERATOR
    # ==========================================================
    generator = gen.build()
    
    generator.classid_cols = classid_cols

    for col in classid_cols:
        rule_str = getattr(gtsrb_rules, f"{col}_rule", None)
        if rule_str is None:
            continue

        eval_scope = {name: 0 for name in generator.feature_names}

        for name in [
            "Black_Symbol","White_Symbol","Symbol_NoEntryGoods","Symbol_Overtaking",
            "Symbol_OvertakingGoods","Symbol_Speed20","Symbol_Speed30","Symbol_Speed50",
            "Symbol_Speed60","Symbol_Speed70","Symbol_Speed80","Symbol_Speed100",
            "Symbol_Speed120","Symbol_Stop",
            "D1a1","D1a4","D1a5","D1a6","D1a7","D2a1","D2a2","D3"
        ]:
            eval_scope[name] = 0

        label_idx = generator.label_names.index(col)
        generator.labels[label_idx] = eval(rule_str, {}, eval_scope)
    
    signclass_names = [
        "SignClass_A",
        "SignClass_B",
        "SignClass_C",
        "SignClass_D"
    ]

    signclass_indices = [generator.label_names.index(name) for name in signclass_names]
    generator.signclass_indices = signclass_indices
    
    generator.classid_cols = classid_cols

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

        row_index = index % len(original_rows)
        row_tuple = original_generate(index, force_valid)

        base_tensor = torch.cat(row_tuple)

        feature_count = len(generator.feature_names)
        label_count = len(generator.label_names)

        features = base_tensor[:feature_count]
        labels = base_tensor[feature_count:feature_count + label_count]
        valid = base_tensor[-1:]

        # -----------------------------
        # Symbol logic
        # -----------------------------
        symbol_tensor = torch.zeros(len(symbol_specific_names), dtype=base_tensor.dtype)

        symbol_idx = generator.feature_names.index("Symbol")

        if features[symbol_idx].item() == 1:
            chosen_idx = torch.randint(0, len(symbol_specific_names), (1,)).item()
            symbol_tensor[chosen_idx] = 1

        # -----------------------------
        # Enforce exactly one SignClass
        # -----------------------------
        
        #label_offset = 0
        #active_classes = []

        #for idx in generator.signclass_indices:
        #    if labels[idx].item() == 1:
        #        active_classes.append(idx)

        #if len(active_classes) > 1:
        #    for idx in active_classes[1:]:
        #        labels[idx] = 0

        #elif len(active_classes) == 0:
        #    labels[random.choice(generator.signclass_indices)] = 1

        # -----------------------------
        # ClassId logic
        # -----------------------------
        class_values = torch.zeros(len(classid_cols), dtype=base_tensor.dtype)

        for i, col in enumerate(classid_cols):
            rule_str = getattr(gtsrb_rules, f"{col}_rule", None)
            if rule_str is None:
                continue

            # Build eval scope using current sample features
            eval_scope = {}

            for j, fname in enumerate(generator.feature_names):
                eval_scope[fname] = bool(features[j].item())

            for j, sname in enumerate(symbol_specific_names):
                eval_scope[sname] = bool(symbol_tensor[j].item())

            class_values[i] = int(eval(rule_str, {}, eval_scope))

        # -----------------------------
        # Final row assembly
        # -----------------------------
        row_tensor = torch.cat([
            features,
            symbol_tensor,
            labels,
            class_values,
            valid
        ])

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

    feature_names = generator.feature_names
    label_names = generator.label_names
    classid_cols = generator.classid_cols
    assert feature_names is not None and label_names is not None

    PATH_OUT = Path("data/gtsrb_ontology.csv")
    if PATH_OUT.exists():
        raise FileExistsError(f"{PATH_OUT} already exists")

    #header = feature_names + label_names + [generator.valid_label]
    header = (
        feature_names
        + symbol_specific_names
        + label_names
        + classid_cols
        + [generator.valid_label]
    )

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