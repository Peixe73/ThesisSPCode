import logging
from typing import Optional
from torch import nn
from core.nn.layers import MakeBinary, NegateMask, Reorder

module_logger = logging.getLogger(__name__)

# Full class names
"""
CLASSES = [
    'ClassId'        # original numeric classes 0..42
    #'SignClass',      # A..D categories
]
"""

"""
SIGNCLASS = [
    'SignClass'      # A..D categories
]
"""

# Short names for convenience
"""
SHORT_CLASSES = [
    'CId',  # for ClassId
    'SC',   # for SignClass
]
"""

# Binary column names
SIGNCLASS_BINARY = ["SignClass_A", "SignClass_B", "SignClass_C", "SignClass_D"]
CLASSID_BINARY = [f"ClassId_{i}" for i in range(43)]

CLASS_COLS = [
    "ClassId_1","ClassId_2","ClassId_3","ClassId_4","ClassId_5",
    "ClassId_6","ClassId_7","ClassId_8","ClassId_9","ClassId_10",
    "ClassId_11","ClassId_12","ClassId_13","ClassId_14","ClassId_15",
    "ClassId_16","ClassId_17","ClassId_18","ClassId_33","ClassId_34",
    "ClassId_35","ClassId_36","ClassId_37","ClassId_38","ClassId_39",
    "ClassId_40","ClassId_41","ClassId_42","ClassId_43",
]



# Concept names for PN attribution
"""
CONCEPTS = [
    "Bar", "Black_Bar", "White_Bar",
    "Border", "Black_Border", "Red_Border",
    "Red_Ground", "White_Ground", "Yellow_Ground",
    "End_Prohibition", "Start_Prohibition",
    "Circular_Shape", "Diamond_Shape", "Triangular_Shape", "Octagonal_Shape",
    "Symbol", "Black_Symbol", "White_Symbol",
    "Symbol_NoEntryGoods", "Symbol_Overtaking", "Symbol_OvertakingGoods",
    "Symbol_Speed20", "Symbol_Speed30", "Symbol_Speed50", "Symbol_Speed60",
    "Symbol_Speed70", "Symbol_Speed80", "Symbol_Speed100", "Symbol_Speed120",
    "Symbol_Stop",
    "Blue", "White_Border", "Post",
    "D1a1", "D1a4", "D1a5", "D1a6", "D1a7",
    "D2a1", "D2a2", "D3"
]
"""

CONCEPTS = [
    "Circular_Shape","Diamond_Shape","Triangular_Shape","Octagonal_Shape",
    "Red_Ground","White_Ground","Yellow_Ground","Blue",
    "Border","Black_Border","Red_Border","White_Border",
    "Bar","Black_Bar","White_Bar",
    "Symbol","Black_Symbol","White_Symbol",
    "Symbol_Stop","Symbol_NoEntryGoods","Symbol_Overtaking",
    "Symbol_OvertakingGoods","Symbol_Speed20","Symbol_Speed30",
    "Symbol_Speed50","Symbol_Speed60","Symbol_Speed70",
    "Symbol_Speed80","Symbol_Speed100","Symbol_Speed120",
    "D1a1","D1a4","D1a5","D1a6","D1a7","D2a1","D2a2","D3",
]

CLASSES = CONCEPTS + CLASS_COLS

SHORT_CONCEPTS = [c.replace('_', '') for c in CONCEPTS]  # simple short names

SHORT_CLASSES = SHORT_CONCEPTS + CLASS_COLS

def log_short_class_correspondence(logger: logging.Logger):
    correspondence = [f'\t{name} -> {short}' for name, short in zip(CLASSES, SHORT_CLASSES)]
    logger.info('Concept names have been shortened for convenience:\n' +
                ('\n'.join(correspondence)))


def class_to_manchester_assertion(cls: str, negate: bool = False) -> str:
    prefix = "__input__ Type: "
    negation = "not " if negate else ""
    if cls in ["ClassId", "SignClass"]:
        concept = f"({cls})"
    elif cls in CONCEPTS:
        concept = f"(has {cls})"
    else:
        raise ValueError(f"Unknown class/concept: {cls}")
    return f"{prefix}{negation}{concept}"


def class_to_latex_cmd(cls: str):
    negate = False
    if cls.startswith('!'):
        cls = cls[1:]
        negate = True
    if cls in SHORT_CLASSES:
        cls = CLASSES[SHORT_CLASSES.index(cls)]
    elif cls not in CLASSES:
        raise ValueError(f"Unknown class: {cls}")
    cmd = f"\\{cls}"
    if negate:
        cmd = f"\\neg {cmd}"
    return f"${cmd}$"


def make_order_from_attribution(attribution: list[str]):
    for c in attribution:
        if c not in SHORT_CONCEPTS:
            raise ValueError(f"Unknown concept {c} in attribution")
    module_logger.info(f"Searching indices for concepts {attribution}")
    indices = [SHORT_CONCEPTS.index(c) for c in attribution]
    str_builder = [f"Found indices {indices}, corresponding to the attribution:\n"]
    for pn_i, rn_i in enumerate(indices):
        str_builder.append(f"\tPN output {pn_i} -> RN input {rn_i} ({SHORT_CONCEPTS[rn_i]})\n")
    str_builder.append("Reordering")
    module_logger.info(''.join(str_builder))
    order = [indices.index(i) for i in range(len(indices))]
    str_builder = [f"Order: {order}\n"]
    for rn_i, pn_i in enumerate(order):
        str_builder.append(f"\tPN output {pn_i} -> RN input {rn_i} ({SHORT_CONCEPTS[rn_i]})\n")
    module_logger.info(''.join(str_builder))
    return order


def prepare_pn_with_attribution(
        pn: nn.Module,
        attribution: list[str],
        binary_threshold: Optional[float]
) -> nn.Sequential:
    negate = [c.startswith('!') for c in attribution]
    concepts = [c[1:] if n else c for c, n in zip(attribution, negate)]

    layers = [pn]
    if binary_threshold is not None:
        module_logger.info(f"Adding binary threshold layer with threshold {binary_threshold}")
        module_logger.warning("PN output values will be discrete: precisely 0 or 1")
        layers.append(MakeBinary(binary_threshold))
    if any(negate):
        module_logger.info(f"Applying negation mask to PN: {negate}")
        layers.append(NegateMask(negate))
    order = make_order_from_attribution(concepts)
    layers.append(Reorder(order))
    return nn.Sequential(*layers)