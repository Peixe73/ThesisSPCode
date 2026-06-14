import logging
from typing import Optional
from torch import nn
from core.nn.layers import MakeBinary, NegateMask, Reorder
from analysis_tools.ontology_mapping import ONTOLOGY_CONCEPT_MAP

module_logger = logging.getLogger(__name__)

# Binary column names
#SIGNCLASS_BINARY = ["SignClass_A", "SignClass_B", "SignClass_C", "SignClass_D"]
#CLASSID_BINARY = [f"ClassId_{i}" for i in range(43)]

"""
CLASS_COLS = [
    "ClassId_1","ClassId_2","ClassId_3","ClassId_4","ClassId_5",
    "ClassId_6","ClassId_7","ClassId_8","ClassId_9","ClassId_10",
    "ClassId_11","ClassId_12","ClassId_13","ClassId_14","ClassId_15",
    "ClassId_16","ClassId_17","ClassId_18","ClassId_33","ClassId_34",
    "ClassId_35","ClassId_36","ClassId_37","ClassId_38","ClassId_39",
    "ClassId_40","ClassId_41","ClassId_42","ClassId_43",
]
"""

CLASS_COLS = [ "C14_20", "C14_30", "C14_50", "C14_60", "C14_70",
               "C14_80", "C17b_80", "C14_100", "C14_120","C13aa",
               "C13bb", "A19a", "B3", "B1", "B2a",
               "C2", "C3e3", "C1a", "A32", "A1a",
               "A1b", "A1c", "A7a", "A9", "A4b2",
               "A16", "A17a", "A33", "A13", "A14",
               "A34", "A15b", "C17a", "D1a5", "D1a4",
               "D1a1", "D1a7", "D1a6", "D2a2", "D2a1",
               "D3", "C17c", "C17d"
             ]

CONCEPTS = [
    "Circular_Shape","Diamond_Shape","Triangular_Shape","Octagonal_Shape",
    "Red_Ground","White_Ground","Yellow_Ground","Blue_Ground",
    "Border","Black_Border","Red_Border","White_Border",
    "Bar","Black_Bar","White_Bar",
    "Symbol","Black_Symbol","White_Symbol",
    "Symbol_Stop","Symbol_NoEntryGoods","Symbol_Overtaking",
    "Symbol_OvertakingGoods","Symbol_Speed20","Symbol_Speed30",
    "Symbol_Speed50","Symbol_Speed60","Symbol_Speed70",
    "Symbol_Speed80","Symbol_Speed100","Symbol_Speed120",
    "Symbol_D1a1","Symbol_D1a4","Symbol_D1a5","Symbol_D1a6",
    "Symbol_D1a7","Symbol_D2a1","Symbol_D2a2","Symbol_D3",
    "Symbol_SingleLeftBend", "Symbol_SingleRightBend",
    "Symbol_TwoOrMoreLeftBend", "Symbol_RightNarrow", "Symbol_RoadDeformities",
    "Symbol_SlipperyRoad", "Symbol_Children", "Symbol_Cyclists",
    "Symbol_WildAnimals", "Symbol_RoadWorks", "Symbol_VerticalLightSignals",
    "Symbol_IntersectionPriority", "Symbol_Danger", "Symbol_Pedestrians",
    "Symbol_IceSnow"
]

CLASSES = CONCEPTS + CLASS_COLS

SHORT_CONCEPTS = [c.replace('_', '') for c in CONCEPTS]  # simple short names

SHORT_CLASSES = SHORT_CONCEPTS + CLASS_COLS

SHORT_TO_FULL = dict(zip(SHORT_CONCEPTS, CONCEPTS))

FULL_SET = set(CONCEPTS)

CLASS_MAP = {
    **{f"ClassId_{i}": f"ClassId_{i}" for i in range(1, 44)}
}

def log_short_class_correspondence(logger: logging.Logger):
    correspondence = [f'\t{name} -> {short}' for name, short in zip(CLASSES, SHORT_CLASSES)]
    logger.info('Concept names have been shortened for convenience:\n' +
                ('\n'.join(correspondence)))
    

def class_to_manchester_assertion(cls : str, negate : bool = False) -> str | None:
    prefix = "__input__ Type: "
    negation = "not " if negate else ""
    concept : str
    match cls:
        case 'A1a' | 'A1b' | 'A1c' | 'A4b2' | 'A7a' | 'A9' | 'A13' | 'A14' | 'A15b' | 'A16' | 'A17a' | 'A19a' | 'A32' | 'A33' | 'A34'|'B1' | 'B2a' | 'B3' | 'C1a' | 'C2' | 'C3e3' | 'C13aa' | 'C13bb' | 'C14_20' | 'C14_30' | 'C14_50' | 'C14_60' | 'C14_70' | 'C14_80' | 'C14_100' | 'C14_120' | 'C17a' | 'C17b_80' | 'C17c' | 'C17d'| 'D1a1'| 'D1a4'| 'D1a5'| 'D1a6'| 'D1a7'| 'D2a1'| 'D2a2'| 'D3':
            concept = f"({cls})"
        case 'Diamond_Shape':
            concept = f"(hasShape some {cls.removesuffix('_Shape')})"
        case 'Triangular_Shape':
            concept = f"(hasShape some Triangle)"
        case 'Circular_Shape':
            concept = f"(hasShape some Circle)"
        case 'Octagonal_Shape':
            concept = f"(hasShape some Octagon)"
        case 'Red_Ground' | 'White_Ground' | 'Yellow_Ground' | 'Blue_Ground':
            concept = f"(hasGround some {cls.removesuffix('_Ground')})"
        #case 'Border':
        #    concept = f"(hasBorder some Thing)"
        case 'Black_Border' | 'Red_Border' | 'White_Border':
            concept = f"(hasBorder some {cls.removesuffix('_Border')})"
        #case 'Bar':
        #    concept = f"(hasBar some Thing)"
        case 'Black_Bar' | 'White_Bar':
            concept = f"(hasBar some {cls.removesuffix('_Bar')})"
        #case 'Symbol':
        #    concept = f"(hasSymbol some Thing)"
        case 'Black_Symbol' | 'White_Symbol':
            concept = f"(hasSymbol some {cls.removesuffix('_Symbol')})"
        case 'Symbol_Stop' | 'Symbol_NoEntryGoods' | 'Symbol_Overtaking' | 'Symbol_OvertakingGoods' | 'Symbol_Speed20' | 'Symbol_Speed30' | 'Symbol_Speed50' | 'Symbol_Speed60' | 'Symbol_Speed70' | 'Symbol_Speed80' | 'Symbol_Speed100' | 'Symbol_Speed120' | 'Symbol_SingleLeftBend' | 'Symbol_SingleRightBend' | 'Symbol_TwoOrMoreLeftBend' | 'Symbol_RightNarrow' | 'Symbol_RoadDeformities' | 'Symbol_SlipperyRoad' | 'Symbol_Children' | 'Symbol_Cyclists' | 'Symbol_WildAnimals' | 'Symbol_RoadWorks' | 'Symbol_VerticalLightSignals' | 'Symbol_IntersectionPriority' | 'Symbol_Danger' | 'Symbol_Pedestrians'| 'Symbol_IceSnow':
            concept = f"(hasSymbol some {cls.removeprefix('Symbol_')})"
        case  'Symbol_D1a1':
            concept = f"(hasSymbol some DirectionStraight)"
        case 'Symbol_D1a4':
            concept = f"(hasSymbol some DirectionLeft)"
        case 'Symbol_D1a5':
            concept = f"(hasSymbol some DirectionRight)"
        case 'Symbol_D1a6':
            concept = f"(hasSymbol some DirectionLeftStraight)"
        case 'Symbol_D1a7':
            concept = f"(hasSymbol some DirectionRightStraight)"
        case 'Symbol_D2a1':
            concept = f"(hasSymbol some PassLeft)"
        case 'Symbol_D2a2':
            concept = f"(hasSymbol some PassRight)"
        case 'Symbol_D3':
            concept = f"(hasSymbol some Roundabout)"
        case 'Border' | 'Bar' | 'Symbol':
            return None  # skip these, as they are not actual concepts but rather placeholders for the existence of a relation
        case _:
            raise ValueError(f"Unknown class: {cls}")
    return f"{prefix}{negation}{concept}"

# Needs Update
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


#Needs update
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
    
'''

"""
def concept_assertion(concept: str, negate: bool = False) -> str:
    prefix = "__input__ Type: "
    negation = "not " if negate else ""

    return f"{prefix}{negation}({concept})"
"""

def concept_assertion(concept: str, negate: bool = False) -> str:
    prefix = "__input__ Type: "

    if concept not in ONTOLOGY_CONCEPT_MAP:
        raise ValueError(f"Unknown concept in ontology mapping: {concept}")

    owl_expr = ONTOLOGY_CONCEPT_MAP[concept]

    if negate:
        owl_expr = f"not ({owl_expr})"

    return f"{prefix}{owl_expr}"

"""
def class_assertion(cls: str, negate: bool = False) -> str:
    prefix = "__input__ Type: "
    negation = "not " if negate else ""

    return f"{prefix}{negation}({cls})"
"""

def class_assertion(cls: str, negate: bool = False) -> str:
    prefix = "__input__ Type: "

    if cls not in CLASS_MAP:
        raise ValueError(f"Unknown class in ontology mapping: {cls}")

    owl_expr = CLASS_MAP[cls]

    if negate:
        owl_expr = f"not ({owl_expr})"

    return f"{prefix}{owl_expr}"
    
# Latex formatting

def class_to_latex_cmd(cls: str):
    negate = False

    if cls.startswith("!"):
        cls = cls[1:]
        negate = True

    if cls in SHORT_TO_FULL:
        cls = SHORT_TO_FULL[cls]

    cmd = f"\\{cls}"

    if negate:
        cmd = f"\\neg {cmd}"

    return f"${cmd}$"

# Attribution ordering

def make_order_from_attribution(attribution: list[str]):

    normalized = []

    for c in attribution:
        neg = c.startswith("!")
        if neg:
            c = c[1:]

        if c in SHORT_TO_FULL:
            c = SHORT_TO_FULL[c]

        if c not in FULL_SET:
            raise ValueError(f"Unknown concept: {c}")

        normalized.append(c)

    module_logger.info(f"Attribution (normalized): {normalized}")

    indices = [CONCEPTS.index(c) for c in normalized]

    order = [indices.index(i) for i in range(len(indices))]

    module_logger.info(f"Reorder: {order}")
    return order

# PN preparation

def prepare_pn_with_attribution(
    pn: nn.Module,
    attribution: list[str],
    binary_threshold: Optional[float],
) -> nn.Sequential:

    negate = [c.startswith("!") for c in attribution]

    concepts = [
        c[1:] if n else c
        for c, n in zip(attribution, negate)
    ]

    layers = [pn]

    if binary_threshold is not None:
        module_logger.info(f"Binary threshold: {binary_threshold}")
        layers.append(MakeBinary(binary_threshold))

    if any(negate):
        layers.append(NegateMask(negate))

    order = make_order_from_attribution(concepts)
    layers.append(Reorder(order))

    return nn.Sequential(*layers)
'''
    