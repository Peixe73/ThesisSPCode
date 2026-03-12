"""
Generated translation of the Traffic Signs Ontology (Vienna Convention 1968)
from the dissertation appendix.

These rules define ontology concepts (A, B, C, D and subclasses).
Dataset classes (ClassId_X) should map to these concepts later.
"""

# ============================================================
# CATEGORY A — Danger Warning Signs
# ============================================================

def A(env):
    return (
        env["Triangular_Shape"]
        and (env["White_Ground"] or env["Yellow_Ground"])
        and env["Red_Border"]
        and env["Symbol"]
    )


def A1a(env):
    return A(env) and env["Symbol_SingleLeftBend"]


def A1b(env):
    return A(env) and env["Symbol_SingleRightBend"]


def A1c(env):
    return A(env) and env["Symbol_TwoOrMoreLeftBend"]


def A4b2(env):
    return A(env) and env["Symbol_RightNarrow"]


def A7a(env):
    return A(env) and env["Symbol_RoadDeformities"]


def A9(env):
    return A(env) and env["Symbol_SlipperyRoad"]


def A13(env):
    return A(env) and env["Symbol_Children"]


def A14(env):
    return A(env) and env["Symbol_Cyclists"]


def A15b(env):
    return A(env) and env["Symbol_WildAnimals"]


def A16(env):
    return A(env) and env["Symbol_RoadWorks"]


def A17a(env):
    return A(env) and env["Symbol_VerticalLightSignals"]


def A19a(env):
    return A(env) and env["Symbol_IntersectionPriority"]


def A32(env):
    return A(env) and env["Symbol_Danger"]


def A33(env):
    return A(env) and env["Symbol_Pedestrians"]


def A34(env):
    return A(env) and env["Symbol_IceSnow"]


# ============================================================
# CATEGORY B — Priority Signs
# ============================================================

def B1(env):
    return (
        env["Triangular_Shape"]
        and (env["White_Ground"] or env["Yellow_Ground"])
        and env["Red_Border"]
        and not env["Symbol"]
    )


def B2a(env):
    return (
        env["Octagonal_Shape"]
        and env["Red_Ground"]
        and env["White_Symbol"]
        and env["Symbol_Stop"]
    )


def B3(env):
    return (
        env["Diamond_Shape"]
        and (env["Yellow_Ground"] or env.get("Orange_Ground", False))
        and env["White_Border"]
    )


# ============================================================
# CATEGORY C — Prohibitory / Restrictive Signs
# ============================================================

def C(env):
    return env["Circular_Shape"]


def StartProhibition(env):
    return (
        (
            env["White_Ground"]
            or env["Yellow_Ground"]
            or (env["Blue"] and env["Red_Border"])
        )
    )


def C1a(env):
    return (
        env["Circular_Shape"]
        and env["Red_Ground"]
        and not env["Symbol"]
        and env["White_Bar"]
    )


def C2(env):
    return (
        env["Circular_Shape"]
        and (env["White_Ground"] or env["Yellow_Ground"])
        and env["Red_Border"]
        and not env["Symbol"]
    )


def C3e3(env):
    return (
        StartProhibition(env)
        and env["Symbol_NoEntryGoods"]
        and not env["Bar"]
    )


def C13aa(env):
    return (
        StartProhibition(env)
        and env["Symbol_Overtaking"]
        and not env["Bar"]
    )


def C13bb(env):
    return (
        StartProhibition(env)
        and env["Symbol_OvertakingGoods"]
        and not env["Bar"]
    )


# Speed limit signs
def C14_20(env):
    return StartProhibition(env) and env["Symbol_Speed20"]


def C14_30(env):
    return StartProhibition(env) and env["Symbol_Speed30"]


def C14_50(env):
    return StartProhibition(env) and env["Symbol_Speed50"]


def C14_60(env):
    return StartProhibition(env) and env["Symbol_Speed60"]


def C14_70(env):
    return StartProhibition(env) and env["Symbol_Speed70"]


def C14_80(env):
    return StartProhibition(env) and env["Symbol_Speed80"]


def C14_100(env):
    return StartProhibition(env) and env["Symbol_Speed100"]


def C14_120(env):
    return StartProhibition(env) and env["Symbol_Speed120"]


def C17a(env):
    return (
        env["Circular_Shape"]
        and (env["White_Ground"] or env["Yellow_Ground"])
        and not env["Border"]
        and env["Black_Bar"]
        and not env["Symbol"]
    )


def C17b_80(env):
    return (
        env["Circular_Shape"]
        and (env["White_Ground"] or env["Yellow_Ground"])
        and not env["Border"]
        and env["Black_Bar"]
        and env["Symbol_Speed80"]
    )


def C17c(env):
    return (
        env["Circular_Shape"]
        and (env["White_Ground"] or env["Yellow_Ground"])
        and not env["Border"]
        and env["Black_Bar"]
        and env["Symbol_Overtaking"]
    )


def C17d(env):
    return (
        env["Circular_Shape"]
        and (env["White_Ground"] or env["Yellow_Ground"])
        and not env["Border"]
        and env["Black_Bar"]
        and env["Symbol_OvertakingGoods"]
    )


# ============================================================
# CATEGORY D — Mandatory Signs
# ============================================================

def D(env):
    return (
        env["Circular_Shape"]
        and (
            (env["Blue"] and env["White_Symbol"])
            or (env["White_Ground"] and env["Black_Symbol"])
        )
    )


def D1a1(env):
    return D(env) and env["Symbol_DirectionStraight"]


def D1a2(env):
    return D(env) and env["Symbol_DirectionLeft"]


def D1a3(env):
    return D(env) and env["Symbol_DirectionRight"]


def D1a6(env):
    return D(env) and env["Symbol_DirectionLeftStraight"]


def D1a7(env):
    return D(env) and env["Symbol_DirectionRightStraight"]


def D2a1(env):
    return D(env) and env["Symbol_PassLeft"]


def D2a2(env):
    return D(env) and env["Symbol_PassRight"]


def D3(env):
    return D(env) and env["Symbol_Roundabout"]