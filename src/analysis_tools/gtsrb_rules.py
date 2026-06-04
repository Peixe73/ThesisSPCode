"""
Translation of the Traffic Signs Ontology (Vienna Convention 1968)
adapted to match the GTSRB concept dataset.

Important adaptation:
The dataset does NOT contain individual danger-sign symbols,
so all A-category subclasses share the same structural rule.
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
        and env["Black_Symbol"]
        and not env["Bar"]
    )

# In Old Version
# Dataset cannot distinguish the specific symbols
# so subclasses collapse to the same rule

def A1a(env): return A(env) and env["Symbol_SingleLeftBend"]
def A1b(env): return A(env) and env["Symbol_SingleRightBend"]
def A1c(env): return A(env) and env["Symbol_TwoOrMoreLeftBend"]
def A4b2(env): return A(env) and env["Symbol_RightNarrow"]
def A7a(env): return A(env) and env["Symbol_RoadDeformities"]
def A9(env): return A(env) and env["Symbol_SlipperyRoad"]
def A13(env): return A(env) and env["Symbol_Children"]
def A14(env): return A(env) and env["Symbol_Cyclists"]
def A15b(env): return A(env) and env["Symbol_WildAnimals"]
def A16(env): return A(env) and env["Symbol_RoadWorks"]
def A17a(env): return A(env) and env["Symbol_VerticalLightSignals"]
def A19a(env): return A(env) and env["Symbol_IntersectionPriority"]
def A32(env): return A(env) and env["Symbol_Danger"]
def A33(env): return A(env) and env["Symbol_Pedestrians"]
def A34(env): return A(env) and env["Symbol_IceSnow"]


# ============================================================
# CATEGORY B — Priority Signs
# ============================================================

# B1 — Priority road (diamond)

"""
def B1(env):
    return (
        env["Diamond_Shape"]
        and env["Yellow_Ground"]
        and env["White_Border"]
    )
"""
def B1(env):
    return (
        env["Triangular_Shape"]
        and (env["Yellow_Ground"] or env["White_Ground"])
        and env["Red_Border"]
        and not env["Symbol"]
        and not env["Bar"]
    )


# B2a — STOP sign

def B2a(env):
    return (
        env["Octagonal_Shape"]
        and env["Red_Ground"]
        and env["White_Symbol"]
        and env["Symbol_Stop"]
        and not env["Bar"]
    )


# B3 — Yield

"""
def B3(env):
    return (
        env["Triangular_Shape"]
        and env["White_Ground"]
        and env["Red_Border"]
        and not env["Symbol"]
    )
"""
def B3(env):
    return (
        env["Diamond_Shape"]
        and env["Yellow_Ground"]
        and env["White_Border"]
        and not env["Symbol"]
        and not env["Bar"]
    )


# ============================================================
# CATEGORY C — Prohibitory / Restrictive Signs
# ============================================================

def C(env):
    return env["Circular_Shape"]


def StartProhibition(env):

    ground_condition = (
        (env["White_Ground"] or env["Yellow_Ground"])
        or (env["Blue_Ground"] and env["Red_Border"])
    )

    symbol_condition = (
        not env["Symbol"]  
        or env["Black_Symbol"]
        #or env["Red_Symbol"]
    )

    bar_condition = (
        not env["Bar"]
        #or env["Red_Bar"]
        or env["White_Bar"]
    )

    return ground_condition and symbol_condition and bar_condition


# No entry

def C1a(env):
    return (
        env["Circular_Shape"]
        and env["Red_Ground"]
        and not env["Symbol"]
        and env["White_Bar"]
    )


# Generic prohibition

def C2(env):
    return (
        env["Circular_Shape"]
        and (env["White_Ground"] or env["Yellow_Ground"])
        and env["Red_Border"]
        and not env["Symbol"]
        and not env["Bar"]
    )


# Overtaking prohibited

def C3e3(env):
    return (
        env["Circular_Shape"]
        and StartProhibition(env)
        and env["Symbol_NoEntryGoods"]
        and not env["Bar"]
    )


# Overtaking

def C13aa(env):
    return (
        env["Circular_Shape"]
        and StartProhibition(env)
        and env["Symbol_Overtaking"]
        and not env["Bar"]
    )


# Overtaking goods vehicles

def C13bb(env):
    return (
        env["Circular_Shape"]
        and StartProhibition(env)
        and env["Symbol_OvertakingGoods"]
        and not env["Bar"]
    )


# ============================================================
# Speed limits
# ============================================================

def C14_20(env): return env["Circular_Shape"] and StartProhibition(env) and env["Symbol_Speed20"]
def C14_30(env): return env["Circular_Shape"] and StartProhibition(env) and env["Symbol_Speed30"]
def C14_50(env): return env["Circular_Shape"] and StartProhibition(env) and env["Symbol_Speed50"]
def C14_60(env): return env["Circular_Shape"] and StartProhibition(env) and env["Symbol_Speed60"]
def C14_70(env): return env["Circular_Shape"] and StartProhibition(env) and env["Symbol_Speed70"]
def C14_80(env): return env["Circular_Shape"] and StartProhibition(env) and env["Symbol_Speed80"]
def C14_100(env): return env["Circular_Shape"] and StartProhibition(env) and env["Symbol_Speed100"]
def C14_120(env): return env["Circular_Shape"] and StartProhibition(env) and env["Symbol_Speed120"]


# ============================================================
# End of prohibition signs
# ============================================================

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
        and env["Black_Symbol"]
    )


def C17d(env):
    return (
        env["Circular_Shape"]
        and (env["White_Ground"] or env["Yellow_Ground"])
        and not env["Border"]
        and env["Black_Bar"]
        and env["Symbol_OvertakingGoods"]
        and env["Black_Symbol"]
    )


# ============================================================
# CATEGORY D — Mandatory Signs
# ============================================================

def D(env):
    return (
        env["Circular_Shape"]
        and ((env["Blue_Ground"] and env["White_Symbol"])
        or (env["White_Ground"] and env["Black_Symbol"]))
        and not env["Border"]
    )


def D1a1(env): return D(env) and env["Symbol_D1a1"]
def D1a4(env): return D(env) and env["Symbol_D1a4"]
def D1a5(env): return D(env) and env["Symbol_D1a5"]
def D1a6(env): return D(env) and env["Symbol_D1a6"]
def D1a7(env): return D(env) and env["Symbol_D1a7"]

def D2a1(env): return D(env) and env["Symbol_D2a1"]
def D2a2(env): return D(env) and env["Symbol_D2a2"]

def D3(env): return D(env) and env["Symbol_D3"]

classid_rules = {

    "C14_20": C14_20,
    "C14_30": C14_30,
    "C14_50": C14_50,
    "C14_60": C14_60,
    "C14_70": C14_70,
    "C14_80": C14_80,
    "C17b_80": C17b_80,
    "C14_100": C14_100,
    "C14_120": C14_120,

    "C13aa": C13aa,
    "C13bb": C13bb,

    "A19a": A19a,

    "B3": B3,
    "B1": B1,
    "B2a": B2a,

    "C2": C2,
    "C3e3": C3e3,
    "C1a": C1a,
    
    "A32": A32,
    "A1a": A1a,
    "A1b": A1b,
    "A1c": A1c,
    "A7a": A7a,
    "A9": A9,
    "A4b2": A4b2,
    "A16": A16,
    "A17a": A17a,
    "A33": A33,
    "A13": A13,
    "A14": A14,
    "A34": A34,
    "A15b": A15b,

    "C17a": C17a,

    "D1a5": D1a5,
    "D1a4": D1a4,
    "D1a1": D1a1,
    "D1a7": D1a7,
    "D1a6": D1a6,
    "D2a2": D2a2,
    "D2a1": D2a1,
    "D3": D3,

    "C17c": C17c,
    "C17d": C17d,
}