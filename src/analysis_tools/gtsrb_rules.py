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
    )


# Dataset cannot distinguish the specific symbols
# so subclasses collapse to the same rule

def A1a(env): return A(env)
def A1b(env): return A(env)
def A1c(env): return A(env)
def A4b2(env): return A(env)
def A7a(env): return A(env)
def A9(env): return A(env)
def A13(env): return A(env)
def A14(env): return A(env)
def A15b(env): return A(env)
def A16(env): return A(env)
def A17a(env): return A(env)
def A19a(env): return A(env)
def A32(env): return A(env)
def A33(env): return A(env)
def A34(env): return A(env)


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
        #and not env["Bar"]
    )


# B2a — STOP sign

def B2a(env):
    return (
        env["Octagonal_Shape"]
        and env["Red_Ground"]
        and env["White_Symbol"]
        and env["Symbol_Stop"]
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
        #and not env["Symbol"]
    )


# ============================================================
# CATEGORY C — Prohibitory / Restrictive Signs
# ============================================================

def C(env):
    return env["Circular_Shape"]


def StartProhibition(env):

    ground_condition = (
        (env["White_Ground"] or env["Yellow_Ground"])
        or (env["Blue"] and env["Red_Border"])
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
        and ((env["Blue"] and env["White_Symbol"])
        or (env["White_Ground"] and env["Black_Symbol"]))
    )


def D1a1(env): return D(env) and env["D1a1"]
def D1a4(env): return D(env) and env["D1a4"]
def D1a5(env): return D(env) and env["D1a5"]
def D1a6(env): return D(env) and env["D1a6"]
def D1a7(env): return D(env) and env["D1a7"]

def D2a1(env): return D(env) and env["D2a1"]
def D2a2(env): return D(env) and env["D2a2"]

def D3(env): return D(env) and env["D3"]

classid_rules = {
    "ClassId_1": C14_20,
    "ClassId_2": C14_30,
    "ClassId_3": C14_50,
    "ClassId_4": C14_60,
    "ClassId_5": C14_70,
    "ClassId_6": C14_80,
    "ClassId_7": C17b_80,
    "ClassId_8": C14_100,
    "ClassId_9": C14_120,

    "ClassId_10": C13aa,
    "ClassId_11": C13bb,
    "ClassId_12": A19a,

    "ClassId_13": B3,
    "ClassId_14": B1,
    "ClassId_15": B2a,

    "ClassId_16": C2,
    "ClassId_17": C3e3,
    "ClassId_18": C1a,

    "ClassId_33": C17a,

    "ClassId_34": D1a5,
    "ClassId_35": D1a4,
    "ClassId_36": D1a1,
    "ClassId_37": D1a7,
    "ClassId_38": D1a6,
    "ClassId_39": D2a2,
    "ClassId_40": D2a1,
    "ClassId_41": D3,

    "ClassId_42": C17c,
    "ClassId_43": C17d,
}