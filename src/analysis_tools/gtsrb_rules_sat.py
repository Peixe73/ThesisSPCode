"""
Translation of the Traffic Signs Ontology (Vienna Convention 1968)
adapted to match the GTSRB concept dataset.

Symbolic version (SAT-compatible)
"""

# ============================================================
# CATEGORY A — Danger Warning Signs
# ============================================================

def A(env):
    return (
        env["Triangular_Shape"]
        & (env["White_Ground"] | env["Yellow_Ground"])
        & env["Red_Border"]
        & env["Symbol"]
    )


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

def B1(env):
    return (
        env["Triangular_Shape"]
        & (env["Yellow_Ground"] | env["White_Ground"])
        & env["Red_Border"]
        & ~env["Symbol"]
    )


def B2a(env):
    return (
        env["Octagonal_Shape"]
        & env["Red_Ground"]
        & env["White_Symbol"]
        & env["Symbol_Stop"]
    )


def B3(env):
    return (
        env["Diamond_Shape"]
        & env["Yellow_Ground"]
        & env["White_Border"]
    )


# ============================================================
# CATEGORY C — Prohibitory / Restrictive Signs
# ============================================================

def C(env):
    return env["Circular_Shape"]


def StartProhibition(env):

    ground_condition = (
        (env["White_Ground"] | env["Yellow_Ground"])
        | (env["Blue"] & env["Red_Border"])
    )

    symbol_condition = (
        ~env["Symbol"]
        | env["Black_Symbol"]
    )

    bar_condition = (
        ~env["Bar"]
        | env["White_Bar"]
    )

    return ground_condition & symbol_condition & bar_condition


def C1a(env):
    return (
        env["Circular_Shape"]
        & env["Red_Ground"]
        & ~env["Symbol"]
        & env["White_Bar"]
    )


def C2(env):
    return (
        env["Circular_Shape"]
        & (env["White_Ground"] | env["Yellow_Ground"])
        & env["Red_Border"]
        & ~env["Symbol"]
    )


def C3e3(env):
    return (
        env["Circular_Shape"]
        & StartProhibition(env)
        & env["Symbol_NoEntryGoods"]
        & ~env["Bar"]
    )


def C13aa(env):
    return (
        env["Circular_Shape"]
        & StartProhibition(env)
        & env["Symbol_Overtaking"]
        & ~env["Bar"]
    )


def C13bb(env):
    return (
        env["Circular_Shape"]
        & StartProhibition(env)
        & env["Symbol_OvertakingGoods"]
        & ~env["Bar"]
    )


# ============================================================
# Speed limits
# ============================================================

def C14_20(env): return env["Circular_Shape"] & StartProhibition(env) & env["Symbol_Speed20"]
def C14_30(env): return env["Circular_Shape"] & StartProhibition(env) & env["Symbol_Speed30"]
def C14_50(env): return env["Circular_Shape"] & StartProhibition(env) & env["Symbol_Speed50"]
def C14_60(env): return env["Circular_Shape"] & StartProhibition(env) & env["Symbol_Speed60"]
def C14_70(env): return env["Circular_Shape"] & StartProhibition(env) & env["Symbol_Speed70"]
def C14_80(env): return env["Circular_Shape"] & StartProhibition(env) & env["Symbol_Speed80"]
def C14_100(env): return env["Circular_Shape"] & StartProhibition(env) & env["Symbol_Speed100"]
def C14_120(env): return env["Circular_Shape"] & StartProhibition(env) & env["Symbol_Speed120"]


# ============================================================
# End of prohibition signs
# ============================================================

def C17a(env):
    return (
        env["Circular_Shape"]
        & (env["White_Ground"] | env["Yellow_Ground"])
        & ~env["Border"]
        & env["Black_Bar"]
        & ~env["Symbol"]
    )


def C17b_80(env):
    return (
        env["Circular_Shape"]
        & (env["White_Ground"] | env["Yellow_Ground"])
        & ~env["Border"]
        & env["Black_Bar"]
        & env["Symbol_Speed80"]
    )


def C17c(env):
    return (
        env["Circular_Shape"]
        & (env["White_Ground"] | env["Yellow_Ground"])
        & ~env["Border"]
        & env["Black_Bar"]
        & env["Symbol_Overtaking"]
    )


def C17d(env):
    return (
        env["Circular_Shape"]
        & (env["White_Ground"] | env["Yellow_Ground"])
        & ~env["Border"]
        & env["Black_Bar"]
        & env["Symbol_OvertakingGoods"]
    )


# ============================================================
# CATEGORY D — Mandatory Signs
# ============================================================

def D(env):
    return (
        env["Circular_Shape"]
        & (
            (env["Blue"] & env["White_Symbol"])
            | (env["White_Ground"] & env["Black_Symbol"])
        )
    )


def D1a1(env): return D(env) & env["D1a1"]
def D1a4(env): return D(env) & env["D1a4"]
def D1a5(env): return D(env) & env["D1a5"]
def D1a6(env): return D(env) & env["D1a6"]
def D1a7(env): return D(env) & env["D1a7"]

def D2a1(env): return D(env) & env["D2a1"]
def D2a2(env): return D(env) & env["D2a2"]

def D3(env): return D(env) & env["D3"]