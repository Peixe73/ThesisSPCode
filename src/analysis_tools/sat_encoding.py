import subprocess
import re
import importlib.util
from itertools import combinations
from pysat.formula import CNF

# ==========================================================
# LOAD RULES
# ==========================================================

spec = importlib.util.spec_from_file_location(
    "rules",
    "src/analysis_tools/gtsrb_rules_sat.py"
)
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)

# ==========================================================
# VARIABLE MANAGEMENT
# ==========================================================

class VarManager:
    def __init__(self):
        self.map = {}
        self.counter = 1

    def get(self, name):
        if name not in self.map:
            self.map[name] = self.counter
            self.counter += 1
        return self.map[name]

vm = VarManager()

# ==========================================================
# FEATURES (adapted from your ontology)
# ==========================================================

feature_names = [
    "Circular_Shape","Diamond_Shape","Triangular_Shape","Octagonal_Shape",
    "Red_Ground","White_Ground","Yellow_Ground","Blue",
    "Border","Black_Border","Red_Border","White_Border",
    "Bar","Black_Bar","White_Bar",
    "Symbol","Black_Symbol","White_Symbol"
]

symbol_specific = [
    "Symbol_Stop","Symbol_NoEntryGoods","Symbol_Overtaking","Symbol_OvertakingGoods",
    "Symbol_Speed20","Symbol_Speed30","Symbol_Speed50","Symbol_Speed60",
    "Symbol_Speed70","Symbol_Speed80","Symbol_Speed100","Symbol_Speed120",
    "D1a1","D1a4","D1a5","D1a6","D1a7","D2a1","D2a2","D3"
]

all_vars = feature_names + symbol_specific

for v in all_vars:
    vm.get(v)

# ==========================================================
# CNF BUILDING
# ==========================================================

def build_cnf():
    cnf = CNF()

    def exactly_one(vars_list):
        cnf.append(vars_list)
        for a, b in combinations(vars_list, 2):
            cnf.append([-a, -b])

    def implies(a, b):
        cnf.append([-a, b])

    # ---- Shape constraint ----
    shape_vars = [vm.get(v) for v in [
        "Circular_Shape","Diamond_Shape","Triangular_Shape","Octagonal_Shape"
    ]]
    exactly_one(shape_vars)

    # ---- Ground constraint ----
    ground_vars = [vm.get(v) for v in [
        "Red_Ground","White_Ground","Yellow_Ground","Blue"
    ]]
    exactly_one(ground_vars)

    # ---- Symbol logic ----
    symbol = vm.get("Symbol")
    black = vm.get("Black_Symbol")
    white = vm.get("White_Symbol")

    implies(black, symbol)
    implies(white, symbol)

    cnf.append([symbol] + [-black, -white])
    cnf.append([-black, -white])  # mutual exclusion

    return cnf

# ==========================================================
# EXPORT CNF
# ==========================================================

def export_cnf(cnf, filename):
    with open(filename, "w") as f:
        f.write(f"p cnf {vm.counter - 1} {len(cnf.clauses)}\n")
        for clause in cnf.clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")

# ==========================================================
# RUN sharpSAT
# ==========================================================

def run_sharpsat(cnf_file):
    cmd = ["./sharpsat", cnf_file]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("\n=== sharpSAT OUTPUT ===")
    print(result.stdout)

    match = re.search(r'Number of satisfying assignments:\s*(\d+)', result.stdout)

    if match:
        return int(match.group(1))
    return None

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    cnf = build_cnf()

    print("Variables:", vm.counter - 1)
    print("Clauses:", len(cnf.clauses))

    export_cnf(cnf, "formula.cnf")

    count = run_sharpsat("formula.cnf")

    print("\n====================")
    print("EXACT #SAT COUNT:", count)
    print("====================")