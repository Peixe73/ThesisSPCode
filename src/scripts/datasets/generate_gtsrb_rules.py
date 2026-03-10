import json
from pathlib import Path

INPUT_JSON = Path("data/gtsrb_class_patterns.json")
OUTPUT_PY = Path("data/generated_gtsrb_rules.py")


def build_rule_expr(must_true, must_false):

    terms = []

    for feat in must_true:
        terms.append(feat)

    for feat in must_false:
        terms.append(f"~{feat}")

    if not terms:
        return "True"

    expr = " &\n    ".join(terms)

    return f"(\n    {expr}\n)"


def main():

    print("Loading patterns...")

    with open(INPUT_JSON) as f:
        patterns = json.load(f)

    print(f"Generating {len(patterns)} rules")

    lines = []
    lines.append("# Auto-generated file\n")

    for class_id, rule_data in patterns.items():

        must_true = rule_data["true"]
        must_false = rule_data["false"]

        rule_expr = build_rule_expr(must_true, must_false)

        lines.append(f'{class_id}_rule = """{rule_expr}"""\n')

    print(f"Writing rule file: {OUTPUT_PY}")

    OUTPUT_PY.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PY, "w") as f:
        f.write("\n".join(lines))

    print("Done.")


if __name__ == "__main__":
    main()