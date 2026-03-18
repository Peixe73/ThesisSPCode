from pathlib import Path
import pandas as pd
import logging

from core.init import DO_SCRIPT_IMPORTS

logger = logging.getLogger(__name__)


# =========================
# PATH
# =========================

VALID_PATH = Path("data/gtsrb_ontology_Valid_Final.csv")


# =========================
# CATEGORY DEFINITIONS
# =========================

SIGN_CATEGORIES = [
    "SignClass_A",
    "SignClass_B",
    "SignClass_C",
    "SignClass_D",
]


CLASS_TO_CATEGORY = {

    # C — prohibition / restriction
    "ClassId_1": "SignClass_C",
    "ClassId_2": "SignClass_C",
    "ClassId_3": "SignClass_C",
    "ClassId_4": "SignClass_C",
    "ClassId_5": "SignClass_C",
    "ClassId_6": "SignClass_C",
    "ClassId_7": "SignClass_C",
    "ClassId_8": "SignClass_C",
    "ClassId_9": "SignClass_C",
    "ClassId_10": "SignClass_C",
    "ClassId_11": "SignClass_C",
    "ClassId_16": "SignClass_C",
    "ClassId_17": "SignClass_C",
    "ClassId_18": "SignClass_C",
    "ClassId_33": "SignClass_C",
    "ClassId_42": "SignClass_C",
    "ClassId_43": "SignClass_C",

    # A — danger
    "ClassId_12": "SignClass_A",

    # B — priority
    "ClassId_13": "SignClass_B",
    "ClassId_14": "SignClass_B",
    "ClassId_15": "SignClass_B",

    # D — mandatory
    "ClassId_34": "SignClass_D",
    "ClassId_35": "SignClass_D",
    "ClassId_36": "SignClass_D",
    "ClassId_37": "SignClass_D",
    "ClassId_38": "SignClass_D",
    "ClassId_39": "SignClass_D",
    "ClassId_40": "SignClass_D",
    "ClassId_41": "SignClass_D",
}


# =========================
# MAIN
# =========================

def main():

    logger.info("Loading valid dataset from %s", VALID_PATH)

    df = pd.read_csv(VALID_PATH)

    # Check if already exists
    if all(c in df.columns for c in SIGN_CATEGORIES):
        logger.info("Category columns already exist. Skipping.")
        return

    logger.info("Adding category columns...")

    # Initialize columns
    for cat in SIGN_CATEGORIES:
        df[cat] = 0

    # Fill them
    for cls, cat in CLASS_TO_CATEGORY.items():
        if cls in df.columns:
            df.loc[df[cls] == 1, cat] = 1

    # Sanity check
    category_sums = df[SIGN_CATEGORIES].sum(axis=1)

    invalid_rows = (category_sums != 1).sum()

    if invalid_rows > 0:
        logger.warning(
            "Found %d rows without exactly one category!", invalid_rows
        )
    else:
        logger.info("All rows correctly mapped to exactly one category.")

    # Save
    df.to_csv(VALID_PATH, index=False)

    logger.info("Saved updated dataset to %s", VALID_PATH)


if __name__ == "__main__":
    main()