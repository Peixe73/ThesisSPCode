import shutil
from pathlib import Path

SOURCE_DIR = Path("C:/Users/simao/Downloads/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images")
DEST_DIR = Path(r"\\wsl.localhost\Ubuntu\home\sp73\thesis-code\data\gtsrb_dataset\train")

DEST_DIR.mkdir(parents=True, exist_ok=True)

total = 0

for class_dir in sorted(SOURCE_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    class_id = class_dir.name  # like "00000"

    for img_path in class_dir.glob("*.ppm"):
        # original name: XXXXX_YYYYY.ppm
        original_name = img_path.stem  # without .ppm

        new_name = f"{class_id}_{original_name}.ppm"
        new_path = DEST_DIR / new_name

        shutil.copy2(img_path, new_path)
        total += 1

print(f"Done. Copied {total} images to {DEST_DIR}")