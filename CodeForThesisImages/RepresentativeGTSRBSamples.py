from PIL import Image
import math

IMAGE_FILES = [
    "data/gtsrb_dataset/train/00000_00000_00000.ppm",
    "data/gtsrb_dataset/train/00001_00001_00000.ppm",
    "data/gtsrb_dataset/train/00002_00002_00000.ppm",
    "data/gtsrb_dataset/train/00003_00003_00000.ppm",
    "data/gtsrb_dataset/train/00004_00004_00000.ppm",
    "data/gtsrb_dataset/train/00005_00005_00000.ppm",
    "data/gtsrb_dataset/train/00006_00006_00000.ppm",
    "data/gtsrb_dataset/train/00007_00007_00000.ppm",
    "data/gtsrb_dataset/train/00008_00008_00000.ppm",
    "data/gtsrb_dataset/train/00009_00009_00000.ppm",
    "data/gtsrb_dataset/train/00010_00010_00000.ppm",
    "data/gtsrb_dataset/train/00011_00011_00000.ppm",
    "data/gtsrb_dataset/train/00012_00012_00000.ppm",
    "data/gtsrb_dataset/train/00013_00013_00000.ppm",
    "data/gtsrb_dataset/train/00014_00014_00000.ppm",
    "data/gtsrb_dataset/train/00015_00015_00000.ppm",
]

THUMB_SIZE = (64, 64)
GRID_COLS = 4
OUTPUT_FILE = "CodeForThesisImages/Images/gtsrb_samples.png"

rows = math.ceil(len(IMAGE_FILES) / GRID_COLS)

grid = Image.new(
    "RGB",
    (GRID_COLS * THUMB_SIZE[0], rows * THUMB_SIZE[1]),
    (255, 255, 255)
)

for i, path in enumerate(IMAGE_FILES):
    img = Image.open(path).convert("RGB")
    img = img.resize(THUMB_SIZE)

    x = (i % GRID_COLS) * THUMB_SIZE[0]
    y = (i // GRID_COLS) * THUMB_SIZE[1]

    grid.paste(img, (x, y))

grid.save(OUTPUT_FILE)
print(f"Saved to {OUTPUT_FILE}")