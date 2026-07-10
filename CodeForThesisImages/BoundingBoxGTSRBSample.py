from math import ceil
from PIL import Image, ImageDraw, ImageOps

# ==========================
# CONFIG
# ==========================

EXAMPLES = [
    {
        "image": "data/gtsrb_dataset/train/00000_00000_00029.ppm",
        "bbox": (12, 13, 132, 136)
    },
    {
        "image": "data/gtsrb_dataset/train/00001_00000_00029.ppm",
        "bbox": (10, 11, 104, 108)
    },
    {
        "image": "data/gtsrb_dataset/train/00002_00000_00029.ppm",
        "bbox": (10, 10, 98, 96)
    },
    {
        "image": "data/gtsrb_dataset/train/00011_00029_00029.ppm",
        "bbox": (13, 12, 131, 118)
    },
    {
        "image": "data/gtsrb_dataset/train/00012_00017_00028.ppm",
        "bbox": (10, 9, 99, 98)
    },
    {
        "image": "data/gtsrb_dataset/train/00013_00068_00027.ppm",
        "bbox": (13, 10, 136, 104)
    },
    {
        "image": "data/gtsrb_dataset/train/00014_00002_00026.ppm",
        "bbox": (5, 5, 59, 59)
    },
    {
        "image": "data/gtsrb_dataset/train/00035_00032_00028.ppm",
        "bbox": (6, 7, 58, 63)
    },
    {
        "image": "data/gtsrb_dataset/train/00042_00000_00029.ppm",
        "bbox": (6, 7, 63, 66)
    },
]

OUTPUT_FILE = "CodeForThesisImages/Images/gtsrb_bbox_grid.png"

BOX_COLOR = "lime"
BOX_THICKNESS = 2

BACKGROUND = "white"

BORDER = 2
PADDING = 2
OUTER_PADDING = 0

GRID_COLUMNS = 3

# ==========================
# Draw bounding box
# ==========================

def create_image(image_path, bbox):

    img = Image.open(image_path).convert("RGB")

    draw = ImageDraw.Draw(img)

    xmin, ymin, xmax, ymax = bbox

    for i in range(BOX_THICKNESS):
        draw.rectangle(
            (xmin-i, ymin-i, xmax+i, ymax+i),
            outline=BOX_COLOR
        )

    return ImageOps.expand(img, border=BORDER, fill=BACKGROUND)


images = [create_image(e["image"], e["bbox"]) for e in EXAMPLES]

img_w = images[0].width
img_h = images[0].height

rows = ceil(len(images) / GRID_COLUMNS)

canvas_width = (
    OUTER_PADDING * 2
    + GRID_COLUMNS * img_w
    + (GRID_COLUMNS - 1) * PADDING
)

canvas_height = (
    OUTER_PADDING * 2
    + rows * img_h
    + (rows - 1) * PADDING
)

canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)

for idx, img in enumerate(images):

    row = idx // GRID_COLUMNS
    col = idx % GRID_COLUMNS

    x = OUTER_PADDING + col * (img_w + PADDING)
    y = OUTER_PADDING + row * (img_h + PADDING)

    canvas.paste(img, (x, y))

canvas.save(OUTPUT_FILE)

print(f"Saved {OUTPUT_FILE}")