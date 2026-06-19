from PIL import Image, ImageOps, ImageDraw, ImageFont

# ==========================
# CONFIG
# ==========================
TITLE = "Appearance Variations Within a Single GTSRB Track"

TRACK_IMAGES = [
    "data/gtsrb_dataset/train/00000_00000_00000.ppm",
    "data/gtsrb_dataset/train/00000_00000_00001.ppm",
    "data/gtsrb_dataset/train/00000_00000_00002.ppm",
    "data/gtsrb_dataset/train/00000_00000_00009.ppm",
    "data/gtsrb_dataset/train/00000_00000_00019.ppm",
    "data/gtsrb_dataset/train/00000_00000_00029.ppm",
]

SELECTED_INDICES = [0, 1, 2, 9, 19, 29]

OUTPUT_FILE = "CodeForThesisImages/Images/gtsrb_track_variations.png"

THUMB_SIZE = (140, 140)
BORDER = 3
SPACING = 12

TITLE_HEIGHT = 50
LABEL_HEIGHT = 30

images = []
for path in TRACK_IMAGES:
    img = Image.open(path).convert("RGB")

    # FORCE SAME SIZE (important for alignment)
    img = ImageOps.fit(img, THUMB_SIZE, method=Image.Resampling.LANCZOS)

    # Add border AFTER resizing
    img = ImageOps.expand(img, border=BORDER, fill="black")

    images.append(img)

n = len(images)

img_w = THUMB_SIZE[0] + 2 * BORDER
img_h = THUMB_SIZE[1] + 2 * BORDER

canvas_width = n * img_w + (n - 1) * SPACING
canvas_height = TITLE_HEIGHT + img_h + LABEL_HEIGHT

canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(canvas)

draw.text(
    (canvas_width // 2, 20),
    TITLE,
    fill="black",
    anchor="ma",
    font_size=16
)

for idx, img in enumerate(images):

    x = idx * (img_w + SPACING)
    y = TITLE_HEIGHT

    canvas.paste(img, (x, y))

    label = f"Frame {SELECTED_INDICES[idx] + 1}"

    bbox = draw.textbbox((0, 0), label)
    label_w = bbox[2] - bbox[0]

    draw.text(
        (
            x + img_w // 2 - label_w // 2,
            TITLE_HEIGHT + img_h + 5
        ),
        label,
        fill="black"
    )

canvas.save(OUTPUT_FILE)
print(f"Saved {OUTPUT_FILE}")