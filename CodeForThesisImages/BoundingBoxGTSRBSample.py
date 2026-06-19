from PIL import Image, ImageDraw, ImageOps

# ==========================
# CONFIG
# ==========================
IMAGE_PATH = "data/gtsrb_dataset/train/00000_00000_00029.ppm"

# (xmin, ymin, xmax, ymax)
BBOX = (12, 13, 132, 136)

OUTPUT_FILE = "CodeForThesisImages/Images/gtsrb_bbox_example.png"

SCALE = 5  # <-- IMPORTANT: zoom factor

BOX_COLOR = "lime"
BOX_THICKNESS = 3

# ==========================
# Load image
# ==========================
img = Image.open(IMAGE_PATH).convert("RGB")

# ==========================
# UPSCALE IMAGE
# ==========================
new_size = (img.width * SCALE, img.height * SCALE)
img = img.resize(new_size, Image.Resampling.NEAREST)

# ==========================
# SCALE BBOX
# ==========================
xmin, ymin, xmax, ymax = BBOX

xmin *= SCALE
ymin *= SCALE
xmax *= SCALE
ymax *= SCALE

# ==========================
# DRAW BBOX
# ==========================
draw = ImageDraw.Draw(img)

for i in range(BOX_THICKNESS):
    draw.rectangle(
        [xmin - i, ymin - i, xmax + i, ymax + i],
        outline=BOX_COLOR
    )

# ==========================
# OPTIONAL: add border for clarity
# ==========================
img = ImageOps.expand(img, border=10, fill="white")

# ==========================
# SAVE
# ==========================
img.save(OUTPUT_FILE)

print(f"Saved {OUTPUT_FILE}")