from pathlib import Path

TARGET_DIR = Path(r"\\wsl.localhost\Ubuntu\home\sp73\ThesisSPCode\sharpSAT\src")

deleted = 0

for path in TARGET_DIR.rglob("*Zone.Identifier*"):
    if path.is_file():
        path.unlink()
        deleted += 1

print(f"Removed {deleted} Zone.Identifier files.")