import os
from pathlib import Path

import pytesseract
from PIL import Image, ImageDraw

# ----------------------------
# Config
# ----------------------------
TESSERACT_EXE = os.getenv("TESSERACT_EXE")
if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# Tuned relative boxes (x1, y1, x2, y2) to FULL IMAGE
BLUE_REL = (0.405, 0.055, 0.47, 0.12)
RED_REL  = (0.532, 0.055, 0.595, 0.12)
TOP_REL  = (0.0, 0.0, 1.0, 0.22)

OCR_CFG = "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "out" / "test-image-crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_rel(img: Image.Image, rel_box):
    """Crop using relative coords (x1,y1,x2,y2) in [0..1]."""
    w, h = img.size
    x1, y1, x2, y2 = rel_box
    l = int(x1 * w)
    t = int(y1 * h)
    r = int(x2 * w)
    b = int(y2 * h)

    # guardrails
    l = max(0, min(l, w - 1))
    r = max(1, min(r, w))
    t = max(0, min(t, h - 1))
    b = max(1, min(b, h))

    if r <= l or b <= t:
        raise ValueError(f"Bad crop box px: ({l},{t},{r},{b})")

    return img.crop((l, t, r, b)), (l, t, r, b)


def main():
    # Default image path in repo
    img_path = BASE / "Test-Match-Images" / "match3.png"
    if not img_path.exists():
        raise FileNotFoundError(
            f"Could not find image at: {img_path}\n"
            f"Put an image there or edit the path in read_one_image.py"
        )

    img = Image.open(img_path)

    top_crop, top_px = crop_rel(img, TOP_REL)
    blue_crop, blue_px = crop_rel(img, BLUE_REL)
    red_crop, red_px = crop_rel(img, RED_REL)

    # Debug overlay
    debug = img.copy()
    draw = ImageDraw.Draw(debug)
    draw.rectangle(top_px, outline="yellow", width=4)
    draw.rectangle(blue_px, outline="cyan", width=4)
    draw.rectangle(red_px, outline="red", width=4)

    debug_path = OUT_DIR / "debug_overlay.png"
    blue_path = OUT_DIR / "blue_crop.png"
    red_path = OUT_DIR / "red_crop.png"

    debug.save(debug_path)
    blue_crop.save(blue_path)
    red_crop.save(red_path)

    blue_text = pytesseract.image_to_string(blue_crop, config=OCR_CFG).strip()
    red_text = pytesseract.image_to_string(red_crop, config=OCR_CFG).strip()

    print("blue:", repr(blue_text))
    print("red: ", repr(red_text))
    print(f"Saved: {debug_path}")
    print(f"Saved: {blue_path}")
    print(f"Saved: {red_path}")


if __name__ == "__main__":
    main()
