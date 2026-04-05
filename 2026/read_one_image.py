import os
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw

TESSERACT_EXE = os.getenv("TESSERACT_EXE")
if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

BLUE_REL = (0.405, 0.055, 0.47, 0.12)
RED_REL  = (0.532, 0.055, 0.595, 0.12)
TOP_REL  = (0.0, 0.0, 1.0, 0.22)
TIMER_REL = (0.469, 0.055, 0.5315, 0.118)

OCR_CFG_SCORE = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1"
OCR_CFG_TIMER = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:"

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "out" / "test-image-crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def crop_rel(img: Image.Image, rel_box):
    w, h = img.size
    x1, y1, x2, y2 = rel_box
    l = int(x1 * w)
    t = int(y1 * h)
    r = int(x2 * w)
    b = int(y2 * h)

    l = max(0, min(l, w - 1))
    r = max(1, min(r, w))
    t = max(0, min(t, h - 1))
    b = max(1, min(b, h))

    if r <= l or b <= t:
        raise ValueError(f"Bad crop box px: ({l},{t},{r},{b})")

    return img.crop((l, t, r, b)), (l, t, r, b)

def preprocess_score(pil_crop: Image.Image) -> Image.Image:
    gray = pil_crop.convert("L")
    gray = gray.resize((gray.width * 3, gray.height * 3), Image.Resampling.BICUBIC)

    arr = np.array(gray)
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if th.mean() < 127:
        th = cv2.bitwise_not(th)

    return Image.fromarray(th)

def parse_timer(text: str) -> int | None:
    text = text.strip().replace(" ", "")
    text = text.replace("O", "0").replace("|", "1")

    m = re.search(r"(\d{1,2}):(\d{2})", text)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    if ss >= 60:
        return None
    return mm * 60 + ss

def clean_score(text: str) -> int | None:
    s = re.sub(r"[^0-9]", "", text.strip())
    if not s:
        return None

    # Prefer 2-digit or 1-digit chunks (fixes 155 -> 15)
    # Try last 2 digits first (most common when noise adds a leading digit)
    if len(s) >= 2:
        cand2 = int(s[-2:])
        if 0 <= cand2 <= 99:
            return cand2

    cand1 = int(s[-1])
    return cand1

def main():
    img_path = BASE / "Test-Match-Images" / "frc-2026-match-pic.png"
    if not img_path.exists():
        raise FileNotFoundError(
            f"Could not find image at: {img_path}\n"
            f"Put an image there or edit the path in read_one_image.py"
        )

    img = Image.open(img_path)

    top_crop, top_px = crop_rel(img, TOP_REL)
    blue_crop, blue_px = crop_rel(img, BLUE_REL)
    red_crop, red_px = crop_rel(img, RED_REL)
    timer_crop, timer_px = crop_rel(img, TIMER_REL)

    blue_proc = preprocess_score(blue_crop)
    red_proc  = preprocess_score(red_crop)

    debug = img.copy()
    draw = ImageDraw.Draw(debug)
    draw.rectangle(top_px, outline="yellow", width=4)
    draw.rectangle(blue_px, outline="cyan", width=4)
    draw.rectangle(red_px, outline="red", width=4)
    draw.rectangle(timer_px, outline="lime", width=4)

    debug_path = OUT_DIR / "debug_overlay.png"
    blue_path = OUT_DIR / "blue_crop.png"
    red_path = OUT_DIR / "red_crop.png"
    timer_path = OUT_DIR / "timer_crop.png"
    blue_proc_path = OUT_DIR / "blue_crop_proc.png"
    red_proc_path = OUT_DIR / "red_crop_proc.png"

    debug.save(debug_path)
    blue_crop.save(blue_path)
    red_crop.save(red_path)
    timer_crop.save(timer_path)
    blue_proc.save(blue_proc_path)
    red_proc.save(red_proc_path)

    blue_raw = pytesseract.image_to_string(blue_proc, config=OCR_CFG_SCORE)
    red_raw = pytesseract.image_to_string(red_proc, config=OCR_CFG_SCORE)

    blue_val = clean_score(blue_raw)
    red_val = clean_score(red_raw)

    print("blue_raw:", repr(blue_raw), "->", blue_val)
    print("red_raw: ", repr(red_raw), "->", red_val)

    timer_text_raw = pytesseract.image_to_string(timer_crop, config=OCR_CFG_TIMER)
    timer_sec = parse_timer(timer_text_raw)

    print("blue:", repr(blue_val))
    print("red: ", repr(red_val))
    print("timer_raw:", repr(timer_text_raw))
    print("timer_sec:", timer_sec)

    print(f"Saved: {debug_path}")
    print(f"Saved: {blue_path}")
    print(f"Saved: {red_path}")
    print(f"Saved: {timer_path}")
    print(f"Saved: {blue_proc_path}")
    print(f"Saved: {red_proc_path}")

if __name__ == "__main__":
    main()