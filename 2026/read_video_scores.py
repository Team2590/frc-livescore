import argparse
import os
import re
from collections import Counter
from pathlib import Path
BASE = Path(__file__).resolve().parent

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / ".git").exists() or (cur / "setup.py").exists() or (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()

TESSERACT_EXE = os.getenv("TESSERACT_EXE")
if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

BLUE_REL = (0.405, 0.055, 0.47, 0.12)
RED_REL = (0.532, 0.055, 0.595, 0.12)
TOP_REL = (0.0, 0.0, 1.0, 0.22)

OCR_CFG = "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1"

DEFAULT_SAMPLES_PER_SEC = 5
DEFAULT_MAX_JUMP_PER_SEC = 30
DEFAULT_ALLOW_RESET = False

SCALE = 3

def crop_rel(pil_img: Image.Image, rel_box):
    w, h = pil_img.size
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

    return pil_img.crop((l, t, r, b)), (l, t, r, b)

def cv2_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def preprocess_for_ocr(pil_crop: Image.Image) -> Image.Image:
    gray = pil_crop.convert("L")
    if SCALE != 1:
        gray = gray.resize(
            (gray.width * SCALE, gray.height * SCALE),
            Image.Resampling.BICUBIC,
        )

    arr = np.array(gray)
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() < 127:
        th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(th)


def ocr_int(pil_img: Image.Image) -> int | None:
    txt = pytesseract.image_to_string(pil_img, config=OCR_CFG).strip()
    txt = re.sub(r"[^0-9]", "", txt)
    if not txt:
        return None
    try:
        return int(txt)
    except ValueError:
        return None

def make_debug_overlay(pil_img: Image.Image, boxes_px: dict, out_path: Path):
    dbg = pil_img.copy()
    draw = ImageDraw.Draw(dbg)
    if "top" in boxes_px:
        draw.rectangle(boxes_px["top"], outline="yellow", width=4)
    if "blue" in boxes_px:
        draw.rectangle(boxes_px["blue"], outline="cyan", width=4)
    if "red" in boxes_px:
        draw.rectangle(boxes_px["red"], outline="red", width=4)
    dbg.save(out_path)

def choose_score_value(
    candidates: list[int | None],
    last: int | None,
    *,
    max_jump: int,
    allow_reset: bool,
) -> int | None:
    vals = [v for v in candidates if v is not None]
    if not vals:
        return None

    if last is not None:
        # drop decreases (unless resets allowed)
        if not allow_reset:
            vals = [v for v in vals if v >= last]
        # drop huge spikes
        vals = [v for v in vals if v <= last + max_jump]

    if not vals:
        return None

    # scores only go up → prefer the max surviving sample
    return max(vals)


def clamp_with_last(value: int | None, last: int | None, *, max_jump: int, allow_reset: bool) -> int | None:
    if value is None:
        return last
    if last is None:
        return value
    if not allow_reset and value < last:
        return last
    if value - last > max_jump:
        return last
    return value

def read_scores_from_frame(frame_bgr: np.ndarray, debug_dir: Path | None = None):
    pil_img = cv2_to_pil(frame_bgr)

    _, top_px = crop_rel(pil_img, TOP_REL)
    blue_crop, blue_px = crop_rel(pil_img, BLUE_REL)
    red_crop, red_px = crop_rel(pil_img, RED_REL)

    blue_proc = preprocess_for_ocr(blue_crop)
    red_proc = preprocess_for_ocr(red_crop)

    blue = ocr_int(blue_proc)
    red = ocr_int(red_proc)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        pil_img.save(debug_dir / "frame.png")
        make_debug_overlay(
            pil_img,
            {"top": top_px, "blue": blue_px, "red": red_px},
            debug_dir / "debug_overlay.png",
        )
        blue_crop.save(debug_dir / "blue_crop_raw.png")
        red_crop.save(debug_dir / "red_crop_raw.png")
        blue_proc.save(debug_dir / "blue_crop_proc.png")
        red_proc.save(debug_dir / "red_crop_proc.png")

    return blue, red

def process_video(
    video_path: Path,
    *,
    samples_per_sec: int,
    max_jump_per_sec: int,
    allow_reset: bool,
    debug_seconds: int,
    out_root: Path,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = int(total_frames / fps) if total_frames else None

    debug_root = out_root / "debug_frames"
    debug_root.mkdir(parents=True, exist_ok=True)

    rows = []
    last_blue = 0
    last_red = 0

    s = 0
    while duration_sec is not None and s <= duration_sec + 1:
        s += 1
    max_s = s if duration_sec is not None else None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    sec = 0

    def target_frames_for_sec(sec_i: int) -> list[int]:
        base = int(round(sec_i * fps))
        if samples_per_sec <= 1:
            return [base]
        step = fps / samples_per_sec
        return [int(round(base + k * step)) for k in range(samples_per_sec)]

    targets = target_frames_for_sec(0)
    next_target_i = 0
    blue_samples: list[int | None] = []
    red_samples: list[int | None] = []

    def finalize_second(sec_i: int):
        nonlocal last_blue, last_red, blue_samples, red_samples

        blue_raw = choose_score_value(blue_samples, last_blue, max_jump=max_jump_per_sec, allow_reset=allow_reset)
        red_raw = choose_score_value(red_samples, last_red, max_jump=max_jump_per_sec, allow_reset=allow_reset)

        blue = clamp_with_last(blue_raw, last_blue, max_jump=max_jump_per_sec, allow_reset=allow_reset)
        red = clamp_with_last(red_raw, last_red, max_jump=max_jump_per_sec, allow_reset=allow_reset)

        rows.append({"t_sec": sec_i, "blue": blue, "red": red})

        last_blue, last_red = blue, red
        blue_samples = []
        red_samples = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        while next_target_i < len(targets) and frame_idx >= targets[next_target_i]:
            debug_dir = None
            if sec < debug_seconds and next_target_i == 0:
                debug_dir = debug_root / f"t{sec:04d}"

            b, r = read_scores_from_frame(frame, debug_dir=debug_dir)
            blue_samples.append(b)
            red_samples.append(r)
            next_target_i += 1

        next_sec_frame = int(round((sec + 1) * fps))
        if frame_idx >= next_sec_frame:
            finalize_second(sec)
            sec += 1
            if max_s is not None and sec > max_s:
                break
            targets = target_frames_for_sec(sec)
            next_target_i = 0

        frame_idx += 1

    if blue_samples or red_samples:
        finalize_second(sec)

    cap.release()

    for i in range(1, len(rows)):
        b0, r0 = rows[i - 1]["blue"], rows[i - 1]["red"]
        b1, r1 = rows[i]["blue"], rows[i]["red"]
        rows[i]["blue_delta"] = None if (b0 is None or b1 is None) else max(0, b1 - b0)
        rows[i]["red_delta"] = None if (r0 is None or r1 is None) else max(0, r1 - r0)

    if rows:
        rows[0]["blue_delta"] = 0
        rows[0]["red_delta"] = 0

    return rows

def write_csv(rows: list[dict], out_path: Path):
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_sec", "blue", "red", "blue_delta", "red_delta"])
        w.writeheader()
        w.writerows(rows)

def looks_like_digits(th_img: Image.Image) -> bool:
    # th_img is your binarized crop (0/255). We expect some white pixels for digits.
    arr = np.array(th_img)
    white_frac = (arr > 0).mean()
    # Tune if needed. Typical digit crops are a few % to maybe ~30% white.
    return 0.01 < white_frac < 0.60

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(script_dir)

    parser = argparse.ArgumentParser(description="Extract Blue/Red scores from match video using OCR.")
    parser.add_argument(
        "--video",
        type=str,
        default=r"2026\Test-Match-Videos\test-match-video.mp4",
        help="Path to match video",
    )
    parser.add_argument("--samples-per-sec", type=int, default=DEFAULT_SAMPLES_PER_SEC)
    parser.add_argument("--max-jump-per-sec", type=int, default=DEFAULT_MAX_JUMP_PER_SEC)
    parser.add_argument("--allow-reset", action="store_true", default=DEFAULT_ALLOW_RESET)
    parser.add_argument("--debug-seconds", type=int, default=5)
    parser.add_argument("--out", type=str, default=str(script_dir / "out"))

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (repo_root / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = process_video(
        video_path=video_path,
        samples_per_sec=args.samples_per_sec,
        max_jump_per_sec=args.max_jump_per_sec,
        allow_reset=args.allow_reset,
        debug_seconds=args.debug_seconds,
        out_root=out_root,
    )

    for row in rows[:20]:
        print(row)

    csv_path = out_root / "score_timeseries.csv"
    write_csv(rows, csv_path)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved debug folders under: {out_root / 'debug_frames'}")

if __name__ == "__main__":
    main()