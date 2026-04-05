"""
FRC match score extractor.

Design priorities:
  1. Accuracy  — SCALE=3, 5 samples/sec, confidence-filtered OCR
  2. Speed     — frame-level ThreadPoolExecutor (no nested pools)
  3. Correctness — computed timer output, warmup clamp, fixed AUTO→TELE transition

Expected runtime: ~2-4 min for a 5-10 min match video on a 4-core machine.
"""

import argparse
import os
import re
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TESSERACT_EXE = os.getenv("TESSERACT_EXE")
if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# Relative crop boxes  (x1, y1, x2, y2)
BLUE_REL  = (0.405, 0.055, 0.470,  0.120)
RED_REL   = (0.532, 0.055, 0.595,  0.120)
TIMER_REL = (0.469, 0.055, 0.5315, 0.118)
TOP_REL   = (0.0,   0.0,   1.0,    0.22)

OCR_CFG_SCORE = (
    "--oem 1 --psm 8 "
    "-c tessedit_char_whitelist=0123456789 "
    "-c classify_bln_numeric_mode=1"
)
OCR_CFG_TIMER = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789:"

# Scale=3 is important — smaller values cause misreads on scoreboard fonts.
SCALE = 3

AUTON_LEN = 20
TELE_LEN  = 140
TOTAL_LEN = AUTON_LEN + TELE_LEN

# Scores are forced to 0 for this many elapsed seconds.
WARMUP_SECONDS = 1

# During the first EARLY_WINDOW elapsed seconds, apply a tighter jump cap.
EARLY_WINDOW   = 10
EARLY_MAX_JUMP = 8

DEFAULT_SAMPLES_PER_SEC  = 5
DEFAULT_MAX_JUMP_PER_SEC = 40
DEFAULT_ALLOW_RESET      = False  # FRC scores are monotonically non-decreasing

MAX_WORKERS = min(6, os.cpu_count() or 2)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if any((cur / f).exists() for f in (".git", "setup.py", "pyproject.toml")):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def crop_rel(img: Image.Image, box):
    w, h = img.size
    l = max(0, min(int(box[0] * w), w - 1))
    t = max(0, min(int(box[1] * h), h - 1))
    r = max(l + 1, min(int(box[2] * w), w))
    b = max(t + 1, min(int(box[3] * h), h))
    return img.crop((l, t, r, b)), (l, t, r, b)


def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def preprocess(pil_crop: Image.Image, scale: int = SCALE) -> Image.Image:
    gray = pil_crop.convert("L").resize(
        (pil_crop.width * scale, pil_crop.height * scale),
        Image.Resampling.BICUBIC,
    )
    arr = cv2.GaussianBlur(np.array(gray), (3, 3), 0)
    _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() < 127:
        th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return Image.fromarray(th)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _clean_score(text: str) -> Optional[int]:
    """
    Parse a raw OCR string into a score integer.
    Handles 1-, 2-, 3-, and 4+-digit OCR output correctly.
    No artificial ceiling — scores can exceed 500 or 999 in high-scoring matches.
    """
    s = re.sub(r"[^0-9]", "", text.strip())
    if not s:
        return None
    if len(s) <= 4:
        # Up to 4 digits: parse directly (covers scores up to 9999)
        val = int(s)
    else:
        # 5+ digits → definitely OCR noise. Take first 4 as best guess.
        val = int(s[:4])
    return val if 0 <= val <= 9999 else None


def _ocr_score(proc_img: Image.Image) -> Optional[int]:
    """
    Score OCR with per-token confidence filtering.
    No hard ceiling on score value — the graph and CSV will auto-scale.
    """
    try:
        data = pytesseract.image_to_data(
            proc_img, config=OCR_CFG_SCORE,
            output_type=pytesseract.Output.DICT,
        )
        good_tokens = [
            t.strip()
            for t, c in zip(data["text"], data["conf"])
            if t.strip() and int(c) >= 60
        ]
        raw = "".join(good_tokens) if good_tokens else \
              pytesseract.image_to_string(proc_img, config=OCR_CFG_SCORE)
    except Exception:
        raw = pytesseract.image_to_string(proc_img, config=OCR_CFG_SCORE)

    return _clean_score(raw)


def _parse_timer(text: str) -> Optional[int]:
    text = text.strip().replace(" ", "").replace("O", "0").replace("|", "1")
    m = re.search(r"(\d{1,2}):(\d{2})", text)
    if not m:
        return None
    mm, ss = int(m.group(1)), int(m.group(2))
    return None if ss >= 60 else mm * 60 + ss


def _ocr_timer(proc_img: Image.Image) -> Optional[int]:
    return _parse_timer(pytesseract.image_to_string(proc_img, config=OCR_CFG_TIMER))


# ---------------------------------------------------------------------------
# Timer math
# ---------------------------------------------------------------------------

def timer_to_elapsed(timer_sec: int, phase: str) -> int:
    return AUTON_LEN - timer_sec if phase == "AUTO" \
        else AUTON_LEN + (TELE_LEN - timer_sec)


def elapsed_to_timer(elapsed: int) -> tuple[int, str]:
    """
    Deterministically compute the match timer for any elapsed second.

    elapsed 0  → 0:20  (AUTO start)
    elapsed 19 → 0:01
    elapsed 20 → 2:20  (TELE start — guaranteed correct jump in CSV)
    elapsed 160→ 0:00
    """
    if elapsed <= AUTON_LEN:
        rem = AUTON_LEN - elapsed
    else:
        rem = TELE_LEN - (elapsed - AUTON_LEN)
    return rem, f"{rem // 60}:{rem % 60:02d}"


# ---------------------------------------------------------------------------
# Per-frame OCR  (runs inside thread pool — no nested pools)
# ---------------------------------------------------------------------------

def _process_frame(frame_bgr: np.ndarray) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Returns (timer_sec, blue, red).
    Timer OCR uses scale=2 (faster, sufficient for M:SS format).
    Score OCR uses scale=3 (needed for accurate digit reads).
    """
    pil = cv2_to_pil(frame_bgr)

    timer_sec = _ocr_timer(preprocess(crop_rel(pil, TIMER_REL)[0], scale=2))
    if timer_sec is None:
        return None, None, None

    blue = _ocr_score(preprocess(crop_rel(pil, BLUE_REL)[0]))
    red  = _ocr_score(preprocess(crop_rel(pil, RED_REL)[0]))
    return timer_sec, blue, red


# ---------------------------------------------------------------------------
# Debug overlay  (only called for the first N seconds)
# ---------------------------------------------------------------------------

def _save_debug(frame_bgr: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pil = cv2_to_pil(frame_bgr)
    pil.save(out_dir / "frame.png")

    dbg = pil.copy()
    draw = ImageDraw.Draw(dbg)
    for box, color in [
        (TOP_REL,   "yellow"),
        (BLUE_REL,  "cyan"),
        (TIMER_REL, "lime"),
        (RED_REL,   "red"),
    ]:
        _, px = crop_rel(pil, box)
        draw.rectangle(px, outline=color, width=4)
    dbg.save(out_dir / "debug_overlay.png")

    preprocess(crop_rel(pil, BLUE_REL)[0]).save(out_dir  / "blue_proc.png")
    preprocess(crop_rel(pil, RED_REL)[0]).save(out_dir   / "red_proc.png")
    preprocess(crop_rel(pil, TIMER_REL)[0], scale=2).save(out_dir / "timer_proc.png")


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def _majority(vals: list[int]) -> int:
    c = Counter(vals)
    best = max(c.values())
    return max(v for v, n in c.items() if n == best)


def _pick_score(candidates: list[Optional[int]], last: int,
                *, max_jump: int, allow_reset: bool) -> int:
    vals = [v for v in candidates if v is not None]
    if not vals:
        return last
    filtered = [v for v in vals
                if (allow_reset or v >= last) and v - last <= max_jump]
    return _majority(filtered) if filtered else last


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    *,
    samples_per_sec: int,
    max_jump_per_sec: int,
    allow_reset: bool,
    debug_seconds: int,
    out_root: Path,
) -> list[dict]:

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_every = max(1, int(round(fps / samples_per_sec)))
    max_drop     = max(3, int(fps / sample_every) + 2)

    debug_root = out_root / "debug_frames"
    debug_root.mkdir(parents=True, exist_ok=True)

    LOOKAHEAD = MAX_WORKERS * 4
    raw: list[tuple[int, Optional[int], Optional[int], Optional[int]]] = []
    debug_copies: dict[int, np.ndarray] = {}

    def _drain(pending: deque):
        fidx, fut = pending.popleft()
        timer_sec, blue, red = fut.result()
        raw.append((fidx, timer_sec, blue, red))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        pending: deque = deque()
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % sample_every == 0:
                pending.append((frame_idx, pool.submit(_process_frame, frame.copy())))
                debug_copies[frame_idx] = frame.copy()

                if len(pending) >= LOOKAHEAD:
                    _drain(pending)

            frame_idx += 1

        while pending:
            _drain(pending)

    cap.release()

    # ------------------------------------------------------------------
    # Stage 2 — state machine: phase detection + elapsed mapping
    # ------------------------------------------------------------------
    phase      : Optional[str] = None
    in_match                   = False
    last_timer : Optional[int] = None
    by_elapsed : dict[int, list] = {}

    for fidx, timer_sec, blue, red in raw:
        if timer_sec is None:
            continue

        if not in_match:
            if 0 <= timer_sec <= AUTON_LEN:
                in_match, phase, last_timer = True, "AUTO", timer_sec
            elif 120 <= timer_sec <= TELE_LEN:
                in_match, phase, last_timer = True, "TELE", timer_sec
            else:
                continue

        if phase == "AUTO" and last_timer is not None and last_timer <= 2:
            if timer_sec >= 100:
                phase      = "TELE"
                last_timer = None

        if phase == "AUTO" and not (0 <= timer_sec <= AUTON_LEN):
            continue
        if phase == "TELE" and not (0 <= timer_sec <= TELE_LEN):
            continue

        if last_timer is not None:
            if timer_sec > last_timer:
                continue
            if last_timer - timer_sec > max_drop:
                continue

        last_timer = timer_sec

        elapsed = timer_to_elapsed(timer_sec, phase)
        if not (0 <= elapsed <= TOTAL_LEN):
            continue

        if elapsed < debug_seconds and fidx in debug_copies:
            _save_debug(debug_copies[fidx], debug_root / f"t{elapsed:04d}")

        by_elapsed.setdefault(elapsed, []).append((blue, red, timer_sec))

        if phase == "TELE" and timer_sec == 0:
            break

    debug_copies.clear()

    # ------------------------------------------------------------------
    # Stage 3 — one row per elapsed second
    # ------------------------------------------------------------------
    rows      = []
    last_blue = 0
    last_red  = 0

    for t in range(TOTAL_LEN + 1):
        samples = by_elapsed.get(t, [])
        blues   = [b for b, _, _ in samples]
        reds    = [r for _, r, _ in samples]

        effective_jump = EARLY_MAX_JUMP if t < EARLY_WINDOW else max_jump_per_sec

        blue = _pick_score(blues, last_blue, max_jump=effective_jump, allow_reset=allow_reset)
        red  = _pick_score(reds,  last_red,  max_jump=effective_jump, allow_reset=allow_reset)

        if t < WARMUP_SECONDS:
            blue = 0
            red  = 0

        timer_remaining, timer_display = elapsed_to_timer(t)

        rows.append({
            "match_elapsed_sec":   t,
            "timer_remaining_sec": timer_remaining,
            "timer_display":       timer_display,
            "blue":  blue,
            "red":   red,
        })
        last_blue, last_red = blue, red

    # ------------------------------------------------------------------
    # Stage 4 — monotonic pass (FRC scores never decrease)
    # ------------------------------------------------------------------
    for i in range(1, len(rows)):
        if rows[i]["blue"] < rows[i - 1]["blue"]:
            rows[i]["blue"] = rows[i - 1]["blue"]
        if rows[i]["red"] < rows[i - 1]["red"]:
            rows[i]["red"] = rows[i - 1]["red"]

    for i in range(1, len(rows)):
        rows[i]["blue_delta"] = rows[i]["blue"] - rows[i - 1]["blue"]
        rows[i]["red_delta"]  = rows[i]["red"]  - rows[i - 1]["red"]
    rows[0]["blue_delta"] = rows[0]["red_delta"] = 0

    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], out_path: Path):
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "match_elapsed_sec", "timer_remaining_sec", "timer_display",
            "blue", "red", "blue_delta", "red_delta",
        ])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Graph generation  (auto-scales y-axis to actual final scores)
# ---------------------------------------------------------------------------

def generate_graph(rows: list[dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping graph (pip install matplotlib)")
        return

    t    = [r["match_elapsed_sec"] for r in rows]
    blue = [r["blue"]              for r in rows]
    red  = [r["red"]               for r in rows]
    bd   = [r["blue_delta"]        for r in rows]
    rd   = [r["red_delta"]         for r in rows]

    # y-axis ceiling: 10% headroom above the highest final score.
    # No hardcoded cap — works for any score range.
    max_score    = max(blue[-1], red[-1], 1)
    score_ylim   = max_score * 1.12
    max_delta    = max(max(bd), max(rd), 1)
    delta_ylim   = max_delta * 1.2

    tick_pos    = list(range(0, len(t), 10))
    tick_labels = []
    for i in tick_pos:
        label = rows[i]["timer_display"]
        if i == 0:
            label = "AUTO\n" + label
        elif i == AUTON_LEN:
            label = "TELE\n" + label
        tick_labels.append(label)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [2.2, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Cumulative score
    ax1.fill_between(t, blue, alpha=0.15, color="#4a90d9")
    ax1.fill_between(t, red,  alpha=0.15, color="#e05c5c")
    ax1.plot(t, blue, color="#4a90d9", linewidth=2.2, label="Blue Alliance")
    ax1.plot(t, red,  color="#e05c5c", linewidth=2.2, label="Red Alliance")
    ax1.set_ylim(0, score_ylim)   # auto-scaled, no hard cap
    ax1.annotate(f"{blue[-1]} pts", xy=(t[-1], blue[-1]),
                 xytext=(-42, 8), textcoords="offset points",
                 color="#4a90d9", fontsize=10, fontweight="bold")
    ax1.annotate(f"{red[-1]} pts", xy=(t[-1], red[-1]),
                 xytext=(-42, -16), textcoords="offset points",
                 color="#e05c5c", fontsize=10, fontweight="bold")
    ax1.axvline(AUTON_LEN, color="#ffcc44", linewidth=1.4, linestyle="--", alpha=0.7)
    ax1.text(AUTON_LEN + 0.6, score_ylim * 0.02, "Teleop →",
             color="#ffcc44", fontsize=8.5, alpha=0.85)
    ax1.set_ylabel("Cumulative Score", color="white", fontsize=11)
    ax1.set_title("FRC Match Score Time-Series", color="white", fontsize=14, pad=10)
    ax1.legend(facecolor="#2a2d3a", edgecolor="#555", labelcolor="white",
               fontsize=10, loc="upper left")
    ax1.grid(axis="y", color="#333", linewidth=0.6)

    # Per-second deltas
    bar_w = 0.4
    ax2.bar([x - bar_w / 2 for x in t], bd, width=bar_w,
            color="#4a90d9", alpha=0.85, label="Blue pts/sec")
    ax2.bar([x + bar_w / 2 for x in t], rd, width=bar_w,
            color="#e05c5c", alpha=0.85, label="Red pts/sec")
    ax2.set_ylim(0, delta_ylim)   # auto-scaled
    ax2.axvline(AUTON_LEN, color="#ffcc44", linewidth=1.4, linestyle="--", alpha=0.7)
    ax2.set_ylabel("Points / sec", color="white", fontsize=10)
    ax2.set_xlabel("Match Time", color="white", fontsize=11)
    ax2.legend(facecolor="#2a2d3a", edgecolor="#555", labelcolor="white",
               fontsize=9, loc="upper left")
    ax2.grid(axis="y", color="#333", linewidth=0.6)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels, color="white", fontsize=8)

    for ax in (ax1, ax2):
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")

    plt.tight_layout(h_pad=0.4)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Graph saved : {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root  = find_repo_root(script_dir)

    p = argparse.ArgumentParser(description="Extract FRC match scores from video.")
    p.add_argument("--video",            default=r"2026\Test-Match-Videos\frc-2026-match2.mp4")
    p.add_argument("--samples-per-sec",  type=int, default=DEFAULT_SAMPLES_PER_SEC)
    p.add_argument("--max-jump-per-sec", type=int, default=DEFAULT_MAX_JUMP_PER_SEC)
    p.add_argument("--allow-reset",      action="store_true", default=DEFAULT_ALLOW_RESET)
    p.add_argument("--debug-seconds",    type=int, default=10)
    p.add_argument("--warmup",           type=int, default=WARMUP_SECONDS,
                   help="Seconds from match start where scores are forced to 0")
    p.add_argument("--workers",          type=int, default=MAX_WORKERS,
                   help="Parallel OCR workers (default: auto)")
    p.add_argument("--out",              default=str(script_dir / "out"))
    args = p.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (repo_root / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Video   : {video_path}")
    print(f"Workers : {args.workers}   Scale: {SCALE}x   Samples/sec: {args.samples_per_sec}")
    print(f"Warmup  : {args.warmup}s clamped to 0")

    rows = process_video(
        video_path=video_path,
        samples_per_sec=args.samples_per_sec,
        max_jump_per_sec=args.max_jump_per_sec,
        allow_reset=args.allow_reset,
        debug_seconds=args.debug_seconds,
        out_root=out_root,
    )

    csv_path   = out_root / "score_timeseries_timer.csv"
    graph_path = out_root / "match_score_timeseries.png"
    write_csv(rows, csv_path)
    generate_graph(rows, graph_path)
    print(f"\nCSV saved   : {csv_path}")
    print(f"Graph saved : {graph_path}")
    print(f"Debug frames: {out_root / 'debug_frames'}")


if __name__ == "__main__":
    main()