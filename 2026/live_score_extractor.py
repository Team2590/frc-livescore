"""
FRC Live Stream Score Extractor
================================
Connects to a YouTube FRC livestream (or a local .mp4 for testing),
automatically detects each match using the on-screen timer, extracts
the blue/red score time-series in real-time, and writes a CSV + graph
for every match as soon as it ends (well before the next one starts).

HOW IT WORKS — 3-state machine
───────────────────────────────
  SCANNING   Watches the stream at ~1fps doing only timer OCR (cheap).
             The scoreboard overlay disappears between matches so no
             valid M:SS timer = no match. No CPU wasted during breaks.

  IN_MATCH   The moment a valid AUTO or TELE timer appears, the machine
             switches to full OCR (timer + blue + red) at ~5fps using a
             thread pool. Each sampled frame is submitted to a worker.
             Results are bucketed by match-elapsed-second exactly like
             the file-based extractor (read_video_scores btw).

  COOLDOWN   When TELE hits 0:00, a background thread takes a snapshot
             of the data and writes the CSV + graph. The main loop
             immediately returns to SCANNING. It never pauses, so no
             match start is ever missed even on short event-day breaks.

TESTING WITHOUT A LIVE STREAM
──────────────────────────────
  python extract_scores_live.py --file path/to/match.mp4
  Same exact stuff, it just reads a local file instead
  Use this to validate everything before pointing it at a real event.
  ( Everything works fine after testing it with a livestream VOD )

  python extract_scores_live.py --url "https://www.youtube.com/watch?v=VOD_ID" --start at HH:MM:SS
  Works on past VODs too
  yt-dlp handles them identically to live.
  --start-at helps for testing when the first match starts quite later in the stream (saves us the wait)

REQUIREMENTS
────────────
  pip install yt-dlp opencv-python pytesseract pillow numpy matplotlib
"""

import argparse
import os
import re
import subprocess
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Config — mirrors read_video_scores.py
# ---------------------------------------------------------------------------

TESSERACT_EXE = os.getenv("TESSERACT_EXE")
if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# ── Scoreboard layout ────────────────────────────────────────────────────────
# BLUE_LEFT = True  → blue score is on the LEFT  (standard broadcast)
# BLUE_LEFT = False → blue score is on the RIGHT (some regional broadcasts)
# Override per-run with --blue-left or --blue-right on the command line
BLUE_LEFT = True

# The two score panel crop boxes (tightened to centre 60% to avoid edge
# elements that caused single digits to read as two digits like "2"->"20")
_LEFT_BOX  = (0.4180, 0.055, 0.4570, 0.120)
_RIGHT_BOX = (0.5446, 0.055, 0.5824, 0.120)

# Crop box for the match timer (M:SS) between the two score panels
TIMER_REL = (0.469, 0.055, 0.5315, 0.118)
TOP_REL = (0.0,   0.0,   1.0,    0.22)

def _assign_boxes(blue_left: bool):
    global BLUE_REL, RED_REL
    BLUE_REL = _LEFT_BOX  if blue_left else _RIGHT_BOX
    RED_REL  = _RIGHT_BOX if blue_left else _LEFT_BOX

_assign_boxes(BLUE_LEFT)

# ── OCR config ───────────────────────────────────────────────────────────────
OCR_CFG_SCORE = (
    "--oem 1 --psm 8 "
    "-c tessedit_char_whitelist=0123456789 "
    "-c classify_bln_numeric_mode=1"
)
OCR_CFG_TIMER = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789:"

SCALE = 3   # upscale factor for score crops (keep at 3 for accuracy)

# ── Match timing ─────────────────────────────────────────────────────────────
AUTON_LEN = 20
TELE_LEN  = 140
TOTAL_LEN = AUTON_LEN + TELE_LEN

# ── Score smoothing ───────────────────────────────────────────────────────────
# During AUTO (first 20 elapsed seconds) uses a tighter per-second jump cap
# 25 pts allows legitimate large auto scores (+20 for a game piece)
# while blocking OCR noise that briefly reads 30+ when the true score is 0-10
# After AUTO the wider TELE cap takes over for fast teleop scoring.
EARLY_WINDOW = 20
EARLY_MAX_JUMP = 25
DEFAULT_MAX_JUMP_PER_SEC = 40
DEFAULT_ALLOW_RESET = False  # FRC scores never decrease

# ── Sampling rates ────────────────────────────────────────────────────────────
SCAN_SAMPLE_EVERY  = 30   # ~1 fps at 30 fps (timer-only, negligible CPU)
MATCH_SAMPLE_EVERY = 6    # ~5 fps at 30 fps (full OCR via thread pool)

# After TELE hits 0:00, keeps collecting for this many extra seconds
POST_MATCH_BUFFER_SEC = 3

# If no valid timer is seen for this many consecutive sampled frames
# it saves whatever was collected and goes back to SCANNING
# At MATCH_SAMPLE_EVERY=6 and 30fps = 5 samples/sec
# 900 means ~180s (3 min) with no valid timer before giving up
# its long enough to survive any camera cut or overlay glitch mid-match
# Keeps track of the number of consecutive descending timer samples required before
# declaring a match started. Filters out referee timer tests that
# flicker briefly and then reset. At 5fps, 5 ticks = ~1 real second
# of confirmed countdown which is enough to rule out any test.
CONFIRM_TICKS = 5

IDLE_TIMEOUT_FRAMES = 900

MAX_WORKERS = min(6, os.cpu_count() or 2)

# ---------------------------------------------------------------------------
# Stream URL extraction
# ---------------------------------------------------------------------------

def get_stream_url(youtube_url: str) -> str:
    """
    Uses yt-dlp to resolve a YouTube URL (live or VOD) into a direct
    stream URL that OpenCV can open with VideoCapture()
    """
    print("[stream] Resolving URL via yt-dlp...")
    try:
        result = subprocess.run(
            ["yt-dlp", "--no-warnings", "-f", "best[ext=mp4]/best", "-g", youtube_url],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not found.  Install with:  pip install yt-dlp")
    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp timed out resolving the URL.")

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr.strip()}")

    url = result.stdout.strip().splitlines()[0]
    print("[stream] URL resolved OK.")
    return url

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def crop_rel(img: Image.Image, box) -> tuple:
    w, h = img.size
    l = max(0, min(int(box[0] * w), w - 1))
    t = max(0, min(int(box[1] * h), h - 1))
    r = max(l + 1, min(int(box[2] * w), w))
    b = max(t + 1, min(int(box[3] * h), h))
    return img.crop((l, t, r, b)), (l, t, r, b)

def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def preprocess(pil_crop: Image.Image, scale: int = SCALE) -> Image.Image:
    """
    Prepares a crop for Tesseract:
    greyscale -> upscale -> blur -> Otsu threshold -> inverts if needed -> denoise
    """
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
    s = re.sub(r"[^0-9]", "", text.strip())
    if not s:
        return None
    val = int(s[:4]) if len(s) > 4 else int(s)
    return val if 0 <= val <= 9999 else None

def _ocr_score(proc_img: Image.Image) -> Optional[int]:
    """Confidence-filtered score OCR (discards low-confidence noise tokens)"""
    try:
        data = pytesseract.image_to_data(
            proc_img, config=OCR_CFG_SCORE,
            output_type=pytesseract.Output.DICT,
        )
        good = [t.strip() for t, c in zip(data["text"], data["conf"])
                if t.strip() and int(c) >= 60]
        raw = "".join(good) if good else \
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

def _ocr_timer_only(pil: Image.Image) -> Optional[int]:
    """Cheap timer-only OCR (used while scanning between matches)"""
    return _parse_timer(
        pytesseract.image_to_string(
            preprocess(crop_rel(pil, TIMER_REL)[0], scale=2),
            config=OCR_CFG_TIMER,
        )
    )

def _ocr_full_frame(frame_bgr: np.ndarray):
    """Full OCR: timer + blue + red
    Runs inside a thread pool worker."""
    pil = cv2_to_pil(frame_bgr)
    timer_sec = _parse_timer(
        pytesseract.image_to_string(
            preprocess(crop_rel(pil, TIMER_REL)[0], scale=2),
            config=OCR_CFG_TIMER,
        )
    )
    if timer_sec is None:
        return None, None, None
    blue = _ocr_score(preprocess(crop_rel(pil, BLUE_REL)[0]))
    red  = _ocr_score(preprocess(crop_rel(pil, RED_REL)[0]))
    return timer_sec, blue, red

# ---------------------------------------------------------------------------
# Timer math
# ---------------------------------------------------------------------------

def timer_to_elapsed(timer_sec: int, phase: str) -> int:
    return AUTON_LEN - timer_sec if phase == "AUTO" \
        else AUTON_LEN + (TELE_LEN - timer_sec)

def elapsed_to_timer(elapsed: int) -> tuple[int, str]:
    rem = (AUTON_LEN - elapsed) if elapsed <= AUTON_LEN \
          else (TELE_LEN - (elapsed - AUTON_LEN))
    return rem, f"{rem // 60}:{rem % 60:02d}"

# ---------------------------------------------------------------------------
# Score aggregation
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
# Post-match output  (runs in a background thread)
# ---------------------------------------------------------------------------

def build_and_save(
    by_elapsed: dict,
    match_num: int,
    started_at: str,
    out_root: Path,
    max_jump_per_sec: int,
    allow_reset: bool,
):
    """Converts raw sample buckets into a CSV + graph and saves to disk."""
    label = f"match_{match_num:03d}_{started_at}"
    match_dir = out_root / label
    match_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    last_blue = 0
    last_red  = 0

    for t in range(TOTAL_LEN + 1):
        samples = by_elapsed.get(t, [])
        blues = [b for b, _, _ in samples]
        reds = [r for _, r, _ in samples]
        ej = EARLY_MAX_JUMP if t < EARLY_WINDOW else max_jump_per_sec

        blue = _pick_score(blues, last_blue, max_jump=ej, allow_reset=allow_reset)
        red  = _pick_score(reds,  last_red,  max_jump=ej, allow_reset=allow_reset)

        tr, td = elapsed_to_timer(t)
        rows.append({
            "match_elapsed_sec":   t,
            "timer_remaining_sec": tr,
            "timer_display":       td,
            "blue": blue,
            "red":  red,
        })
        last_blue, last_red = blue, red

    # Monotonic pass — FRC scores physically cannot decrease
    for i in range(1, len(rows)):
        if rows[i]["blue"] < rows[i-1]["blue"]: rows[i]["blue"] = rows[i-1]["blue"]
        if rows[i]["red"]  < rows[i-1]["red"]:  rows[i]["red"]  = rows[i-1]["red"]

    for i in range(1, len(rows)):
        rows[i]["blue_delta"] = rows[i]["blue"] - rows[i-1]["blue"]
        rows[i]["red_delta"]  = rows[i]["red"]  - rows[i-1]["red"]
    rows[0]["blue_delta"] = rows[0]["red_delta"] = 0

    _write_csv(rows, match_dir / "score_timeseries.csv")
    _generate_graph(rows, match_dir / "match_score_timeseries.png", label)
    print(f"\n[match {match_num}] ✓ Saved → {match_dir}")

def _write_csv(rows: list[dict], out_path: Path):
    import csv
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "match_elapsed_sec", "timer_remaining_sec", "timer_display",
            "blue", "red", "blue_delta", "red_delta",
        ])
        w.writeheader()
        w.writerows(rows)

def _generate_graph(rows: list[dict], out_path: Path, title_suffix: str = ""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping graph)")
        return

    t = [r["match_elapsed_sec"] for r in rows]
    blue = [r["blue"]  for r in rows]
    red = [r["red"]   for r in rows]
    bd = [r["blue_delta"] for r in rows]
    rd = [r["red_delta"]  for r in rows]

    max_score = max(blue[-1], red[-1], 1)
    max_delta = max(max(bd), max(rd), 1)

    tick_pos, tick_labels = [], []
    for i in range(0, len(t), 10):
        lbl = rows[i]["timer_display"]
        if i == 0: lbl = "AUTO\n" + lbl
        elif i == AUTON_LEN: lbl = "TELE\n" + lbl
        tick_pos.append(i)
        tick_labels.append(lbl)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True,
    )
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white", labelsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor("#444")

    ax1.fill_between(t, blue, alpha=0.15, color="#4a90d9")
    ax1.fill_between(t, red,  alpha=0.15, color="#e05c5c")
    ax1.plot(t, blue, color="#4a90d9", lw=2.2, label="Blue Alliance")
    ax1.plot(t, red,  color="#e05c5c", lw=2.2, label="Red Alliance")
    ax1.set_ylim(0, max_score * 1.12)
    ax1.annotate(f"{blue[-1]} pts", xy=(t[-1], blue[-1]),
                 xytext=(-42, 8),  textcoords="offset points",
                 color="#4a90d9", fontsize=10, fontweight="bold")
    ax1.annotate(f"{red[-1]} pts", xy=(t[-1], red[-1]),
                 xytext=(-42, -16), textcoords="offset points",
                 color="#e05c5c", fontsize=10, fontweight="bold")
    ax1.axvline(AUTON_LEN, color="#ffcc44", lw=1.4, linestyle="--", alpha=0.7)
    ax1.text(AUTON_LEN + 0.6, max_score * 0.02, "Teleop →",
             color="#ffcc44", fontsize=8.5, alpha=0.85)
    ax1.set_ylabel("Cumulative Score", color="white", fontsize=11)
    ax1.set_title(f"FRC Match Score Time-Series — {title_suffix}",
                  color="white", fontsize=13, pad=10)
    ax1.legend(facecolor="#2a2d3a", edgecolor="#555",
               labelcolor="white", fontsize=10, loc="upper left")
    ax1.grid(axis="y", color="#333", lw=0.6)

    bar_w = 0.4
    ax2.bar([x - bar_w/2 for x in t], bd, width=bar_w,
            color="#4a90d9", alpha=0.85, label="Blue pts/sec")
    ax2.bar([x + bar_w/2 for x in t], rd, width=bar_w,
            color="#e05c5c", alpha=0.85, label="Red pts/sec")
    ax2.set_ylim(0, max_delta * 1.2)
    ax2.axvline(AUTON_LEN, color="#ffcc44", lw=1.4, linestyle="--", alpha=0.7)
    ax2.set_ylabel("Points / sec", color="white", fontsize=10)
    ax2.set_xlabel("Match Time",   color="white", fontsize=11)
    ax2.legend(facecolor="#2a2d3a", edgecolor="#555",
               labelcolor="white", fontsize=9, loc="upper left")
    ax2.grid(axis="y", color="#333", lw=0.6)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels, color="white", fontsize=8)

    for ax in (ax1, ax2):
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")

    plt.tight_layout(h_pad=0.4)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

# ---------------------------------------------------------------------------
# Core loop  (shared by live and file modes)
# ---------------------------------------------------------------------------

def parse_start_at(s: str) -> float:
    """
    Parse --start-at value into seconds
    Accepts: 33:00 / 2:05:30 / 1980 (raw seconds)
    """
    s = s.strip()
    if ":" in s:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 2: return parts[0] * 60 + parts[1]
        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(s)

def _save_debug(frame_bgr: np.ndarray, out_dir: Path):
    """Saves a debug overlay image showing the crop box positions"""
    try:
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
    except Exception:
        pass   # never crash the main loop over a debug save

def _build_rows(by_elapsed: dict, max_jump_per_sec: int, allow_reset: bool) -> list[dict]:
    """Builds the full monotonic row list from raw sample buckets"""
    rows = []
    last_blue = 0
    last_red  = 0
    for t in range(TOTAL_LEN + 1):
        samples = by_elapsed.get(t, [])
        blues = [b for b, _, _ in samples]
        reds = [r for _, r, _ in samples]
        ej = EARLY_MAX_JUMP if t < EARLY_WINDOW else max_jump_per_sec
        blue = _pick_score(blues, last_blue, max_jump=ej, allow_reset=allow_reset)
        red = _pick_score(reds,  last_red,  max_jump=ej, allow_reset=allow_reset)
        tr, td = elapsed_to_timer(t)
        rows.append({"match_elapsed_sec": t, "timer_remaining_sec": tr,
                     "timer_display": td, "blue": blue, "red": red})
        last_blue, last_red = blue, red
    # Monotonic pass
    for i in range(1, len(rows)):
        if rows[i]["blue"] < rows[i-1]["blue"]: rows[i]["blue"] = rows[i-1]["blue"]
        if rows[i]["red"]  < rows[i-1]["red"]:  rows[i]["red"]  = rows[i-1]["red"]
    # Deltas
    for i in range(1, len(rows)):
        rows[i]["blue_delta"] = rows[i]["blue"] - rows[i-1]["blue"]
        rows[i]["red_delta"]  = rows[i]["red"]  - rows[i-1]["red"]
    rows[0]["blue_delta"] = rows[0]["red_delta"] = 0
    return rows

def _write_csv(rows: list[dict], out_path: Path):
    import csv
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "match_elapsed_sec", "timer_remaining_sec", "timer_display",
            "blue", "red", "blue_delta", "red_delta",
        ])
        w.writeheader()
        w.writerows(rows)

def run_loop(
    source: str,
    out_root: Path,
    max_jump_per_sec: int,
    allow_reset: bool,
    is_live: bool,
    start_at_sec: float = 0.0,
):
    """
    Core loop got three states:

    WAITING    Stream is open. Timer-only OCR at ~1fps
               Sits here indefinitely (no timeout) through pre-match
               ceremonies, technical delays, scoreboard-up-but-frozen,
               or anything else between matches
               Transitions to READY when a valid timer value appears

    READY      Scoreboard is visible but timer hasn't started counting
               down yet (timer frozen at 0:20 or 0:00 pre-match)
               Full OCR starts here so we don't miss the first second
               Transitions to IN_MATCH the moment the timer moves

    IN_MATCH   Timer is actively counting down. Full OCR at ~5fps via
               thread pool. Each elapsed second is written to the CSV
               immediately as it is confirmed, so the file grows live.
               Transitions back to WAITING when TELE hits 0:00 (or the
               scoreboard disappears) then finalises the graph.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {source[:120]}")

    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mode = "LIVE" if is_live else "FILE"
    max_drop = max(3, int(fps / (fps / MATCH_SAMPLE_EVERY)) + 2)

    print(f"[{mode}] Opened.  FPS: {fps:.1f}")
    print(f"[{mode}] Workers: {MAX_WORKERS}   Scale: {SCALE}x")
    print(f"[{mode}] Layout : blue on {'LEFT' if BLUE_REL == _LEFT_BOX else 'RIGHT'}")
    print(f"[{mode}] Waiting for scoreboard...\n")

    # Seek to start position if requested
    if start_at_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_at_sec * 1000)
        actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        print(f"[{mode}] Seeked to {actual:.1f}s")

    state = "WAITING"
    match_num = 0
    phase = None
    last_timer = None
    frozen_timer = None   # timer value when scoreboard first appears
    confirm_ticks = 0      # consecutive descending timer readings seen
    prev_ready_timer = None   # last timer value seen in READY state
    ready_buffer = []     # full OCR results buffered during READY
    by_elapsed = {}
    started_at = ""
    frame_idx = 0
    post_buf = 0
    csv_writer = None
    csv_file = None
    match_dir = None
    last_written_elapsed = -1   # tracks which seconds have been written live

    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pending = deque()

    # ------------------------------------------------------------------
    def _open_live_csv(mdir: Path):
        """Open the CSV for writing and write the header immediately."""
        nonlocal csv_writer, csv_file
        import csv as _csv
        csv_file = (mdir / "score_timeseries.csv").open("w", newline="")
        csv_writer = _csv.DictWriter(csv_file, fieldnames=[
            "match_elapsed_sec", "timer_remaining_sec", "timer_display",
            "blue", "red", "blue_delta", "red_delta",
        ])
        csv_writer.writeheader()
        csv_file.flush()

    def _write_live_rows(rows: list[dict]):
        """Append any newly confirmed rows to the open CSV."""
        nonlocal last_written_elapsed
        for row in rows:
            if row["match_elapsed_sec"] > last_written_elapsed:
                csv_writer.writerow(row)
                last_written_elapsed = row["match_elapsed_sec"]
        csv_file.flush()

    def _close_live_csv():
        nonlocal csv_writer, csv_file, last_written_elapsed
        if csv_file:
            csv_file.close()
        csv_writer = None
        csv_file = None
        last_written_elapsed = -1

    # ------------------------------------------------------------------
    def _ingest(timer_sec, blue, red):
        nonlocal phase, last_timer, post_buf

        if timer_sec is None:
            return

        # AUTO -> TELE transition
        if phase == "AUTO" and last_timer is not None and last_timer <= 2:
            if timer_sec >= 100:
                phase = "TELE"
                last_timer = None

        if phase == "AUTO" and not (0 <= timer_sec <= AUTON_LEN): return
        if phase == "TELE" and not (0 <= timer_sec <= TELE_LEN):  return

        if last_timer is not None:
            if timer_sec > last_timer:            return
            if last_timer - timer_sec > max_drop: return

        last_timer = timer_sec
        elapsed = timer_to_elapsed(timer_sec, phase)
        if not (0 <= elapsed <= TOTAL_LEN): return

        by_elapsed.setdefault(elapsed, []).append((blue, red, timer_sec))

        # Writes confirmed rows live to CSV
        if csv_writer and elapsed > 0:
            _flush_rows_up_to(elapsed - 1)

        if phase == "TELE" and timer_sec == 0:
            post_buf = int(fps * POST_MATCH_BUFFER_SEC / MATCH_SAMPLE_EVERY)

    def _flush_rows_up_to(up_to: int):
        """Builds and writes all rows up to up_to that aren't written yet."""
        if csv_writer is None:
            return
        new_rows = []
        prev_blue = by_elapsed.get(last_written_elapsed, [(0,0,None)])[-1][0] or 0
        prev_red  = by_elapsed.get(last_written_elapsed, [(0,None,0)])[-1][1] or 0
        # walks from last_written+1 to up_to
        for t in range(last_written_elapsed + 1, up_to + 1):
            samples = by_elapsed.get(t, [])
            blues = [b for b, _, _ in samples]
            reds = [r for _, r, _ in samples]
            ej = EARLY_MAX_JUMP if t < EARLY_WINDOW else max_jump_per_sec
            blue = _pick_score(blues, prev_blue, max_jump=ej, allow_reset=allow_reset)
            red = _pick_score(reds,  prev_red,  max_jump=ej, allow_reset=allow_reset)
            tr, td = elapsed_to_timer(t)
            delta_b = max(0, blue - prev_blue)
            delta_r = max(0, red  - prev_red)
            new_rows.append({
                "match_elapsed_sec": t,
                "timer_remaining_sec": tr,
                "timer_display": td,
                "blue": blue,
                "red": red,
                "blue_delta": delta_b,
                "red_delta":  delta_r,
            })
            prev_blue, prev_red = blue, red
        _write_live_rows(new_rows)

    def flush_pending():
        while pending:
            _, fut = pending.popleft()
            _ingest(*fut.result())

    def _start_match(detected_phase: str, timer_val: int):
        nonlocal match_num, phase, last_timer, by_elapsed
        nonlocal started_at, post_buf, frozen_timer, match_dir
        match_num += 1
        phase = detected_phase
        last_timer = timer_val
        by_elapsed = {}
        started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        post_buf = 0
        frozen_timer = None
        match_dir = out_root / f"match_{match_num:03d}_{started_at}"
        match_dir.mkdir(parents=True, exist_ok=True)
        _open_live_csv(match_dir)
        print(f"[match {match_num}] ● Started  phase={phase}  "
              f"timer={timer_val}s  at {started_at}")
        print(f"[match {match_num}]   Live CSV: {match_dir / 'score_timeseries.csv'}")

    def _end_match(reason: str):
        nonlocal state, phase, last_timer, by_elapsed
        nonlocal post_buf, frozen_timer, match_dir
        flush_pending()

        # Writes all remaining rows to CSV
        _flush_rows_up_to(TOTAL_LEN)
        _close_live_csv()

        # Monotonic pass + graph in background thread
        snap = dict(by_elapsed)
        mdir = match_dir
        mn = match_num
        sa = started_at

        def _finalise(snap, mdir, mn, sa):
            rows = _build_rows(snap, max_jump_per_sec, allow_reset)
            # Overwrites CSV with monotonic-corrected version
            _write_csv(rows, mdir / "score_timeseries.csv")
            _generate_graph(rows, mdir / "match_score_timeseries.png",
                            f"match_{mn:03d}_{sa}")
            print(f"\n[match {mn}] ✓ Finalised → {mdir}")

        threading.Thread(target=_finalise, args=(snap,mdir,mn,sa),
                         daemon=True).start()

        phase = None
        last_timer = None
        by_elapsed = {}
        post_buf = 0
        frozen_timer = None
        match_dir = None
        state = "WAITING"
        print(f"[{mode}] Waiting for next match...")

    # ------------------------------------------------------------------
    while True:
        ok, frame = cap.read()

        if not ok:
            if is_live:
                print(f"[{mode}] ⚠ Frame read failed. Reconnecting in 5s...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    print(f"[{mode}] ✗ Reconnect failed. Exiting.")
                    break
                print(f"[{mode}] Reconnected.")
                frame_idx = 0
                continue
            else:
                if state == "IN_MATCH" and by_elapsed:
                    _end_match("File ended")
                break

        # ── WAITING ──────────────────────────────────────────────────
        # Cheap timer-only OCR at ~1fps
        # No timeout (waits forever)
        if state == "WAITING":
            if frame_idx % SCAN_SAMPLE_EVERY != 0:
                frame_idx += 1
                continue

            timer_sec = _ocr_timer_only(cv2_to_pil(frame))

            if timer_sec is not None:
                if   0 <= timer_sec <= AUTON_LEN:  det_phase = "AUTO"
                elif 100 <= timer_sec <= TELE_LEN: det_phase = "TELE"
                else:
                    frame_idx += 1
                    continue

                # Scoreboard is visible -> moves to READY
                frozen_timer = timer_sec
                state = "READY"
                print(f"[{mode}] Scoreboard detected (timer={timer_sec}s). "
                      f"Waiting for match to start...")

        # ── READY ────────────────────────────────────────────────────
        # Scoreboard visible. We require CONFIRM_TICKS consecutive descending
        # timer samples before declaring a match started
        # filters out referee tests that flicker briefly then reset.
        #
        # Critically: it runs FULL OCR on every sampled frame during READY and
        # buffer the results. When the match is confirmed it replays the buffer
        # into by_elapsed so no early AUTO data is lost.
        elif state == "READY":
            if frame_idx % MATCH_SAMPLE_EVERY != 0:
                frame_idx += 1
                continue

            # Full OCR so it buffers score data alongside the timer
            timer_sec, blue, red = _ocr_full_frame(frame)

            if timer_sec is None:
                # Scoreboard gone -> discard buffer so it goes back to WAITING
                frozen_timer = None
                confirm_ticks = 0
                prev_ready_timer = None
                ready_buffer = []
                state = "WAITING"
                print(f"[{mode}] Scoreboard gone. Waiting...")
                frame_idx += 1
                continue

            if   0 <= timer_sec <= AUTON_LEN:  det_phase = "AUTO"
            elif 100 <= timer_sec <= TELE_LEN: det_phase = "TELE"
            else:
                confirm_ticks = 0
                prev_ready_timer = None
                ready_buffer = []
                frame_idx += 1
                continue

            if frozen_timer is None:
                frozen_timer = timer_sec
                prev_ready_timer = timer_sec
                confirm_ticks = 0
                ready_buffer = []

            elif timer_sec == frozen_timer:
                # Still frozen -> pre-match hold so keeps buffer empty
                confirm_ticks = 0
                prev_ready_timer = timer_sec
                ready_buffer = []

            elif prev_ready_timer is not None and timer_sec < prev_ready_timer:
                # Timer ticked down -> buffers this sample and counts
                ready_buffer.append((timer_sec, blue, red, det_phase))
                confirm_ticks += 1
                prev_ready_timer = timer_sec
                print(f"[{mode}] Timer counting ({timer_sec}s, "
                      f"confirmed {confirm_ticks}/{CONFIRM_TICKS})...")

                if confirm_ticks >= CONFIRM_TICKS:
                    # Real match confirmed -> replay buffer before switching
                    _start_match(det_phase, ready_buffer[0][0])
                    for buf_timer, buf_blue, buf_red, buf_phase in ready_buffer:
                        # Manually ingest buffered frames as if they arrived live
                        elapsed = timer_to_elapsed(buf_timer, buf_phase)
                        if 0 <= elapsed <= TOTAL_LEN:
                            by_elapsed.setdefault(elapsed, []).append(
                                (buf_blue, buf_red, buf_timer)
                            )
                    print(f"[match {match_num}] Replayed {len(ready_buffer)} "
                          f"buffered frames (elapsed 0–{len(ready_buffer)-1}s recovered)")
                    confirm_ticks    = 0
                    prev_ready_timer = None
                    ready_buffer     = []
                    state            = "IN_MATCH"

            elif timer_sec == prev_ready_timer:
                # Same value again -> OCR reads the same frame twice so it js skips
                pass

            elif timer_sec > prev_ready_timer:
                # Timer went UP -> genuine referee test/reset
                print(f"[{mode}] Timer jumped ({prev_ready_timer}→{timer_sec}s) "
                      f"— looks like a test. Resetting...")
                frozen_timer     = timer_sec
                prev_ready_timer = timer_sec
                confirm_ticks    = 0
                ready_buffer     = []

        # ── IN_MATCH ───────────────────────────────────────────────
        # Full OCR at ~5fps via thread pool
        # CSV grows live.
        elif state == "IN_MATCH":
            if frame_idx % MATCH_SAMPLE_EVERY != 0:
                frame_idx += 1
                continue

            # Debug overlays for the first ~5 seconds of each match
            if len(by_elapsed) < 25:
                elapsed_est = timer_to_elapsed(last_timer, phase) \
                    if last_timer is not None and phase else 0
                _save_debug(frame,
                            out_root / f"match_{match_num:03d}_debug"
                                      / f"t{elapsed_est:04d}")

            pending.append((frame_idx,
                            pool.submit(_ocr_full_frame, frame.copy())))

            # Drain completed futures
            while pending and pending[0][1].done():
                _, fut = pending.popleft()
                _ingest(*fut.result())

            # Post-match buffer
            if post_buf > 0:
                post_buf -= 1
                if post_buf == 0:
                    _end_match("TELE reached 0:00")

            # If scoreboard disappears mid-match it saves partially and waits
            elif timer_sec_gone(frame, fps, frame_idx):
                _end_match("Scoreboard lost mid-match")

        frame_idx += 1

    pool.shutdown(wait=False)
    cap.release()
    print(f"\n[{mode}] Done.  Outputs in: {out_root}")


def timer_sec_gone(frame, fps, frame_idx,
                   _counter=[0], _last=[0]):
    """
    Returns True only after IDLE_TIMEOUT_FRAMES consecutive sampled
    frames with no valid timer -> avoids false triggers on brief glitches.
    """
    pil = cv2_to_pil(frame)
    t   = _ocr_timer_only(pil)
    if t is not None:
        _counter[0] = 0
    else:
        _counter[0] += 1
    return _counter[0] > IDLE_TIMEOUT_FRAMES

def main():
    p = argparse.ArgumentParser(
        description="FRC score extractor -> live stream or local file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live event stream
  python extract_scores_live.py --url "https://www.youtube.com/watch?v=LIVE_ID"

  # Past VOD (great for testing — same pipeline as live)
  python extract_scores_live.py --url "https://www.youtube.com/watch?v=VOD_ID"

  # Local .mp4 (fastest way to test without internet)
  python extract_scores_live.py --file path/to/match.mp4

  # Blue score on the RIGHT side of the scoreboard overlay
  python extract_scores_live.py --file match.mp4 --blue-right
        """,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url",  help="YouTube live or VOD URL")
    src.add_argument("--file", help="Local .mp4 file (test mode)")

    p.add_argument("--out",              default="./live_out",
                   help="Output root directory (default: ./live_out)")
    p.add_argument("--max-jump-per-sec", type=int, default=DEFAULT_MAX_JUMP_PER_SEC,
                   help="Max pts/sec before a sample is rejected as noise (default: 40)")
    p.add_argument("--allow-reset",      action="store_true", default=DEFAULT_ALLOW_RESET,
                   help="Allow scores to decrease (off by default)")
    p.add_argument("--start-at", default=None,
                   help="Seek to this timestamp before scanning. "
                        "Formats: 33:00 / 2:05:30 / 1980 (seconds). "
                        "Useful for skipping pre-match content when testing on VODs.")

    layout = p.add_mutually_exclusive_group()
    layout.add_argument("--blue-left",  dest="blue_left", action="store_true",  default=None,
                        help="Blue alliance score is on the LEFT  (default)")
    layout.add_argument("--blue-right", dest="blue_left", action="store_false",
                        help="Blue alliance score is on the RIGHT (some regional broadcasts)")

    args = p.parse_args()

    if args.blue_left is not None:
        _assign_boxes(args.blue_left)

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.url:
        source  = get_stream_url(args.url)
        is_live = True
    else:
        source = str(Path(args.file).resolve())
        if not Path(source).exists():
            raise FileNotFoundError(f"File not found: {source}")
        is_live = False
        print(f"[FILE] Test mode — source: {source}")

    print(f"[{'LIVE' if is_live else 'FILE'}] Output root: {out_root}\n")

    start_at_sec = parse_start_at(args.start_at) if args.start_at else 0.0
    if start_at_sec > 0:
        m, s = int(start_at_sec // 60), int(start_at_sec % 60)
        print(f"[{'LIVE' if is_live else 'FILE'}] Seeking to {m}:{s:02d} "
              f"({start_at_sec:.0f}s) before scanning...\n")

    try:
        run_loop(source, out_root, args.max_jump_per_sec, args.allow_reset,
                 is_live, start_at_sec=start_at_sec)
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()