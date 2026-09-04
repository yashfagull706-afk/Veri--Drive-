"""
ANPR Module - Automatic Number Plate Recognition (Veri-Drive)

Detects license plates in images and reads the text using EasyOCR.
Two ways to read a plate:
  1. detect_plate(image_path)          -> best effort single string (files)
  2. read_plate_candidates(image)      -> candidate list from a live frame
  3. match_plate(text, plates)         -> matches OCR text against the
                                          registered vehicle list

Usage:
    python -m modules.anpr path/to/image.jpg
"""

import re
import sys
import os
import threading
from difflib import SequenceMatcher

import cv2
import numpy as np
import easyocr

# Import project config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Initialize EasyOCR reader (load model once, reuse for all calls)
# ---------------------------------------------------------------------------
_reader = None
# One shared EasyOCR reader is used by the gate-scanner thread and by
# vehicle-registration requests. Serialize inference so the two can never
# run readtext() on the same model at the same instant.
_reader_lock = threading.Lock()


def _get_reader():
    """Lazy-load the EasyOCR reader so the model loads only once."""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _readtext(reader, img):
    """Thread-safe wrapper around reader.readtext()."""
    with _reader_lock:
        return reader.readtext(img)


# ---------------------------------------------------------------------------
# Plate detection using contour-based approach
# ---------------------------------------------------------------------------
def _detect_plate_region(image):
    """
    Attempts to find the license plate region in an image using
    edge detection + contour filtering.

    Args:
        image: BGR image (numpy array from cv2.imread)

    Returns:
        Cropped plate image (numpy array), or None if no plate found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection
    edges = cv2.Canny(filtered, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first) - plates are usually large rectangles
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    plate_region = None
    for contour in contours:
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)

        # License plates are roughly rectangular (4 vertices)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # License plates typically have aspect ratio between 2 and 6
            if 1.5 <= aspect_ratio <= 7:
                plate_region = gray[y : y + h, x : x + w]
                break

    return plate_region


# ---------------------------------------------------------------------------
# OCR on plate region
# ---------------------------------------------------------------------------
def _read_plate_text(plate_image):
    """
    Runs EasyOCR on a cropped plate image and returns the detected text.

    Args:
        plate_image: Grayscale cropped plate image (numpy array)

    Returns:
        Tuple of (plate_text, confidence) or (None, 0) if no text found.
    """
    reader = _get_reader()

    # Run OCR
    results = _readtext(reader, plate_image)

    if not results:
        return None, 0.0

    # Combine all detected text blocks into one plate string
    # EasyOCR returns list of (bbox, text, confidence)
    best_text = ""
    best_confidence = 0.0

    for bbox, text, confidence in results:
        if confidence > best_confidence:
            best_confidence = confidence
        best_text += text

    # Clean up the plate text: uppercase, remove spaces/special chars
    plate_text = best_text.upper().replace(" ", "")

    if not plate_text:
        return None, 0.0

    return plate_text, best_confidence


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def detect_plate(image_path):
    """
    Detects and reads the license plate from an image.

    Args:
        image_path: Path to the image file.

    Returns:
        Plate number as a string (e.g. "LEA1234"),
        or None if no plate could be detected/read.
    """
    if not os.path.exists(image_path):
        print(f"[ANPR] Error: Image not found: {image_path}")
        return None

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ANPR] Error: Could not read image: {image_path}")
        return None

    # Step 1: Detect the plate region
    plate_region = _detect_plate_region(image)

    if plate_region is None:
        # Fallback: try OCR on the full image (plate might fill the frame)
        print("[ANPR] No plate region detected via contours, trying full image OCR...")
        plate_region = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Read text from the plate
    plate_text, confidence = _read_plate_text(plate_region)

    if plate_text is None:
        print("[ANPR] Could not read any text from the plate region.")
        return None

    if confidence < config.ANPR_CONFIDENCE_THRESHOLD:
        print(f"[ANPR] Low confidence ({confidence:.2f}), plate text may be unreliable: {plate_text}")

    print(f"[ANPR] Detected plate: {plate_text} (confidence: {confidence:.2f})")
    return plate_text


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def normalize_text(text):
    """Uppercase + convert OCR separator artifacts into hyphens."""
    text = text.strip().upper()
    text = re.sub(r'[·•._/]', '-', text)
    text = re.sub(r'[^A-Z0-9\s\-]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def clean_plate(s):
    """Plate text without spaces/hyphens, used for comparison."""
    return s.replace(" ", "").replace("-", "")


def levenshtein(a, b):
    """Edit distance between two strings (small DP, plates are short)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_likely_plate(text):
    """Does this OCR string look like a license plate?"""
    text = normalize_text(text)
    cleaned = clean_plate(text)
    if not (config.MIN_PLATE_LENGTH <= len(cleaned) <= config.MAX_PLATE_LENGTH):
        return False
    if not (any(c.isdigit() for c in cleaned)
            and any(c.isalpha() for c in cleaned)):
        return False
    if not re.match(r'^[A-Za-z0-9\s\-]+$', text):
        return False
    return True


def match_plate(text, registered_plates):
    """
    Compares OCR text against the registered vehicle plates.

    Fuzzy matching (SequenceMatcher >= config.MATCH_THRESHOLD) tolerates OCR
    noise, with one guard: when the reading fuzzy-matches TWO registered
    plates at once (e.g. near-duplicates like BDE-759 vs BDE-758 read as
    "BDE75?"), the match is AMBIGUOUS and is rejected unless it is exact.
    This prevents silently logging the wrong car.

    Args:
        text:              raw OCR reading.
        registered_plates: iterable of registered plate strings.

    Returns:
        The registered plate string if confidently matched, otherwise None.
    """
    target = clean_plate(normalize_text(text or ""))
    if not target:
        return None
    ranked = []
    for plate in registered_plates:
        ref = clean_plate(plate)
        if not ref:
            continue
        ratio = (1.0 if target == ref
                 else SequenceMatcher(None, target, ref).ratio())
        ranked.append((plate, ratio))
    if not ranked:
        return None
    ranked.sort(key=lambda pr: pr[1], reverse=True)
    best_name, best_ratio = ranked[0]

    if best_ratio < config.MATCH_THRESHOLD:
        return None

    if best_ratio < 1.0 and len(ranked) > 1:
        second_ratio = ranked[1][1]
        # Two registered plates both fuzzy-match the reading -> ambiguous.
        # Require an exact (post-cleaning) match to log anything.
        if second_ratio >= config.MATCH_THRESHOLD:
            print(f"[ANPR] ambiguous plate '{target}': "
                  f"{best_name} ({best_ratio:.2f}) vs "
                  f"{ranked[1][0]} ({second_ratio:.2f}) - rejected")
            return None
        # Best match is a near-duplicate of another registered plate
        # (edit distance 1): only accept a fuzzy reading if it is very close.
        if levenshtein(best_name.replace(" ", "").replace("-", ""),
                       ranked[1][0].replace(" ", "").replace("-", "")) <= 1 \
                and best_ratio < 0.95:
            print(f"[ANPR] near-duplicate plates ({best_name} / "
                  f"{ranked[1][0]}): weak reading '{target}' rejected")
            return None
    return best_name


# ---------------------------------------------------------------------------
# Image enhancement for live frames
# ---------------------------------------------------------------------------
def _enhance_image(img, max_dim=None):
    """Grayscale + CLAHE contrast boost, capped resolution."""
    if max_dim is None:
        max_dim = config.MAX_IMAGE_DIM
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _resize_color(img, max_dim=None):
    """Resolution cap but keep colors (fallback pass for embossed plates)."""
    if max_dim is None:
        max_dim = config.MAX_IMAGE_DIM
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        return cv2.resize(img, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    return img


def _analyze(ocr_results):
    """Filters OCR results into plate candidates sorted by confidence."""
    candidates = []
    for _bbox, text, confidence in ocr_results:
        if confidence < config.ANPR_CONFIDENCE_THRESHOLD:
            continue
        clean_text = normalize_text(text)
        if is_likely_plate(clean_text):
            candidates.append((clean_text, float(confidence)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def read_plate_candidates(img, fast=False):
    """
    Full ANPR pipeline on one in-memory image (camera frame or decoded file):
    grayscale+CLAHE pass first, color fallback if nothing found.

    Args:
        fast: when True (and config.ENABLE_FAST_PASS is on), uses a smaller
              resolution cap for the automatic scanner so each scan takes a
              fraction of the time.  Manual/upload passes should stay False.

    Returns:
        List of (plate_text, confidence), best first. May be empty.
    """
    reader = _get_reader()
    max_dim = None
    if fast and getattr(config, "ENABLE_FAST_PASS", False):
        max_dim = getattr(config, "FAST_MAX_IMAGE_DIM", config.MAX_IMAGE_DIM)
    try:
        candidates = _analyze(_readtext(reader, _enhance_image(img, max_dim)))
        if not candidates:
            candidates = _analyze(_readtext(reader, _resize_color(img, max_dim)))
    except Exception as exc:
        print(f"[ANPR] OCR error skipped: {exc}")
        candidates = []
    return candidates


def read_plate_from_image(img):
    """Best plate reading from an in-memory image, or None."""
    candidates = read_plate_candidates(img)
    return candidates[0][0] if candidates else None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m modules.anpr <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = detect_plate(image_path)

    if result:
        print(f"\nResult: {result}")
    else:
        print("\nResult: No plate detected.")
