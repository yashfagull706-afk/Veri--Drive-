"""
face_id.py — Standalone Face Recognition for Driver Identification (Veri-Drive)

Self-contained file. Only needs: opencv-python, numpy
No dlib, no tensorflow, no face_recognition library.

Usage in your pipeline:
    from face_id import load_drivers, identify_driver

    load_drivers("data/drivers")          # load reference photos
    name = identify_driver("photo.jpg")   # returns "Ali" or None

Register drivers via the Veri-Drive web portal (Register page).
Test with webcam:             py modules/face_id.py --live
"""

import os
import sys
import threading
import cv2
import numpy as np

# Read global settings from config.py when available (the file still works
# standalone with the fallback defaults below).
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
try:
    import config as _config
except ImportError:
    _config = None


# ===================================================================
# CONFIGURATION (edit these to tune behavior)
# ===================================================================
MATCH_THRESHOLD = getattr(_config, "FACE_MATCH_THRESHOLD", 0.50)  # Max distance to accept a match
MATCH_MARGIN = getattr(_config, "FACE_MATCH_MARGIN", 0.08)        # Best must beat 2nd-best by at least this much
FACE_SIZE = (112, 112)    # Standard face size for encoding.
DEBUG = "--debug" in sys.argv  # Show distances in terminal

# Bump when the encoding pipeline changes: saved .npz files from older
# versions are rejected on load (mixing incompatible encodings silently
# degrades matching), forcing clean re-enrollment.
ENCODING_VERSION = 2

DATABASE_FILE = os.path.join(_SCRIPT_DIR, "data", "veri_drivers.npz")
DRIVERS_DIR = os.path.join(_SCRIPT_DIR, "data", "drivers")
YUNET_MODEL = os.path.join(_SCRIPT_DIR, "data", "face_detection_yunet_2023mar.onnx")


# ===================================================================
# INTERNAL STATE
# ===================================================================
_known_encodings = []    # list of feature vectors (numpy arrays)
_known_names = []        # driver names (same order as encodings)
_loaded = False
_detector = None
# The YuNet detector is a single shared cv2.dnn model. OpenCV's DNN graph
# engine keeps per-instance buffers, so setInputSize()+detect() from two
# threads at once (gate scanner thread vs. a Register-page request) corrupts
# them -> "(-215) buf.shape() == m.shape()" and a missed face. Serialize it.
_detector_lock = threading.Lock()
_last_face_boxes = []    # list of (x, y, w, h) for all detected faces


# ===================================================================
# FACE DETECTION (YuNet DNN — supports multiple faces)
# ===================================================================
def _get_detector():
    """Lazy-load the YuNet face detector."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None and os.path.exists(YUNET_MODEL):
                _detector = cv2.FaceDetectorYN_create(
                    YUNET_MODEL, "", (320, 320),
                    0.7,   # score threshold
                    0.3,   # NMS threshold
                    5000   # top_k
                )
    return _detector


def _detect_all_faces(image):
    """
    Detect ALL faces in an image using YuNet.
    Returns list of (box, face_crop) where box=(x,y,w,h) and face_crop is grayscale.
    """
    global _last_face_boxes
    _last_face_boxes = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    bgr = image if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape[:2]
    results = []

    det = _get_detector()
    if det is not None:
        # hold the lock across setInputSize+detect: both mutate the shared
        # model's internal buffers and must not interleave across threads.
        with _detector_lock:
            det.setInputSize((w, h))
            _, faces = det.detect(bgr)

        if faces is not None and len(faces) > 0:
            for face_data in faces:
                fx, fy, fw, fh = face_data[:4].astype(int)
                _last_face_boxes.append((fx, fy, fw, fh))

                # Add padding
                pad = int(max(fw, fh) * 0.2)
                x1 = max(0, fx - pad)
                y1 = max(0, fy - pad)
                x2 = min(w, fx + fw + pad)
                y2 = min(h, fy + fh + pad)

                face_crop = gray[y1:y2, x1:x2]
                if face_crop.size > 0:
                    results.append(((fx, fy, fw, fh), face_crop))

    return results


def _detect_face(image):
    """
    Detects the first (largest) face in the image.
    Returns a grayscale face crop, or None.
    """
    all_faces = _detect_all_faces(image)
    if not all_faces:
        return None

    # Return the largest face
    best = max(all_faces, key=lambda f: f[0][2] * f[0][3])
    return best[1]


# ===================================================================
# FEATURE EXTRACTION
# ===================================================================
def _compute_lbp(img):
    """Compute Local Binary Pattern histogram (lighting-invariant texture feature)."""
    # 8-neighbor LBP
    h, w = img.shape
    lbp = np.zeros_like(img)
    # Offsets for 8 neighbors (clockwise from top-left)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, 1),  (1, 1),  (1, 0),
               (1, -1),  (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        shifted = np.zeros_like(img)
        sy = max(0, dy)
        sx = max(0, dx)
        ey = h + min(0, dy)
        ex = w + min(0, dx)
        shifted[sy:ey, sx:ex] = img[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
        lbp |= (shifted >= img).astype(np.uint8) << bit
    # Only use inner pixels to avoid border artifacts
    inner = lbp[2:-2, 2:-2]
    hist = cv2.calcHist([inner], [0], None, [256], [0, 256]).flatten()
    return hist / (hist.sum() + 1e-7)


def _edge_orientation_hist(img, bins=8):
    """Histogram of edge orientations (captures facial structure)."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.degrees(np.arctan2(gy, gx)) % 180

    hist = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        lo = i * (180.0 / bins)
        hi = (i + 1) * (180.0 / bins)
        mask = (angle >= lo) & (angle < hi) & (mag > 20)
        hist[i] = mask.sum()
    return hist / (hist.sum() + 1e-7)


def _normalize_lighting(gray):
    """CLAHE local-contrast + global histogram equalization.

    Reduces sensitivity to harsh gate lighting, backlight and shadow
    without pulling in a heavy deep-learning stack.
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.equalizeHist(clahe.apply(gray))


def _encode_face(face):
    """
    Converts a grayscale face into a robust feature vector.
    Features:
      1. LBP histogram (lighting-invariant texture)
      2. Edge orientation histogram (facial structure)
      3. Smoothed downsampled pixel grid
      4. Intensity histogram (coarse color distribution)
      5. Quadrant LBP histograms (spatial texture)
    """
    face = cv2.resize(face, FACE_SIZE)
    face = cv2.GaussianBlur(face, (3, 3), 0)
    face = _normalize_lighting(face)
    features = []
    half = FACE_SIZE[0] // 2

    # 1. LBP histogram (256 bins) — lighting invariant
    features.append(_compute_lbp(face))

    # 2. Edge orientation histogram (8 bins) — structural
    features.append(_edge_orientation_hist(face, bins=8))

    # 3. Smoothed downsampled grid (24x24)
    small = cv2.resize(cv2.GaussianBlur(face, (5, 5), 0), (24, 24)).astype(np.float64) / 255.0
    features.append(small.flatten())

    # 4. Coarse intensity histogram (64 bins)
    h = cv2.calcHist([face], [0], None, [64], [0, 256]).flatten()
    features.append(h / (h.sum() + 1e-7))

    # 5. Quadrant LBP histograms (spatial texture)
    for i in range(2):
        for j in range(2):
            q = face[i * half : (i + 1) * half, j * half : (j + 1) * half]
            features.append(_compute_lbp(q))

    return np.concatenate(features)


# ===================================================================
# SAVE / LOAD DATABASE
# ===================================================================
def save_database(filepath=None):
    """
    Saves all registered drivers (names + encodings) to a .npz file
    so they persist between runs without re-processing photos.

    Args:
        filepath: Where to save. Defaults to data/drivers_db.npz.
    """
    if filepath is None:
        filepath = DATABASE_FILE

    if not _known_names:
        # No drivers left -> remove the saved database so the next start is fresh
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass
        print("[Face ID] No drivers registered - face database cleared.")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Stack encodings into a 2D array, names into a 1D array
    encodings_array = np.array(_known_encodings)   # shape: (N, feature_length)
    names_array = np.array(_known_names)            # shape: (N,)

    np.savez_compressed(filepath, encodings=encodings_array, names=names_array,
                        version=np.array([ENCODING_VERSION]))
    print(f"[Face ID] Saved {len(_known_names)} driver(s) to {filepath}")


def load_database(filepath=None):
    """
    Loads previously saved driver database from the .npz file.
    Much faster than re-processing photos every time.

    Args:
        filepath: Where to load from. Defaults to data/drivers_db.npz.

    Returns:
        True if loaded successfully, False if file doesn't exist.
    """
    global _known_encodings, _known_names, _loaded

    if filepath is None:
        filepath = DATABASE_FILE

    if not os.path.exists(filepath):
        return False

    data = np.load(filepath, allow_pickle=False)

    # Reject databases saved by an older encoding pipeline — mixing versions
    # silently degrades match quality.  Drivers must re-enroll (seed photos
    # are re-applied automatically on startup).
    saved_version = int(data["version"][0]) if "version" in data else 1
    if saved_version != ENCODING_VERSION:
        print(f"[Face ID] Ignoring {os.path.basename(filepath)}: encoding "
              f"version {saved_version} != current {ENCODING_VERSION}. "
              f"Drivers need re-enrollment.")
        return False

    encodings_array = data["encodings"]
    names_array = data["names"]

    _known_encodings = [encodings_array[i] for i in range(len(names_array))]
    _known_names = list(names_array)
    _loaded = True

    print(f"[Face ID] Loaded {len(_known_names)} driver(s) from database: {', '.join(_known_names)}")
    return True


# ===================================================================
# DRIVER REGISTRY
# ===================================================================
def load_drivers(drivers_dir=None):
    """
    Load all registered drivers. Tries the saved database first (fast),
    then falls back to processing photos from the drivers folder.

    Supports multiple photos per driver — averages all encodings
    for a more robust reference.

    Args:
        drivers_dir: Path to the drivers folder.
    """
    global _known_encodings, _known_names, _loaded

    # Try saved database first
    if load_database():
        return

    # Fall back to scanning photo folders
    if drivers_dir is None:
        drivers_dir = DRIVERS_DIR

    _known_encodings = []
    _known_names = []

    if not os.path.exists(drivers_dir):
        print(f"[Face ID] Folder not found: {drivers_dir}")
        return

    for name in sorted(os.listdir(drivers_dir)):
        folder = os.path.join(drivers_dir, name)
        if not os.path.isdir(folder):
            continue

        encodings_for_this_driver = []

        for photo in sorted(os.listdir(folder)):
            if not photo.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            img = cv2.imread(os.path.join(folder, photo))
            if img is None:
                continue

            face = _detect_face(img)
            if face is not None:
                encodings_for_this_driver.append(_encode_face(face))
                print(f"  Loaded: {name} ({photo})")

        if encodings_for_this_driver:
            # Average all encodings for this driver
            avg_encoding = np.mean(encodings_for_this_driver, axis=0)
            _known_encodings.append(avg_encoding)
            _known_names.append(name)

    _loaded = True

    # Save the database so next time it loads instantly
    if _known_names:
        save_database()

    print(f"[Face ID] {len(_known_names)} driver(s): {', '.join(_known_names)}\n")


def register_driver(name, image_path):
    """
    Add a driver at runtime and save to database.

    Args:
        name:       Driver's name.
        image_path: Path to a clear face photo.

    Returns:
        True if successful.
    """
    global _loaded

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Face ID] Cannot read: {image_path}")
        return False

    face = _detect_face(img)
    if face is None:
        print("[Face ID] No face detected in image.")
        return False

    _known_encodings.append(_encode_face(face))
    _known_names.append(name)
    _loaded = True

    # Save updated database
    save_database()

    print(f"[Face ID] Registered: {name}")
    return True


def remove_driver(name):
    """
    Remove a driver from the database.

    Args:
        name: Driver's name to remove.

    Returns:
        True if removed, False if not found.
    """
    global _known_encodings, _known_names
    if name not in _known_names:
        print(f"[Face ID] Driver '{name}' not found.")
        return False

    # Remove all entries for this driver
    keep_indices = [i for i, n in enumerate(_known_names) if n != name]
    _known_encodings = [_known_encodings[i] for i in keep_indices]
    _known_names = [_known_names[i] for i in keep_indices]

    save_database()
    print(f"[Face ID] Removed: {name}")
    return True


# ===================================================================
# FACE COMPARISON
# ===================================================================
def _distance(enc1, enc2):
    """
    Pearson correlation distance between two encodings.
    Returns 0 (identical) to 2 (opposite). Lower = better match.
    """
    if np.std(enc1) == 0 or np.std(enc2) == 0:
        return 1.0
    corr = np.corrcoef(enc1, enc2)[0, 1]
    return 1.0 - corr


# ===================================================================
# MATCHING (with margin check — rejects unknown faces)
# ===================================================================
def _match_encoding(query):
    """
    Match a face encoding against all known drivers.

    A driver may have several enrollment photos, so the comparison is
    done PER DRIVER (best distance among that driver's photos).
    Requires:
      1. Best driver distance < MATCH_THRESHOLD
      2. Best driver beats the 2nd-best DIFFERENT driver by MATCH_MARGIN

    Returns:
        (name, best_dist) if matched, (None, best_dist) if not.
    """
    if not _known_encodings:
        return None, 1.0

    distances = [_distance(query, k) for k in _known_encodings]

    # Collapse to the best distance for each driver name.
    best_by_name = {}
    for name, d in zip(_known_names, distances):
        if name not in best_by_name or d < best_by_name[name]:
            best_by_name[name] = d

    ranked = sorted(best_by_name.items(), key=lambda kv: kv[1])
    best_name, best_dist = ranked[0]
    second_best = ranked[1][1] if len(ranked) > 1 else 1.0
    margin = second_best - best_dist

    if DEBUG:
        for name, d in ranked:
            print(f"  {name}: {d:.3f}")
        print(f"  margin={margin:.3f}")

    # Must be under threshold AND beat the next DIFFERENT driver by margin
    if best_dist < MATCH_THRESHOLD and margin >= MATCH_MARGIN:
        return best_name, best_dist

    return None, best_dist


# ===================================================================
# IDENTIFICATION (from image file)
# ===================================================================
def identify_driver(image_path):
    """
    Identify the driver in a photo.

    Args:
        image_path: Path to the gate camera image.

    Returns:
        Driver name (str), or None if no match.
    """
    if not _known_encodings:
        print("[Face ID] No drivers loaded. Call load_drivers() first.")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Face ID] Cannot read: {image_path}")
        return None

    face = _detect_face(img)
    if face is None:
        print("[Face ID] No face in image.")
        return None

    query = _encode_face(face)
    name, dist = _match_encoding(query)

    if name:
        print(f"[Face ID] Matched: {name} (distance: {dist:.3f})")
        return name

    print(f"[Face ID] No match (distance: {dist:.3f}, threshold: {MATCH_THRESHOLD})")
    return None


# ===================================================================
# IDENTIFY ALL FACES IN A FRAME (multi-person support)
# ===================================================================
def identify_all_in_frame(frame):
    """
    Detect and identify ALL faces in a camera frame.

    Results are sorted LARGEST FACE FIRST: at a gate camera the driver sits
    closest to the lens, so the biggest recognized face is the best available
    driver-vs-passenger heuristic with a single camera.  (True driver/seat
    detection would need camera placement constraints or a second camera —
    this ordering is a heuristic, not a guarantee.)

    Args:
        frame: BGR image from cv2.VideoCapture.read().

    Returns:
        List of (box, name_or_None, distance) tuples, largest face first.
        box = (x, y, w, h)
    """
    all_faces = _detect_all_faces(frame)
    all_faces.sort(key=lambda f: f[0][2] * f[0][3], reverse=True)
    results = []

    for box, face_crop in all_faces:
        query = _encode_face(face_crop)
        name, dist = _match_encoding(query)
        results.append((box, name, dist))

    return results


# ===================================================================
# IDENTIFICATION (from camera frame — single face, backward compatible)
# ===================================================================
def identify_from_frame(frame):
    """
    Identify the driver from a live camera frame (numpy array).
    Returns the best match only. Use identify_all_in_frame() for multi-face.

    Args:
        frame: BGR image from cv2.VideoCapture.read().

    Returns:
        Driver name (str), or None if no match.
    """
    results = identify_all_in_frame(frame)
    # Return the first matched name
    for box, name, dist in results:
        if name is not None:
            return name
    return None


# ===================================================================
# LIVE WEBCAM RECOGNITION
# ===================================================================
def recognize_live(camera_index=0):
    """
    Opens the webcam and identifies ALL drivers in real time.
    Supports multiple people in frame simultaneously.
    Press Q to quit.

    Shows:
      - Green box + name when a driver is recognized (after 2+ confirmations)
      - Red box + "Unknown" when face detected but no match

    Args:
        camera_index: Which camera to use (0 = default).
    """
    if not _known_encodings:
        print("[Face ID] No drivers loaded. Call load_drivers() first.")
        return

    print(f"\n[Face ID] Starting live recognition with {len(_known_names)} driver(s)")
    print(f"  Drivers: {', '.join(_known_names)}")
    print("  Press Q to quit.\n")

    # Try multiple camera backends (MSMF often fails on Windows)
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "Auto"),
    ]
    cap = None
    for backend_id, backend_name in backends:
        cap = cv2.VideoCapture(camera_index, backend_id)
        if cap.isOpened():
            print(f"  Using camera backend: {backend_name}")
            break
        cap.release()
        print(f"  Backend {backend_name} failed, trying next...")
    else:
        print("[Face ID] Error: Cannot open camera with any backend.")
        print("  Make sure no other app is using the camera.")
        return

    frame_count = 0
    fail_count = 0

    # Warm up the camera (first few frames are often black/blank)
    for _ in range(5):
        cap.read()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            fail_count += 1
            if fail_count > 10:
                print("[Face ID] Camera read failed repeatedly.")
                break
            continue
        fail_count = 0

        frame_count += 1
        h, w = frame.shape[:2]

        # Detect and identify ALL faces every 3rd frame
        if frame_count % 3 == 0:
            results = identify_all_in_frame(frame)

            if DEBUG and results and frame_count % 9 == 0:
                for box, name, dist in results:
                    status = name if name else "Unknown"
                    print(f"  Face at ({box[0]},{box[1]}): {status} ({dist:.3f})")

            # Draw all faces
            for box, name, dist in results:
                fx, fy, fw, fh = box
                if name is not None:
                    color = (0, 255, 0)  # green
                    label = f"{name} ({dist:.2f})"
                else:
                    color = (0, 0, 255)  # red
                    label = "Unknown"

                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (fx, fy - th - 12), (fx + tw + 4, fy), color, -1)
                cv2.putText(frame, label, (fx + 2, fy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show driver list at bottom
        cv2.putText(frame, f"Registered: {', '.join(_known_names)}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Faces: {len(_last_face_boxes)}",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Fleet AI - Driver Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Face ID] Live recognition stopped.")


def get_all_driver_names():
    """Returns list of currently loaded driver names."""
    return list(_known_names)


# ===================================================================
# CLI
# ===================================================================
if __name__ == "__main__":
    print("Loading drivers...")
    load_drivers()

    if not _known_names:
        print("\nNo drivers found. Register drivers on the Veri-Drive")
        print("web portal (Register page).\n")
        sys.exit(1)

    # --live flag: open webcam for real-time recognition
    if "--live" in sys.argv:
        recognize_live()

    # --save flag: just rebuild the database from photos
    elif "--save" in sys.argv:
        print("Database saved.")

    # --list flag: show all drivers
    elif "--list" in sys.argv:
        print(f"\nRegistered drivers ({len(_known_names)}):")
        for i, name in enumerate(_known_names, 1):
            print(f"  {i}. {name}")

    # Default: identify from an image file (find first non-flag arg)
    else:
        image_arg = [a for a in sys.argv[1:] if not a.startswith("--")]
        if image_arg:
            result = identify_driver(image_arg[0])
            print(f"\n{'Match: ' + result if result else 'No match found.'}")
        else:
            print("\nUsage:")
            print("  py face_id.py <image.jpg> [--debug]  Identify driver from a photo")
            print("  py face_id.py --live [--debug]        Real-time webcam recognition")
            print("  py face_id.py --list                  Show all registered drivers")
            print("  py face_id.py --save                  Rebuild database from photos")
