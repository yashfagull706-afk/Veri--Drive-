"""
Veri-Drive - Configuration
All project-wide settings in one place.

Veri-Drive is a gate security system for transport companies:
registered cars + registered drivers -> automatic entry/exit logging,
anything unregistered -> alarm.
"""

import os
import sys

# Fix Windows console encoding for EasyOCR progress bars and Unicode output
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
DATABASE_PATH = os.path.join(BASE_DIR, "data", "veri_drive.db")  # unified SQLite DB
DRIVERS_DIR   = os.path.join(BASE_DIR, "data", "drivers")        # driver face photos
VEHICLES_DIR  = os.path.join(BASE_DIR, "data", "vehicles")       # car plate photos
CAPTURES_DIR  = os.path.join(BASE_DIR, "data", "captures")       # gate event snapshots
YUNET_MODEL   = os.path.join(BASE_DIR, "data",
                             "face_detection_yunet_2023mar.onnx")

# ---------------------------------------------------------------------------
# ANPR (plate reading) settings
# ---------------------------------------------------------------------------
ANPR_CONFIDENCE_THRESHOLD = 0.25  # minimum OCR confidence to trust a reading
MIN_PLATE_LENGTH = 4              # shortest valid plate string
MAX_PLATE_LENGTH = 12             # longest valid plate string
MATCH_THRESHOLD = 0.80            # similarity needed to confirm a registered plate
MAX_IMAGE_DIM = 1024              # cap image size so CPU OCR stays fast

# Fast pass: smaller frames for the automatic scanner (CPU stays responsive).
# Recognition accuracy is unchanged for the manual/upload passes, which
# always use the full MAX_IMAGE_DIM pipeline.
ENABLE_FAST_PASS  = True          # automatic scans try a low-res pass first
FAST_MAX_IMAGE_DIM = 640          # resolution cap for the fast pass

# ---------------------------------------------------------------------------
# Face recognition settings
# ---------------------------------------------------------------------------
# face_id.py reads these (falls back to its own defaults if missing).
FACE_MATCH_THRESHOLD = 0.50       # max Pearson distance to accept a match
FACE_MATCH_MARGIN    = 0.08       # best driver must beat the next DIFFERENT
                                  # driver by at least this much

# ---------------------------------------------------------------------------
# Live gate camera settings
# ---------------------------------------------------------------------------
CAMERA_INDEX    = 0      # local webcam index (fallback source)
# CAMERA_SOURCE accepts an int (webcam index) OR a string (RTSP/IP-cam URL),
# e.g. "rtsp://user:pass@192.168.1.50:554/stream1". Env var overrides both.
CAMERA_SOURCE   = os.environ.get("VERIDRIVE_CAMERA", CAMERA_INDEX)
SCAN_COOLDOWN   = 3      # seconds between automatic real-time scans
SCAN_COOLDOWN_ACTIVE = 1 # faster rescans while a plate is in view (so a car
                         # lingering at the gate is not skipped between scans)
EVENT_GAP       = 15     # same plate is not logged twice within this many
                         # seconds (shortened from 30: a real second pass of
                         # the same car within 30s was being suppressed)
ALARM_GAP       = 8      # same alarm is not re-raised within this many seconds
                         # (shortened from 15: two different unknown cars
                         # within 15s were collapsed into one alarm)
FACE_SCAN_EVERY = 5      # live feed: run face ID on every Nth frame

# ---------------------------------------------------------------------------
# Weekly report settings
# ---------------------------------------------------------------------------
WEEK_DAYS = 7            # a weekly report covers 7 days

# ---------------------------------------------------------------------------
# Security: login (P0)
# ---------------------------------------------------------------------------
# Override with env vars VERIDRIVE_USER / VERIDRIVE_PASS in production.
ADMIN_USERNAME  = os.environ.get("VERIDRIVE_USER", "admin")
ADMIN_PASSWORD  = os.environ.get("VERIDRIVE_PASS", "veridrive2026")
SECRET_KEY_FILE = os.path.join(BASE_DIR, "data", "secret_key.txt")

# CORS: ONLY these origins may poll the read-only APIs (hosted demo shells).
# Registration/delete endpoints never send CORS headers, regardless of origin.
# The Vercel/Cloud Run pages are static demos with no camera or real data.
CORS_ALLOWED_ORIGINS = [
    # "https://veri-drive-axwq8hci3-yashfagull706-2265.vercel.app",
]

# HTTPS (optional): drop a self-signed cert + key at these paths and the
# server will use them automatically.  Generate with:
#   openssl req -x509 -newkey rsa:2048 -nodes -keyout data/key.pem
#           -out data/cert.pem -days 365 -subj "/CN=veri-drive"
HTTPS_CERT = os.path.join(BASE_DIR, "data", "cert.pem")
HTTPS_KEY  = os.path.join(BASE_DIR, "data", "key.pem")

# ---------------------------------------------------------------------------
# Production readiness (P2)
# ---------------------------------------------------------------------------
USE_WAITRESS = True       # serve via waitress when installed (Windows-friendly)
CAPTURE_RETENTION_DAYS = 30   # snapshots older than this are auto-deleted
RETENTION_CHECK_MINUTES = 60  # how often the cleanup job runs
BACKUP_DIR  = os.path.join(BASE_DIR, "data", "backups")
BACKUP_KEEP = 10          # number of backups retained

# ---------------------------------------------------------------------------
# Logs UI (P3)
# ---------------------------------------------------------------------------
LOG_PAGE_SIZE = 25        # rows per page in the Logs tables

# ---------------------------------------------------------------------------
# Pre-registered (seed) company data
# Loaded automatically ONLY when the database is empty (fresh install),
# so manual registrations/deletions from the Register page are never touched.
# ---------------------------------------------------------------------------
SEED_DRIVERS = [
    ("Yashfa", os.path.join(BASE_DIR, "data", "seed", "drivers", "Yashfa.jpg")),
    ("Hadia",  os.path.join(BASE_DIR, "data", "seed", "drivers", "Hadia.jpg")),
    ("Hadia",  os.path.join(BASE_DIR, "data", "seed", "drivers", "Hadia2.jpg")),
]
SEED_VEHICLES = [
    ("BDE-759", "Car 1"),
    ("AYY-911", "Car 2"),
]

# ---------------------------------------------------------------------------
# Barrier (gate arm) settings
# ---------------------------------------------------------------------------
BARRIER_OPEN_SECONDS = 10   # barrier stays open this long after authorization
