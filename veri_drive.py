"""
====================================================================
VERI-DRIVE — GATE SECURITY SYSTEM
====================================================================
A real-time gate system for transport companies / security teams.

  * Register cars (plates) and drivers (faces).
  * Any REGISTERED driver may drive any REGISTERED car.
  * When a registered car passes the gate its ENTRY / EXIT time is
    logged automatically, together with the driver behind the wheel.
  * An event is ONLY logged when a plate is read.  A driver passing
    with no readable plate is NOT logged.
  * Unregistered car or unregistered driver -> ALARM (sound + banner
    in the web UI and a record in the alarm log).
  * Every driver gets personal logs (how many times he drove, which
    cars) and driver-car logs; weekly reports are generated for all
    of it.

Run:      py veri_drive.py
Then open http://127.0.0.1:5000  (or the Network URL shown).
====================================================================
"""

import os
import re
import sys
import json
import time
import shutil
import base64
import socket
import secrets
import tempfile
import threading
from functools import wraps
from datetime import datetime, timedelta

import cv2
import numpy as np
from flask import (Flask, request, jsonify, render_template_string,
                   Response, send_from_directory, send_file,
                   redirect, session)
from werkzeug.security import generate_password_hash, check_password_hash

# Project root + modules on the import path
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "modules"))

import config
import face_id
import anpr
import database as db

app = Flask(__name__)
app.permanent_session_lifetime = timedelta(days=7)


# ===================================================================
# AUTH (P0): simple session login in front of the Register page and
# every destructive endpoint.  Viewing pages/APIs stay open on the LAN.
# ===================================================================
def _load_secret_key():
    """Persistent random Flask secret (sessions survive restarts)."""
    try:
        with open(config.SECRET_KEY_FILE) as fh:
            key = fh.read().strip()
        if key:
            return key
    except OSError:
        pass
    key = secrets.token_hex(32)
    os.makedirs(os.path.dirname(config.SECRET_KEY_FILE), exist_ok=True)
    with open(config.SECRET_KEY_FILE, "w") as fh:
        fh.write(key)
    return key


app.secret_key = _load_secret_key()
_ADMIN_PASS_HASH = generate_password_hash(config.ADMIN_PASSWORD)
if config.ADMIN_PASSWORD == "veridrive2026":
    print("[SECURITY] Default admin password is in use! Set the env var "
          "VERIDRIVE_PASS (and optionally VERIDRIVE_USER) before real use.")


def _is_logged_in():
    return session.get("user") == config.ADMIN_USERNAME


def login_required_api(fn):
    """JSON 401 for API endpoints that change data."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _is_logged_in():
            return jsonify({"ok": False,
                            "message": "Login required."}), 401
        return fn(*args, **kwargs)
    return wrapper


def login_required_page(fn):
    """Redirect to /login for protected pages."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _is_logged_in():
            return redirect("/login")
        return fn(*args, **kwargs)
    return wrapper


# ===================================================================
# CORS (P0): locked down.  ONLY origins listed in config.CORS_ALLOWED_ORIGINS
# may poll, and ONLY the read-only endpoints below.  Registration/delete
# endpoints never send CORS headers.  (The hosted Vercel/Cloud Run pages are
# static demo shells with no camera or real data - they are the only
# legitimate cross-origin consumers.)
# ===================================================================
READ_ONLY_API = (
    "/api/status", "/api/events", "/api/gate_log", "/api/alarms",
    "/api/drivers", "/api/vehicles", "/api/driver_logs",
    "/api/driver_vehicle_logs", "/api/report",
)


@app.before_request
def _cors_preflight():
    if (request.method == "OPTIONS" and request.path in READ_ONLY_API
            and request.headers.get("Origin", "") in config.CORS_ALLOWED_ORIGINS):
        return "", 204


@app.after_request
def _add_cors(resp):
    origin = request.headers.get("Origin", "")
    if origin in config.CORS_ALLOWED_ORIGINS and request.path in READ_ONLY_API:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

# Initialize database + load registered drivers (fresh system = empty).
# Veri-Drive only trusts its own saved face database - registration is
# done through the web portal, never from loose photo folders.
db.init_db()
print("Loading driver face database...")
if not face_id.load_database():
    print("[Face ID] No registered drivers yet - starting fresh.")

for d in (config.DRIVERS_DIR, config.VEHICLES_DIR, config.CAPTURES_DIR):
    os.makedirs(d, exist_ok=True)


def ensure_app_icon():
    """Generates the Veri-Drive app icon once (used by PWA/home screen)."""
    path = os.path.join(BASE, "data", "app_icon.png")
    if os.path.exists(path):
        return path
    icon = np.zeros((512, 512, 3), dtype=np.uint8)
    icon[:] = (32, 18, 11)                          # dark navy background
    shield = np.array([(256, 48), (432, 112), (432, 280),
                       (256, 464), (80, 280), (80, 112)], dtype=np.int32)
    cv2.fillPoly(icon, [shield], (255, 163, 77))    # blue outer shield
    inner = np.array([(256, 84), (400, 136), (400, 268),
                      (256, 424), (112, 268), (112, 136)], dtype=np.int32)
    cv2.fillPoly(icon, [inner], (46, 28, 18))       # dark inner shield
    cv2.putText(icon, "VD", (150, 300), cv2.FONT_HERSHEY_SIMPLEX,
                3.4, (106, 196, 53), 16)            # green VD mark
    cv2.imwrite(path, icon)
    return path


# ===================================================================
# CAMERA MANAGER  (one shared camera for stream + scanner + capture)
# ===================================================================
class CameraManager(threading.Thread):
    """Continuously grabs frames so stream and scanner never fight.

    Source (config.CAMERA_SOURCE) may be an int (local webcam index) or a
    string (RTSP / IP-camera URL), so the camera does not have to be
    physically attached to the gate PC.
    """

    def __init__(self, source=None):
        super().__init__(daemon=True)
        self.source = config.CAMERA_SOURCE if source is None else source
        self._lock = threading.Lock()
        self.frame = None
        self.frame_time = 0.0
        self.ok = False
        self.error = ""
        self._cap = None

    def _open(self):
        if isinstance(self.source, str):
            # RTSP / IP camera URL
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                self._cap = cap
                return True
            cap.release()
            self.error = (f"IP camera could not be reached: {self.source}")
            return False
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.source, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cap = cap
                return True
            cap.release()
        return False

    def run(self):
        if not self._open():
            self.error = ("Camera could not be opened. Close other apps "
                          "using the webcam and reload.")
            return
        # warm-up: webcams send black frames for the first second
        for _ in range(20):
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.mean() > 10:
                with self._lock:
                    self.frame = frame
                    self.frame_time = time.time()
        self.ok = True
        fail = 0
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fail += 1
                if fail > 60:
                    self.error = "Camera stopped sending frames."
                    self.ok = False
                    time.sleep(2)
                    if self._open():
                        self.ok = True
                        self.error = ""
                        fail = 0
                    continue
                time.sleep(0.05)
                continue
            fail = 0
            if frame.mean() < 10:      # skip black frames
                continue
            with self._lock:
                self.frame = frame
                self.frame_time = time.time()

    def get_frame(self, max_age=5.0):
        """Latest fresh frame or (None, reason)."""
        with self._lock:
            if self.frame is None:
                return None, self.error or "no frame yet"
            if time.time() - self.frame_time > max_age:
                return None, "camera feed is stale"
            return self.frame.copy(), ""


camera = CameraManager()
camera.start()


# ===================================================================
# GATE SCANNER  (real-time background scanning + security rules)
# ===================================================================
def save_capture(frame, label):
    """Save an event snapshot into data/captures. Returns filename."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r'[^A-Za-z0-9]+', '-', label or "unknown").strip('-') or "unknown"
    name = f"{stamp}_{safe}.jpg"
    try:
        cv2.imwrite(os.path.join(config.CAPTURES_DIR, name), frame)
    except Exception as exc:
        print(f"[SNAP] could not save capture: {exc}")
        return None
    return name


def scan_frame(frame, fast=False):
    """
    Reads plate + faces from one frame.

    Args:
        fast: low-resolution first pass for the automatic scanner (see
              config.ENABLE_FAST_PASS).  Manual/upload scans use fast=False.

    Returns dict:
        plate_text       : raw OCR reading (or None)
        plate            : matched REGISTERED plate (or None)
        driver           : registered driver name (or None) — the LARGEST
                           matched face (closest to the gate camera)
        occupants        : all recognized names in the frame
        unknown_face     : True if a face was seen but not registered
        faces            : [(box, name_or_None, dist), ...] largest first
    """
    try:
        candidates = anpr.read_plate_candidates(frame, fast=fast)
    except Exception as exc:
        print(f"[SCAN] plate OCR failed: {exc}")
        candidates = []
    plate_text = candidates[0][0] if candidates else None

    plate = None
    if plate_text:
        plate = anpr.match_plate(plate_text, db.get_registered_plates())

    # Enhanced retry for stylized / low-contrast plates (holo plates etc):
    # 2x upscale + strong CLAHE, only when the first pass matched nothing.
    if plate is None:
        try:
            up = cv2.resize(frame, None, fx=2, fy=2,
                            interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            enh = cv2.cvtColor(cv2.createCLAHE(3.0, (8, 8)).apply(gray),
                               cv2.COLOR_GRAY2BGR)
            for text, _conf in anpr.read_plate_candidates(enh):
                if not plate_text:
                    plate_text = text
                m = anpr.match_plate(text, db.get_registered_plates())
                if m:
                    plate, plate_text = m, text
                    break
        except Exception as exc:
            print(f"[SCAN] enhanced plate pass failed: {exc}")

    try:
        raw_faces = face_id.identify_all_in_frame(frame)
    except Exception as exc:
        print(f"[SCAN] face detection failed: {exc}")
        raw_faces = []

    driver = None
    occupants = []
    unknown_face = False
    for _box, name, _dist in raw_faces:   # largest face first (see face_id)
        if name is not None:
            occupants.append(name)
            if driver is None:
                driver = name            # largest matched face = driver seat
        else:
            unknown_face = True

    return {
        "plate_text": plate_text,
        "plate": plate,
        "driver": driver,
        "occupants": occupants,
        "unknown_face": unknown_face,
        "faces": raw_faces,
    }


# -------------------------------------------------------------------
# Barrier (gate arm) state.
# The barrier opens ONLY when a registered car AND a registered
# driver are seen together at the gate (the M x N rule).  It closes
# automatically after BARRIER_OPEN_SECONDS.
# -------------------------------------------------------------------
_barrier_lock = threading.Lock()
_barrier = {"open": False, "opened_at": 0.0, "reason": ""}


def open_barrier(reason):
    with _barrier_lock:
        _barrier["open"] = True
        _barrier["opened_at"] = time.time()
        _barrier["reason"] = reason


def close_barrier(reason=""):
    """Lock the barrier immediately (unknown face / unregistered vehicle)."""
    with _barrier_lock:
        _barrier["open"] = False
        _barrier["opened_at"] = 0.0
        _barrier["reason"] = reason


def barrier_state():
    """Current barrier state (auto-closes once the open window expires)."""
    with _barrier_lock:
        if _barrier["open"] and time.time() - _barrier["opened_at"] \
                > config.BARRIER_OPEN_SECONDS:
            _barrier["open"] = False
            _barrier["reason"] = ""
        return dict(_barrier)


class GateScanner(threading.Thread):
    """
    Real-time security engine.  Every SCAN_COOLDOWN seconds it scans
    the latest camera frame and applies the Veri-Drive rules:

      registered plate  -> entry/exit is logged (with the driver)
      unknown plate     -> ALARM, nothing logged
      no plate          -> nothing logged (alarm only for unknown faces)
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.scan_requested = threading.Event()
        self.last_outcome = None            # latest scan result for the UI
        self.last_scan_time = None
        self._last_event = {}               # plate -> time (EVENT_GAP dedup)
        self._last_alarm = {}               # key  -> time (ALARM_GAP dedup)

    # ---------- rules ----------
    def handle(self, frame, source="auto"):
        s = scan_frame(frame, fast=(source == "auto"))
        now = time.time()
        outcome = {
            "source": source,
            "time": datetime.now().strftime("%I:%M:%S %p"),
            "plate_text": s["plate_text"],
            "plate": s["plate"],
            "driver": s["driver"],
            "occupants": s["occupants"],
            "faces_seen": len(s["faces"]),
            "event": None,
            "alarm": None,
            "barrier": "CLOSED",
            "status": "idle",
            "message": "Gate idle - nothing detected.",
            "snapshot": None,
        }

        # ---- 1. registered car -> log entry/exit ----
        if s["plate"] is not None:
            last = self._last_event.get(s["plate"], 0)
            if now - last >= config.EVENT_GAP or source != "auto":
                snapshot = save_capture(frame, s["plate"])
                event = db.log_gate_event(s["plate"], s["driver"],
                                          snapshot=snapshot)
                outcome["event"] = event
                outcome["snapshot"] = snapshot
                self._last_event[s["plate"]] = now
                outcome["status"] = "authorized"
                outcome["message"] = event["message"]

                # registered car, UNREGISTERED face behind the wheel
                if s["unknown_face"]:
                    self._alarm(outcome, frame, "UNKNOWN_DRIVER",
                                s["plate"], "Unknown",
                                f"Unregistered driver in registered car "
                                f"{s['plate']}")
            else:
                outcome["status"] = "authorized"
                outcome["message"] = (f"{s['plate']} already logged moments "
                                      f"ago ({int(config.EVENT_GAP)}s window).")

            # registered car but no face recognized at all
            if s["driver"] is None and not s["unknown_face"] \
                    and outcome["event"]:
                outcome["message"] += "  (driver not identified)"

            # ---- barrier: opens only for registered car + registered driver
            if s["driver"] is not None and not s["unknown_face"]:
                open_barrier(f"{s['plate']} + {s['driver']}")
                outcome["barrier"] = "OPEN"
                outcome["message"] += "  |  BARRIER OPEN"
            elif s["unknown_face"]:
                close_barrier("unknown driver in registered car")
                outcome["barrier"] = "CLOSED"
                outcome["message"] += "  |  BARRIER CLOSED (driver not registered)"

            # more than one recognized person in the car: show all of them
            if len(s["occupants"]) > 1:
                outcome["message"] += (f"  |  occupants: "
                                       f"{', '.join(s['occupants'])}")

        # ---- 2. plate read but NOT registered -> alarm ----
        elif s["plate_text"] is not None:
            kind = ("UNKNOWN_BOTH" if s["unknown_face"] or s["driver"] is None
                    else "UNKNOWN_VEHICLE")
            if s["driver"] is not None:
                kind = "UNKNOWN_VEHICLE"
            self._alarm(outcome, frame, kind, s["plate_text"],
                        s["driver"] or ("Unknown" if s["unknown_face"] else None),
                        f"Unregistered vehicle '{s['plate_text']}' at the gate.")
            outcome["status"] = "alarm"
            close_barrier("unregistered vehicle at gate")
            outcome["barrier"] = "CLOSED"
            outcome["message"] += "  |  BARRIER CLOSED"

        # ---- 3. no plate at all -> never log ----
        else:
            if s["unknown_face"]:
                self._alarm(outcome, frame, "UNKNOWN_BOTH", None, "Unknown",
                            "Unknown person at the gate, no readable plate.")
                outcome["status"] = "alarm"
                close_barrier("unknown person at gate")
                outcome["barrier"] = "CLOSED"
                outcome["message"] += "  |  BARRIER CLOSED"
            elif s["driver"] is not None:
                outcome["status"] = "no-plate"
                outcome["message"] = (f"{s['driver']} seen but no plate read "
                                      f"- nothing logged.")
            else:
                outcome["status"] = "nothing"

        outcome["_faces"] = s["faces"]   # used only for stream annotation
        self.last_outcome = outcome
        self.last_scan_time = datetime.now()
        return outcome

    def _alarm(self, outcome, frame, kind, plate, driver, details):
        """Raise an alarm unless the same one fired within ALARM_GAP."""
        key = f"{kind}|{plate or ''}|{driver or ''}"
        now = time.time()
        if now - self._last_alarm.get(key, 0) < config.ALARM_GAP:
            outcome["message"] += "  (alarm already raised recently)"
            return
        snapshot = outcome.get("snapshot") or save_capture(frame, kind)
        alarm = db.raise_alarm(kind, plate=plate, driver=driver,
                               details=details, snapshot=snapshot)
        outcome["alarm"] = alarm
        outcome["snapshot"] = snapshot
        outcome["status"] = "alarm"
        outcome["message"] = f"ALARM: {details}"
        self._last_alarm[key] = now

    # ---------- loop ----------
    def run(self):
        next_scan = time.time() + 2
        while True:
            forced = self.scan_requested.wait(timeout=0.2)
            if forced:
                self.scan_requested.clear()
            elif time.time() < next_scan:
                continue

            frame, err = camera.get_frame()
            if frame is None:
                next_scan = time.time() + config.SCAN_COOLDOWN
                continue
            outcome = None
            try:
                outcome = self.handle(frame,
                                      source="manual" if forced else "auto")
            except Exception as exc:
                print(f"[SCAN] error skipped: {exc}")

            # The cooldown is measured from the END of a scan, so a slow OCR
            # pass never makes the scanner skip its next slot (a car waiting
            # at the gate is rescanned immediately after a long scan).
            # While a plate is in view, rescan even faster so a lingering
            # car is tracked continuously.
            gap = config.SCAN_COOLDOWN
            if not forced and outcome and (outcome.get("plate_text")
                                           or outcome.get("plate")):
                gap = config.SCAN_COOLDOWN_ACTIVE
            next_scan = time.time() + gap


scanner = GateScanner()
scanner.start()


# ===================================================================
# HOUSEKEEPING  (P2: capture retention + daily database backups)
# ===================================================================
class Housekeeping(threading.Thread):
    """Periodic chores: delete old snapshots, back up the SQLite DB."""

    def run(self):
        while True:
            self._cleanup_captures()
            self._daily_backup()
            time.sleep(config.RETENTION_CHECK_MINUTES * 60)

    def _cleanup_captures(self):
        cutoff = time.time() - config.CAPTURE_RETENTION_DAYS * 86400
        removed = 0
        try:
            for f in os.listdir(config.CAPTURES_DIR):
                p = os.path.join(config.CAPTURES_DIR, f)
                if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                    os.remove(p)
                    removed += 1
        except OSError as exc:
            print(f"[HOUSE] capture cleanup error: {exc}")
        if removed:
            print(f"[HOUSE] removed {removed} snapshot(s) older than "
                  f"{config.CAPTURE_RETENTION_DAYS} days.")

    def _daily_backup(self):
        """One backup per calendar day (pruned to config.BACKUP_KEEP)."""
        today = datetime.now().strftime("%Y%m%d")
        try:
            existing = os.path.isdir(config.BACKUP_DIR) and any(
                f.startswith(f"veri_drive_{today}")
                for f in os.listdir(config.BACKUP_DIR))
        except OSError:
            existing = False
        if not existing:
            db.backup_database("daily")


housekeeping = Housekeeping()
housekeeping.start()


# ===================================================================
# IMAGE HELPERS
# ===================================================================
def annotate(frame, outcome):
    """Draw faces + status banner onto a frame."""
    out = frame.copy()
    if outcome:
        for box, name, dist in outcome.get("_faces", []):
            x, y, w, h = [int(v) for v in box]
            color = (0, 255, 0) if name else (0, 0, 255)
            label = f"{name} ({dist:.2f})" if name else "Unknown"
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        status = outcome.get("status", "idle")
        msg = outcome.get("message", "")[:80]
        if status == "alarm":
            bar = (0, 0, 200)
        elif status == "authorized":
            bar = (0, 140, 0)
        else:
            bar = (60, 60, 60)
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), bar, -1)
        cv2.putText(out, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)

        plate = outcome.get("plate") or outcome.get("plate_text")
        if plate:
            tag = f"PLATE: {plate}" + ("  (REGISTERED)" if outcome.get("plate")
                                       else "  (NOT REGISTERED)")
            cv2.putText(out, tag, (8, out.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    else:
        cv2.putText(out, "VERI-DRIVE  -  scanning...", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return out


def encode_jpg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return None if not ok else buf.tobytes()


def decode_upload(file_storage):
    """Decode an uploaded image file -> (img, error)."""
    try:
        data = file_storage.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "Uploaded file is not a readable image."
        return img, ""
    except Exception:
        return None, "Could not read the uploaded file."


# ===================================================================
# REGISTRATION HELPERS (with error checking)
# ===================================================================
def _save_driver_photo(name, img):
    folder = os.path.join(config.DRIVERS_DIR, re.sub(r'[^\w\- ]', '_', name).strip())
    os.makedirs(folder, exist_ok=True)
    n = len([f for f in os.listdir(folder) if f.endswith(".jpg")]) + 1
    path = os.path.join(folder, f"photo{n}.jpg")
    cv2.imwrite(path, img)
    return path


def register_driver_from_image(name, phone, img):
    """Full driver registration with validation. Returns (ok, message)."""
    name = (name or "").strip()
    if not name:
        return False, "Please enter the driver's name."

    if img is None:
        return False, "No photo provided - capture or upload a clear face photo."

    faces = face_id._detect_all_faces(img)
    if not faces:
        return False, ("No face detected in the photo. Take a clearer, "
                       "front-facing photo and try again.")
    if len(faces) > 1:
        return False, ("Multiple faces detected. Register one driver at a "
                       "time - please use a photo with a single face.")

    is_new = not db.is_driver_registered(name)
    if is_new:
        ok, msg = db.add_driver(name, phone)
        if not ok:
            return False, msg

    # write a temp file because face_id.register_driver reads from disk
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        tmp_path = tmp.name
    try:
        if not face_id.register_driver(name, tmp_path):
            if is_new:
                db.remove_driver(name)     # roll back the DB record
            return False, "Face registration failed - try a clearer photo."
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    _save_driver_photo(name, img)
    if is_new:
        return True, f"Driver '{name}' registered successfully."
    return True, f"Additional photo added for driver '{name}'."


def register_vehicle_from_image(plate, label, img):
    """Full vehicle registration with validation. Returns (ok, message)."""
    plate = (plate or "").strip().upper()
    if img is None:
        return False, "No photo provided - capture or upload a photo of the car/plate."

    if not plate:
        # try OCR as a last resort
        plate = anpr.read_plate_from_image(img) or ""

    ok, msg = db.add_vehicle(plate, label)
    if not ok:
        return False, msg

    safe = re.sub(r'[^A-Za-z0-9]+', '-', plate).strip('-')
    try:
        cv2.imwrite(os.path.join(config.VEHICLES_DIR, f"{safe}.jpg"), img)
    except Exception as exc:
        print(f"[REG] could not save vehicle photo: {exc}")
    return True, f"Vehicle '{plate}' registered successfully."


SEED_STATE_FILE = os.path.join(BASE, "data", "seed_state.json")


def seed_initial_data():
    """Pre-register the company's drivers and cars.

    Idempotent: every seed photo is applied exactly once (tracked in
    data/seed_state.json), so extra reference photos are added even on
    an existing install, while anything registered or removed through
    the Register page is never overwritten.
    """
    state = {"photos": []}
    if os.path.exists(SEED_STATE_FILE):
        try:
            with open(SEED_STATE_FILE) as fh:
                state = json.load(fh)
        except Exception:
            state = {"photos": []}
    # If the face-encoding format changed, the npz was invalidated - the
    # seed photos must be re-applied even though seed_state says "done".
    if state.get("encoding_version") != face_id.ENCODING_VERSION:
        print(f"[SEED] face encoding version changed "
              f"({state.get('encoding_version')} -> {face_id.ENCODING_VERSION});"
              f" re-enrolling seed drivers")
        state = {"photos": []}
    done = set(state.get("photos", []))

    for name, photo in config.SEED_DRIVERS:
        key = f"{name}|{os.path.basename(photo)}"
        if key in done or not os.path.exists(photo):
            continue
        if not db.is_driver_registered(name):
            img = cv2.imread(photo)
            if img is None:
                print(f"[SEED] unreadable photo for driver '{name}': {photo}")
                continue
            ok, msg = register_driver_from_image(name, "", img)
            print(f"[SEED] driver -> {msg}")
        else:
            # driver already exists: add this as an extra enrollment photo
            ok = face_id.register_driver(name, photo)
            print(f"[SEED] extra photo for '{name}' -> "
                  f"{'ok' if ok else 'failed'}")
        if ok:
            done.add(key)

    if not db.get_vehicles():
        for plate, label in config.SEED_VEHICLES:
            _ok, msg = db.add_vehicle(plate, label)
            print(f"[SEED] vehicle -> {msg}")

    state["photos"] = sorted(done)
    state["encoding_version"] = face_id.ENCODING_VERSION
    try:
        with open(SEED_STATE_FILE, "w") as fh:
            json.dump(state, fh)
    except Exception as exc:
        print(f"[SEED] could not save seed state: {exc}")


seed_initial_data()


# ===================================================================
# HTML / CSS / JS
# ===================================================================
_BASE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Veri-Drive</title>
<link rel="manifest" href="/manifest.json">
<link rel="icon" type="image/png" href="/icon.png">
<link rel="apple-touch-icon" href="/icon.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Veri-Drive">
<meta name="theme-color" content="#121c2e">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0b1220;--surface:#121c2e;--surface2:#1a2740;--border:#253350;
 --text:#e8eef7;--muted:#8fa0b8;--accent:#4da3ff;--ok:#35c46a;--bad:#ff5252;
 --warn:#ffb020}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);
 color:var(--text);min-height:100vh}
.header{background:var(--surface);border-bottom:1px solid var(--border);
 padding:14px 20px;display:flex;justify-content:space-between;align-items:center;
 position:sticky;top:0;z-index:50;min-height:56px}
.header h1{font-size:20px;color:var(--accent)}
.header h1 span{color:var(--text)}
.camstat{font-size:13px;color:var(--muted)}
.layout{display:flex;align-items:flex-start}
.nav{background:var(--surface);border-right:1px solid var(--border);
 display:flex;flex-direction:column;gap:2px;padding:12px 0;width:212px;
 min-width:212px;position:sticky;top:56px;align-self:flex-start;
 max-height:calc(100vh - 56px);overflow-y:auto}
.nav a{color:var(--muted);text-decoration:none;padding:13px 20px;font-size:14px;
 font-weight:600;border-left:3px solid transparent;display:flex;
 align-items:center;gap:10px}
.nav a.active,.nav a:hover{color:var(--text);border-left-color:var(--accent);
 background:var(--surface2)}
.container{flex:1;min-width:0;padding:20px;max-width:1200px;margin:0}
.card{background:var(--surface);border:1px solid var(--border);
 border-radius:12px;padding:20px;margin-bottom:20px}
.card h2{font-size:16px;margin-bottom:14px}
.grid{display:grid;gap:16px}
.stats{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:16px}
.stat .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;
 letter-spacing:.6px;margin-bottom:6px}
.stat .val{font-size:30px;font-weight:700}
.val.b{color:var(--accent)}.val.g{color:var(--ok)}.val.r{color:var(--bad)}
.val.o{color:var(--warn)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:10px;color:var(--muted);border-bottom:1px solid var(--border)}
td{padding:10px;border-bottom:1px solid var(--surface2)}
.badge{display:inline-block;padding:3px 10px;border-radius:14px;font-size:11px;
 font-weight:700}
.b-entry{background:rgba(77,163,255,.15);color:var(--accent)}
.b-exit{background:rgba(53,196,106,.15);color:var(--ok)}
.b-alarm{background:rgba(255,82,82,.18);color:var(--bad)}
.b-ok{background:rgba(53,196,106,.15);color:var(--ok)}
.b-muted{background:rgba(143,160,184,.15);color:var(--muted)}
.btn{display:inline-flex;align-items:center;gap:6px;padding:11px 18px;border:none;
 border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;
 text-decoration:none;color:#fff;background:var(--accent)}
.btn:hover{filter:brightness(1.15)}
.btn.gray{background:var(--surface2);border:1px solid var(--border)}
.btn.red{background:var(--bad)}
.btn.small{padding:5px 10px;font-size:12px}
input,select{background:var(--surface2);border:1px solid var(--border);
 color:var(--text);border-radius:8px;padding:10px 12px;font-size:14px;width:100%}
input[type=file]{display:none}
label.fbtn{display:inline-flex;align-items:center;padding:11px 18px;
 border-radius:8px;background:var(--surface2);border:1px solid var(--border);
 font-size:14px;font-weight:600;cursor:pointer;color:var(--text)}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.two{grid-template-columns:1fr 1fr}
@media(max-width:800px){.two{grid-template-columns:1fr}
 .layout{flex-direction:column}
 .nav{flex-direction:row;width:100%;min-width:0;position:static;max-height:none;
  overflow-x:auto;border-right:none;border-bottom:1px solid var(--border);
  padding:0 8px;gap:4px}
 .nav a{border-left:none;border-bottom:2px solid transparent;white-space:nowrap;
  padding:12px 14px}
 .nav a.active,.nav a:hover{border-left:none;
  border-bottom-color:var(--accent);background:transparent}}
.preview{width:100%;max-width:420px;border-radius:10px;border:2px solid var(--border);
 background:#000;margin-top:10px}
.livefeed{width:100%;border-radius:12px;border:2px solid var(--border);background:#000}
.msg{padding:10px 14px;border-radius:8px;font-size:14px;margin-top:12px;display:none}
.msg.ok{display:block;background:rgba(53,196,106,.12);color:var(--ok)}
.msg.err{display:block;background:rgba(255,82,82,.12);color:var(--bad)}
#alarmBanner{display:none;position:fixed;top:0;left:0;right:0;z-index:200;
 background:var(--bad);color:#fff;text-align:center;padding:12px;font-weight:700;
 font-size:16px;animation:flash .6s infinite alternate}
@keyframes flash{from{opacity:1}to{opacity:.55}}
.tabs{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.tabs button{background:var(--surface2);border:1px solid var(--border);
 color:var(--muted);padding:8px 16px;border-radius:8px;cursor:pointer;
 font-weight:600;font-size:13px}
.tabs button.active{color:var(--text);border-color:var(--accent)}
.muted{color:var(--muted);font-size:13px}
body.light{--bg:#eef2f8;--surface:#ffffff;--surface2:#e6ecf5;
 --border:#ccd6e5;--text:#16233a;--muted:#5b6b84}
</style></head><body>
<div id="alarmBanner">&#128680; ALARM: <span id="alarmText"></span></div>
<div class="header">
  <h1>&#128737; Veri<span>-Drive</span></h1>
  <div class="row">
    <button class="btn small gray" id="soundBtn" onclick="toggleSound()">&#128266; Alarm sound: ON</button>
    <button class="btn small gray" id="themeBtn" onclick="toggleTheme()">&#127769; Theme</button>
    {% if session.get('user') %}<button class="btn small gray" onclick="logout()">&#128274; Sign out ({{ session.get('user') }})</button>{% endif %}
    <span class="camstat" id="camstat">camera: ...</span>
  </div>
</div>
<div class="layout">
<nav class="nav">
  <a href="/" class="{% if page=='dash' %}active{% endif %}">&#128202; Dashboard</a>
  <a href="/live" class="{% if page=='live' %}active{% endif %}">&#128249; Live Gate</a>
  <a href="/logs" class="{% if page=='logs' %}active{% endif %}">&#128218; Logs</a>
  <a href="/reports" class="{% if page=='reports' %}active{% endif %}">&#128197; Weekly Reports</a>
  <a href="/register" class="{% if page=='register' %}active{% endif %}">&#10133; Register</a>
</nav>
<div class="container">{{ content|safe }}</div>
</div>
<script>
var soundOn = true, lastAlarmId = 0, alarmTimer = null;
function toggleSound(){
  soundOn = !soundOn;
  document.getElementById('soundBtn').innerHTML = soundOn
    ? '&#128266; Alarm sound: ON' : '&#128263; Alarm sound: OFF';
}
function beep(){
  try{
    var ctx = new (window.AudioContext||window.webkitAudioContext)();
    var t = ctx.currentTime;
    for(var i=0;i<3;i++){
      var o = ctx.createOscillator(), g = ctx.createGain();
      o.type='square'; o.frequency.value = 880;
      g.gain.setValueAtTime(.25, t+i*.4);
      g.gain.exponentialRampToValueAtTime(.001, t+i*.4+.3);
      o.connect(g); g.connect(ctx.destination);
      o.start(t+i*.4); o.stop(t+i*.4+.32);
    }
  }catch(e){}
}
function fmtTime(iso){
  if(!iso) return '-';
  var d = new Date(iso);
  return d.toLocaleDateString([], {day:'2-digit',month:'short'}) + ' ' +
         d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
}
function applyStatus(st){
    document.getElementById('camstat').textContent =
      st.camera_ok ? 'camera: online' : 'camera: OFFLINE';
    if(st.last_alarm && st.last_alarm.id > lastAlarmId){
      if(lastAlarmId > 0){
        var a = st.last_alarm;
        document.getElementById('alarmText').textContent =
          a.kind.replace('_',' ') + (a.plate ? ' - plate ' + a.plate : '') +
          (a.driver ? ' - driver ' + a.driver : '') + ' @ ' + a.time;
        document.getElementById('alarmBanner').style.display = 'block';
        if(soundOn) beep();
        clearTimeout(alarmTimer);
        alarmTimer = setTimeout(function(){
          document.getElementById('alarmBanner').style.display='none';
        }, 15000);
      }
      lastAlarmId = st.last_alarm.id;
    }
}
function pollStatus(){
  fetch('/api/status').then(function(r){return r.json()})
    .then(applyStatus).catch(function(){});
}
// Prefer Server-Sent Events (push, fewer requests); fall back to polling
// automatically if the stream is unavailable.
if(window.EventSource){
  var es = new EventSource('/api/events');
  es.onmessage = function(e){
    try{ applyStatus(JSON.parse(e.data)); }catch(err){}
  };
  es.onerror = function(){
    es.close();
    setInterval(pollStatus, 2000);
  };
} else {
  setInterval(pollStatus, 2000);
}
pollStatus();
function toggleTheme(){
  document.body.classList.toggle('light');
  var light = document.body.classList.contains('light');
  try{ localStorage.setItem('vdTheme', light ? 'light' : 'dark'); }catch(e){}
  document.getElementById('themeBtn').innerHTML =
    light ? '&#9728; Theme' : '&#127769; Theme';
}
try{
  if(localStorage.getItem('vdTheme') === 'light'){
    document.body.classList.add('light');
    document.getElementById('themeBtn').innerHTML = '&#9728; Theme';
  }
}catch(e){}
function logout(){
  fetch('/api/logout',{method:'POST'}).then(function(){ location.href='/'; });
}
{{ script|safe }}
</script>
</body></html>
"""

_DASH = """
<div class="grid stats" style="margin-bottom:20px">
  <div class="stat"><div class="lbl">Registered Drivers</div>
    <div class="val g" id="st-drivers">-</div></div>
  <div class="stat"><div class="lbl">Registered Cars</div>
    <div class="val b" id="st-cars">-</div></div>
  <div class="stat"><div class="lbl">Trips Today</div>
    <div class="val b" id="st-trips">-</div></div>
  <div class="stat"><div class="lbl">Cars Inside Now</div>
    <div class="val o" id="st-active">-</div></div>
  <div class="stat"><div class="lbl">Alarms Today</div>
    <div class="val r" id="st-alarms">-</div></div>
</div>
<div class="grid two">
  <div class="card"><h2>&#128663; Recent Gate Activity</h2>
    <div style="overflow-x:auto"><table>
      <thead><tr><th>Plate</th><th>Driver</th><th>Type</th><th>Time</th><th>Duration</th></tr></thead>
      <tbody id="recentTrips"></tbody></table></div>
  </div>
  <div class="card"><h2>&#128680; Recent Alarms</h2>
    <div style="overflow-x:auto"><table>
      <thead><tr><th>Type</th><th>Plate</th><th>Driver</th><th>Time</th></tr></thead>
      <tbody id="recentAlarms"></tbody></table></div>
  </div>
</div>
"""

_DASH_JS = """
function refreshDash(){
  fetch('/api/status').then(function(r){return r.json()}).then(function(st){
    var s = st.stats;
    document.getElementById('st-drivers').textContent = s.registered_drivers;
    document.getElementById('st-cars').textContent = s.registered_vehicles;
    document.getElementById('st-trips').textContent = s.trips_today;
    document.getElementById('st-active').textContent = s.active_vehicles;
    document.getElementById('st-alarms').textContent = s.alarms_today;
  }).catch(function(){});
  fetch('/api/gate_log').then(function(r){return r.json()}).then(function(rows){
    var html = '';
    rows.slice(0,8).forEach(function(t){
      html += '<tr><td><strong>'+t.plate+'</strong></td><td>'+
        (t.driver||'Unknown')+'</td><td>'+
        (t.exit_time ? '<span class="badge b-exit">EXIT</span>'
                     : '<span class="badge b-entry">ENTRY</span>')+'</td><td>'+
        fmtTime(t.entry_time)+'</td><td>'+
        (t.duration_minutes ? t.duration_minutes+' min' : 'in progress')+
        '</td></tr>';
    });
    document.getElementById('recentTrips').innerHTML =
      html || '<tr><td colspan="5" class="muted">No trips logged yet.</td></tr>';
  }).catch(function(){});
  fetch('/api/alarms').then(function(r){return r.json()}).then(function(rows){
    var html = '';
    rows.slice(0,8).forEach(function(a){
      html += '<tr><td><span class="badge b-alarm">'+a.kind.replace('_',' ')+
        '</span></td><td>'+(a.plate||'-')+'</td><td>'+(a.driver||'-')+
        '</td><td>'+fmtTime(a.time)+'</td></tr>';
    });
    document.getElementById('recentAlarms').innerHTML =
      html || '<tr><td colspan="4" class="muted">No alarms. All clear.</td></tr>';
  }).catch(function(){});
}
setInterval(refreshDash, 4000); refreshDash();
"""

_LIVE = """
<div class="grid two">
  <div class="card">
    <h2>&#128249; Live Gate Camera</h2>
    <img class="livefeed" src="/api/stream" alt="live gate feed">
    <p class="muted" style="margin-top:8px">
      Automatic scan every {{ scan_gap }}s. Registered cars are logged
      automatically; anything unregistered raises an alarm.</p>
  </div>
  <div class="card">
    <h2>&#9889; Current Scan</h2>
    <div id="scanPanel" class="muted">Waiting for first scan...</div>
    <div style="margin-top:14px" class="row">
      <button class="btn" onclick="scanNow()">&#128269; Scan Now</button>
      <label class="fbtn" for="gateUpload">&#128247; Check a Photo</label>
      <input type="file" id="gateUpload" accept="image/*" onchange="gatePhoto(this)">
    </div>
    <div class="msg" id="liveMsg"></div>
    <h2 style="margin-top:20px">&#128220; Last Events</h2>
    <div style="overflow-x:auto"><table>
      <thead><tr><th>Plate</th><th>Driver</th><th>Type</th><th>Time</th></tr></thead>
      <tbody id="liveEvents"></tbody></table></div>
  </div>
</div>
"""

_LIVE_JS = """
function showLiveMsg(ok, text){
  var m = document.getElementById('liveMsg');
  m.className = 'msg ' + (ok ? 'ok' : 'err');
  m.textContent = text;
}
function renderOutcome(o){
  if(!o){ return; }
  var chip = 'b-muted', label = 'Idle';
  if(o.status==='alarm'){ chip='b-alarm'; label='ALARM'; }
  else if(o.status==='authorized'){ chip='b-ok'; label='AUTHORIZED'; }
  else if(o.status==='no-plate'){ chip='b-entry'; label='NO PLATE READ'; }
  var html =
    '<div style="margin-bottom:8px"><span class="badge '+chip+'">'+label+
    '</span> <span class="muted">'+o.time+'</span></div>' +
    '<table style="font-size:14px">' +
    '<tr><td class="muted">Plate read</td><td><strong>'+
      (o.plate_text||'none')+'</strong></td></tr>' +
    '<tr><td class="muted">Registered car</td><td>'+
      (o.plate ? '<span class="badge b-ok">'+o.plate+'</span>' : '-')+'</td></tr>' +
    '<tr><td class="muted">Driver</td><td>'+
      (o.driver ? '<span class="badge b-ok">'+o.driver+'</span>' :
       (o.faces_seen ? '<span class="badge b-alarm">Unknown</span>' : 'none seen'))+
    '</td></tr>'+
    (o.occupants && o.occupants.length > 1 ?
      '<tr><td class="muted">Occupants</td><td>'+o.occupants.join(', ')+
      '</td></tr>' : '')+
    '</table>' +
    '<p style="margin-top:10px">'+o.message+'</p>';
  if(o.event && o.event.inferred){
    html += '<p class="muted" style="margin-top:6px">Direction inferred '+
            'from trip state - a single gate camera cannot see direction '+
            'of travel.</p>';
  }
  if(o.snapshot){
    html += '<img class="preview" style="max-width:280px" src="/snapshots/'+
            o.snapshot+'">';
  }
  document.getElementById('scanPanel').innerHTML = html;
}
function refreshLive(){
  fetch('/api/status').then(function(r){return r.json()}).then(function(st){
    renderOutcome(st.outcome);
  }).catch(function(){});
  fetch('/api/gate_log').then(function(r){return r.json()}).then(function(rows){
    var html='';
    rows.slice(0,6).forEach(function(t){
      html += '<tr><td><strong>'+t.plate+'</strong></td><td>'+
        (t.driver||'Unknown')+'</td><td>'+
        (t.exit_time?'<span class="badge b-exit">EXIT</span>'
                    :'<span class="badge b-entry">ENTRY</span>')+'</td><td>'+
        fmtTime(t.entry_time)+'</td></tr>';
    });
    document.getElementById('liveEvents').innerHTML =
      html || '<tr><td colspan="4" class="muted">Nothing yet.</td></tr>';
  }).catch(function(){});
}
function scanNow(){
  showLiveMsg(true, 'Scanning...');
  fetch('/api/scan_now',{method:'POST'}).then(function(r){return r.json()})
  .then(function(o){ renderOutcome(o); showLiveMsg(true, o.message); })
  .catch(function(e){ showLiveMsg(false, 'Scan failed: '+e.message); });
}
function gatePhoto(input){
  var f = input.files[0]; if(!f) return;
  showLiveMsg(true, 'Processing photo...');
  var fd = new FormData(); fd.append('photo', f);
  fetch('/api/gate_process',{method:'POST', body:fd})
  .then(function(r){return r.json()}).then(function(o){
    if(o.error){ showLiveMsg(false, o.error); return; }
    renderOutcome(o); showLiveMsg(true, o.message);
  }).catch(function(e){ showLiveMsg(false, 'Error: '+e.message); });
  input.value='';
}
setInterval(refreshLive, 3000); refreshLive();
"""

_LOGS = """
<div class="card">
  <h2>&#128218; System Logs</h2>
  <div class="tabs">
    <button class="active" onclick="showTab('gate',this)">Gate Log</button>
    <button onclick="showTab('alarms',this)">Alarms</button>
    <button onclick="showTab('drivers',this)">Driver Logs</button>
    <button onclick="showTab('dcar',this)">Driver-Car Logs</button>
  </div>
  <div class="row" style="margin-bottom:12px">
    <input id="logSearch" placeholder="Search plate / driver / details..."
      style="max-width:280px" onkeydown="if(event.key==='Enter')doSearch()">
    <button class="btn gray" onclick="doSearch()">&#128269; Search</button>
    <button class="btn gray" onclick="exportCsv()">&#11015; CSV Export</button>
    <button class="btn red" onclick="clearRecords()">&#128465; Delete</button>
    <span class="muted" id="logCount"></span>
  </div>
  <div style="overflow-x:auto"><table id="logTable"></table></div>
  <div class="row" style="margin-top:12px;justify-content:center">
    <button class="btn gray small" onclick="changePage(-1)">&#9664; Prev</button>
    <span class="muted" id="pageInfo"></span>
    <button class="btn gray small" onclick="changePage(1)">Next &#9654;</button>
  </div>
</div>
"""

_LOGS_JS = """
var curTab = 'gate', curPage = 1, curQ = '';
function showTab(t, btn){
  curTab = t; curPage = 1; curQ = '';
  document.getElementById('logSearch').value = '';
  var btns = document.querySelectorAll('.tabs button');
  for(var i=0;i<btns.length;i++) btns[i].className='';
  btn.className='active';
  refreshLogs();
}
function doSearch(){
  curQ = document.getElementById('logSearch').value.trim();
  curPage = 1;
  refreshLogs();
}
function changePage(d){
  curPage = Math.max(1, curPage + d);
  refreshLogs();
}
function exportCsv(){
  window.location.href = '/api/export/' + curTab + '.csv' +
    (curQ ? '?q=' + encodeURIComponent(curQ) : '');
}
function inferredBadge(t){
  return t.inferred
    ? ' <span class="badge b-muted" title="Direction inferred from trip '+
      'state - a single gate camera cannot see direction of travel">'+
      'inferred</span>'
    : '';
}
function emptyRow(cols, text){
  return '<tr><td colspan="'+cols+'" class="muted">'+text+'</td></tr>';
}
function refreshLogs(){
  var tbl = document.getElementById('logTable');
  var url = '/api/logs/' + curTab + '?page=' + curPage +
            (curQ ? '&q=' + encodeURIComponent(curQ) : '');
  fetch(url).then(function(r){return r.json()}).then(function(res){
    var rows = res.rows, h = '';
    document.getElementById('logCount').textContent = res.total + ' record(s)';
    document.getElementById('pageInfo').textContent =
      'Page ' + res.page + ' / ' + res.pages;
    if(curTab==='gate'){
      h = '<thead><tr><th>#</th><th>Plate</th><th>Driver</th><th>Entry</th>'+
        '<th>Exit</th><th>Duration</th><th>Snapshot</th></tr></thead><tbody>';
      rows.forEach(function(t){
        h += '<tr><td>'+t.id+'</td><td><strong>'+t.plate+'</strong></td><td>'+
          (t.driver||'Unknown')+'</td><td>'+fmtTime(t.entry_time)+
          inferredBadge(t)+'</td><td>'+
          (t.exit_time?fmtTime(t.exit_time):'<span class="badge b-entry">inside</span>')+
          '</td><td>'+(t.duration_minutes?t.duration_minutes+' min':'-')+'</td><td>'+
          (t.snapshot?'<a href="/snapshots/'+t.snapshot+'" target="_blank">view</a>':'-')+
          '</td></tr>';
      });
      if(!rows.length) h += emptyRow(7,'No trips found.');
      tbl.innerHTML = h+'</tbody>';
    } else if(curTab==='alarms'){
      h = '<thead><tr><th>#</th><th>Type</th><th>Plate</th><th>Driver</th>'+
        '<th>Details</th><th>Time</th><th>Snapshot</th></tr></thead><tbody>';
      rows.forEach(function(a){
        h += '<tr><td>'+a.id+'</td><td><span class="badge b-alarm">'+
          a.kind.replace('_',' ')+'</span></td><td>'+(a.plate||'-')+'</td><td>'+
          (a.driver||'-')+'</td><td class="muted">'+(a.details||'')+'</td><td>'+
          fmtTime(a.time)+'</td><td>'+
          (a.snapshot?'<a href="/snapshots/'+a.snapshot+'" target="_blank">view</a>':'-')+
          '</td></tr>';
      });
      if(!rows.length) h += emptyRow(7,'No alarms found.');
      tbl.innerHTML = h+'</tbody>';
    } else if(curTab==='drivers'){
      h = '<thead><tr><th>Driver</th><th>Total Trips</th><th>Cars Driven</th>'+
        '<th>Total Time</th><th>First Trip</th><th>Last Seen</th></tr></thead><tbody>';
      rows.forEach(function(d){
        h += '<tr><td><strong>'+d.driver+'</strong></td><td>'+d.total_trips+
          '</td><td>'+d.cars_used+'</td><td>'+d.total_minutes+' min</td><td>'+
          fmtTime(d.first_trip)+'</td><td>'+fmtTime(d.last_seen)+'</td></tr>';
      });
      if(!rows.length) h += emptyRow(6,'No driver activity found.');
      tbl.innerHTML = h+'</tbody>';
    } else {
      h = '<thead><tr><th>Driver</th><th>Car</th><th>Times Driven</th>'+
        '<th>Total Time</th><th>Last Used</th></tr></thead><tbody>';
      rows.forEach(function(d){
        h += '<tr><td><strong>'+d.driver+'</strong></td><td><strong>'+d.plate+
          '</strong></td><td>'+d.times_driven+'</td><td>'+d.total_minutes+
          ' min</td><td>'+fmtTime(d.last_used)+'</td></tr>';
      });
      if(!rows.length) h += emptyRow(5,'No driver-car activity found.');
      tbl.innerHTML = h+'</tbody>';
    }
  }).catch(function(){});
}
function clearRecords(){
  var labels = {gate:'ALL gate trips', alarms:'ALL alarm records',
    drivers:'ALL gate trips (driver summaries are computed from trips)',
    dcar:'ALL gate trips (driver-car summaries are computed from trips)'};
  if(!confirm('Delete ' + (labels[curTab]||'these records') +
    '? This cannot be undone. A database backup is saved first.')) return;
  fetch('/api/clear/' + curTab, {method:'POST', body:new FormData()})
  .then(function(r){
    if(r.status===401){ alert('Please sign in to delete records.');
      location.href='/login'; return null; }
    return r.json();
  })
  .then(function(res){ if(res){ alert(res.message); curPage=1; refreshLogs(); } })
  .catch(function(e){ alert('Delete failed: '+e.message); });
}
setInterval(refreshLogs, 5000); refreshLogs();
"""

_REGISTER = """
<div class="grid two">
  <div class="card">
    <h2>&#128100; Register Driver</h2>
    <p class="muted" style="margin-bottom:12px">Enter the name, take a photo
      with the gate camera (or upload one), then register.
      <strong>Tip:</strong> add 2-3 photos per driver (different angles /
      lighting) by registering the same name again - recognition gets
      noticeably more reliable.</p>
    <div style="display:grid;gap:10px">
      <input id="drvName" placeholder="Driver name (e.g. Ali)" maxlength="40">
      <input id="drvPhone" placeholder="Phone (optional)" maxlength="20">
      <div class="row">
        <button class="btn" onclick="captureDriver()">&#128247; Take Photo</button>
        <label class="fbtn" for="drvFile">&#128193; Upload Photo</label>
        <input type="file" id="drvFile" accept="image/*">
      </div>
      <img id="drvPreview" class="preview" style="display:none" alt="driver preview">
      <button class="btn" style="background:var(--ok);color:#000"
        onclick="submitDriver()">&#9989; Register Driver</button>
      <div class="msg" id="drvMsg"></div>
    </div>
    <h2 style="margin-top:22px">Registered Drivers</h2>
    <div style="overflow-x:auto"><table>
      <thead><tr><th>Name</th><th>Phone</th><th>Trips</th><th></th></tr></thead>
      <tbody id="drvList"></tbody></table></div>
  </div>
  <div class="card">
    <h2>&#128663; Register Car (Plate)</h2>
    <p class="muted" style="margin-bottom:12px">Take a photo of the car/plate.
      Veri-Drive reads the plate automatically - correct it if needed.</p>
    <div style="display:grid;gap:10px">
      <div class="row">
        <button class="btn" onclick="captureCar()">&#128247; Take Photo</button>
        <label class="fbtn" for="carFile">&#128193; Upload Photo</label>
        <input type="file" id="carFile" accept="image/*">
      </div>
      <img id="carPreview" class="preview" style="display:none" alt="car preview">
      <input id="carPlate" placeholder="Plate number (e.g. LEB-1234)" maxlength="12"
        style="text-transform:uppercase">
      <input id="carLabel" placeholder="Car label (optional, e.g. Car 1)">
      <button class="btn" style="background:var(--ok);color:#000"
        onclick="submitCar()">&#9989; Register Car</button>
      <div class="msg" id="carMsg"></div>
    </div>
    <h2 style="margin-top:22px">Registered Cars</h2>
    <div style="overflow-x:auto"><table>
      <thead><tr><th>Plate</th><th>Label</th><th>Trips</th><th></th></tr></thead>
      <tbody id="carList"></tbody></table></div>
  </div>
</div>
"""

_REG_JS = """
var drvBlob = null, carBlob = null;
function msg(id, ok, text){
  var m = document.getElementById(id);
  m.className = 'msg ' + (ok ? 'ok' : 'err');
  m.textContent = text;
}
function captureDriver(){
  msg('drvMsg', true, 'Capturing photo...');
  fetch('/api/capture_frame').then(function(r){
    if(!r.ok) throw new Error('camera unavailable');
    return r.blob();
  }).then(function(b){
    drvBlob = b;
    var img = document.getElementById('drvPreview');
    img.src = URL.createObjectURL(b); img.style.display='block';
    msg('drvMsg', true, 'Photo captured. Check it, then register.');
  }).catch(function(e){ msg('drvMsg', false, 'Capture failed: '+e.message); });
}
document.getElementById('drvFile').addEventListener('change', function(){
  var f = this.files[0]; if(!f) return;
  drvBlob = f;
  var img = document.getElementById('drvPreview');
  img.src = URL.createObjectURL(f); img.style.display='block';
  msg('drvMsg', true, 'Photo loaded. Check it, then register.');
});
function submitDriver(){
  var name = document.getElementById('drvName').value.trim();
  if(!name){ msg('drvMsg', false, 'Please enter the driver name.'); return; }
  if(!drvBlob){ msg('drvMsg', false, 'Please capture or upload a photo first.'); return; }
  msg('drvMsg', true, 'Registering...');
  var fd = new FormData();
  fd.append('name', name);
  fd.append('phone', document.getElementById('drvPhone').value.trim());
  fd.append('photo', drvBlob, 'photo.jpg');
  fetch('/api/register/driver',{method:'POST', body:fd})
  .then(function(r){return r.json()}).then(function(res){
    msg('drvMsg', res.ok, res.message);
    if(res.ok){
      drvBlob = null;
      document.getElementById('drvPreview').style.display='none';
      document.getElementById('drvName').value='';
      document.getElementById('drvPhone').value='';
      refreshReg();
    }
  }).catch(function(e){ msg('drvMsg', false, 'Error: '+e.message); });
}
function captureCar(){
  msg('carMsg', true, 'Capturing photo and reading plate...');
  fetch('/api/capture_frame').then(function(r){
    if(!r.ok) throw new Error('camera unavailable');
    return r.blob();
  }).then(function(b){
    carBlob = b;
    var img = document.getElementById('carPreview');
    img.src = URL.createObjectURL(b); img.style.display='block';
    return ocrBlob(b);
  }).catch(function(e){ msg('carMsg', false, 'Capture failed: '+e.message); });
}
document.getElementById('carFile').addEventListener('change', function(){
  var f = this.files[0]; if(!f) return;
  carBlob = f;
  var img = document.getElementById('carPreview');
  img.src = URL.createObjectURL(f); img.style.display='block';
  msg('carMsg', true, 'Reading plate...');
  ocrBlob(f);
});
function ocrBlob(b){
  var fd = new FormData(); fd.append('photo', b);
  return fetch('/api/ocr_photo',{method:'POST', body:fd})
  .then(function(r){return r.json()}).then(function(res){
    if(res.plate){
      document.getElementById('carPlate').value = res.plate;
      msg('carMsg', true, 'Plate read as "'+res.plate+'" - confirm or correct it.');
    } else {
      msg('carMsg', true, 'Could not auto-read the plate - type it manually.');
    }
  }).catch(function(){ msg('carMsg', true, 'Type the plate manually.'); });
}
function submitCar(){
  if(!carBlob){ msg('carMsg', false, 'Please capture or upload a photo first.'); return; }
  msg('carMsg', true, 'Registering...');
  var fd = new FormData();
  fd.append('plate', document.getElementById('carPlate').value.trim());
  fd.append('label', document.getElementById('carLabel').value.trim());
  fd.append('photo', carBlob, 'photo.jpg');
  fetch('/api/register/vehicle',{method:'POST', body:fd})
  .then(function(r){return r.json()}).then(function(res){
    msg('carMsg', res.ok, res.message);
    if(res.ok){
      carBlob = null;
      document.getElementById('carPreview').style.display='none';
      document.getElementById('carPlate').value='';
      document.getElementById('carLabel').value='';
      refreshReg();
    }
  }).catch(function(e){ msg('carMsg', false, 'Error: '+e.message); });
}
function refreshReg(){
  fetch('/api/drivers').then(function(r){return r.json()}).then(function(rows){
    var h='';
    rows.forEach(function(d){
      h += '<tr><td><strong>'+d.name+'</strong></td><td>'+(d.phone||'-')+
        '</td><td>'+d.trips+'</td><td>'+
        '<button class="btn small red" data-n="'+d.name+'" '+
        'onclick="delDriver(this.dataset.n)">Delete</button></td></tr>';
    });
    document.getElementById('drvList').innerHTML =
      h || '<tr><td colspan="4" class="muted">No drivers registered yet.</td></tr>';
  });
  fetch('/api/vehicles').then(function(r){return r.json()}).then(function(rows){
    var h='';
    rows.forEach(function(v){
      h += '<tr><td><strong>'+v.plate+'</strong></td><td>'+(v.label||'-')+
        '</td><td>'+v.trips+'</td><td>'+
        '<button class="btn small red" data-n="'+v.plate+'" '+
        'onclick="delVehicle(this.dataset.n)">Delete</button></td></tr>';
    });
    document.getElementById('carList').innerHTML =
      h || '<tr><td colspan="4" class="muted">No cars registered yet.</td></tr>';
  });
}
function delDriver(name){
  if(!confirm('Remove driver "'+name+'"? Their history stays in the logs.')) return;
  var fd = new FormData(); fd.append('name', name);
  fetch('/api/delete/driver',{method:'POST', body:fd})
  .then(function(r){return r.json()}).then(function(res){
    msg('drvMsg', res.ok, res.message); refreshReg();
  });
}
function delVehicle(plate){
  if(!confirm('Remove vehicle "'+plate+'"? Its history stays in the logs.')) return;
  var fd = new FormData(); fd.append('plate', plate);
  fetch('/api/delete/vehicle',{method:'POST', body:fd})
  .then(function(r){return r.json()}).then(function(res){
    msg('carMsg', res.ok, res.message); refreshReg();
  });
}
refreshReg();
"""

_REPORTS = """
<div class="card">
  <h2>&#128197; Weekly Report</h2>
  <div class="row">
    <label class="muted">Week ending on:</label>
    <input type="date" id="repDate" style="width:auto">
    <button class="btn" onclick="genReport()">Generate Report</button>
    <button class="btn gray" onclick="window.print()">&#128424; Print</button>
  </div>
</div>
<div id="reportOut"></div>
"""

_REP_JS = """
function tableHTML(head, rows, empty){
  if(!rows.length) return '<p class="muted">'+empty+'</p>';
  var h = '<table><thead><tr>';
  head.forEach(function(c){ h += '<th>'+c+'</th>'; });
  h += '</tr></thead><tbody>';
  rows.forEach(function(r){
    h += '<tr>'; r.forEach(function(c){ h += '<td>'+c+'</td>'; }); h += '</tr>';
  });
  return h + '</tbody></table>';
}
function genReport(){
  var end = document.getElementById('repDate').value;
  fetch('/api/report' + (end ? '?end='+end : ''))
  .then(function(r){return r.json()}).then(function(rep){
    var t = rep.totals;
    var h =
    '<div class="grid stats" style="margin-bottom:20px">'+
      '<div class="stat"><div class="lbl">Trips</div><div class="val b">'+t.trips+'</div></div>'+
      '<div class="stat"><div class="lbl">Active Drivers</div><div class="val g">'+t.drivers_active+'</div></div>'+
      '<div class="stat"><div class="lbl">Active Cars</div><div class="val b">'+t.vehicles_active+'</div></div>'+
      '<div class="stat"><div class="lbl">Alarms</div><div class="val r">'+t.alarms+'</div></div>'+
    '</div>'+
    '<p class="muted" style="margin-bottom:16px">Week: '+rep.start+' &rarr; '+rep.end+'</p>'+

    '<div class="card"><h2>&#9201; Log Times (entry / exit)</h2>'+
      tableHTML(['Plate','Driver','Entry','Exit','Duration'],
        rep.trips.map(function(x){ return [
          '<strong>'+x.plate+'</strong>', x.driver||'Unknown',
          fmtTime(x.entry_time),
          x.exit_time?fmtTime(x.exit_time):'in progress',
          x.duration_minutes?x.duration_minutes+' min':'-']; }),
        'No trips this week.')+'</div>'+

    '<div class="card"><h2>&#128100; Driver Logs</h2>'+
      tableHTML(['Driver','Trips','Cars Driven','Total Time'],
        rep.driver_logs.map(function(x){ return [
          '<strong>'+x.driver+'</strong>', x.total_trips, x.cars_used,
          x.total_minutes+' min']; }),
        'No driver activity this week.')+'</div>'+

    '<div class="card"><h2>&#128100;&#128663; Driver-Car Logs</h2>'+
      tableHTML(['Driver','Car','Times Driven','Total Time'],
        rep.driver_vehicle.map(function(x){ return [
          '<strong>'+x.driver+'</strong>', '<strong>'+x.plate+'</strong>',
          x.times_driven, x.total_minutes+' min']; }),
        'No driver-car activity this week.')+'</div>'+

    '<div class="card"><h2>&#128663; Car Logs</h2>'+
      tableHTML(['Car','Trips','Different Drivers','Total Time'],
        rep.vehicle_logs.map(function(x){ return [
          '<strong>'+x.plate+'</strong>', x.total_trips, x.drivers_used,
          x.total_minutes+' min']; }),
        'No car activity this week.')+'</div>'+

    '<div class="card"><h2>&#128680; Alarms This Week</h2>'+
      tableHTML(['Type','Plate','Driver','Details','Time'],
        rep.alarms.map(function(x){ return [
          '<span class="badge b-alarm">'+x.kind.replace('_',' ')+'</span>',
          x.plate||'-', x.driver||'-', x.details||'', fmtTime(x.time)]; }),
        'No alarms this week. All clear.')+'</div>';
    document.getElementById('reportOut').innerHTML = h;
  }).catch(function(e){
    document.getElementById('reportOut').innerHTML =
      '<div class="card msg err">Could not generate report: '+e.message+'</div>';
  });
}
(function(){ var d=new Date();
  document.getElementById('repDate').value = d.getFullYear()+'-'+
    String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
})();
genReport();
"""


_LOGIN = """
<div style="max-width:380px;margin:70px auto">
  <div class="card">
    <h2>&#128274; Veri-Drive Login</h2>
    <p class="muted" style="margin:8px 0 14px">Required to register or delete
      drivers and cars. Viewing pages stay open.</p>
    <div style="display:grid;gap:10px">
      <input id="loginUser" placeholder="Username" autocomplete="username">
      <input id="loginPass" type="password" placeholder="Password"
        autocomplete="current-password">
      <button class="btn" onclick="doLogin()">Sign in</button>
      <div class="msg" id="loginMsg"></div>
    </div>
  </div>
</div>
"""

_LOGIN_JS = """
function doLogin(){
  var fd = new FormData();
  fd.append('username', document.getElementById('loginUser').value.trim());
  fd.append('password', document.getElementById('loginPass').value);
  var m = document.getElementById('loginMsg');
  m.className = 'msg ok'; m.textContent = 'Signing in...';
  fetch('/api/login',{method:'POST',body:fd})
  .then(function(r){return r.json()}).then(function(res){
    if(res.ok){ location.href = res.next || '/register'; }
    else { m.className='msg err'; m.textContent=res.message; }
  }).catch(function(e){ m.className='msg err'; m.textContent='Error: '+e.message; });
}
document.getElementById('loginPass').addEventListener('keydown', function(e){
  if(e.key === 'Enter') doLogin();
});
"""


# ===================================================================
# PAGE ROUTES
# ===================================================================
@app.route("/")
def page_dashboard():
    return render_template_string(_BASE, page="dash",
                                  content=_DASH, script=_DASH_JS)


@app.route("/live")
def page_live():
    content = _LIVE.replace("{{ scan_gap }}", str(config.SCAN_COOLDOWN))
    return render_template_string(_BASE, page="live",
                                  content=content, script=_LIVE_JS)


@app.route("/logs")
def page_logs():
    return render_template_string(_BASE, page="logs",
                                  content=_LOGS, script=_LOGS_JS)


@app.route("/register")
@login_required_page
def page_register():
    return render_template_string(_BASE, page="register",
                                  content=_REGISTER, script=_REG_JS)


@app.route("/login")
def page_login():
    if _is_logged_in():
        return redirect("/register")
    return render_template_string(_BASE, page="login",
                                  content=_LOGIN, script=_LOGIN_JS)


@app.route("/reports")
def page_reports():
    return render_template_string(_BASE, page="reports",
                                  content=_REPORTS, script=_REP_JS)


@app.route("/snapshots/<path:filename>")
def snapshot(filename):
    return send_from_directory(config.CAPTURES_DIR, filename)


@app.route("/icon.png")
def app_icon():
    return send_file(ensure_app_icon(), mimetype="image/png")


@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name": "Veri-Drive Gate Security",
        "short_name": "Veri-Drive",
        "description": "Gate security: ANPR + driver recognition + trip logs",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": "#0b1220",
        "theme_color": "#121c2e",
        "icons": [{"src": "/icon.png", "sizes": "512x512",
                   "type": "image/png", "purpose": "any"}],
    })

# ===================================================================
# API
# ===================================================================
def _status_payload():
    """Shared status snapshot used by /api/status and the SSE stream."""
    outcome = None
    if scanner.last_outcome:
        outcome = {k: v for k, v in scanner.last_outcome.items()
                   if k != "_faces"}
    return {
        "camera_ok": camera.ok,
        "camera_error": camera.error,
        "barrier": barrier_state(),
        "last_scan_time": (scanner.last_scan_time.isoformat(timespec="seconds")
                           if scanner.last_scan_time else None),
        "outcome": outcome,
        "last_alarm": db.get_last_alarm(),
        "stats": db.get_summary_stats(),
    }


@app.route("/api/status")
def api_status():
    return jsonify(_status_payload())


@app.route("/api/events")
def api_events():
    """Server-Sent Events: pushes the status snapshot whenever something
    changes (new scan result, alarm, barrier move), so browsers no longer
    need to poll /api/status every 2s."""
    def generate():
        last_stamp = None
        while True:
            st = _status_payload()
            alarm = st.get("last_alarm") or {}
            stamp = (st.get("last_scan_time"), alarm.get("id"),
                     st.get("barrier", {}).get("open"))
            if stamp != last_stamp:
                yield f"data: {json.dumps(st)}\n\n"
                last_stamp = stamp
            time.sleep(0.5)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ---------- auth ----------
@app.route("/api/login", methods=["POST"])
def api_login():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    if (username == config.ADMIN_USERNAME
            and check_password_hash(_ADMIN_PASS_HASH, password)):
        session.permanent = True
        session["user"] = username
        return jsonify({"ok": True, "next": "/register"})
    return jsonify({"ok": False, "message": "Wrong username or password."}), 401


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/stream")
def api_stream():
    def generate():
        while True:
            frame, _err = camera.get_frame(max_age=3.0)
            if frame is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                text = ("CAMERA OFFLINE" if camera.error
                        else "waiting for camera...")
                cv2.putText(blank, text, (140, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                buf = encode_jpg(blank, 60)
            else:
                buf = encode_jpg(annotate(frame, scanner.last_outcome), 70)
            if buf:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf + b"\r\n")
            time.sleep(0.08)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/capture_frame")
def api_capture_frame():
    frame, err = camera.get_frame()
    if frame is None:
        return jsonify({"error": err or "camera unavailable"}), 503
    buf = encode_jpg(frame, 90)
    return Response(buf, mimetype="image/jpeg")


@app.route("/api/scan_now", methods=["POST"])
def api_scan_now():
    frame, err = camera.get_frame()
    if frame is None:
        return jsonify({"error": err or "camera unavailable"}), 503
    outcome = scanner.handle(frame, source="manual")
    outcome.pop("_faces", None)
    return jsonify(outcome)


@app.route("/api/gate_process", methods=["POST"])
def api_gate_process():
    """Manual check: run the full gate rules on an uploaded photo."""
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded."}), 400
    img, err = decode_upload(request.files["photo"])
    if img is None:
        return jsonify({"error": err}), 400
    outcome = scanner.handle(img, source="upload")
    outcome.pop("_faces", None)
    return jsonify(outcome)


@app.route("/api/ocr_photo", methods=["POST"])
def api_ocr_photo():
    """Read a plate from an uploaded photo (vehicle registration helper)."""
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded."}), 400
    img, err = decode_upload(request.files["photo"])
    if img is None:
        return jsonify({"error": err}), 400
    candidates = anpr.read_plate_candidates(img)
    return jsonify({
        "plate": candidates[0][0] if candidates else None,
        "all": [c[0] for c in candidates[:3]],
    })

# ---------- registration (login required) ----------
@app.route("/api/register/driver", methods=["POST"])
@login_required_api
def api_register_driver():
    if "photo" not in request.files:
        return jsonify({"ok": False,
                        "message": "No photo received - try again."}), 400
    img, err = decode_upload(request.files["photo"])
    if img is None:
        return jsonify({"ok": False, "message": err}), 400
    ok, message = register_driver_from_image(
        request.form.get("name"), request.form.get("phone"), img)
    return jsonify({"ok": ok, "message": message}), (200 if ok else 400)


@app.route("/api/register/vehicle", methods=["POST"])
@login_required_api
def api_register_vehicle():
    if "photo" not in request.files:
        return jsonify({"ok": False,
                        "message": "No photo received - try again."}), 400
    img, err = decode_upload(request.files["photo"])
    if img is None:
        return jsonify({"ok": False, "message": err}), 400
    ok, message = register_vehicle_from_image(
        request.form.get("plate"), request.form.get("label"), img)
    return jsonify({"ok": ok, "message": message}), (200 if ok else 400)


@app.route("/api/delete/driver", methods=["POST"])
@login_required_api
def api_delete_driver():
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "message": "Missing driver name."}), 400
    db.backup_database("before-delete-driver")   # P2: backup before destroy
    ok, message = db.remove_driver(name)
    if ok:
        try:
            face_id.remove_driver(name)
        except Exception as exc:
            print(f"[DEL] face removal issue: {exc}")
        folder = os.path.join(config.DRIVERS_DIR,
                              re.sub(r'[^\w\- ]', '_', name).strip())
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
    return jsonify({"ok": ok, "message": message}), (200 if ok else 404)


@app.route("/api/delete/vehicle", methods=["POST"])
@login_required_api
def api_delete_vehicle():
    plate = (request.form.get("plate") or "").strip().upper()
    if not plate:
        return jsonify({"ok": False, "message": "Missing plate."}), 400
    db.backup_database("before-delete-vehicle")   # P2: backup before destroy
    ok, message = db.remove_vehicle(plate)
    if ok:
        safe = re.sub(r'[^A-Za-z0-9]+', '-', plate).strip('-')
        photo = os.path.join(config.VEHICLES_DIR, f"{safe}.jpg")
        if os.path.exists(photo):
            try:
                os.remove(photo)
            except OSError:
                pass
    return jsonify({"ok": ok, "message": message}), (200 if ok else 404)


@app.route("/api/clear/<kind>", methods=["POST"])
@login_required_api
def api_clear_records(kind):
    """Delete log records for a Logs tab (gate|alarms|drivers|dcar).
    Destructive: login required and a DB backup is taken first. Deliberately
    NOT in the read-only CORS allowlist."""
    before = (request.form.get("before") or "").strip() or None
    db.backup_database(f"before-clear-{kind}")
    n, msg = db.clear_records(kind, before)
    return jsonify({"ok": True, "deleted": n, "message": msg})


# ---------- data ----------
@app.route("/api/drivers")
def api_drivers():
    return jsonify(db.get_drivers())


@app.route("/api/vehicles")
def api_vehicles():
    return jsonify(db.get_vehicles())


@app.route("/api/gate_log")
def api_gate_log():
    return jsonify(db.get_gate_log())


@app.route("/api/alarms")
def api_alarms():
    return jsonify(db.get_alarms())


@app.route("/api/driver_logs")
def api_driver_logs():
    return jsonify(db.get_driver_logs())


@app.route("/api/driver_vehicle_logs")
def api_driver_vehicle_logs():
    return jsonify(db.get_driver_vehicle_logs())


@app.route("/api/report")
def api_report():
    end = request.args.get("end") or None
    if end:
        try:
            datetime.fromisoformat(end)
        except ValueError:
            return jsonify({"error": "Invalid date format."}), 400
    return jsonify(db.get_weekly_report(end))


# ---------- paged logs + CSV export (P3) ----------
@app.route("/api/logs/<kind>")
def api_logs_paged(kind):
    """Paged/searchable log data.  kind: gate | alarms | drivers | dcar."""
    try:
        page = max(1, int(request.args.get("page", 1) or 1))
    except ValueError:
        page = 1
    q = (request.args.get("q") or "").strip()
    size = config.LOG_PAGE_SIZE

    if kind == "gate":
        total = db.count_gate_log(q)
        rows = db.get_gate_log(limit=size, offset=(page - 1) * size, search=q)
    elif kind == "alarms":
        total = db.count_alarms(q)
        rows = db.get_alarms(limit=size, offset=(page - 1) * size, search=q)
    elif kind == "drivers":
        rows = db.get_driver_logs()
        if q:
            ql = q.lower()
            rows = [r for r in rows if ql in str(r.get("driver", "")).lower()]
        total = len(rows)
    elif kind == "dcar":
        rows = db.get_driver_vehicle_logs()
        if q:
            ql = q.lower()
            rows = [r for r in rows
                    if ql in str(r.get("driver", "")).lower()
                    or ql in str(r.get("plate", "")).lower()]
        total = len(rows)
    else:
        return jsonify({"error": "Unknown log kind."}), 404

    return jsonify({
        "rows": rows,
        "total": total,
        "page": page,
        "pages": max(1, (total + size - 1) // size),
    })


_EXPORT_COLUMNS = {
    "gate": ["id", "plate", "driver", "entry_time", "exit_time",
             "duration_minutes", "inferred", "snapshot"],
    "alarms": ["id", "time", "kind", "plate", "driver", "details", "snapshot"],
    "drivers": ["driver", "total_trips", "cars_used", "total_minutes",
                "first_trip", "last_seen"],
    "dcar": ["driver", "plate", "times_driven", "total_minutes", "last_used"],
}


@app.route("/api/export/<kind>.csv")
def api_export_csv(kind):
    """CSV download of a full log table (respects the current search)."""
    if kind not in _EXPORT_COLUMNS:
        return jsonify({"error": "Unknown export kind."}), 404
    q = (request.args.get("q") or "").strip()
    if kind == "gate":
        rows = db.get_gate_log(limit=1000000, search=q)
    elif kind == "alarms":
        rows = db.get_alarms(limit=1000000, search=q)
    elif kind == "drivers":
        rows = db.get_driver_logs()
        if q:
            ql = q.lower()
            rows = [r for r in rows if ql in str(r.get("driver", "")).lower()]
    else:
        rows = db.get_driver_vehicle_logs()
        if q:
            ql = q.lower()
            rows = [r for r in rows
                    if ql in str(r.get("driver", "")).lower()
                    or ql in str(r.get("plate", "")).lower()]
    csv_text = db.rows_to_csv(rows, _EXPORT_COLUMNS[kind])
    stamp = datetime.now().strftime("%Y%m%d")
    return Response(csv_text, mimetype="text/csv",
                    headers={"Content-Disposition":
                             f"attachment; filename=veridrive_{kind}_{stamp}.csv"})


# ===================================================================
# MAIN
# ===================================================================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    port = 5000
    ip = get_local_ip()
    https = (os.path.exists(config.HTTPS_CERT)
             and os.path.exists(config.HTTPS_KEY))
    scheme = "https" if https else "http"
    print("=" * 58)
    print("  VERI-DRIVE  -  Gate Security System")
    print("=" * 58)
    print(f"  PC browser:  {scheme}://127.0.0.1:{port}")
    print(f"  Phone/app:   {scheme}://{ip}:{port}")
    print()
    print(f"  Admin login: user '{config.ADMIN_USERNAME}' - needed for the")
    print("  Register page and all delete/reset operations.")
    if not https:
        print("  NOTE: serving plain HTTP. For use beyond a trusted LAN,")
        print("  add data/cert.pem + data/key.pem (see config.py) or a")
        print("  private tunnel (Tailscale).")
    print()
    print("  iPhone: open the Phone URL in Safari, then")
    print("  Share -> Add to Home Screen  (installs as an app)")
    print("  Android: open the URL, menu -> Add to Home screen")
    print()
    print("  Start by registering your cars and drivers on the")
    print(f"  Register page.  The gate scans every {config.SCAN_COOLDOWN}s.")
    print("=" * 58)
    if https:
        # Flask's built-in server handles TLS directly.
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True,
                ssl_context=(config.HTTPS_CERT, config.HTTPS_KEY))
    elif config.USE_WAITRESS:
        try:
            from waitress import serve
            print("  Serving with waitress (production WSGI server).")
            # threads must cover: MJPEG viewers + SSE viewers + API calls
            serve(app, host="0.0.0.0", port=port, threads=16)
        except ImportError:
            print("  waitress not installed (pip install waitress) -")
            print("  falling back to the Flask development server.")
            app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    else:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
