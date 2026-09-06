# Veri-Drive

**AI dual-factor gate security for fleets, depots, and campuses.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?logo=flask&logoColor=white)
![Vision](https://img.shields.io/badge/Vision-EasyOCR%20%2B%20OpenCV%20YuNet-orange)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-success)

Veri-Drive turns a single ordinary webcam into an intelligent gate guard. It reads each vehicle's **number plate** (ANPR via EasyOCR) and recognizes the **driver's face** (OpenCV YuNet) in real time. The barrier opens **only** when a *registered car* carries a *registered driver*; anything else raises a classified alarm with a captured snapshot. Every entry and exit is logged, searchable, and reportable — with no cloud dependency and no extra hardware.

---

## Table of Contents
- [Why Veri-Drive](#why-veri-drive)
- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [First Run & Default Login](#first-run--default-login)
- [Using the System](#using-the-system)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Deployment](#deployment)
- [Security & Privacy](#security--privacy)
- [Troubleshooting](#troubleshooting)

---

## Why Veri-Drive
Manual gatekeeping is slow, error-prone, and leaves no audit trail. A guard eyeballs a plate, waves through a familiar face, and records nothing. Stolen vehicles slip out, unauthorized drivers slip in, and there is no evidence when something goes wrong.

Veri-Drive closes that gap by **verifying two independent identities — the vehicle and the human — before granting access**, and by logging every event automatically. One camera, one laptop, no subscription.

## How It Works
1. A gate camera scans continuously (every **3 seconds** by default).
2. When a vehicle is present, **EasyOCR** reads the plate and **OpenCV YuNet** detects and recognizes the driver's face.
3. Both are cross-checked against the registered database:
   - **Registered car + registered driver** → the barrier opens and the trip is logged as an **entry** or **exit** (direction is inferred from trip state).
   - **Unknown plate, unknown face, or a mismatched pair** → a classified alarm fires with a snapshot.
4. Alarms are de-duplicated and surfaced live (banner + sound) and in the Logs page.

Alarm classifications: `UNKNOWN_VEHICLE`, `UNKNOWN_DRIVER`, `UNKNOWN_BOTH`.

## Key Features
- **Dual-factor verification** — plate (ANPR) + face; both must be registered.
- **Real-time live gate** — MJPEG camera stream, live status, and an instant alarm banner with sound.
- **Automatic trip logging** — entry/exit with duration; a single camera infers direction from trip state.
- **Searchable logs** — Gate, Alarms, Driver, and Driver-Car tabs; paginated, full-text search, and CSV export.
- **Weekly reports** — trips, active drivers/cars, alarms, and per-driver / per-car summaries for any 7-day window.
- **Self-healing data safety** — a database backup is taken automatically before every destructive action.
- **Secure by default** — admin login gates registration and all delete/reset operations.
- **Deployable anywhere** — runs on an ordinary laptop, LAN-accessible, installable as a PWA, optional HTTPS.
- **Thread-safe inference** — shared vision models are lock-serialized so the live scanner and manual uploads never race.

## Tech Stack
| Layer | Technology |
| --- | --- |
| Web / API | Flask 3 (server-rendered UI + REST + SSE + MJPEG streaming) |
| Production server | Waitress (used automatically when installed) |
| Plate recognition (ANPR) | EasyOCR |
| Face detection / recognition | OpenCV YuNet (ONNX) + NumPy encodings |
| Storage | SQLite (single file: `data/veri_drive.db`) |
| Face store | `data/veri_drivers.npz` (generated at runtime) |
| Client | Installable PWA (manifest + icon), responsive UI |

## Getting Started

### Prerequisites
- **Python 3.10 or newer** (developed on Python 3.14)
- A webcam, or an RTSP / IP-camera URL
- Internet access on **first run only** — EasyOCR downloads its recognition models once and caches them locally

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/yashfagull706-afk/Veri--Drive-.git
cd Veri--Drive-

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run
```bash
# Windows (opens the dashboard automatically)
start_veri_drive.bat

# macOS / Linux
./start_veri_drive.sh

# Or directly, on any platform
python veri_drive.py
```

Then open:
- **This PC:** http://127.0.0.1:5000
- **Phone / other devices (same network):** `http://<your-LAN-IP>:5000` — the exact URL is printed in the startup banner.

## First Run & Default Login
On first launch Veri-Drive initializes the database and **seeds demo data** so it is ready to evaluate immediately:
- **Drivers:** Yashfa, Hadia (from `data/seed/drivers/`)
- **Vehicles:** `BDE-759` (Car 1), `AYY-911` (Car 2)

Seeding runs **only when the database is empty**, so your own registrations and deletions are never overwritten.

**Admin login (required to register or delete):**

| Field | Default |
| --- | --- |
| Username | `admin` |
| Password | `veridrive2026` |

> **Security:** change this immediately for any real deployment. Set the environment variables `VERIDRIVE_USER` and `VERIDRIVE_PASS` before starting the server. The app prints a warning while the default password is in use.

## Using the System
1. **Register** (`/register`, login required) — add drivers (a clear, single-face photo) and vehicles (a plate photo; the plate is OCR-assisted and editable).
2. **Live Gate** (`/live`) — watch the camera stream while the system scans automatically, or upload a photo to process a single event.
3. **Dashboard** (`/`) — live counts of active vehicles, drivers, open trips, and recent alarms.
4. **Logs** (`/logs`) — search, paginate, export CSV, and (with admin login) delete records across the Gate / Alarms / Driver / Driver-Car tabs.
5. **Weekly Reports** (`/reports`) — pick any week-ending date, generate, and print a full activity report.

## Configuration
All settings live in [`config.py`](config.py). Highlights:

| Setting | Default | Purpose |
| --- | --- | --- |
| `CAMERA_SOURCE` | `0` | Webcam index, or an RTSP / IP-cam URL (env: `VERIDRIVE_CAMERA`) |
| `SCAN_COOLDOWN` | `3` | Seconds between automatic scans |
| `EVENT_GAP` | `15` | Suppresses re-logging the same plate within N seconds |
| `ALARM_GAP` | `8` | Suppresses duplicate alarms within N seconds |
| `ANPR_CONFIDENCE_THRESHOLD` | `0.25` | Minimum OCR confidence to trust a plate |
| `MIN_PLATE_LENGTH` / `MAX_PLATE_LENGTH` | `4` / `12` | Valid plate-length bounds |
| `MATCH_THRESHOLD` | `0.80` | Plate similarity required to confirm a registered vehicle |
| `FACE_MATCH_THRESHOLD` | `0.50` | Maximum face distance to accept a match |
| `FACE_MATCH_MARGIN` | `0.08` | Best match must beat the next different driver by this margin |
| `WEEK_DAYS` | `7` | Weekly-report window |
| `LOG_PAGE_SIZE` | `25` | Rows per Logs page |
| `BACKUP_KEEP` | `10` | Automatic backups retained |
| `CAPTURE_RETENTION_DAYS` | `30` | Snapshots older than this are auto-deleted |
| `BARRIER_OPEN_SECONDS` | `10` | How long the barrier stays open after authorization |

**Environment variables**

| Variable | Purpose |
| --- | --- |
| `VERIDRIVE_USER` / `VERIDRIVE_PASS` | Override the admin credentials |
| `VERIDRIVE_CAMERA` | Override the camera index / RTSP URL |

**Optional HTTPS:** place a certificate and key at `data/cert.pem` and `data/key.pem`; the server uses them automatically.

## Project Structure
```
Veri-Drive/
├── veri_drive.py            # Flask app: routes, embedded UI, streaming, seeding
├── config.py                # All settings (paths, thresholds, seed data)
├── requirements.txt         # Python dependencies
├── start_veri_drive.bat     # Windows launcher
├── start_veri_drive.sh      # macOS / Linux launcher
├── reset_fresh.py           # Reset to a clean, freshly-seeded state
├── modules/
│   ├── database.py          # SQLite layer: trips, alarms, logs, reports, backups
│   ├── anpr.py              # EasyOCR plate reading (thread-safe)
│   └── face_id.py           # YuNet face detection + recognition (thread-safe)
├── tests/
│   ├── test_anpr.py         # Plate-reading unit tests
│   ├── test_database.py     # Trip / alarm / report / clear unit tests
│   ├── test_face_id.py      # Face-match + margin-rule unit tests
│   └── e2e_sequence.ps1     # End-to-end: register -> entry -> exit -> alarm -> delete
└── data/
    ├── face_detection_yunet_2023mar.onnx   # Face model (included)
    ├── app_icon.png                        # PWA icon (included)
    ├── seed/drivers/                       # Demo seed photos (included)
    ├── veri_drive.db                       # SQLite DB (generated, git-ignored)
    ├── veri_drivers.npz                    # Face encodings (generated, git-ignored)
    ├── backups/                            # Automatic DB backups (git-ignored)
    ├── captures/                           # Event snapshots (git-ignored)
    └── secret_key.txt                      # Flask session key (generated, git-ignored)
```

## Testing
```bash
# Unit tests (ANPR, database, face ID)
python -m unittest discover tests -v

# Full end-to-end sequence (PowerShell): register -> entry -> exit -> alarm -> delete
./tests/e2e_sequence.ps1
```

## Deployment
- **Production server:** Waitress is used automatically when installed (`USE_WAITRESS = True`); the app falls back to the Flask dev server otherwise.
- **LAN access:** the startup banner prints the phone / LAN URL; devices on the same network can open it directly.
- **Install as an app:** on a phone, open the LAN URL and choose "Add to Home Screen" (PWA manifest + icon included).
- **HTTPS / internet:** add `data/cert.pem` + `data/key.pem`, or place the app behind a reverse proxy or a private tunnel (e.g., Tailscale).

## Security & Privacy
- **Secrets are never committed.** `data/secret_key.txt`, all databases (`*.db`), face encodings (`*.npz`), backups, snapshots, and personal driver photos are excluded via [`.gitignore`](.gitignore).
- **Biometric data is sensitive.** Face photos and encodings are stored locally only. The `data/seed/drivers/` demo photos are included solely to make a fresh clone evaluable — **remove them for any real or public deployment.**
- **Change the default admin password** (`VERIDRIVE_USER` / `VERIDRIVE_PASS`) before real use.
- **Backups before destructive actions.** Every delete / clear / reset snapshots the database first; the last `BACKUP_KEEP` are retained in `data/backups/`.
- **Plain HTTP by default** — use HTTPS or a private tunnel beyond a trusted LAN.

## Troubleshooting
| Symptom | Fix |
| --- | --- |
| "No face detected" on a photo | Use a clear, front-facing, single-person image with good lighting. |
| Camera shows nothing | Check `CAMERA_SOURCE` / `VERIDRIVE_CAMERA`; another app may be holding the webcam. |
| First run is slow | EasyOCR is downloading its models once; later runs use the cache. |
| "Using CPU" warning | Expected — the system runs on CPU by design (no GPU required). |
| Port 5000 already in use | Stop the other service, or change the port in `veri_drive.py`. |
| Seed drivers missing after an upgrade | Seeding re-enrolls automatically when the face-encoding version changes. |

---

### Acknowledgements
Built with [EasyOCR](https://github.com/JaidedAI/EasyOCR) for plate recognition and [OpenCV](https://opencv.org/) YuNet for face detection and recognition.
