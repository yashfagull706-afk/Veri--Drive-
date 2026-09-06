---
title: Veri-Drive Gate Security
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 5000
pinned: false
license: other
short_description: AI dual-factor gate security - ANPR plate reading + face recognition
---

# Veri-Drive - Cloud Demo

Veri-Drive is an AI gate-security system that verifies **both** the vehicle
(plate, via EasyOCR ANPR) **and** the driver (face, via OpenCV YuNet) before
opening the barrier. A registered car with a registered driver is logged as an
entry/exit; anything else raises a classified alarm with a captured snapshot.

## About this Space
This is the containerized demo. Cloud containers have **no webcam**, so the
Live Gate runs in **photo-upload mode**: upload a gate photo to process a
single event. The full real-time camera system is the Flask app in the source
repository.

## Default login
- Username: `admin`
- Password: `veridrive2026`

Login is required only to register or delete drivers/vehicles and to clear
logs. All viewing pages stay open. (Set `VERIDRIVE_USER` / `VERIDRIVE_PASS`
under Settings -> Variables and secrets to override for a public Space.)

## Seeded demo data
On first boot the Space seeds two drivers (Yashfa, Hadia) and two vehicles
(`BDE-759`, `AYY-911`) so it is ready to try immediately. Container storage is
ephemeral - data resets when the Space rebuilds or restarts.

## Using a real IP camera
Set the `VERIDRIVE_CAMERA` variable (Settings -> Variables and secrets) to an
RTSP URL, e.g. `rtsp://user:pass@host:554/stream1`, to drive the Live Gate from
a real network camera instead of upload mode.
