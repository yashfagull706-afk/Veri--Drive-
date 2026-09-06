# Veri-Drive - Hugging Face Space (Docker SDK) / generic container image.
# Runs the Flask + Waitress gate-security app. Cloud containers have NO webcam,
# so the Live Gate falls back to photo-upload mode; set VERIDRIVE_CAMERA to an
# RTSP/IP-camera URL to drive it from a real network camera.

FROM python:3.11-slim

# OpenCV (cv2) needs libGL + glib at runtime in a slim image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    HOME=/root

WORKDIR /app

# Install CPU-only PyTorch FIRST so EasyOCR does not pull the ~2 GB CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the EasyOCR English model at build time so the first plate read
# is instant (avoids a runtime download stall in front of judges).
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

# Application code + assets (staged into this folder by prepare_space.ps1).
COPY . .

EXPOSE 5000
CMD ["python", "veri_drive.py"]
