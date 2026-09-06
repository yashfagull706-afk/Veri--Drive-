# Stage the Veri-Drive app into this folder so it can be pushed to a
# Hugging Face Space (Docker SDK). Only NON-sensitive files are copied:
# source, the YuNet model, the PWA icon, and the demo seed photos.
# Databases, .npz biometrics, secret_key.txt, captures and backups are NEVER
# copied.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File deploy\hf-space\prepare_space.ps1
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $here '..\..')).Path

Write-Host "Staging Veri-Drive from $root -> $here"

# Core application files
Copy-Item (Join-Path $root 'veri_drive.py')    $here -Force
Copy-Item (Join-Path $root 'config.py')        $here -Force
Copy-Item (Join-Path $root 'requirements.txt') $here -Force

# modules package (anpr.py, database.py, face_id.py, __init__.py)
New-Item -ItemType Directory -Force -Path (Join-Path $here 'modules') | Out-Null
Copy-Item (Join-Path $root 'modules\*.py') (Join-Path $here 'modules') -Force

# data assets the container needs: face model, PWA icon, demo seed photos
New-Item -ItemType Directory -Force -Path (Join-Path $here 'data') | Out-Null
Copy-Item (Join-Path $root 'data\face_detection_yunet_2023mar.onnx') (Join-Path $here 'data') -Force
if (Test-Path (Join-Path $root 'data\app_icon.png')) {
  Copy-Item (Join-Path $root 'data\app_icon.png') (Join-Path $here 'data') -Force
}
if (Test-Path (Join-Path $root 'data\seed')) {
  Copy-Item (Join-Path $root 'data\seed') (Join-Path $here 'data') -Recurse -Force
}

Write-Host ""
Write-Host "Staged. Next steps:"
Write-Host "  cd '$here'"
Write-Host "  git init"
Write-Host "  git add ."
Write-Host "  git commit -m 'Veri-Drive Hugging Face Space'"
Write-Host "  git remote add space https://huggingface.co/spaces/<YOUR_USER>/<YOUR_SPACE>"
Write-Host "  git push -f space main"
