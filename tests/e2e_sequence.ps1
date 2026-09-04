# Veri-Drive manual test sequence (rule 5):
# register -> entry -> exit -> unknown plate alarm -> delete driver (+ npz check)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$base = "http://127.0.0.1:5000"
$data = "c:\Users\HP\Desktop\fleet_ai_system\data"
$pass = 0; $fail = 0
function Check($name, $cond) {
    if ($cond) { $script:pass++; Write-Host "PASS  $name" }
    else       { $script:fail++; Write-Host "FAIL  $name" }
}

function Send-Multipart($url, $fields, $files, $session) {
    $boundary = [guid]::NewGuid().ToString()
    $ms = New-Object System.IO.MemoryStream
    $enc = [System.Text.Encoding]::UTF8
    function W($s) { $b = $enc.GetBytes($s); $ms.Write($b, 0, $b.Length) }
    foreach ($k in $fields.Keys) {
        W "--$boundary`r`nContent-Disposition: form-data; name=`"$k`"`r`n`r`n$($fields[$k])`r`n"
    }
    foreach ($k in $files.Keys) {
        $path = $files[$k]
        $fname = Split-Path $path -Leaf
        W "--$boundary`r`nContent-Disposition: form-data; name=`"$k`"; filename=`"$fname`"`r`nContent-Type: image/jpeg`r`n`r`n"
        $fb = [System.IO.File]::ReadAllBytes($path)
        $ms.Write($fb, 0, $fb.Length)
        W "`r`n"
    }
    W "--$boundary--`r`n"
    $body = $ms.ToArray(); $ms.Dispose()
    $ct = "multipart/form-data; boundary=$boundary"
    if ($session) { return Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType $ct -WebSession $session }
    return Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType $ct
}

Write-Host "`n--- 0. auth wall ---"
try {
    Send-Multipart "$base/api/register/driver" @{name="Ghost"} @{} $null | Out-Null
    Check "register without login rejected" $false
} catch {
    Check "register without login rejected (HTTP $([int]$_.Exception.Response.StatusCode))" ([int]$_.Exception.Response.StatusCode -eq 401)
}
$regPage = Invoke-WebRequest "$base/register" -UseBasicParsing -MaximumRedirection 0 -ErrorAction SilentlyContinue
Check "/register redirects to login" ($regPage.StatusCode -eq 302 -or $regPage.Headers.Location -match "login")

Write-Host "`n--- 1. login ---"
$login = Invoke-RestMethod "$base/api/login" -Method Post -Body @{username="admin"; password="veridrive2026"} -SessionVariable sess
Check "admin login ok" ($login.ok -eq $true)

# best-effort cleanup so this script is re-runnable after an aborted run
try { Invoke-RestMethod "$base/api/delete/driver" -Method Post -Body @{name="TestE2E"} -WebSession $sess | Out-Null } catch {}
try { Invoke-RestMethod "$base/api/delete/vehicle" -Method Post -Body @{plate="DEF9999"} -WebSession $sess | Out-Null } catch {}

Write-Host "`n--- 2. register driver (TestE2E) ---"
$r = Send-Multipart "$base/api/register/driver" @{name="TestE2E"} @{photo="$data\seed\drivers\Yashfa.jpg"} $sess
Check "driver registered" ($r.ok -eq $true)
Write-Host "      $($r.message)"
$veh = Invoke-RestMethod "$base/api/vehicles" -WebSession $sess
Check "vehicle BDE-759 registered (seeded)" (($veh | ForEach-Object { $_.plate }) -contains "BDE-759")
# plate_def9999.jpg reads as DEF9999 - register it so entry/exit can be tested
$v = Send-Multipart "$base/api/register/vehicle" @{plate="DEF9999"; label="E2E Test Car"} @{photo="$data\test_images\plate_def9999.jpg"} $sess
Check "test vehicle DEF9999 registered" ($v.ok -eq $true)

Write-Host "`n--- 3. gate pass: entry ---"
$e1 = Send-Multipart "$base/api/gate_process" @{} @{photo="$data\test_images\plate_def9999.jpg"} $sess
Write-Host "      plate=$($e1.plate) event=$($e1.event.event_type) barrier=$($e1.barrier) msg=$($e1.message)"
Check "entry logged for DEF9999" ($e1.event.event_type -eq "entry")
Check "entry marked inferred" ($e1.event.inferred -eq $true)
Check "barrier stays closed (no driver face in plate-only photo)" ($e1.barrier -eq "CLOSED")

Write-Host "`n--- 4. gate pass: exit (after EVENT_GAP) ---"
Start-Sleep -Seconds 17
$e2 = Send-Multipart "$base/api/gate_process" @{} @{photo="$data\test_images\plate_def9999.jpg"} $sess
Write-Host "      plate=$($e2.plate) event=$($e2.event.event_type) msg=$($e2.message)"
Check "exit logged for DEF9999" ($e2.event.event_type -eq "exit")

Write-Host "`n--- 5. unknown plate alarm ---"
Start-Sleep -Seconds 10
$a1 = Send-Multipart "$base/api/gate_process" @{} @{photo="$data\test_images\plate_xyz5678.jpg"} $sess
Write-Host "      plate=$($a1.plate) alarm=$($a1.alarm.kind) msg=$($a1.message)"
Check "unknown plate raised alarm" ($null -ne $a1.alarm -or $a1.message -match "alarm|unknown|ALERT")
Check "barrier stayed closed for unknown" ($a1.barrier -eq "CLOSED")

Write-Host "`n--- 6. delete driver + npz cleanup ---"
$before = (Get-Item "$data\veri_drivers.npz" -ErrorAction SilentlyContinue).LastWriteTime
$d = Send-Multipart "$base/api/delete/driver" @{name="TestE2E"} @{} $sess
Check "delete driver ok" ($d.ok -eq $true)
Start-Sleep -Seconds 2
$drivers = Invoke-RestMethod "$base/api/drivers" -WebSession $sess
Check "TestE2E gone from driver list" ((($drivers | ForEach-Object { $_.name }) -contains "TestE2E") -eq $false)
if (Test-Path "$data\veri_drivers.npz") {
    # read names.npy via numpy (raw-text parsing fails: numpy stores strings
    # in a padded unicode dtype, so names are not plain ASCII substrings)
    $names = (py -c "import numpy as np; d=np.load(r'$data\veri_drivers.npz', allow_pickle=True); print('|'.join(str(x) for x in d['names']))" 2>$null)
    Write-Host "      npz names: $names"
    Check "npz no longer contains TestE2E" ($names -notmatch "TestE2E")
    Check "npz still contains seeded drivers" ($names -match "Yashfa")
} else {
    Check "npz exists after delete (seeded drivers remain)" $false
}
# cleanup: remove the test vehicle
$dv = Invoke-RestMethod "$base/api/delete/vehicle" -Method Post -Body @{plate="DEF9999"} -WebSession $sess
Check "test vehicle DEF9999 removed" ($dv.ok -eq $true)

Write-Host "`n--- 7. P3 spot checks ---"
$logs = Invoke-RestMethod "$base/api/logs/gate?page=1" -WebSession $sess
Check "paged logs endpoint works" ($null -ne $logs.rows -and $logs.page -eq 1)
$csv = Invoke-WebRequest "$base/api/export/gate.csv" -UseBasicParsing -WebSession $sess
Check "CSV export works" ($csv.Headers["Content-Type"] -match "csv" -and $csv.Content.Length -gt 10)
# NOTE: /api/events is SSE (never ends) - verified separately, not here.

Write-Host "`n=== RESULT: $pass passed, $fail failed ==="
